"""Explicit, provider-neutral attachment and proposal publication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

from filelock import FileLock

from . import git_store


class RemoteError(ValueError):
    """An attachment or remote operation could not be completed safely."""


class _Timeout(RemoteError):
    pass


_TIMEOUT = 60
_REMOTE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_MODES = {"read", "propose", "write"}
_PROTOCOLS = ("file", "https", "ssh")


def _transport_dir(root: Path) -> tuple[Path, Path]:
    private = root / ".memory-graph"
    if private.is_symlink():
        raise RemoteError("Transport directory must not be a symlink")
    private.mkdir(mode=0o700, exist_ok=True)
    if private.stat().st_mode & 0o077:
        raise RemoteError("Transport directory must be private")
    path = private / "transport.git"
    objects = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-path", "objects"))
    object_format = _git(root, "rev-parse", "--show-object-format")
    if object_format not in {"sha1", "sha256"}:
        raise RemoteError("Unsupported Git object format")
    config = (
        "[core]\n\trepositoryformatversion = "
        + ("1" if object_format == "sha256" else "0")
        + "\n\tbare = true\n"
        + ("\n[extensions]\n\tobjectformat = sha256\n" if object_format == "sha256" else "")
    )
    if not path.exists():
        path.mkdir(mode=0o700)
        (path / "objects").mkdir(mode=0o700)
        (path / "refs").mkdir(mode=0o700)
        (path / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (path / "config").write_text(config, encoding="utf-8")
        (path / "config").chmod(0o400)
        path.chmod(0o500)
    if (
        path.is_symlink()
        or path.stat().st_mode & 0o277
        or (path / "config").is_symlink()
        or (path / "config").stat().st_mode & 0o277
        or (path / "config").read_text(encoding="utf-8") != config
    ):
        raise RemoteError("Unsafe transport repository configuration")
    return path, objects


def _git(
    root: Path,
    *args: str,
    timeout: int = _TIMEOUT,
    transport: bool = False,
    credentials: tuple[tuple[str, str], ...] = (),
) -> str:
    env = os.environ.copy()
    for key in list(env):
        if key in {
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_COMMON_DIR",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG",
            "GIT_CONFIG_PARAMETERS",
            "GIT_REPLACE_REF_BASE",
        } or key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            env.pop(key, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    allowed = env.get("GIT_ALLOW_PROTOCOL")
    env["GIT_ALLOW_PROTOCOL"] = ":".join(
        name for name in _PROTOCOLS if allowed is None or name in allowed.split(":")
    )
    if transport:
        directory, objects = _transport_dir(root)
        env = {key: value for key, value in env.items() if not key.upper().startswith("GIT_")}
        for key in list(env):
            if key.lower() in {
                "http_proxy",
                "https_proxy",
                "all_proxy",
                "no_proxy",
                "ssl_cert_file",
                "ssl_cert_dir",
                "sslkeylogfile",
                "curl_ca_bundle",
                "ssh_askpass",
            }:
                env.pop(key, None)
        env.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_SYSTEM": os.devnull,
                "GIT_OBJECT_DIRECTORY": str(objects),
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_ALLOW_PROTOCOL": ":".join(
                    name for name in _PROTOCOLS if allowed is None or name in allowed.split(":")
                ),
                "GIT_SSH_COMMAND": (
                    f"ssh -F {os.devnull} -o StrictHostKeyChecking=yes "
                    "-o ProxyCommand=none -o ProxyJump=none"
                ),
                "GIT_SSH_VARIANT": "ssh",
            }
        )
        env["GIT_CONFIG_COUNT"] = str(len(credentials))
        for index, (key, value) in enumerate(credentials):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
        command = ["git", "--git-dir", str(directory)]
    else:
        env = git_store._environment()
        command = git_store._command(root, [])
    command += [
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "maintenance.auto=false",
        "-c",
        "gc.auto=0",
        *(["-c", "core.alternateRefsCommand=true"] if transport else []),
        "-c",
        "http.sslVerify=true",
        "-c",
        "http.followRedirects=false",
        "-c",
        "http.proxy=",
        "-c",
        "push.gpgSign=false",
        "-c",
        "push.followTags=false",
        *args,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _Timeout("Git operation timed out; remote outcome may be unknown") from exc
    except OSError as exc:
        raise RemoteError("Could not run Git") from exc
    if result.returncode:
        # Git's diagnostics can contain remote URLs and credentials.
        if not transport and b"unknown option: --no-lazy-fetch" in result.stderr:
            raise RemoteError("Git must support --no-lazy-fetch (upstream Git 2.45 or newer)")
        raise RemoteError(f"Git {args[0]} failed (exit {result.returncode})")
    try:
        return result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RemoteError("Git returned invalid output") from exc


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _uuid(value: object) -> str:
    if not isinstance(value, str):
        raise RemoteError("Expected a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise RemoteError("Expected a canonical UUID") from exc
    if str(parsed) != value or parsed.version != 4:
        raise RemoteError("Expected a canonical UUID")
    return value


def _remote(root: Path, remote: object) -> tuple[str, bool]:
    if not isinstance(remote, str) or not _REMOTE_NAME.fullmatch(remote):
        raise RemoteError("Invalid remote name")
    if remote not in _git(root, "remote").splitlines():
        raise RemoteError("Selected Git remote does not exist")
    _git(root, "check-ref-format", f"refs/remotes/{remote}/main")
    local_config = _git(root, "config", "--null", "--local", "--includes", "--list")
    for item in local_config.split("\0"):
        key = item.partition("\n")[0].lower()
        if key in {"core.sshcommand", "core.askpass", "credential.helper"} or (
            key.startswith("credential.") and key.endswith(".helper")
        ):
            raise RemoteError("Unsafe repository-local Git command configuration")
    all_config = _git(root, "config", "--null", "--includes", "--list")
    keys = (item.partition("\n")[0].lower() for item in all_config.split("\0"))
    if f"remote.{remote.lower()}.vcs" in keys:
        raise RemoteError("Git remote helper overrides are not supported")
    return _remote_url(root, remote)


def _remote_url(root: Path, remote: str) -> tuple[str, bool]:
    config = _git(root, "config", "--null", "--local", "--no-includes", "--list")
    values: dict[str, list[str]] = {"url": [], "pushurl": []}
    for item in config.split("\0"):
        key, separator, value = item.partition("\n")
        for field, entries in values.items():
            if separator and key == f"remote.{remote}.{field}":
                entries.append(value)
    if len(values["url"]) != 1 or len(values["pushurl"]) > 1:
        raise RemoteError("The selected remote must have one explicit local URL")
    url, local = _endpoint(root, values["url"][0])
    if values["pushurl"] and _endpoint(root, values["pushurl"][0]) != (url, local):
        raise RemoteError("Push destination differs from the attached endpoint")
    return url, local


def _endpoint(root: Path, url: str) -> tuple[str, bool]:
    if not url or url != url.strip() or any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise RemoteError("Invalid remote endpoint")
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9+.-]*::", url):
        raise RemoteError("Git remote helper transports are not supported")
    if "://" in url:
        try:
            parsed = urlsplit(url)
            scheme = parsed.scheme
            port = parsed.port
        except ValueError as exc:
            raise RemoteError("Invalid remote endpoint") from exc
        if scheme not in _PROTOCOLS:
            raise RemoteError("Git remote transport is not supported")
        if parsed.password is not None or parsed.query or parsed.fragment:
            raise RemoteError("Remote endpoints must not contain credentials or query parameters")
        if scheme == "file":
            if parsed.hostname not in (None, "", "localhost") or parsed.username or port:
                raise RemoteError("Invalid local remote endpoint")
            return str(Path(unquote(parsed.path)).resolve(strict=True)), True
        if not parsed.hostname or parsed.hostname.startswith("-"):
            raise RemoteError("Invalid remote endpoint")
        return url, False
    is_scp = re.match(r"^[^/:]+:", url) and not re.match(r"^[A-Za-z]:[\\/]", url)
    if is_scp:
        host, _, path = url.partition(":")
        if host.startswith("-") or not path:
            raise RemoteError("Invalid remote endpoint")
        return url, False
    try:
        return str((root / url).resolve(strict=True)), True
    except OSError as exc:
        raise RemoteError("Local remote endpoint does not exist") from exc


def _endpoint_id(endpoint: str) -> str:
    return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()


def _credentials(root: Path, endpoint: str) -> tuple[tuple[str, str], ...]:
    if not endpoint.startswith("https://"):
        return ()
    home = Path.home()
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config")))
    paths = (xdg / "git" / "config", home / ".gitconfig")
    env = git_store._environment()
    result: list[tuple[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            process = subprocess.run(
                git_store._command(
                    root, ["config", "--file", str(path), "--no-includes", "--null", "--list"]
                ),
                capture_output=True,
                env=env,
                timeout=_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RemoteError("Could not read trusted Git credential configuration") from exc
        if process.returncode:
            if b"unknown option: --no-lazy-fetch" in process.stderr:
                raise RemoteError("Git must support --no-lazy-fetch (upstream Git 2.45 or newer)")
            raise RemoteError("Could not read trusted Git credential configuration")
        try:
            entries = process.stdout.decode("utf-8").split("\0")
        except UnicodeDecodeError as exc:
            raise RemoteError("Invalid trusted Git credential configuration") from exc
        for entry in entries:
            key, separator, value = entry.partition("\n")
            lower = key.lower()
            if separator and (
                lower in {"credential.helper", "credential.username", "credential.usehttppath"}
                or (
                    lower.startswith("credential.")
                    and lower.endswith((".helper", ".username", ".usehttppath"))
                )
            ):
                result.append((key, value))
    return tuple(result)


def _ref(remote: str) -> str:
    return f"refs/remotes/{remote}/main"


def _snapshot(root: Path, ref: str) -> tuple[str, dict[str, bytes], str]:
    try:
        # service imports this module, so defer importing its full snapshot validator.
        from . import service

        commit, files = git_store.load_snapshot(root, ref=ref)
        manifest = json.loads(files["store.json"], object_pairs_hook=_object)
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"schema", "id"}
            or type(manifest.get("schema")) is not int
        ):
            raise ValueError("invalid manifest")
        if manifest["schema"] != 1:
            raise ValueError("unsupported manifest schema")
        store_id = _uuid(manifest.get("id"))
        snapshot, _ = service.parse_snapshot(commit, files, None)
        service._require_evidence(root, snapshot)
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, git_store.GitStoreError) as exc:
        raise RemoteError("Invalid or unavailable store snapshot") from exc
    if not _OID.fullmatch(commit):
        raise RemoteError("Git returned an invalid commit ID")
    return commit, files, store_id


def _config_path(root: Path) -> Path:
    return root / ".memory-graph" / "config.json"


def _root(root: Path) -> Path:
    try:
        root = Path(root).resolve(strict=True)
    except OSError as exc:
        raise RemoteError("Store root does not exist") from exc
    if not root.is_dir():
        raise RemoteError("Store root is not a directory")
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise RemoteError("Store root must be the repository root")
    return root


def get_config(root: Path) -> dict | None:
    path = _config_path(Path(root))
    try:
        data = json.loads(path.read_bytes(), object_pairs_hook=_object)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise RemoteError("Invalid local attachment configuration") from exc
    if not isinstance(data, dict):
        raise RemoteError("Invalid local attachment configuration")
    keys = {"remote", "mode", "store_id", "fetched_at", "fetched_commit", "last_fetch"}
    if set(data) == keys:
        raise RemoteError("Attachment has no pinned endpoint; explicitly reattach")
    if set(data) != keys | {"endpoint_id"}:
        raise RemoteError("Invalid local attachment configuration")
    if not isinstance(data["remote"], str) or not _REMOTE_NAME.fullmatch(data["remote"]):
        raise RemoteError("Invalid local attachment configuration")
    if not isinstance(data["mode"], str) or data["mode"] not in _MODES:
        raise RemoteError("Invalid local attachment configuration")
    _uuid(data["store_id"])
    if not isinstance(data["endpoint_id"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", data["endpoint_id"]
    ):
        raise RemoteError("Invalid local attachment configuration")
    if data["fetched_at"] is not None and not isinstance(data["fetched_at"], str):
        raise RemoteError("Invalid local attachment configuration")
    if data["last_fetch"] != data["fetched_at"]:
        raise RemoteError("Invalid local attachment configuration")
    if data["fetched_commit"] is not None and (
        not isinstance(data["fetched_commit"], str) or not _OID.fullmatch(data["fetched_commit"])
    ):
        raise RemoteError("Invalid local attachment configuration")
    return data


def _write_config(root: Path, data: dict) -> None:
    path = _config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".config-{uuid.uuid4()}.tmp"
    try:
        with temp.open("x", encoding="utf-8") as handle:
            os.chmod(temp, 0o600)
            json.dump(data, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _lock(root: Path) -> FileLock:
    directory = root / ".memory-graph"
    if directory.is_symlink():
        raise RemoteError("Local configuration directory must not be a symlink")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    return FileLock(str(directory / "write.lock"), timeout=_TIMEOUT)


def _attached(root: Path) -> tuple[dict, str, bool]:
    config = get_config(root)
    if config is None:
        raise RemoteError("No remote is attached")
    endpoint, local = _remote(root, config["remote"])
    if _endpoint_id(endpoint) != config["endpoint_id"]:
        raise RemoteError("Remote endpoint changed; explicitly reattach")
    return config, endpoint, local


def _validate_pair(root: Path, config: dict, ref: str) -> str:
    _, _, local_id = _snapshot(root, "refs/heads/main")
    commit, _, remote_id = _snapshot(root, ref)
    if local_id != remote_id or config["store_id"] != local_id:
        raise RemoteError("Store identity does not match the attached remote")
    return commit


def attach(root: Path, remote: str, mode: str) -> dict:
    root = _root(root)
    if not isinstance(mode, str) or mode not in _MODES:
        raise RemoteError("Mode must be read, propose, or write")
    with _lock(root):
        endpoint, _ = _remote(root, remote)
        _, _, local_id = _snapshot(root, "refs/heads/main")
        commit, _, remote_id = _snapshot(root, _ref(remote))
        if local_id != remote_id:
            raise RemoteError("Store identity does not match the selected remote")
        config = {
            "remote": remote,
            "mode": mode,
            "store_id": local_id,
            "fetched_at": None,
            "last_fetch": None,
            "fetched_commit": commit,
            "endpoint_id": _endpoint_id(endpoint),
        }
        _write_config(root, config)
        return config


def fetch(root: Path) -> dict:
    root = _root(root)
    with _lock(root):
        config, endpoint, local = _attached(root)
        credentials = _credentials(root, endpoint)
        remote = config["remote"]
        ref = _ref(remote)
        _, _, local_id = _snapshot(root, "refs/heads/main")
        if local_id != config["store_id"]:
            raise RemoteError("Store identity does not match the attachment")
        old_commit = _git(root, "for-each-ref", "--format=%(objectname)", ref)
        if old_commit and not _OID.fullmatch(old_commit):
            raise RemoteError("Invalid tracking ref")
        temporary_ref = f"refs/memory-graph/fetch/{uuid.uuid4()}"
        fetched = None
        temporary_created = False
        try:
            upload_pack = (
                "git -c core.alternateRefsCommand=true upload-pack" if local else "git-upload-pack"
            )
            advertised = _git(
                root,
                "ls-remote",
                f"--upload-pack={upload_pack}",
                "--refs",
                "--",
                endpoint,
                "refs/heads/main",
                transport=True,
                credentials=credentials,
            ).splitlines()
            if len(advertised) != 1:
                raise RemoteError("Remote main ref is missing or ambiguous")
            fields = advertised[0].split("\t")
            if len(fields) != 2 or not _OID.fullmatch(fields[0]) or fields[1] != "refs/heads/main":
                raise RemoteError("Remote main ref is invalid")
            fetched = fields[0]
            _git(
                root,
                "fetch",
                "--no-tags",
                "--no-recurse-submodules",
                "--no-write-fetch-head",
                "--refmap=",
                f"--upload-pack={upload_pack}",
                "--",
                endpoint,
                fetched,
                transport=True,
                credentials=credentials,
            )
            _git(root, "update-ref", temporary_ref, fetched, "0" * len(fetched))
            temporary_created = True
            commit = _validate_pair(root, config, temporary_ref)
            _git(root, "update-ref", ref, commit, old_commit or "0" * len(commit))
            config["fetched_at"] = datetime.now(UTC).isoformat()
            config["last_fetch"] = config["fetched_at"]
            config["fetched_commit"] = commit
            try:
                _write_config(root, config)
            except OSError as exc:
                try:
                    if old_commit:
                        _git(root, "update-ref", ref, old_commit, commit)
                    else:
                        _git(root, "update-ref", "-d", ref, commit)
                except RemoteError as rollback_error:
                    raise RemoteError(
                        "Could not save fetch configuration or restore tracking ref"
                    ) from rollback_error
                raise RemoteError("Could not save fetch configuration") from exc
            return config
        finally:
            if temporary_created and fetched is not None:
                _git(root, "update-ref", "-d", temporary_ref, fetched)


def status(root: Path) -> dict:
    root = _root(root)
    config, _, _ = _attached(root)
    commit = _validate_pair(root, config, _ref(config["remote"]))
    branches = _git(
        root,
        "for-each-ref",
        "--format=%(refname) %(objectname)",
        "refs/heads/memory-graph/",
    )
    return {
        **config,
        "observed_commit": commit,
        "freshness": "unknown",
        "pending": [
            {"ref": line.split(" ", 1)[0], "commit": line.split(" ", 1)[1]}
            for line in branches.splitlines()
            if " " in line
        ],
    }


def publish(root: Path, operation_id: str) -> dict:
    """Publish a proposal; local endpoints fail closed rather than bypass server policy."""
    root = _root(root)
    operation_id = _uuid(operation_id)
    branch = f"refs/heads/memory-graph/{operation_id}"
    with _lock(root):
        config, endpoint, local = _attached(root)
        if config["mode"] == "read":
            raise RemoteError("Read mode does not permit publication")
        if local:
            raise RemoteError(
                "Local publication is unavailable: server policy cannot be enforced safely"
            )
        credentials = _credentials(root, endpoint)
        _validate_pair(root, config, _ref(config["remote"]))
        commit, files, store_id = _snapshot(root, branch)
        if store_id != config["store_id"]:
            raise RemoteError("Proposal store identity does not match attachment")
        if f"operations/{operation_id}.json" not in files:
            raise RemoteError("Proposal does not contain its operation receipt")
        result = {
            "remote": config["remote"],
            "ref": branch,
            "commit": commit,
            "review_created": False,
            "accepted": False,
        }
        try:
            found = _git(
                root,
                "ls-remote",
                "--upload-pack=git-upload-pack",
                "--refs",
                "--",
                endpoint,
                branch,
                transport=True,
                credentials=credentials,
            )
        except _Timeout:
            return {**result, "status": "unknown"}
        if found:
            lines = found.splitlines()
            if len(lines) != 1 or lines[0].split("\t") != [commit, branch]:
                raise RemoteError("Remote proposal ref conflicts with the local proposal")
            return {**result, "status": "reconciled"}
        try:
            # The empty expected value makes this a create-only compare-and-swap.
            output = _git(
                root,
                "push",
                "--no-verify",
                "--porcelain",
                f"--force-with-lease={branch}:",
                "--receive-pack=git-receive-pack",
                "--no-follow-tags",
                "--no-signed",
                "--",
                endpoint,
                f"{commit}:{branch}",
                transport=True,
                credentials=credentials,
            )
        except _Timeout:
            return {**result, "status": "unknown"}
        except RemoteError:
            # Reconcile a possible concurrent creation or ambiguous transport failure.
            try:
                found = _git(
                    root,
                    "ls-remote",
                    "--upload-pack=git-upload-pack",
                    "--refs",
                    "--",
                    endpoint,
                    branch,
                    transport=True,
                    credentials=credentials,
                )
            except RemoteError:
                return {**result, "status": "unknown"}
            if found:
                lines = found.splitlines()
                if len(lines) != 1 or lines[0].split("\t") != [commit, branch]:
                    raise RemoteError("Remote proposal ref conflicts with the local proposal")
                return {**result, "status": "reconciled"}
            return {**result, "status": "unknown"}
        updates = [line.split("\t") for line in output.splitlines() if "\t" in line]
        if len(updates) != 1 or len(updates[0]) != 3 or updates[0][1] != f"{commit}:{branch}":
            return {**result, "status": "unknown"}
        if updates[0][0] == "=":
            return {**result, "status": "reconciled"}
        return {**result, "status": "proposed" if updates[0][0] == "*" else "unknown"}
