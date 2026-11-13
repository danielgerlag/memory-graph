"""Git object storage for memory stores.

Publishing advances a ref only. It never refreshes an existing checkout or
index; callers must report that distinction and coordinate their own writes.
Git history is not an authenticated or immutable audit log.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import uuid
from collections import Counter
from pathlib import Path


class GitStoreError(RuntimeError):
    """The repository, snapshot, or requested Git operation is invalid."""


class StaleRefError(GitStoreError):
    """The publication ref no longer has the expected value."""


class RefNotFound(ValueError):
    """A requested Git ref does not exist."""


_TIMEOUT = 30
_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_MAX_BLOB_BYTES = 8 * 1024 * 1024
_MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
_MAX_TREE_ENTRIES = 200_000
_MAX_TREE_LISTING_BYTES = 64 * 1024 * 1024
_METADATA_BATCH_SIZE = 4096


def _environment(index: Path | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_NO_LAZY_FETCH="1",
        GIT_ALLOW_PROTOCOL="",
        GIT_ATTR_NOSYSTEM="1",
        GIT_AUTHOR_NAME="Memory Graph",
        GIT_AUTHOR_EMAIL="memory-graph@localhost",
        GIT_COMMITTER_NAME="Memory Graph",
        GIT_COMMITTER_EMAIL="memory-graph@localhost",
        LC_ALL="C",
    )
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    return env


def _command(root: Path, args: list[str]) -> list[str]:
    return [
        "git",
        "--no-lazy-fetch",
        "-C",
        str(root),
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.ext.allow=never",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.attributesFile={os.devnull}",
        "-c",
        f"core.alternateRefsCommand={os.devnull}",
        "-c",
        "commit.gpgSign=false",
        "-c",
        "core.sparseCheckout=false",
        "-c",
        "core.splitIndex=false",
        "-c",
        "gc.auto=0",
        "-c",
        "maintenance.auto=false",
        *args,
    ]


def _run(
    root: Path,
    args: list[str],
    *,
    data: bytes | None = None,
    index: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            _command(root, args),
            input=data,
            capture_output=True,
            env=_environment(index),
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitStoreError(f"Git command failed or timed out: {args[0]}: {exc}") from exc
    if check and result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        if "unknown option: --no-lazy-fetch" in detail:
            raise GitStoreError("Git must support --no-lazy-fetch (upstream Git 2.45 or newer)")
        raise GitStoreError(f"Git {args[0]} failed: {detail}")
    return result


def _bounded_output(
    root: Path,
    args: list[str],
    max_bytes: int,
    *,
    data: bytes | None = None,
    max_records: int | None = None,
) -> bytes:
    """Drain Git output with fixed bounds, including while its stdin is being written."""
    try:
        process = subprocess.Popen(
            _command(root, args),
            stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_environment(),
        )
    except OSError as exc:
        raise GitStoreError(f"Git command failed: {args[0]}: {exc}") from exc

    stderr = process.stderr
    assert process.stdout is not None and stderr is not None
    errors = bytearray()

    def drain_stderr() -> None:
        try:
            while chunk := stderr.read(65536):
                if len(errors) < 65536:
                    errors.extend(chunk[: 65536 - len(errors)])
        except OSError:
            pass

    def write_stdin() -> None:
        assert process.stdin is not None and data is not None
        try:
            with process.stdin:
                process.stdin.write(data)
        except (BrokenPipeError, OSError):
            pass

    def kill_process() -> None:
        try:
            process.kill()
        except ProcessLookupError:
            pass

    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        kill_process()

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()
    stdin_thread = None
    if data is not None:
        stdin_thread = threading.Thread(target=write_stdin, daemon=True)
        stdin_thread.start()
    timer = threading.Timer(_TIMEOUT, expire)
    timer.daemon = True
    timer.start()
    output = bytearray()
    count = 0
    failure = None
    try:
        while chunk := process.stdout.read(min(65536, max_bytes - len(output) + 1)):
            if len(output) + len(chunk) > max_bytes:
                failure = f"Git {args[0]} output exceeds the {max_bytes}-byte limit"
                kill_process()
                break
            if max_records is not None:
                count += chunk.count(b"\0")
                if count > max_records:
                    failure = f"Git tree exceeds the {max_records} entry limit"
                    kill_process()
                    break
            output.extend(chunk)
    finally:
        process.stdout.close()
        process.wait()
        timer.cancel()
        stderr_thread.join()
        if stdin_thread is not None:
            stdin_thread.join()
        stderr.close()
    if timed_out.is_set():
        raise GitStoreError(f"Git command timed out: {args[0]}")
    if failure:
        raise GitStoreError(failure)
    if process.returncode:
        detail = errors.decode("utf-8", "replace").strip()
        raise GitStoreError(f"Git {args[0]} failed: {detail}")
    return bytes(output)


def _root(root: Path) -> Path:
    try:
        root = Path(root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GitStoreError(f"Invalid store root: {exc}") from exc
    if not root.is_dir():
        raise GitStoreError("Store root must be a directory")
    result = _run(root, ["rev-parse", "--show-toplevel"])
    try:
        top = Path(os.fsdecode(result.stdout.removesuffix(b"\n"))).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GitStoreError(f"Invalid Git top-level directory: {exc}") from exc
    if top != root:
        raise GitStoreError("Store root must be exactly the Git top-level directory")
    return root


def _git_dir(root: Path) -> Path:
    result = _run(root, ["rev-parse", "--absolute-git-dir"])
    return Path(os.fsdecode(result.stdout.removesuffix(b"\n")))


def _hash_length(root: Path) -> int:
    fmt = _run(root, ["rev-parse", "--show-object-format"]).stdout.strip()
    if fmt == b"sha1":
        return 40
    if fmt == b"sha256":
        return 64
    raise GitStoreError("Unsupported Git object format")


def _full_oid(value: object, length: int, *, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or not _OID.fullmatch(value) or len(value) != length:
        raise GitStoreError("Expected a full lowercase Git object ID")
    if not allow_zero and value == "0" * length:
        raise GitStoreError("Zero is not a commit object ID")
    return value


def _ref(root: Path, ref: object) -> str:
    if not isinstance(ref, str) or not ref.startswith("refs/") or "\x00" in ref:
        raise GitStoreError("Expected a full Git ref")
    if _run(root, ["check-ref-format", ref], check=False).returncode:
        raise GitStoreError(f"Invalid Git ref: {ref!r}")
    return ref


def _missing_ref(root: Path, ref: str) -> bool:
    result = _run(root, ["show-ref", "--exists", ref], check=False)
    if result.returncode == 0:
        return False
    if result.returncode == 2:
        return True
    if result.returncode == 129:
        raise GitStoreError("Git must support show-ref --exists to distinguish missing refs")
    detail = result.stderr.decode("utf-8", "replace").strip()
    raise GitStoreError(f"Cannot determine whether Git ref {ref} exists: {detail}")


def _resolve(root: Path, ref: str) -> str:
    length = _hash_length(root)
    is_oid = (
        isinstance(ref, str)
        and len(ref) == length
        and re.fullmatch(r"[0-9a-fA-F]+", ref) is not None
    )
    if is_oid:
        ref = ref.lower()
    else:
        _ref(root, ref)
    result = _run(
        root, ["rev-parse", "--verify", "--end-of-options", ref + "^{commit}"], check=False
    )
    if result.returncode:
        if not is_oid and _missing_ref(root, ref):
            raise RefNotFound(f"Git ref does not exist: {ref}")
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise GitStoreError(f"Git rev-parse failed: {detail}")
    commit = _full_oid(result.stdout.decode("ascii").strip(), length)
    if is_oid and commit != ref:
        raise GitStoreError("Object ID must identify a commit, not a tag or other object")
    return commit


def _path(path: object) -> str:
    if not isinstance(path, str) or not path:
        raise GitStoreError("Snapshot paths must be nonempty strings")
    try:
        path.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise GitStoreError("Snapshot paths must be UTF-8") from exc
    parts = path.split("/")
    if (
        "\\" in path
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
        or any(part in ("", ".", "..") or part.rstrip(" .").casefold() == ".git" for part in parts)
        or parts[0].rstrip(" .").casefold() == ".memory-graph"
        or any(":" in part for part in parts)
    ):
        raise GitStoreError(f"Unsafe snapshot path: {path!r}")
    return path


def _entries(root: Path, commit: str) -> dict[str, tuple[str, str]]:
    listing = _bounded_output(
        root,
        ["ls-tree", "-r", "-t", "-z", "--full-tree", commit],
        _MAX_TREE_LISTING_BYTES,
        max_records=_MAX_TREE_ENTRIES,
    )
    entries: dict[str, tuple[str, str]] = {}
    for record in listing.split(b"\0"):
        if not record:
            continue
        try:
            meta, raw_path = record.split(b"\t", 1)
            mode, kind, oid = meta.decode("ascii").split(" ")
            path = _path(raw_path.decode("utf-8", "strict"))
        except (ValueError, UnicodeError) as exc:
            raise GitStoreError("Invalid Git tree entry") from exc
        _full_oid(oid, len(commit))
        if mode == "040000" and kind == "tree":
            continue
        if mode not in ("100644", "100755") or kind != "blob":
            raise GitStoreError(f"Unsupported non-regular Git tree entry: {path!r} ({mode} {kind})")
        if path in entries:
            raise GitStoreError(f"Duplicate snapshot path: {path!r}")
        entries[path] = (mode, oid)
    return entries


def _read_blobs(root: Path, entries: dict[str, tuple[str, str]]) -> dict[str, bytes]:
    if not entries:
        return {}
    counts = Counter(oid for _, oid in entries.values())
    oids = list(counts)
    sizes: dict[str, int] = {}
    total = 0
    content_output_limit = 0
    for start in range(0, len(oids), _METADATA_BATCH_SIZE):
        batch = oids[start : start + _METADATA_BATCH_SIZE]
        metadata = _bounded_output(
            root,
            ["cat-file", "--batch-check"],
            sum(len(oid) + 128 for oid in batch),
            data=("\n".join(batch) + "\n").encode("ascii"),
        )
        lines = metadata.splitlines()
        if len(lines) != len(batch):
            raise GitStoreError("Invalid Git blob metadata batch")
        for oid, line in zip(batch, lines, strict=True):
            header = line.split(b" ")
            if len(header) != 3 or header[0] != oid.encode("ascii") or header[1] != b"blob":
                raise GitStoreError(f"Invalid or missing Git blob: {oid}")
            try:
                size = int(header[2])
            except ValueError as exc:
                raise GitStoreError(f"Invalid Git blob size: {oid}") from exc
            if size < 0:
                raise GitStoreError(f"Invalid Git blob size: {oid}")
            if size > _MAX_BLOB_BYTES:
                raise GitStoreError(f"Git blob {oid} exceeds the 8 MiB per-blob limit")
            sizes[oid] = size
            total += size * counts[oid]
            if total > _MAX_SNAPSHOT_BYTES:
                raise GitStoreError("Snapshot contents exceed the 512 MiB total limit")
            content_output_limit += len(oid) + 8 + len(str(size)) + size
    stream = _bounded_output(
        root,
        ["cat-file", "--batch"],
        content_output_limit,
        data=("\n".join(oids) + "\n").encode("ascii"),
    )
    offset = 0
    blobs: dict[str, bytes] = {}
    for oid in oids:
        end = stream.find(b"\n", offset)
        if end < 0:
            raise GitStoreError("Truncated Git blob batch")
        header = stream[offset:end].split(b" ")
        if len(header) != 3 or header[0] != oid.encode("ascii") or header[1] != b"blob":
            raise GitStoreError(f"Invalid or missing Git blob: {oid}")
        try:
            size = int(header[2])
        except ValueError as exc:
            raise GitStoreError("Invalid Git blob size") from exc
        if size != sizes[oid]:
            raise GitStoreError(f"Git blob {oid} size changed after metadata check")
        start = end + 1
        end = start + size
        if size < 0 or end >= len(stream) or stream[end : end + 1] != b"\n":
            raise GitStoreError("Truncated Git blob batch")
        blobs[oid] = stream[start:end]
        offset = end + 1
    if offset != len(stream):
        raise GitStoreError("Unexpected Git blob batch output")
    return {path: blobs[oid] for path, (_, oid) in entries.items()}


def load_snapshot(root: Path, ref: str = "refs/heads/main") -> tuple[str, dict[str, bytes]]:
    """Read one committed tree, never the checkout.

    Limits are 8 MiB per blob, 512 MiB total (counting each path), 200,000
    tree entries including directories, and 64 MiB of ls-tree output. These
    bound retained output, not Git's internal object decompression or peak RSS.
    """
    root = _root(root)
    commit = _resolve(root, ref)
    return commit, _read_blobs(root, _entries(root, commit))


class _Index:
    def __init__(self, root: Path):
        self.root = root
        self.path = _git_dir(root) / f"memory-graph-index-{uuid.uuid4().hex}"

    def __enter__(self) -> Path:
        # A valid empty index reserves the name exclusively, without a temporary directory.
        index = b"DIRC" + (2).to_bytes(4, "big") + (0).to_bytes(4, "big")
        algorithm = "sha256" if _hash_length(self.root) == 64 else "sha1"
        try:
            with self.path.open("xb") as file:
                file.write(index + hashlib.new(algorithm, index).digest())
        except OSError as exc:
            raise GitStoreError(f"Cannot create isolated Git index: {exc}") from exc
        return self.path

    def __exit__(self, *args: object) -> None:
        self.path.unlink(missing_ok=True)
        self.path.with_name(self.path.name + ".lock").unlink(missing_ok=True)


def _index_changes(
    root: Path, index: Path, changes: dict[str, bytes | None], modes: dict[str, str], length: int
) -> dict[str, tuple[str, str]]:
    records: list[bytes] = []
    updated: dict[str, tuple[str, str]] = {}
    for path, content in sorted(changes.items()):
        if content is None:
            records.append(b"0 " + b"0" * length + b"\t" + path.encode("utf-8") + b"\0")
    for path, content in sorted(changes.items()):
        if content is not None:
            result = _run(root, ["hash-object", "--no-filters", "-w", "--stdin"], data=content)
            oid = _full_oid(result.stdout.decode("ascii").strip(), length)
            mode = modes.get(path, "100644")
            updated[path] = (mode, oid)
            records.append(f"{mode} {oid}\t".encode("ascii") + path.encode("utf-8") + b"\0")
    if records:
        _run(root, ["update-index", "-z", "--index-info"], data=b"".join(records), index=index)
    return updated


def _publish(root: Path, ref: str, commit: str, expected: str) -> None:
    result = _run(root, ["update-ref", "--no-deref", ref, commit, expected], check=False)
    if not result.returncode:
        return
    missing = _missing_ref(root, ref)
    actual_oid = (
        None
        if missing
        else _run(root, ["show-ref", "--verify", "--hash", ref]).stdout.decode("ascii").strip()
    )
    if actual_oid != (None if set(expected) == {"0"} else expected):
        raise StaleRefError(f"Ref {ref} changed or does not exist; expected {expected}")
    detail = result.stderr.decode("utf-8", "replace").strip()
    raise GitStoreError(f"Git update-ref failed: {detail}")


def commit_snapshot(
    root: Path,
    base: str,
    changes: dict[str, bytes | None],
    message: str,
    ref: str = "refs/heads/main",
    expected_ref: str | None = None,
) -> str:
    """Commit explicit changes on base and CAS-publish ref without touching the checkout."""
    root = _root(root)
    length = _hash_length(root)
    base = _full_oid(base, length)
    if _resolve(root, base) != base:
        raise GitStoreError("Base must be a commit ID")
    ref = _ref(root, ref)
    expected = _full_oid(base if expected_ref is None else expected_ref, length, allow_zero=True)
    if not isinstance(message, str) or not message.strip() or "\x00" in message:
        raise GitStoreError("Commit message must be a nonempty string without NUL")
    try:
        message_data = message.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise GitStoreError("Commit message must be UTF-8") from exc
    if not isinstance(changes, dict):
        raise GitStoreError("Changes must be a mapping of paths to bytes or None")
    for path, content in changes.items():
        _path(path)
        if content is not None and not isinstance(content, bytes):
            raise GitStoreError(f"Change content must be bytes or None: {path!r}")
    entries = _entries(root, base)
    paths = (set(entries) - {p for p, content in changes.items() if content is None}) | {
        p for p, content in changes.items() if content is not None
    }
    for path in paths:
        parts = path.split("/")
        if any("/".join(parts[:i]) in paths for i in range(1, len(parts))):
            raise GitStoreError(f"File/directory path collision: {path!r}")
    with _Index(root) as index:
        _run(root, ["read-tree", base], index=index)
        updated = _index_changes(
            root, index, changes, {p: mode for p, (mode, _) in entries.items()}, length
        )
        tree = _run(root, ["write-tree"], index=index).stdout.decode("ascii").strip()
        _full_oid(tree, length)
        expected_entries = {p: entry for p, entry in entries.items() if p not in changes}
        expected_entries.update(updated)
        if _entries(root, tree) != expected_entries:
            raise GitStoreError("Git index produced an unexpected candidate tree")
        commit = (
            _run(root, ["commit-tree", tree, "-p", base], data=message_data)
            .stdout.decode("ascii")
            .strip()
        )
        _full_oid(commit, length)
        _publish(root, ref, commit, expected)
    return commit


def init_store(root: Path) -> str:
    """Initialize a dedicated repository without replacing any pre-existing files."""
    _run(Path.cwd(), ["--version"])
    root = Path(root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise GitStoreError("Store root must be a directory")
        for name in (".git", ".gitignore", "store.json"):
            if os.path.lexists(root / name):
                raise GitStoreError(f"Cannot initialize: {name} already exists")
        (root / ".git").mkdir(mode=0o700)
        with (root / ".gitignore").open("xb") as file:
            file.write(b".memory-graph/\n")
        with (root / "store.json").open("xb") as file:
            file.write((json.dumps({"schema": 1, "id": str(uuid.uuid4())}) + "\n").encode("utf-8"))
    except OSError as exc:
        raise GitStoreError(f"Cannot initialize store: {exc}") from exc
    _run(root, ["init", "--initial-branch=main", "--template="])
    root = _root(root)
    length = _hash_length(root)
    changes: dict[str, bytes | None] = {
        name: (root / name).read_bytes() for name in (".gitignore", "store.json")
    }
    with _Index(root) as index:
        _run(root, ["read-tree", "--empty"], index=index)
        _index_changes(root, index, changes, {}, length)
        tree = _run(root, ["write-tree"], index=index).stdout.decode("ascii").strip()
        commit = (
            _run(root, ["commit-tree", tree], data=b"Initialize memory store\n")
            .stdout.decode("ascii")
            .strip()
        )
        _full_oid(commit, length)
        _publish(root, "refs/heads/main", commit, "0" * length)
    _run(root, ["read-tree", commit])
    return commit
