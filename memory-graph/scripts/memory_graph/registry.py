"""Named memory stores bound to a project or to the user, as a folder or a Git remote.

A registry file maps names to stores. The project file ``.memory-graph.json`` is found by
searching upward from the working directory and may be committed; the global file lives in the
user's configuration directory. A project entry shadows a global entry of the same name.
Every entry pins the store ID, so a replaced or retargeted store is refused. Resolution never
merges stores: each command uses exactly one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from filelock import FileLock

from . import git_store, remotes, service

PROJECT_FILE = ".memory-graph.json"
MODES = ("read", "propose", "write")
SCOPES = ("project", "global")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_MAX_BYTES = 1_000_000


class RegistryError(service.GraphError):
    """The store registry is invalid or the requested store cannot be used."""


class Entry(NamedTuple):
    name: str
    scope: str
    spec: dict[str, Any]
    base: Path


def _environment_path(variable: str, xdg: str, fallback: Path) -> Path:
    override = os.environ.get(variable)
    if override:
        return Path(override).expanduser()
    base = os.environ.get(xdg)
    if base and Path(base).is_absolute():
        return Path(base) / "memory-graph"
    return fallback / "memory-graph"


def config_dir() -> Path:
    return _environment_path("MEMORY_GRAPH_CONFIG_DIR", "XDG_CONFIG_HOME", Path.home() / ".config")


def data_dir() -> Path:
    return _environment_path(
        "MEMORY_GRAPH_DATA_DIR", "XDG_DATA_HOME", Path.home() / ".local" / "share"
    )


def global_file() -> Path:
    return config_dir() / "stores.json"


def find_project_file(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / PROJECT_FILE
        if candidate.is_symlink():
            raise RegistryError(f"{candidate} must not be a symlink")
        if candidate.is_file():
            return candidate
    return None


def _project_target(start: Path) -> Path:
    existing = find_project_file(start)
    if existing is not None:
        return existing
    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory / PROJECT_FILE
    return start / PROJECT_FILE


def _validate_name(name: object) -> str:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise RegistryError("store name must use letters, digits, '.', '_' or '-' (max 64)")
    return name


def _validate(value: object, label: str) -> dict:
    if (
        not isinstance(value, dict)
        or not set(value) <= {"schema", "default", "stores"}
        or type(value.get("schema")) is not int
        or value["schema"] != 1
        or not isinstance(value.get("stores"), dict)
    ):
        raise RegistryError(f"{label}: invalid store registry")
    for name, spec in value["stores"].items():
        _validate_name(name)
        if (
            not isinstance(spec, dict)
            or not set(spec) <= {"path", "remote", "mode", "store_id"}
            or ("path" in spec) == ("remote" in spec)
            or not all(isinstance(item, str) and item for item in spec.values())
        ):
            raise RegistryError(f"{label}: invalid entry {name!r}")
        if "remote" in spec and spec.get("mode") not in MODES:
            raise RegistryError(f"{label}: remote entry {name!r} needs a mode")
        if "path" in spec and "mode" in spec:
            raise RegistryError(f"{label}: path entry {name!r} must not set a mode")
        try:
            remotes._uuid(spec.get("store_id"))
        except remotes.RemoteError as exc:
            raise RegistryError(f"{label}: entry {name!r} needs a pinned store_id") from exc
    default = value.get("default")
    if default is not None and default not in value["stores"]:
        raise RegistryError(f"{label}: default store {default!r} is not registered")
    return value


def _load(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"schema": 1, "default": None, "stores": {}}
    if path.is_symlink() or not path.is_file():
        raise RegistryError(f"{path} must be a regular file")
    with path.open("rb") as stream:
        data = stream.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise RegistryError(f"{path} is too large")
    try:
        value = json.loads(data, object_pairs_hook=remotes._object)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RegistryError(f"{path}: not valid JSON without duplicate keys") from exc
    value = _validate(value, str(path))
    value.setdefault("default", None)
    return value


def _write(path: Path, registry: dict, *, private: bool) -> None:
    path.parent.mkdir(mode=0o700 if private else 0o755, parents=True, exist_ok=True)
    text = json.dumps(registry, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".stores-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600 if private else 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _lock(path: Path) -> FileLock:
    directory = config_dir() / "locks"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return FileLock(str(directory / (hashlib.sha256(str(path).encode()).hexdigest() + ".lock")), 30)


def _entries(cwd: Path) -> tuple[list[Entry], Path | None, dict, dict]:
    project_file = find_project_file(cwd)
    project = _load(project_file)
    glob = _load(global_file())
    entries = [
        Entry(name, "project", spec, project_file.parent if project_file else cwd)
        for name, spec in project["stores"].items()
    ]
    entries += [
        Entry(name, "global", spec, global_file().parent) for name, spec in glob["stores"].items()
    ]
    return entries, project_file, project, glob


def _clone_path(remote: str, mode: str) -> Path:
    key = hashlib.sha256(f"{remote}\0{mode}".encode()).hexdigest()[:24]
    return data_dir() / "clones" / key


def _location(entry: Entry) -> Path:
    if "remote" in entry.spec:
        return _clone_path(entry.spec["remote"], entry.spec["mode"])
    path = Path(entry.spec["path"]).expanduser()
    return path if path.is_absolute() else Path(os.path.abspath(entry.base / path))


def _identity(root: Path) -> str:
    root = git_store._root(root)
    result = git_store._run(root, ["show", "refs/heads/main:store.json"])
    manifest = json.loads(result.stdout, object_pairs_hook=remotes._object)
    return remotes._uuid(manifest["id"])


def _state(entry: Entry, root: Path) -> str:
    if not root.exists():
        return "not-cloned" if "remote" in entry.spec else "missing"
    try:
        identity = _identity(root)
    except (git_store.GitStoreError, ValueError, KeyError, TypeError, OSError):
        return "invalid"
    return "ready" if identity == entry.spec["store_id"] else "identity-mismatch"


def _effective_default(project: dict, glob: dict) -> tuple[str | None, str | None]:
    if project["default"] is not None:
        return project["default"], "project"
    if glob["default"] is not None:
        return glob["default"], "global"
    return None, None


def resolve(name: str | None, cwd: Path) -> Path:
    """Return the root of the one selected store, refusing anything not ready."""
    entries, _, project, glob = _entries(cwd)
    if name is None:
        name, _ = _effective_default(project, glob)
        if name is None:
            available = ", ".join(sorted({entry.name for entry in entries})) or "none"
            raise RegistryError(
                "no store selected: pass --store NAME or --root, or set a default with "
                f"'stores default NAME' (registered: {available})"
            )
    else:
        _validate_name(name)
    entry = next((e for e in entries if e.name == name), None)
    if entry is None:
        available = ", ".join(sorted({e.name for e in entries})) or "none"
        raise RegistryError(f"unknown store {name!r} (registered: {available})")
    root = _location(entry)
    state = _state(entry, root)
    if state == "not-cloned":
        raise RegistryError(f"remote store {name!r} is not cloned yet; run: stores sync {name}")
    if state != "ready":
        raise RegistryError(f"store {name!r} is not usable: {state}")
    return root


def _normalize_path(raw: str, cwd: Path) -> Path:
    path = Path(raw).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else cwd / path))


def add(args: Any, cwd: Path) -> dict:
    name = _validate_name(args.name)
    if args.mode is not None and args.remote is None:
        raise RegistryError("--mode applies only to --remote stores")
    if args.init and args.path is None:
        raise RegistryError("--init applies only to --path stores")
    target = _project_target(cwd) if args.scope == "project" else global_file()
    with _lock(target):
        registry = _load(target)
        if name in registry["stores"]:
            raise RegistryError(f"{name!r} already exists in the {args.scope} registry")
        if args.path is not None:
            root = _normalize_path(args.path, cwd)
            if args.init and (not root.exists() or not any(root.iterdir())):
                git_store.init_store(root)
            try:
                store_id = _identity(root)
            except (git_store.GitStoreError, ValueError, KeyError, TypeError, OSError) as exc:
                raise RegistryError(
                    f"{root} is not a memory store (use --init to create one)"
                ) from exc
            service.load(root)
            base = target.parent
            stored = "./" + os.path.relpath(root, base) if root.is_relative_to(base) else str(root)
            spec: dict[str, Any] = {"path": stored, "store_id": store_id}
        else:
            endpoint, _ = remotes._endpoint(cwd, args.remote)
            mode = args.mode or "propose"
            destination = _clone_path(endpoint, mode)
            if destination.exists():
                config = remotes.get_config(destination)
                if config is None:
                    raise RegistryError(f"{destination} exists but is not a managed clone")
            else:
                config = remotes.clone(endpoint, destination, mode)
            spec = {"remote": endpoint, "mode": mode, "store_id": config["store_id"]}
        registry["stores"][name] = spec
        if args.default or registry["default"] is None:
            registry["default"] = name
        _write(target, registry, private=args.scope == "global")
    return {"added": name, "scope": args.scope, "file": str(target), **spec}


def _existing_target(scope: str, cwd: Path) -> Path:
    target = find_project_file(cwd) if scope == "project" else global_file()
    return target or cwd / PROJECT_FILE


def remove(args: Any, cwd: Path) -> dict:
    target = _existing_target(args.scope, cwd)
    with _lock(target):
        registry = _load(target)
        spec = registry["stores"].pop(args.name, None)
        if spec is None:
            raise RegistryError(f"{args.name!r} is not in the {args.scope} registry")
        if registry["default"] == args.name:
            registry["default"] = None
        _write(target, registry, private=args.scope == "global")
    result: dict[str, Any] = {"removed": args.name, "scope": args.scope}
    if "remote" in spec:
        result["clone_kept_at"] = str(_clone_path(spec["remote"], spec["mode"]))
    return result


def set_default(args: Any, cwd: Path) -> dict:
    target = _existing_target(args.scope, cwd)
    with _lock(target):
        registry = _load(target)
        if args.name not in registry["stores"]:
            raise RegistryError(f"{args.name!r} is not in the {args.scope} registry")
        registry["default"] = args.name
        _write(target, registry, private=args.scope == "global")
    return {"default": args.name, "scope": args.scope}


def sync(args: Any, cwd: Path) -> dict:
    entries, _, _, _ = _entries(cwd)
    entry = next((e for e in entries if e.name == args.name), None)
    if entry is None:
        raise RegistryError(f"unknown store {args.name!r}")
    if "remote" not in entry.spec:
        raise RegistryError(f"{args.name!r} is a folder store; only remote stores sync")
    root = _location(entry)
    if root.exists():
        if _state(entry, root) != "ready":
            raise RegistryError(f"store {args.name!r} is not usable: {_state(entry, root)}")
        config = remotes.fetch(root)
        action = "fetched"
    else:
        config = remotes.clone(
            entry.spec["remote"], root, entry.spec["mode"], entry.spec["store_id"]
        )
        action = "cloned"
    return {
        "synced": args.name,
        "action": action,
        "fetched_commit": config["fetched_commit"],
        "fetched_at": config["fetched_at"],
    }


def list_stores(cwd: Path) -> dict:
    entries, project_file, project, glob = _entries(cwd)
    default, _ = _effective_default(project, glob)
    seen: set[str] = set()
    rows = []
    for entry in entries:
        root = _location(entry)
        shadowed = entry.name in seen
        row: dict[str, Any] = {
            "name": entry.name,
            "scope": entry.scope,
            "default": entry.name == default and not shadowed,
            "shadowed": shadowed,
            "kind": "remote" if "remote" in entry.spec else "path",
            "location": entry.spec.get("remote") or entry.spec["path"],
            "local_path": str(root),
            "store_id": entry.spec["store_id"],
            "state": _state(entry, root),
        }
        seen.add(entry.name)
        if "remote" in entry.spec:
            row["mode"] = entry.spec["mode"]
            if row["state"] == "ready":
                try:
                    config = remotes.get_config(root) or {}
                except remotes.RemoteError:
                    config = {}
                row["fetched_at"] = config.get("fetched_at")
                row["fetched_commit"] = config.get("fetched_commit")
        rows.append(row)
    return {
        "stores": rows,
        "default": default,
        "project_file": str(project_file) if project_file else None,
        "global_file": str(global_file()),
    }


def run(args: Any, cwd: Path) -> dict:
    command = args.stores_command
    if command == "add":
        return add(args, cwd)
    if command == "list":
        return list_stores(cwd)
    if command == "remove":
        return remove(args, cwd)
    if command == "default":
        return set_default(args, cwd)
    if command == "sync":
        return sync(args, cwd)
    raise RegistryError("unsupported stores command")
