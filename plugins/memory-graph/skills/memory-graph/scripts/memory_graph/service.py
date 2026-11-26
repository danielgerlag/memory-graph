"""Snapshot-based graph operations. The Git tree, not the checkout, is authoritative."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from jsonschema import Draft202012Validator, FormatChecker

from . import cache, git_store, remotes, schema

MAX_BATCH = 100
MAX_RESPONSE = 65536
MAX_SUMMARY_PAYLOAD = MAX_RESPONSE - 32  # Reserve space for the CLI envelope and newline.
NODE_FILE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.md$")
RECEIPT_FILE = re.compile(
    r"^operations/([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\.json$"
)
ASSETS = Path(__file__).resolve().parents[2] / "assets"
STORE_VALIDATOR = Draft202012Validator(json.loads((ASSETS / "store.schema.json").read_text()))
RECEIPT_VALIDATOR = Draft202012Validator(
    json.loads((ASSETS / "operation.schema.json").read_text()), format_checker=FormatChecker()
)


class GraphError(ValueError):
    """A user-visible graph contract error."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def now_utc() -> datetime:
    return datetime.now(UTC)


def timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise GraphError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def valid_uuid(value: Any, name: str = "id") -> str:
    if not isinstance(value, str):
        raise GraphError(f"{name} must be a UUIDv4 string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise GraphError(f"{name} must be a UUIDv4 string") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise GraphError(f"{name} must be a canonical lowercase UUIDv4")
    return value


def _json_file(data: bytes, path: str) -> dict:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise GraphError(f"{path}: duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GraphError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GraphError(f"{path}: expected an object")
    return value


def _parse_receipt(data: bytes, path: str) -> dict:
    match = RECEIPT_FILE.fullmatch(path)
    if not match:
        raise GraphError("invalid receipt path")
    receipt = _json_file(data, path)
    problem = next(RECEIPT_VALIDATOR.iter_errors(receipt), None)
    if problem:
        raise GraphError(f"invalid operation receipt: {problem.message}")
    try:
        parse_time(receipt["resolved_at"])
    except ValueError as exc:
        raise GraphError(f"invalid operation receipt timestamp: {exc}") from exc
    if receipt["operation_id"] != match.group(1):
        raise GraphError("receipt operation id does not match its filename")
    return receipt


@dataclass
class Snapshot:
    commit: str
    store_id: str
    files: dict[str, bytes]
    nodes: dict[str, dict]
    bodies: dict[str, bytes]
    revisions: dict[str, str]
    receipts: dict[str, dict]
    config: dict | None


def parse_snapshot(
    commit: str,
    files: dict[str, bytes],
    config: dict | None,
    *,
    diagnostic: bool = False,
    parse_cache: cache.ParseCache | None = None,
) -> tuple[Snapshot, list[dict]]:
    errors: list[dict] = []
    nodes: dict[str, dict] = {}
    bodies: dict[str, bytes] = {}
    revisions: dict[str, str] = {}
    receipts: dict[str, dict] = {}
    store_id = ""

    def issue(path: str, error: Exception | str) -> None:
        errors.append(
            {
                "path": path,
                "error": str(error),
                "type": type(error).__name__,
                "sha256": digest(files[path]) if path in files else None,
            }
        )

    try:
        manifest = _json_file(files["store.json"], "store.json")
        problem = next(STORE_VALIDATOR.iter_errors(manifest), None)
        if problem:
            raise GraphError(f"store.json: invalid store manifest: {problem.message}")
        store_id = valid_uuid(manifest["id"], "store id")
        if config is not None and config.get("store_id") != store_id:
            raise GraphError("attached store identity does not match the configured store")
    except (KeyError, ValueError) as exc:
        issue("store.json", exc)
    for path, data in sorted(files.items()):
        if "/" not in path and path.endswith(".md"):
            if not NODE_FILE.fullmatch(path):
                issue(path, "node filename must be a canonical UUIDv4 followed by .md")
                continue
            try:
                node, body = (
                    parse_cache.parse(data, path) if parse_cache else schema.parse_node(data, path)
                )
                node_id = path[:-3]
                if node["id"] != node_id:
                    raise GraphError("node id does not match its filename")
                nodes[node_id], bodies[node_id], revisions[node_id] = node, body, digest(data)
            except (ValueError, KeyError) as exc:
                issue(path, exc)
        elif path.startswith("operations/"):
            match = RECEIPT_FILE.fullmatch(path)
            if not match:
                issue(path, "invalid receipt path")
                continue
            try:
                receipt = (
                    parse_cache.parse_receipt(data, path, _parse_receipt)
                    if parse_cache
                    else _parse_receipt(data, path)
                )
                receipts[match.group(1)] = receipt
            except ValueError as exc:
                issue(path, exc)
    if not errors:
        try:
            schema.validate_graph(nodes)
        except ValueError as exc:
            issue("<graph>", exc)
    snap = Snapshot(commit, store_id, files, nodes, bodies, revisions, receipts, config)
    if errors and not diagnostic:
        raise GraphError(
            "invalid main snapshot: " + "; ".join(f"{e['path']}: {e['error']}" for e in errors)
        )
    return snap, errors


def load(root: Path, *, diagnostic: bool = False) -> tuple[Snapshot, list[dict]]:
    config = remotes.get_config(root)
    ref = f"refs/remotes/{config['remote']}/main" if config else "refs/heads/main"
    commit, files = git_store.load_snapshot(root, ref)
    with cache.ParseCache(root) as parse_cache:
        parse_cache.prime(
            [(path, data) for path, data in files.items() if NODE_FILE.fullmatch(path)]
        )
        snap, errors = parse_snapshot(
            commit, files, config, diagnostic=True, parse_cache=parse_cache
        )
    errors.extend(_evidence_errors(root, snap))
    if errors and not diagnostic:
        raise GraphError(
            "invalid main snapshot: " + "; ".join(f"{e['path']}: {e['error']}" for e in errors)
        )
    return snap, errors


def _evidence_errors(root: Path, snap: Snapshot) -> list[dict]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for node_id, node in snap.nodes.items():
        for source in node.get("sources", []):
            if source["kind"] == "memory":
                grouped.setdefault(source["revision"], []).append((node_id, source["node_id"]))
    errors = []

    def record(node_id: str, exc: Exception) -> None:
        path = f"{node_id}.md"
        errors.append(
            {
                "path": path,
                "error": str(exc),
                "type": type(exc).__name__,
                "sha256": digest(snap.files[path]),
            }
        )

    for revision, references in grouped.items():
        try:
            _require_ancestor(root, revision, snap.commit)
            _, files = git_store.load_snapshot(root, revision)
            manifest = _json_file(files["store.json"], "store.json")
            if manifest.get("schema") != 1 or manifest.get("id") != snap.store_id:
                raise GraphError("memory source is from a different store")
        except (ValueError, KeyError) as exc:
            for node_id, _ in references:
                record(node_id, exc)
            continue
        for node_id, target in references:
            try:
                path = f"{target}.md"
                if path not in files:
                    raise GraphError(
                        f"memory source {node_id} cites missing node {target} at {revision}"
                    )
                schema.parse_node(files[path], path)
            except (ValueError, KeyError) as exc:
                record(node_id, exc)
    return errors


def _require_evidence(root: Path, snap: Snapshot) -> None:
    errors = _evidence_errors(root, snap)
    if errors:
        raise GraphError("invalid memory source: " + "; ".join(error["error"] for error in errors))


def provenance(snap: Snapshot) -> dict:
    result: dict[str, Any] = {"store_id": snap.store_id, "commit": snap.commit, "source": "main"}
    if snap.config:
        result.update(
            {
                "remote": snap.config["remote"],
                "last_fetch": snap.config.get("last_fetch"),
                "freshness": "unknown",
            }
        )
    return result


def _visible(
    node: dict,
    when: datetime,
    scope: str | None,
    include_unscoped: bool,
    include_inactive: bool = False,
) -> bool:
    if (
        scope is not None
        and node.get("scope") != scope
        and not (include_unscoped and node.get("scope") is None)
    ):
        return False
    if not include_inactive:
        if node["status"] == "superseded":
            return False
        if node.get("expires_at") and schema.timestamp_reached(node["expires_at"], when):
            return False
    return True


def summary(snap: Snapshot, node_id: str, when: datetime) -> dict:
    node = snap.nodes[node_id]
    return {
        "id": node_id,
        "label": node["label"],
        "kind": node["kind"],
        "status": node["status"],
        "scope": node.get("scope"),
        "revision": snap.revisions[node_id],
        "expired": bool(
            node.get("expires_at") and schema.timestamp_reached(node["expires_at"], when)
        ),
        "review_due": bool(
            node.get("review_after") and schema.timestamp_reached(node["review_after"], when)
        ),
    }


def historical_snapshot(root: Path, main: Snapshot, commit: str) -> Snapshot:
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise GraphError("historical commit must be a full lowercase Git object ID")
    _require_ancestor(root, commit, main.commit)
    resolved, files = git_store.load_snapshot(root, commit)
    historical, _ = parse_snapshot(resolved, files, main.config)
    _require_evidence(root, historical)
    return historical


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", ancestor) or not re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", descendant
    ):
        raise GraphError("commit must be a full lowercase Git object ID")
    result = git_store._run(
        root, ["merge-base", "--is-ancestor", ancestor, descendant], check=False
    )
    if result.returncode == 1:
        raise GraphError("historical commit is not in the selected main history")
    if result.returncode != 0:
        raise GraphError(
            f"cannot verify historical commit: {result.stderr.decode('utf-8', 'replace').strip()}"
        )


def history(root: Path, limit: int) -> dict:
    if not 1 <= limit <= 200:
        raise GraphError("history limit must be between 1 and 200")
    snap, _ = load(root)
    result = git_store._run(
        root, ["rev-list", f"--max-count={limit + 1}", snap.commit], check=False
    )
    if result.returncode:
        raise GraphError(
            f"cannot read main history: {result.stderr.decode('utf-8', 'replace').strip()}"
        )
    commits = result.stdout.decode("ascii").splitlines()
    return {
        "snapshot": provenance(snap),
        "commits": commits[:limit],
        "truncated": len(commits) > limit,
    }


def export_view(root: Path, destination: Path) -> dict:
    requested = destination.expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise GraphError("view destination already exists")
    destination = requested.resolve()
    if destination.is_relative_to(root.resolve()):
        raise GraphError("view destination cannot be inside the memory store")
    if destination.is_relative_to(Path(__file__).resolve().parents[2]):
        raise GraphError("view destination cannot be inside the installed skill")
    snap, _ = load(root)
    destination.mkdir(mode=0o700)

    def write(path: Path, data: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)

    for path, data in sorted(snap.files.items()):
        if NODE_FILE.fullmatch(path):
            write(destination / path, data)
    write(destination / "store.json", snap.files["store.json"])
    info = {"snapshot": provenance(snap), "includes_inactive": True, "live_store": False}
    write(destination / ".memory-graph-snapshot.json", canonical(info) + b"\n")
    return {"destination": str(destination), **info, "node_count": len(snap.nodes)}


def check_merge(root: Path, base_commit: str, candidate_commit: str) -> dict:
    """Validate a proposed integration tree against the actual protected base."""
    _require_ancestor(root, base_commit, candidate_commit)
    config = remotes.get_config(root)
    resolved_base, base_files = git_store.load_snapshot(root, base_commit)
    resolved_candidate, candidate_files = git_store.load_snapshot(root, candidate_commit)
    base, base_errors = parse_snapshot(resolved_base, base_files, config, diagnostic=True)
    candidate, _ = parse_snapshot(resolved_candidate, candidate_files, config)
    base_errors.extend(_evidence_errors(root, base))
    _require_evidence(root, candidate)
    if any(
        error["path"] == "store.json"
        or error["path"].startswith("operations/")
        or error["type"] == "UnsupportedSchema"
        for error in base_errors
    ):
        raise GraphError("base store identity, receipts, or schema require operator recovery")
    if base.store_id != candidate.store_id or base_files.get("store.json") != candidate_files.get(
        "store.json"
    ):
        raise GraphError("candidate changes the store identity or manifest")
    for operation_id in base.receipts:
        path = f"operations/{operation_id}.json"
        if candidate_files.get(path) != base_files[path]:
            raise GraphError(f"candidate changes or removes existing receipt {operation_id}")
    new_operations = sorted(set(candidate.receipts) - set(base.receipts))
    if base_errors and (
        not new_operations
        or any(candidate.receipts[op]["action"] != "repair" for op in new_operations)
    ):
        raise GraphError("an invalid base requires an explicit repair operation")
    declared: set[str] = set()
    for operation_id in new_operations:
        receipt = candidate.receipts[operation_id]
        _require_ancestor(root, receipt.get("base", ""), base_commit)
        preconditions = receipt.get("preconditions")
        changed = receipt.get("changed_hashes")
        if not isinstance(preconditions, dict) or not isinstance(changed, dict):
            raise GraphError(f"receipt {operation_id} lacks preconditions or changed hashes")
        for key, expected in preconditions.items():
            path = f"{key}.md" if isinstance(key, str) and not key.endswith(".md") else key
            if (
                not isinstance(path, str)
                or not NODE_FILE.fullmatch(path)
                or path not in base_files
                or digest(base_files[path]) != expected
            ):
                raise GraphError(f"stale affected-node precondition for {key}")
        for path, expected in changed.items():
            if not isinstance(path, str) or (
                not NODE_FILE.fullmatch(path) and not path.startswith(f"quarantine/{operation_id}/")
            ):
                raise GraphError(f"invalid changed path in receipt {operation_id}")
            if path in declared:
                raise GraphError(f"multiple operations change {path}")
            declared.add(path)
            actual = digest(candidate_files[path]) if path in candidate_files else None
            if expected != actual:
                raise GraphError(f"candidate content differs from receipt for {path}")
            if NODE_FILE.fullmatch(path):
                node_id = path[:-3]
                if (
                    path in base_files
                    and node_id not in preconditions
                    and path not in preconditions
                ):
                    raise GraphError(f"existing node {path} lacks a precondition")
    actual_changes = {
        path
        for path in set(base_files) | set(candidate_files)
        if base_files.get(path) != candidate_files.get(path) and not RECEIPT_FILE.fullmatch(path)
    }
    if actual_changes != declared:
        raise GraphError("candidate contains undeclared or missing file changes")
    return {
        "valid": True,
        "store_id": base.store_id,
        "base": base_commit,
        "candidate": candidate_commit,
        "new_operations": new_operations,
    }


def show(
    root: Path,
    node_id: str,
    scope: str | None = None,
    include_unscoped: bool = False,
    include_inactive: bool = False,
    commit: str | None = None,
) -> dict:
    valid_uuid(node_id)
    snap, _ = load(root)
    main_commit = snap.commit
    if commit is not None:
        snap = historical_snapshot(root, snap, commit)
    when = now_utc()
    node = snap.nodes.get(node_id)
    if node is None or not _visible(node, when, scope, include_unscoped, include_inactive):
        raise GraphError("node not found in the selected visible snapshot")
    info = provenance(snap)
    if commit is not None:
        info.update({"source": "history", "main_commit": main_commit})
    visible_node = {
        **node,
        "edges": [
            edge
            for edge in node["edges"]
            if _visible(snap.nodes[edge["to"]], when, scope, include_unscoped, include_inactive)
        ],
    }
    return {
        "snapshot": info,
        "node": visible_node,
        "body": snap.bodies[node_id].decode("utf-8"),
        "revision": snap.revisions[node_id],
        "lifecycle": summary(snap, node_id, when),
    }


def _bounded_output(items: list[dict], limit: int, envelope: dict) -> dict:
    result: list[dict] = []
    truncated = len(items) > limit
    for item in items[:limit]:
        candidate = {**envelope, "results": result + [item], "truncated": truncated}
        if len(canonical(candidate)) > MAX_SUMMARY_PAYLOAD:
            truncated = True
            break
        result.append(item)
    return {**envelope, "results": result, "truncated": truncated}


def find(
    root: Path,
    query: str,
    scope: str | None,
    include_unscoped: bool,
    limit: int,
    tag: str | None = None,
    kind: str | None = None,
) -> dict:
    if not 1 <= limit <= 200:
        raise GraphError("find limit must be between 1 and 200")
    if not query.strip() and tag is None and kind is None:
        raise GraphError("provide a nonempty query, --tag, or --kind")
    if (tag is not None and not tag.strip()) or (kind is not None and not kind.strip()):
        raise GraphError("tag and kind filters must not be empty")
    if len(query.encode("utf-8")) > 4096:
        raise GraphError("query must not exceed 4096 UTF-8 bytes")
    if any(len(value.encode("utf-8")) > 4096 for value in (tag, kind) if value is not None):
        raise GraphError("tag and kind filters must not exceed 4096 UTF-8 bytes")
    snap, _ = load(root)
    when = now_utc()
    needle = query.casefold() if query.strip() else ""
    found = []
    for node_id, node in snap.nodes.items():
        if not _visible(node, when, scope, include_unscoped):
            continue
        if tag is not None and tag not in node.get("tags", []):
            continue
        if kind is not None and node["kind"] != kind:
            continue
        reasons = []
        rank = 0
        if needle:
            fields = [
                ("label", node["label"]),
                ("aliases", " ".join(node.get("aliases", []))),
                ("tags", " ".join(node.get("tags", []))),
                (
                    "properties",
                    json.dumps(node.get("properties", {}), ensure_ascii=False, sort_keys=True),
                ),
                ("body", snap.bodies[node_id].decode("utf-8")),
            ]
            reasons = [name for name, value in fields if needle in value.casefold()]
            if not reasons:
                continue
            rank = min(
                ["label", "aliases", "tags", "properties", "body"].index(name) for name in reasons
            )
        reasons += (["tag_filter"] if tag is not None else []) + (
            ["kind_filter"] if kind is not None else []
        )
        found.append(
            (
                rank,
                -node.get("importance", 0),
                node_id,
                {**summary(snap, node_id, when), "match_reasons": reasons},
            )
        )
    found.sort(key=lambda x: x[:3])
    return _bounded_output(
        [row[3] for row in found],
        limit,
        {"snapshot": provenance(snap), "query": query, "tag": tag, "kind": kind},
    )


def neighbors(
    root: Path,
    node_id: str,
    direction: str,
    edge_label: str | None,
    scope: str | None,
    include_unscoped: bool,
    depth: int,
    limit: int,
) -> dict:
    if not 1 <= depth <= 8 or not 1 <= limit <= 500:
        raise GraphError("depth must be 1..8 and limit must be 1..500")
    snap, _ = load(root)
    when = now_utc()
    visible = {
        key for key, node in snap.nodes.items() if _visible(node, when, scope, include_unscoped)
    }
    if node_id not in visible:
        raise GraphError("node not found in the selected visible snapshot")
    adjacency: dict[str, list[tuple[str, str, str, dict]]] = {key: [] for key in visible}
    for source in sorted(visible):
        for edge in snap.nodes[source]["edges"]:
            target, label = edge["to"], edge["label"]
            if target not in visible or (edge_label is not None and label != edge_label):
                continue
            if direction in ("out", "both"):
                adjacency[source].append((target, label, "out", edge))
            if direction in ("in", "both"):
                adjacency[target].append((source, label, "in", edge))
    seen = {node_id}
    queue = [(node_id, 0)]
    found: list[dict[str, Any]] = []
    relationships: dict[tuple[str, str, str], dict] = {}
    relationships_truncated = False
    traversal_truncated = False
    for current, level in queue:
        if level >= depth:
            if any(target not in seen for target, _, _, _ in adjacency[current]):
                traversal_truncated = True
            continue
        for target, label, edge_direction, edge in sorted(
            adjacency[current], key=lambda item: item[:3]
        ):
            if target not in seen and len(found) >= limit:
                traversal_truncated = True
                continue
            source_id, target_id = (
                (current, target) if edge_direction == "out" else (target, current)
            )
            key = (source_id, label, target_id)
            if key not in relationships:
                if len(relationships) < 1000:
                    relationships[key] = {
                        "source": source_id,
                        "target": target_id,
                        **{name: value for name, value in edge.items() if name != "to"},
                    }
                else:
                    relationships_truncated = True
            if target in seen:
                continue
            seen.add(target)
            queue.append((target, level + 1))
            found.append(
                {
                    **summary(snap, target, when),
                    "depth": level + 1,
                    "via": current,
                    "edge_label": label,
                    "direction": edge_direction,
                }
            )
    result = _bounded_output(
        found, limit, {"snapshot": provenance(snap), "root": node_id, "relationships": []}
    )
    allowed = {node_id, *(item["id"] for item in result["results"])}
    response_size = len(canonical(result))
    for key in sorted(relationships):
        source_id, _, target_id = key
        if source_id not in allowed or target_id not in allowed:
            continue
        relationship = relationships[key]
        added = len(canonical(relationship)) + bool(result["relationships"])
        if response_size + added > MAX_SUMMARY_PAYLOAD:
            relationships_truncated = True
            break
        result["relationships"].append(relationship)
        response_size += added
    result["truncated"] = result["truncated"] or relationships_truncated or traversal_truncated
    return result


def review(
    root: Path, limit: int, scope: str | None = None, include_unscoped: bool = False
) -> dict:
    if not 1 <= limit <= 200:
        raise GraphError("review limit must be between 1 and 200")
    snap, _ = load(root)
    when = now_utc()
    entries = [
        summary(snap, key, when)
        for key in sorted(snap.nodes)
        if _visible(snap.nodes[key], when, scope, include_unscoped, include_inactive=True)
    ]
    return _bounded_output(
        [
            entry
            for entry in entries
            if entry["expired"] or entry["review_due"] or entry["status"] != "active"
        ],
        limit,
        {"snapshot": provenance(snap)},
    )


def _created_node(spec: dict, when: datetime, *, require_id: bool) -> tuple[dict, bytes]:
    if not isinstance(spec, dict):
        raise GraphError("create input must be an object")
    spec = dict(spec)
    body = spec.pop("body", "")
    ttl_supplied = "ttl_seconds" in spec
    ttl = spec.pop("ttl_seconds", None)
    id_supplied = "id" in spec
    node_id = spec.pop("id", None)
    if id_supplied and node_id is None:
        raise GraphError("id must be a UUIDv4 string when supplied")
    if require_id and node_id is None:
        raise GraphError("batch-created nodes require an explicit id")
    node_id = valid_uuid(node_id) if node_id is not None else str(uuid.uuid4())
    if any(field in spec for field in ("schema", "created_at", "updated_at", "superseded_by")):
        raise GraphError("system-owned fields cannot be supplied on creation")
    if spec.get("status", "active") == "superseded":
        raise GraphError("use supersede to enter superseded status")
    if not isinstance(body, str):
        raise GraphError("body must be a string")
    node = {
        "schema": 1,
        "id": node_id,
        **spec,
        "status": spec.get("status", "active"),
        "created_at": timestamp(when),
        "updated_at": timestamp(when),
    }
    node.setdefault("aliases", [])
    node.setdefault("tags", [])
    node.setdefault("edges", [])
    if ttl_supplied:
        if "expires_at" in spec:
            raise GraphError("ttl_seconds and expires_at are mutually exclusive")
        node["expires_at"] = _expiry(when, ttl)
    return node, body.encode("utf-8")


def _expiry(when: datetime, seconds: Any) -> str:
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not 0 <= seconds <= 315576000
    ):
        raise GraphError("ttl_seconds must be a nonnegative number no greater than ten years")
    duration = Decimal(str(seconds))
    exponent = duration.as_tuple().exponent
    if not isinstance(exponent, int):
        raise GraphError("ttl_seconds must be finite")
    with localcontext() as context:
        context.prec = max(40, len(duration.as_tuple().digits) + abs(exponent) + 8)
        amount = duration + Decimal(when.astimezone(UTC).microsecond).scaleb(-6)
        whole = int(amount)
        fraction = format(amount - whole, "f").removeprefix("0.").rstrip("0")
    try:
        result = when.astimezone(UTC).replace(microsecond=0) + timedelta(seconds=whole)
    except OverflowError as exc:
        raise GraphError("ttl_seconds produces an out-of-range timestamp") from exc
    suffix = f".{fraction}" if fraction else ""
    return result.isoformat(timespec="seconds").replace("+00:00", "") + suffix + "Z"


def _patch_node(
    node: dict, body: bytes, patch: dict, when: datetime, *, allow_supersede: bool = False
) -> tuple[dict, bytes]:
    if not isinstance(patch, dict):
        raise GraphError("patch must be an object")
    patch = dict(patch)
    protected = {"schema", "id", "created_at", "updated_at"}
    if protected.intersection(patch):
        raise GraphError("system-owned fields cannot be patched")
    if not allow_supersede and (
        "superseded_by" in patch
        or patch.get("status") == "superseded"
        or (
            node["status"] == "superseded" and "status" in patch and patch["status"] != "superseded"
        )
    ):
        raise GraphError("use supersede to change supersession state")
    if "ttl_seconds" in patch:
        if "expires_at" in patch:
            raise GraphError("ttl_seconds and expires_at are mutually exclusive")
        patch["expires_at"] = _expiry(when, patch.pop("ttl_seconds"))
    if "body" in patch:
        value = patch.pop("body")
        if not isinstance(value, str):
            raise GraphError("body must be a string")
        body = value.encode("utf-8")
    return {**node, **patch, "updated_at": timestamp(when)}, body


def _require_revision(snap: Snapshot, node_id: str, revision: Any) -> None:
    valid_uuid(node_id)
    if node_id not in snap.nodes:
        raise GraphError(f"node {node_id} does not exist")
    if not isinstance(revision, str) or snap.revisions[node_id] != revision:
        raise GraphError(f"stale or missing expected revision for {node_id}")


def mutate(root: Path, operation_id: str, action: str, request: dict) -> dict:
    valid_uuid(operation_id, "operation id")
    if not isinstance(request, dict):
        raise GraphError("mutation input must be a JSON object")
    request_hash = digest(canonical({"action": action, "request": request}))
    # Validate the root before creating local state and refuse redirected locks.
    load(root, diagnostic=action == "repair")
    state_dir = root / ".memory-graph"
    if state_dir.is_symlink():
        raise GraphError("store state directory must not be a symlink")
    state_dir.mkdir(exist_ok=True)
    lock_path = state_dir / "write.lock"
    if lock_path.is_symlink():
        raise GraphError("store lock must not be a symlink")
    try:
        with FileLock(str(lock_path), timeout=10):
            return _mutate_locked(root, operation_id, action, request, request_hash)
    except Timeout as exc:
        raise GraphError(f"timed out acquiring store lock for operation {operation_id}") from exc


def _journal(
    root: Path, operation_id: str, action: str, request: dict, request_hash: str, snap: Snapshot
) -> dict:
    path = root / ".memory-graph" / f"{operation_id}.json"
    if path.is_symlink():
        raise GraphError("operation journal must not be a symlink")
    if path.exists():
        journal = _json_file(path.read_bytes(), str(path))
        if journal.get("request_hash") != request_hash or journal.get("store_id") != snap.store_id:
            raise GraphError("operation id was already used for a different request or store")
        if journal.get("base") != snap.commit:
            raise GraphError(
                "operation base changed; reconcile and use a new operation after replanning"
            )
        return journal
    generated: dict[str, str] = {}
    if action == "create" and "id" not in request:
        generated["create"] = str(uuid.uuid4())
    if action == "supersede" and request.get("replacement_id") is None:
        replacement = request.get("replacement")
        if isinstance(replacement, dict) and "id" not in replacement:
            generated["replacement"] = str(uuid.uuid4())
    journal = {
        "schema": 1,
        "store_id": snap.store_id,
        "request_hash": request_hash,
        "base": snap.commit,
        "resolved_at": timestamp(now_utc()),
        "generated_ids": generated,
    }
    fd, temporary = tempfile.mkstemp(prefix=".operation-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(journal) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return journal


def _mutate_locked(
    root: Path, operation_id: str, action: str, request: dict, request_hash: str
) -> dict:
    snap, errors = load(root, diagnostic=action == "repair")
    if snap.config and snap.config["mode"] == "read":
        raise GraphError("read-only store does not permit mutations")
    branch = f"refs/heads/memory-graph/{operation_id}"
    receipt = snap.receipts.get(operation_id)
    if receipt is None and snap.config:
        try:
            proposal_commit, proposal_files = git_store.load_snapshot(root, branch)
        except git_store.RefNotFound:
            pass
        else:
            proposal, _ = parse_snapshot(proposal_commit, proposal_files, snap.config)
            receipt = proposal.receipts.get(operation_id)
    if receipt is not None:
        if receipt["request_hash"] != request_hash:
            raise GraphError("operation id was already used for a different request")
        accepted = operation_id in snap.receipts
        result: dict[str, Any] = {
            "operation_id": operation_id,
            "state": "accepted" if accepted else "prepared",
            "replayed": True,
            "receipt": receipt,
            "snapshot": provenance(snap),
        }
        if accepted:
            result["reverted_by"] = _reversions(snap, operation_id)
        else:
            result["publication"] = "unknown"
        return result
    if action == "repair":
        return _repair_locked(root, snap, errors, operation_id, request, request_hash, branch)
    if action == "revert":
        return _revert_locked(root, snap, operation_id, request, request_hash, branch)
    journal = _journal(root, operation_id, action, request, request_hash, snap)
    when = parse_time(journal["resolved_at"])
    nodes = {key: dict(value) for key, value in snap.nodes.items()}
    bodies = dict(snap.bodies)
    changed: set[str] = set()
    preconditions: dict[str, str] = {}

    def create(spec: dict, require_id: bool = False) -> str:
        node, body = _created_node(spec, when, require_id=require_id)
        node_id = node["id"]
        if node_id in nodes:
            raise GraphError(f"node {node_id} already exists")
        nodes[node_id], bodies[node_id] = node, body
        changed.add(node_id)
        return node_id

    def update(node_id: str, expected: Any, patch: dict, allow_supersede: bool = False) -> None:
        _require_revision(snap, node_id, expected)
        if node_id in changed:
            raise GraphError(f"node {node_id} appears more than once in the change set")
        preconditions[node_id] = expected
        nodes[node_id], bodies[node_id] = _patch_node(
            nodes[node_id], bodies[node_id], patch, when, allow_supersede=allow_supersede
        )
        changed.add(node_id)

    if action == "create":
        create(
            {
                **request,
                **(
                    {"id": journal["generated_ids"]["create"]}
                    if "create" in journal["generated_ids"]
                    else {}
                ),
            }
        )
    elif action == "update":
        update(request["id"], request.get("expected_revision"), request["patch"])
    elif action == "apply":
        batch_changes = request.get("changes")
        if not isinstance(batch_changes, list) or not 1 <= len(batch_changes) <= MAX_BATCH:
            raise GraphError(f"changes must contain 1..{MAX_BATCH} items")
        for change in batch_changes:
            if not isinstance(change, dict):
                raise GraphError("each change must be an object")
            if change.get("op") == "create":
                create(change["node"], require_id=True)
            elif change.get("op") == "update":
                update(change["id"], change.get("expected_revision"), change["patch"])
            else:
                raise GraphError("batch changes must be create or update")
    elif action in ("link", "unlink"):
        node_id = request["id"]
        _require_revision(snap, node_id, request.get("expected_revision"))
        edge = request["edge"]
        if (
            not isinstance(edge, dict)
            or not isinstance(edge.get("label"), str)
            or not isinstance(edge.get("to"), str)
        ):
            raise GraphError("edge requires label and to")
        edges = list(nodes[node_id]["edges"])
        existing = [e for e in edges if e["label"] == edge["label"] and e["to"] == edge["to"]]
        if action == "link":
            if existing and existing[0] != edge:
                raise GraphError("edge exists with different attributes; update explicitly")
            if existing:
                preconditions[node_id] = request["expected_revision"]
            else:
                edges.append(edge)
                update(node_id, request.get("expected_revision"), {"edges": edges})
        else:
            if not existing:
                raise GraphError("edge does not exist")
            edges.remove(existing[0])
            update(node_id, request.get("expected_revision"), {"edges": edges})
    elif action == "supersede":
        old = request["id"]
        _require_revision(snap, old, request.get("expected_revision"))
        if nodes[old]["status"] == "superseded":
            raise GraphError("node is already superseded")
        replacement = request.get("replacement_id")
        if replacement is None:
            replacement_spec = request["replacement"]
            if "replacement" in journal["generated_ids"]:
                replacement_spec = {
                    **replacement_spec,
                    "id": journal["generated_ids"]["replacement"],
                }
            replacement = create(replacement_spec)
        else:
            valid_uuid(replacement, "replacement id")
            if replacement not in nodes:
                raise GraphError("replacement node does not exist")
        update(
            old,
            request.get("expected_revision"),
            {"status": "superseded", "superseded_by": replacement},
            allow_supersede=True,
        )
    else:
        raise GraphError(f"unsupported mutation action: {action}")

    schema.validate_graph(nodes)
    changes: dict[str, bytes | None] = {}
    for node_id in sorted(changed):
        raw = schema.dump_node(nodes[node_id], bodies[node_id])
        schema.parse_node(raw, f"{node_id}.md")
        changes[f"{node_id}.md"] = raw
    receipt = {
        "schema": 1,
        "operation_id": operation_id,
        "request_hash": request_hash,
        "action": action,
        "base": snap.commit,
        "preconditions": preconditions,
        "node_ids": sorted(changed),
        "resolved_at": timestamp(when),
        "generated_ids": journal["generated_ids"],
        "changed_hashes": {
            path: digest(data) for path, data in changes.items() if data is not None
        },
    }
    changes[f"operations/{operation_id}.json"] = canonical(receipt) + b"\n"
    candidate = dict(snap.files)
    for path, data in changes.items():
        if data is None:
            raise GraphError("ordinary mutations cannot delete files")
        candidate[path] = data
    candidate_snapshot, _ = parse_snapshot(snap.commit, candidate, snap.config)
    _require_evidence(root, candidate_snapshot)
    message = (
        f"memory: {action}\n\nMemory-Commit-Schema: 1\nMemory-Operation: {action}\nMemory-Operation-Id: {operation_id}\nMemory-Store: {snap.store_id}\n"
        + "".join(f"Memory-Node: {node_id}\n" for node_id in sorted(changed))
    )
    if snap.config:
        oid_length = len(snap.commit)
        commit = git_store.commit_snapshot(
            root, snap.commit, changes, message, ref=branch, expected_ref="0" * oid_length
        )
        state = "prepared"
    else:
        commit = git_store.commit_snapshot(root, snap.commit, changes, message)
        state = "accepted"
    return {
        "operation_id": operation_id,
        "state": state,
        "replayed": False,
        "commit": commit,
        "receipt": receipt,
        "checkout_updated": False,
    }


def _revert_locked(
    root: Path, snap: Snapshot, operation_id: str, request: dict, request_hash: str, branch: str
) -> dict:
    target_id = valid_uuid(request.get("target_operation"), "target operation id")
    target = snap.receipts.get(target_id)
    if target is None:
        raise GraphError("target operation is not accepted on main")
    if target.get("action") in ("repair", "revert"):
        raise GraphError(
            "repair and revert operations require explicit corrective changes, not automatic revert"
        )
    if any(
        receipt.get("action") == "revert" and receipt.get("target_operation") == target_id
        for receipt in snap.receipts.values()
    ):
        raise GraphError("target operation has already been reverted")
    hashes = target.get("changed_hashes")
    if not isinstance(hashes, dict) or any(not NODE_FILE.fullmatch(path) for path in hashes):
        raise GraphError("target receipt has unsupported changed paths")
    previous = historical_snapshot(root, snap, target["base"])
    changes: dict[str, bytes | None] = {}
    preconditions: dict[str, str] = {}
    for path, expected in hashes.items():
        if (
            not isinstance(expected, str)
            or path not in snap.files
            or digest(snap.files[path]) != expected
        ):
            raise GraphError(f"changed node {path} has been modified since the target operation")
        preconditions[path] = expected
        changes[path] = previous.files.get(path)
    candidate = dict(snap.files)
    for path, data in changes.items():
        if data is None:
            candidate.pop(path)
        else:
            candidate[path] = data
    candidate_snapshot, _ = parse_snapshot(snap.commit, candidate, snap.config)
    _require_evidence(root, candidate_snapshot)
    journal = _journal(root, operation_id, "revert", request, request_hash, snap)
    receipt = {
        "schema": 1,
        "operation_id": operation_id,
        "request_hash": request_hash,
        "action": "revert",
        "target_operation": target_id,
        "base": snap.commit,
        "preconditions": preconditions,
        "node_ids": sorted(path[:-3] for path in changes),
        "resolved_at": journal["resolved_at"],
        "generated_ids": {},
        "changed_hashes": {
            path: digest(data) if data is not None else None for path, data in changes.items()
        },
    }
    changes[f"operations/{operation_id}.json"] = canonical(receipt) + b"\n"
    message = (
        f"memory: revert\n\nMemory-Commit-Schema: 1\nMemory-Operation: revert\nMemory-Operation-Id: {operation_id}\nMemory-Store: {snap.store_id}\n"
        + "".join(f"Memory-Node: {node_id}\n" for node_id in receipt["node_ids"])
    )
    if snap.config:
        commit = git_store.commit_snapshot(
            root, snap.commit, changes, message, ref=branch, expected_ref="0" * len(snap.commit)
        )
        state = "prepared"
    else:
        commit = git_store.commit_snapshot(root, snap.commit, changes, message)
        state = "accepted"
    return {
        "operation_id": operation_id,
        "state": state,
        "replayed": False,
        "commit": commit,
        "receipt": receipt,
        "checkout_updated": False,
    }


def _repair_locked(
    root: Path,
    snap: Snapshot,
    errors: list[dict],
    operation_id: str,
    request: dict,
    request_hash: str,
    branch: str,
) -> dict:
    if not snap.store_id or any(
        error["path"] == "store.json" or error["path"].startswith("operations/") for error in errors
    ):
        raise GraphError(
            "store identity or receipt history is damaged; operator recovery is required"
        )
    if any(error["type"] == "UnsupportedSchema" for error in errors):
        raise GraphError("unsupported future schema requires a compatible runtime, not repair")
    repairs = request.get("changes")
    if not isinstance(repairs, list) or not 1 <= len(repairs) <= MAX_BATCH:
        raise GraphError(f"repair changes must contain 1..{MAX_BATCH} items")
    journal = _journal(root, operation_id, "repair", request, request_hash, snap)
    changes: dict[str, bytes | None] = {}
    preconditions: dict[str, str] = {}
    for item in repairs:
        if not isinstance(item, dict):
            raise GraphError("each repair must be an object")
        path = item.get("path")
        if not isinstance(path, str) or not NODE_FILE.fullmatch(path):
            raise GraphError("repair paths must name root UUIDv4 Markdown files")
        if path in changes:
            raise GraphError(f"duplicate repair path {path}")
        if path not in snap.files or item.get("expected_sha256") != digest(snap.files[path]):
            raise GraphError(f"missing or stale raw hash for {path}")
        preconditions[path] = item["expected_sha256"]
        if item.get("action") == "replace":
            content = item.get("content")
            if not isinstance(content, str):
                raise GraphError("replacement content must be a UTF-8 string")
            changes[path] = content.encode("utf-8")
        elif item.get("action") == "quarantine":
            target = f"quarantine/{operation_id}/{path}"
            if target in snap.files:
                raise GraphError("quarantine destination already exists")
            changes[target] = snap.files[path]
            changes[path] = None
        else:
            raise GraphError("repair action must be replace or quarantine")
    candidate = dict(snap.files)
    for path, data in changes.items():
        if data is None:
            candidate.pop(path)
        else:
            candidate[path] = data
    candidate_snapshot, _ = parse_snapshot(snap.commit, candidate, snap.config)
    _require_evidence(root, candidate_snapshot)
    receipt = {
        "schema": 1,
        "operation_id": operation_id,
        "request_hash": request_hash,
        "action": "repair",
        "base": snap.commit,
        "preconditions": preconditions,
        "node_ids": sorted(path[:-3] for path in preconditions),
        "resolved_at": journal["resolved_at"],
        "generated_ids": {},
        "changed_hashes": {
            path: digest(data) if data is not None else None for path, data in changes.items()
        },
    }
    changes[f"operations/{operation_id}.json"] = canonical(receipt) + b"\n"
    message = (
        f"memory: repair\n\nMemory-Commit-Schema: 1\nMemory-Operation: repair\nMemory-Operation-Id: {operation_id}\nMemory-Store: {snap.store_id}\n"
        + "".join(f"Memory-Node: {node_id}\n" for node_id in receipt["node_ids"])
    )
    if snap.config:
        commit = git_store.commit_snapshot(
            root, snap.commit, changes, message, ref=branch, expected_ref="0" * len(snap.commit)
        )
        state = "prepared"
    else:
        commit = git_store.commit_snapshot(root, snap.commit, changes, message)
        state = "accepted"
    return {
        "operation_id": operation_id,
        "state": state,
        "replayed": False,
        "commit": commit,
        "receipt": receipt,
        "checkout_updated": False,
    }


def _reversions(snap: Snapshot, operation_id: str) -> list[str]:
    return sorted(
        key
        for key, receipt in snap.receipts.items()
        if receipt.get("action") == "revert" and receipt.get("target_operation") == operation_id
    )


def operation_status(root: Path, operation_id: str) -> dict:
    valid_uuid(operation_id, "operation id")
    snap, errors = load(root, diagnostic=True)
    if not snap.store_id or any(
        error["path"] == "store.json" or error["path"].startswith("operations/") for error in errors
    ):
        raise GraphError(
            "store identity or receipt history is damaged; operator recovery is required"
        )
    snapshot_info = {
        "snapshot": provenance(snap),
        "snapshot_valid": not errors,
        "diagnostic_error_count": len(errors),
    }
    if operation_id in snap.receipts:
        return {
            "operation_id": operation_id,
            "state": "accepted",
            "receipt": snap.receipts[operation_id],
            "reverted_by": _reversions(snap, operation_id),
            **snapshot_info,
        }
    if snap.config:
        branch = f"refs/heads/memory-graph/{operation_id}"
        try:
            commit, files = git_store.load_snapshot(root, branch)
        except git_store.RefNotFound:
            pass
        else:
            proposal, _ = parse_snapshot(commit, files, snap.config)
            if operation_id in proposal.receipts:
                return {
                    "operation_id": operation_id,
                    "state": "prepared",
                    "publication": "unknown",
                    "commit": commit,
                    "receipt": proposal.receipts[operation_id],
                    **snapshot_info,
                }
    journal_path = root / ".memory-graph" / f"{operation_id}.json"
    if journal_path.is_symlink():
        raise GraphError("operation journal must not be a symlink")
    if journal_path.exists():
        journal = _json_file(journal_path.read_bytes(), str(journal_path))
        if journal.get("store_id") != snap.store_id:
            raise GraphError("operation journal belongs to another store")
        return {
            "operation_id": operation_id,
            "state": "journaled",
            "journal": journal,
            **snapshot_info,
        }
    return {"operation_id": operation_id, "state": "not_found", **snapshot_info}
