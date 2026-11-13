"""Portable node frontmatter parsing, serialization, and graph validation."""

from __future__ import annotations

import copy
import json
import math
import re
from calendar import monthrange
from datetime import UTC, datetime
from functools import lru_cache
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator
from ruamel.yaml import YAML
from ruamel.yaml.events import (
    AliasEvent,
    CollectionEndEvent,
    CollectionStartEvent,
    DocumentStartEvent,
)
from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode

MAX_NODE_BYTES = 256 * 1024
MAX_FRONTMATTER_BYTES = 64 * 1024
MAX_DEPTH = 32

_FIELDS = (
    "schema",
    "id",
    "label",
    "kind",
    "aliases",
    "tags",
    "scope",
    "status",
    "superseded_by",
    "created_at",
    "updated_at",
    "review_after",
    "expires_at",
    "importance",
    "confidence",
    "evidence_mode",
    "properties",
    "edges",
    "sources",
)
_TIMESTAMPS = {"created_at", "updated_at", "review_after", "expires_at"}
_RFC3339 = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[Tt](\d{2}):(\d{2}):(\d{2})(\.\d+)?"
    r"([Zz]|[+-]\d{2}:\d{2})$",
    re.ASCII,
)
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")
_YAML_TAGS = {
    "tag:yaml.org,2002:" + name for name in ("str", "null", "bool", "int", "float", "seq", "map")
}


class UnsupportedSchema(ValueError):
    """The node declares a newer schema version than this runtime supports."""


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    resource = resources.files("memory_graph").joinpath("assets", "node.schema.json")
    if resource.is_file():
        text = resource.read_text(encoding="utf-8")
    else:
        text = (Path(__file__).resolve().parents[2] / "assets" / "node.schema.json").read_text(
            encoding="utf-8"
        )
    schema = json.loads(text)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _where(path: tuple) -> str:
    text = "metadata"
    for part in path:
        text += f"[{part}]" if isinstance(part, int) else f".{part}"
    return text


def _check_json(value: object, path: tuple = (), depth: int = 0, active: set | None = None) -> None:
    if active is None:
        active = set()
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ValueError(f"{_where(path)} contains an invalid Unicode surrogate")
        return
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{_where(path)} must be finite")
        return
    if type(value) not in (dict, list):
        raise ValueError(f"{_where(path)} must contain only JSON-compatible values")
    if depth >= MAX_DEPTH:
        raise ValueError(f"{_where(path)} exceeds maximum nesting depth {MAX_DEPTH}")
    if id(value) in active:
        raise ValueError(f"{_where(path)} contains a cycle")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{_where(path)} has a non-string mapping key")  # noqa: TRY004
                _check_json(key, path, depth + 1, active)
                _check_json(child, path + (key,), depth + 1, active)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                _check_json(child, path + (index,), depth + 1, active)
    finally:
        active.remove(id(value))


def _utc_timestamp(value: str, where: str) -> str:
    match = _RFC3339.fullmatch(value)
    if not match:
        raise ValueError(f"{where} must be an RFC 3339 timestamp with an offset")
    date, hour, minute, second, fraction, offset = match.groups()
    if int(second) > 60:
        raise ValueError(f"{where} is not a valid RFC 3339 timestamp")
    if offset.upper() == "Z":
        offset = "+00:00"
    elif int(offset[1:3]) > 23 or int(offset[4:]) > 59:
        raise ValueError(f"{where} has an invalid UTC offset")
    try:
        # Retain arbitrary fractional precision, including a leap-second spelling.
        parsed = datetime.fromisoformat(
            f"{date}T{hour}:{minute}:{min(int(second), 59):02d}{offset}"
        ).astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{where} is not a valid RFC 3339 timestamp") from exc
    if second == "60" and (
        parsed.hour != 23
        or parsed.minute != 59
        or parsed.day != monthrange(parsed.year, parsed.month)[1]
    ):
        raise ValueError(f"{where} has an invalid leap second")
    sec = "60" if second == "60" else f"{parsed.second:02d}"
    return (
        f"{parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d}T"
        f"{parsed.hour:02d}:{parsed.minute:02d}:{sec}{fraction or ''}Z"
    )


def timestamp_reached(value: str, now: datetime) -> bool:
    """Return whether an RFC 3339 instant has been reached by an aware datetime.

    Leap seconds sort after :59 and before the following minute. Python datetime
    cannot itself represent a time within a leap second.
    """
    if not isinstance(value, str):
        raise ValueError("timestamp must be an RFC 3339 string")  # noqa: TRY004
    if not isinstance(now, datetime):
        raise ValueError("now must be an aware datetime")  # noqa: TRY004
    try:
        if now.utcoffset() is None:
            raise ValueError("now must be an aware datetime")
        current = now.astimezone(UTC)
    except (OverflowError, TypeError) as exc:
        raise ValueError("now must be a valid aware datetime") from exc

    normalized = _utc_timestamp(value, "timestamp")
    target_second = (
        int(normalized[0:4]),
        int(normalized[5:7]),
        int(normalized[8:10]),
        int(normalized[11:13]),
        int(normalized[14:16]),
        int(normalized[17:19]),
    )
    current_second = (
        current.year,
        current.month,
        current.day,
        current.hour,
        current.minute,
        current.second,
    )
    if current_second != target_second:
        return current_second > target_second
    target_fraction = normalized[19:-1].removeprefix(".").rstrip("0")
    current_fraction = f"{current.microsecond:06d}".rstrip("0")
    return current_fraction >= target_fraction


def _absolute_uri(value: str, where: str) -> None:
    try:
        parts = urlsplit(value)
        valid = (
            bool(_SCHEME.fullmatch(parts.scheme))
            and bool(value[len(parts.scheme) + 1 :])
            and bool(re.fullmatch(r"[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+", value))
            and not re.search(r"%(?![0-9A-Fa-f]{2})", value)
        )
        if parts.scheme.lower() in ("http", "https") and not parts.netloc:
            valid = False
        if parts.netloc:
            _ = parts.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{where} must be an absolute URI")


def _validate_node(node: dict) -> None:
    if isinstance(node, dict) and type(node.get("schema")) is int and node["schema"] > 1:
        raise UnsupportedSchema(f"unsupported schema version {node['schema']}")
    _check_json(node)
    if isinstance(node, dict) and "schema" in node and type(node["schema"]) is not int:
        raise ValueError("metadata.schema must be an integer schema version")
    errors = sorted(
        _validator().iter_errors(node),
        key=lambda error: (list(map(str, error.absolute_path)), error.message),
    )
    if errors:
        error = errors[0]
        if error.validator == "oneOf" and isinstance(error.instance, dict):
            kinds = ("repository_file", "web", "interaction", "memory")
            kind = error.instance.get("kind")
            if kind in kinds:
                branch = kinds.index(kind)
                relevant = [item for item in error.context if item.schema_path[0] == branch]
                if relevant:
                    error = relevant[0]
                    if error.validator == "anyOf" and kind == "web":
                        raise ValueError(
                            f"{_where(tuple(error.absolute_path))}: web source requires "
                            "content_hash, version, or mutable: true"
                        )
            else:
                raise ValueError(
                    f"{_where(tuple(error.absolute_path))}: unknown source kind {kind!r}"
                )
        raise ValueError(f"{_where(tuple(error.absolute_path))}: {error.message}")
    for name in _TIMESTAMPS:
        if node.get(name) is not None:
            _utc_timestamp(node[name], f"metadata.{name}")
    seen = set()
    for index, edge in enumerate(node["edges"]):
        key = (edge["label"], edge["to"])
        if key in seen:
            raise ValueError(f"metadata.edges[{index}]: duplicate edge {key!r}")
        seen.add(key)
    for index, source in enumerate(node.get("sources", [])):
        where = f"metadata.sources[{index}]"
        _utc_timestamp(source["observed_at"], f"{where}.observed_at")
        if "uri" in source:
            _absolute_uri(source["uri"], f"{where}.uri")
        if source["kind"] == "repository_file":
            path = source["path"]
            if (
                path.startswith("/")
                or re.match(r"^[A-Za-z]:", path)
                or "\\" in path
                or any(part in ("", ".", "..") for part in path.split("/"))
            ):
                raise ValueError(
                    f"{where}.path must be a repository-relative path without traversal"
                )


def _check_yaml_tree(node: object, path: tuple = ()) -> None:
    if isinstance(node, MappingNode):
        for key, value in node.value:
            if key.tag == "tag:yaml.org,2002:merge":
                raise ValueError(f"{_where(path)}: YAML merge keys are not allowed")
            key_name = key.value if isinstance(key, ScalarNode) else None
            child_path = path + (key_name,)
            timestamp = (
                len(path) == 0
                and key_name in _TIMESTAMPS
                or len(path) == 2
                and path[0] == "sources"
                and key_name == "observed_at"
            )
            if timestamp and not (
                isinstance(value, ScalarNode)
                and (value.tag == "tag:yaml.org,2002:null" or value.style in ("'", '"'))
            ):
                raise ValueError(f"{_where(child_path)} must be a quoted RFC 3339 string")
            _check_yaml_tree(value, child_path)
    elif isinstance(node, SequenceNode):
        for index, item in enumerate(node.value):
            _check_yaml_tree(item, path + (index,))


def _yaml() -> YAML:
    yaml = YAML(typ="safe", pure=True)
    yaml.version = (1, 2)
    yaml.allow_duplicate_keys = False
    return yaml


def _parse_metadata(raw: bytes) -> dict:
    text = raw.decode("utf-8")
    try:
        depth = 0
        forbidden = None
        for event in _yaml().parse(text):
            if isinstance(event, DocumentStartEvent) and event.version not in (None, (1, 2)):
                raise ValueError("only YAML 1.2 is supported")
            if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
                forbidden = forbidden or "YAML anchors and aliases are not allowed"
            tag = getattr(event, "tag", None)
            if tag is not None and tag not in _YAML_TAGS:
                forbidden = forbidden or f"YAML custom or unsupported tag {tag!r} is not allowed"
            if isinstance(event, CollectionStartEvent):
                depth += 1
                if depth > MAX_DEPTH:
                    raise ValueError(f"metadata exceeds maximum nesting depth {MAX_DEPTH}")
            elif isinstance(event, CollectionEndEvent):
                depth -= 1
        tree = _yaml().compose(text)
        if isinstance(tree, MappingNode):
            for key, value in tree.value:
                if (
                    isinstance(key, ScalarNode)
                    and key.value == "schema"
                    and isinstance(value, ScalarNode)
                    and value.tag == "tag:yaml.org,2002:int"
                ):
                    version = _yaml().load(value.value)
                    if type(version) is int and version > 1:
                        raise UnsupportedSchema(f"unsupported schema version {version}")
        if forbidden:
            raise ValueError(forbidden)
        _check_yaml_tree(tree)
        node = _yaml().load(text)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(node, dict):
        raise ValueError("frontmatter must be a mapping")  # noqa: TRY004
    _validate_node(node)
    return node


def parse_node(data: bytes, path: str) -> tuple[dict, bytes]:
    """Parse and validate a UTF-8 node, retaining its exact Markdown body bytes."""
    if not isinstance(data, bytes):
        raise ValueError("node data must be bytes")  # noqa: TRY004
    if len(data) > MAX_NODE_BYTES:
        raise ValueError(f"{path}: node exceeds {MAX_NODE_BYTES} bytes")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: node must be UTF-8") from exc
    if data.startswith(b"---\n"):
        start = 4
    elif data.startswith(b"---\r\n"):
        start = 5
    else:
        raise ValueError(f"{path}: frontmatter must start with ---")
    position = start
    while position <= len(data):
        end = data.find(b"\n", position)
        next_position = len(data) if end < 0 else end + 1
        line = data[position:next_position].rstrip(b"\r\n")
        if line.rstrip(b" \t") == b"---":
            raw = data[start:position]
            if len(raw) > MAX_FRONTMATTER_BYTES:
                raise ValueError(f"{path}: frontmatter exceeds {MAX_FRONTMATTER_BYTES} bytes")
            try:
                node = _parse_metadata(raw)
            except UnsupportedSchema as exc:
                raise UnsupportedSchema(f"{path}: {exc}") from exc
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
            if Path(path).name != f"{node['id']}.md":
                raise ValueError(f"{path}: filename must match node id {node['id']}.md")
            return node, data[next_position:]
        if next_position - start > MAX_FRONTMATTER_BYTES:
            raise ValueError(f"{path}: frontmatter exceeds {MAX_FRONTMATTER_BYTES} bytes")
        if end < 0:
            break
        position = next_position
    raise ValueError(f"{path}: missing closing frontmatter delimiter")


def dump_node(node: dict, body: bytes) -> bytes:
    """Write deterministic frontmatter in UTC followed by the unchanged body."""
    if not isinstance(body, bytes):
        raise ValueError("body must be bytes")  # noqa: TRY004
    try:
        body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("body must be UTF-8") from exc
    _validate_node(node)
    canonical = copy.deepcopy(node)
    for name in _TIMESTAMPS:
        if canonical.get(name) is not None:
            canonical[name] = _utc_timestamp(canonical[name], f"metadata.{name}")
    for index, source in enumerate(canonical.get("sources", [])):
        source["observed_at"] = _utc_timestamp(
            source["observed_at"], f"metadata.sources[{index}].observed_at"
        )

    def json_value(value: object) -> str:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(", ", ": "), allow_nan=False
        )

    lines = []
    for name in _FIELDS:
        if name not in canonical:
            continue
        value = canonical[name]
        if name in ("edges", "sources") and value:
            lines.append(f"{name}:")
            lines.extend(f"  - {json_value(item)}" for item in value)
        else:
            lines.append(f"{name}: {json_value(value)}")
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    if len(raw) > MAX_FRONTMATTER_BYTES:
        raise ValueError(f"frontmatter exceeds {MAX_FRONTMATTER_BYTES} bytes")
    result = b"---\n" + raw + b"---\n" + body
    if len(result) > MAX_NODE_BYTES:
        raise ValueError(f"node exceeds {MAX_NODE_BYTES} bytes")
    return result


def validate_graph(nodes: dict[str, dict]) -> None:
    """Validate all node IDs, outgoing references, and supersession chains."""
    if not isinstance(nodes, dict):
        raise ValueError("graph must be a mapping from node IDs to metadata")  # noqa: TRY004
    for node_id, node in nodes.items():
        try:
            _validate_node(node)
        except UnsupportedSchema as exc:
            raise UnsupportedSchema(f"node {node_id}: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"node {node_id}: {exc}") from exc
        if node_id != node["id"]:
            raise ValueError(f"node key {node_id!r} does not match id {node['id']!r}")
    for node_id, node in nodes.items():
        for edge in node["edges"]:
            if edge["to"] not in nodes:
                raise ValueError(
                    f"node {node_id}: edge {edge['label']!r} targets missing node {edge['to']}"
                )
        target = node.get("superseded_by")
        if target is not None:
            if target not in nodes:
                raise ValueError(f"node {node_id}: superseded_by targets missing node {target}")
            if target == node_id:
                raise ValueError(f"node {node_id}: cannot supersede itself")
            if node.get("scope") != nodes[target].get("scope"):
                raise ValueError(f"node {node_id}: replacement {target} must have the same scope")
    done = set()
    for start in nodes:
        trail = set()
        current: str | None = start
        while current is not None and current not in done:
            if current in trail:
                raise ValueError(f"node {start}: supersession cycle includes {current}")
            trail.add(current)
            current = nodes[current].get("superseded_by")
        done.update(trail)
