"""Load declarative cases and real graph fixtures into the legacy dataset contract."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from functools import lru_cache
from pathlib import Path
from string import Template

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.events import AliasEvent, ScalarEvent

ROOT = Path(__file__).parent
DATASET = ROOT / "scenarios.json"
LOCK = ROOT / "corpus.lock.json"
STORE_ID = "52bd8904-9a68-4391-9c0c-03af22bf2211"
FIELDS = {
    "id",
    "label",
    "scope",
    "kind",
    "status",
    "aliases",
    "sources",
    "body",
    "expires_at",
    "review_at",
}
KINDS = {
    "create",
    "reuse",
    "retry",
    "correction",
    "shared-read",
    "offline-stale",
    "accepted-after-fetch",
    "timed-review",
    "timed-read",
    "malformed",
    "isolation",
    "find",
    "multi",
    "traversal",
    "alias",
    "conflict",
    "hostile",
}
SPEC_FIELDS = {
    "kind",
    "query",
    "targets",
    "facts",
    "forbidden",
    "sources",
    "notices",
    "root",
    "edge",
    "depth",
    "instruction",
    "seed_request",
}


def semantic_id(case_id, alias):
    digest = hashlib.sha256(f"behavior-v1:{case_id}:{alias}".encode()).digest()[:16]
    return str(uuid.UUID(bytes=digest, version=4))


def load_yaml(raw, label):
    try:
        text = raw.decode("utf-8")
        yaml = YAML(typ="safe", pure=True)
        yaml.allow_duplicate_keys = False
        for event in yaml.parse(text):
            if (
                isinstance(event, AliasEvent)
                or getattr(event, "anchor", None) is not None
                or getattr(event, "tag", None) is not None
                or (isinstance(event, ScalarEvent) and event.style is None and event.value == "<<")
            ):
                raise ValueError("YAML aliases, anchors, tags and merges are not allowed")
        return yaml.load(text)
    except (UnicodeError, YAMLError, RecursionError) as exc:
        raise ValueError(f"invalid YAML in {label}: {exc}") from exc


def parse_node(content, label):
    if not content.startswith(b"---\n"):
        raise ValueError(f"missing frontmatter in {label}")
    metadata, delimiter, body = content[4:].partition(b"\n---\n")
    if not delimiter:
        raise ValueError(f"missing closing frontmatter in {label}")
    node = load_yaml(metadata, label)
    if not isinstance(node, dict):
        raise ValueError(f"invalid node frontmatter in {label}")  # noqa: TRY004
    return node, body.decode("utf-8")


def _sources():
    paths = [ROOT / "corpus.yaml"]
    for directory in (ROOT / "cases", ROOT / "fixtures" / "graphs"):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"missing or invalid corpus directory: {directory}")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"corpus symlinks are not allowed: {path}")
            if path.is_file():
                paths.append(path)
    result = []
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid corpus file: {path}")
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("cases/") and path.suffix != ".yaml":
            raise ValueError(f"unexpected case file: {relative}")
        if (
            relative.startswith("fixtures/graphs/")
            and path.name not in {".gitignore", "store.json"}
            and path.suffix != ".md"
        ):
            raise ValueError(f"unexpected fixture file: {relative}")
        data = path.read_bytes()
        if len(data) > 1024 * 1024:
            raise ValueError(f"oversized corpus file: {relative}")
        result.append((relative, data))
    return tuple(result)


def source_hash(sources):
    digest = hashlib.sha256()
    for name, raw in sources:
        digest.update(name.encode("utf-8") + b"\0" + hashlib.sha256(raw).digest())
    return digest.hexdigest()


def _validate_spec(case_id, spec, ids, nodes):
    kind = spec["kind"]
    for field in ("facts", "targets", "forbidden", "sources", "notices"):
        if field in spec and (
            not isinstance(spec[field], list)
            or any(not isinstance(value, str) or not value for value in spec[field])
        ):
            raise ValueError(f"invalid controlled {field}: {case_id}")
    if "facts" not in spec or (kind != "create" and "targets" not in spec):
        raise ValueError(f"missing controlled facts or targets: {case_id}")
    if kind not in {"create", "retry"} and (
        not isinstance(spec.get("query"), str) or not spec["query"].strip()
    ):
        raise ValueError(f"missing controlled query: {case_id}")
    if kind in {"create", "correction"} and not spec.get("sources"):
        raise ValueError(f"missing controlled source: {case_id}")
    if kind in {"reuse", "retry", "correction"} and len(spec["targets"]) != 1:
        raise ValueError(f"expected one controlled target: {case_id}")
    if any(alias not in ids for alias in spec.get("targets", []) + spec.get("forbidden", [])):
        raise ValueError(f"unknown controlled node alias: {case_id}")
    if kind == "traversal" and (
        spec.get("root") not in ids
        or not isinstance(spec.get("edge"), str)
        or not spec["edge"]
        or type(spec.get("depth")) is not int
        or spec["depth"] < 1
    ):
        raise ValueError(f"invalid controlled traversal: {case_id}")
    if kind == "retry" and spec["seed_request"]["id_alias"] not in ids:
        raise ValueError(f"unknown retry seed alias: {case_id}")
    if kind == "retry":
        seed = spec["seed_request"]
        node = nodes[seed["id_alias"]]
        if any(
            seed[field] != node.get(field) for field in ("label", "kind", "scope", "expires_at")
        ):
            raise ValueError(f"retry seed differs from fixture: {case_id}")


@lru_cache(maxsize=2)
def _assemble(sources):
    files = dict(sources)
    metadata = load_yaml(files["corpus.yaml"], "corpus.yaml")
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema_version",
        "dataset_version",
        "description",
        "trial_count",
    }:
        raise ValueError("invalid corpus metadata")
    entries = []
    specs = {}
    blocked = {}
    fixtures = {}
    used = {"corpus.yaml"}
    for path, raw in sources:
        if not path.startswith("cases/"):
            continue
        doc = load_yaml(raw, path)
        if not isinstance(doc, dict):
            raise ValueError(f"invalid case: {path}")  # noqa: TRY004
        order = doc.pop("order", None)
        case_id = doc.get("id")
        lane = doc.get("lane")
        if (
            type(order) is not int
            or order < 0
            or not isinstance(case_id, str)
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", case_id)
            or Path(path).stem != case_id
            or lane not in {"behavior", "native"}
        ):
            raise ValueError(f"invalid case identity: {path}")
        controlled = doc.pop("controlled", None)
        if lane == "behavior":
            if not isinstance(controlled, dict):
                raise ValueError(f"missing controlled specification: {case_id}")
            if (
                set(controlled) == {"blocked"}
                and isinstance(controlled["blocked"], str)
                and controlled["blocked"]
            ):
                blocked[case_id] = controlled["blocked"]
            elif (
                isinstance(controlled.get("kind"), str)
                and controlled["kind"] in KINDS
                and set(controlled) <= SPEC_FIELDS
            ):
                instruction = controlled.get("instruction")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError(f"missing controlled instruction: {case_id}")
                variables = {"operation_id": "id", "clock": "clock"}
                if controlled.get("sources"):
                    variables["source"] = "source"
                if controlled.get("kind") == "retry":
                    seed = controlled.get("seed_request")
                    if (
                        not isinstance(seed, dict)
                        or set(seed) != {"id_alias", "label", "kind", "scope", "expires_at"}
                        or not all(isinstance(value, str) and value for value in seed.values())
                    ):
                        raise ValueError(f"invalid retry seed: {case_id}")
                    variables["seed_node_id"] = "id"
                    variables.update(
                        {f"seed_{key}": value for key, value in seed.items() if key != "id_alias"}
                    )
                elif "seed_request" in controlled:
                    raise ValueError(f"unexpected retry seed: {case_id}")
                try:
                    Template(instruction).substitute(variables)
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"invalid controlled instruction: {case_id}") from exc
                specs[case_id] = controlled
            else:
                raise ValueError(f"invalid controlled specification: {case_id}")
            fixture = doc.get("fixture")
            if not isinstance(fixture, dict) or not isinstance(fixture.get("main"), dict):
                raise ValueError(f"invalid fixture: {case_id}")
            main = fixture["main"]
            directory = f"fixtures/graphs/{case_id}"
            prefix = directory + "/"
            graph_files = {
                name[len(prefix) :]: value for name, value in sources if name.startswith(prefix)
            }
            used.update(prefix + name for name in graph_files)
            if graph_files.get(".gitignore") != b".memory-graph/\n":
                raise ValueError(f"invalid fixture ignore file: {case_id}")
            try:
                manifest = json.loads(graph_files["store.json"])
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid fixture store: {case_id}") from exc
            if manifest != {"schema": 1, "id": STORE_ID} or type(manifest["schema"]) is not int:
                raise ValueError(f"invalid fixture store: {case_id}")
            if "invalid_nodes" in main:
                if set(main) != {"invalid_nodes"} or set(graph_files) != {
                    ".gitignore",
                    "store.json",
                    f"{semantic_id(case_id, 'broken')}.md",
                }:
                    raise ValueError(f"invalid malformed fixture: {case_id}")
                fixture["main"] = {"invalid_nodes": main["invalid_nodes"]}
                fixtures[case_id] = {
                    "files": graph_files,
                    "ids": {"broken": semantic_id(case_id, "broken")},
                    "nodes": {},
                }
            else:
                descriptors = main.get("nodes")
                if set(main) != {"nodes"} or not isinstance(descriptors, dict):
                    raise ValueError(f"invalid fixture nodes: {case_id}")
                ids = {}
                nodes = {}
                bodies = {}
                fields_by_alias = {}
                for alias, descriptor in descriptors.items():
                    if not isinstance(descriptor, dict) or set(descriptor) - (
                        FIELDS - {"id"} | {"edges"}
                    ):
                        raise ValueError(f"invalid fixture node descriptor: {case_id}")
                    if (
                        not isinstance(alias, str)
                        or not alias
                        or alias in ids
                        or not isinstance(descriptor.get("label"), str)
                        or not descriptor["label"]
                    ):
                        raise ValueError(f"invalid fixture node fields: {case_id}")
                    node_id = semantic_id(case_id, alias)
                    name = f"{node_id}.md"
                    if name not in graph_files:
                        raise ValueError(f"missing fixture node {name}")
                    node, body = parse_node(graph_files[name], name)
                    if node.get("id") != node_id:
                        raise ValueError(f"fixture node identity mismatch: {name}")
                    if node.get("label") != descriptor["label"] or node.get(
                        "scope"
                    ) != descriptor.get("scope", doc["scope"]):
                        raise ValueError(f"fixture node preview differs from Markdown: {case_id}")
                    for field in FIELDS - {"id", "label", "scope"}:
                        if field == "body":
                            actual = body
                            default = ""
                        elif field == "sources":
                            actual = [source["ref"] for source in node.get("sources", [])]
                            default = []
                        else:
                            default = {"status": "active", "kind": "claim", "aliases": []}.get(
                                field
                            )
                            actual = node.get(
                                "review_after" if field == "review_at" else field, default
                            )
                        if actual != descriptor.get(field, default):
                            raise ValueError(
                                f"fixture node {field} preview differs from Markdown: {case_id}"
                            )
                    fields = ["label", "scope", "id"] + [
                        field for field in descriptor if field in FIELDS - {"id", "label", "scope"}
                    ]
                    ids[alias], nodes[alias], bodies[alias], fields_by_alias[alias] = (
                        node_id,
                        node,
                        body,
                        fields,
                    )
                if set(graph_files) != {".gitignore", "store.json"} | {
                    f"{x}.md" for x in ids.values()
                }:
                    raise ValueError(f"unexpected fixture files: {case_id}")
                reverse = {value: alias for alias, value in ids.items()}
                facts = []
                edges = []
                for alias, node in nodes.items():
                    fact = {}
                    for field in fields_by_alias[alias]:
                        if field == "id":
                            fact[field] = alias
                        elif field == "body":
                            fact[field] = bodies[alias]
                        elif field == "review_at":
                            fact[field] = node["review_after"]
                        elif field == "sources":
                            fact[field] = [source["ref"] for source in node["sources"]]
                        else:
                            fact[field] = node[field]
                    facts.append(fact)
                    actual_edges = []
                    for edge in node["edges"]:
                        if set(edge) != {"label", "to"} or edge["to"] not in reverse:
                            raise ValueError(f"invalid fixture edge: {case_id}")
                        actual_edges.append({"label": edge["label"], "to": reverse[edge["to"]]})
                        edges.append({"from": alias, **actual_edges[-1]})
                    if actual_edges != descriptors[alias].get("edges", []):
                        raise ValueError(f"fixture edge preview differs from Markdown: {case_id}")
                fixture["main"] = {"facts": facts, "edges": edges}
                fixtures[case_id] = {"files": graph_files, "ids": ids, "nodes": nodes}
            if case_id in specs:
                _validate_spec(
                    case_id, specs[case_id], fixtures[case_id]["ids"], fixtures[case_id]["nodes"]
                )
        elif controlled is not None:
            raise ValueError(f"native case cannot define a controlled adapter: {case_id}")
        keys = (
            "id",
            "lane",
            "category",
            "prompts",
            "fixture",
            "scope",
            "clock",
            *(("should_activate", "critical") if lane == "native" else ()),
            "allowed_mutations",
            "forbidden_actions",
            "checks",
        )
        doc = {
            **{key: doc[key] for key in keys if key in doc},
            **{key: value for key, value in doc.items() if key not in keys},
        }
        entries.append((order, doc))
        used.add(path)
    if {name for name, _ in sources} != used:
        raise ValueError("unreferenced corpus fixture or case file")
    entries.sort(key=lambda item: item[0])
    if [order for order, _ in entries] != list(range(len(entries))):
        raise ValueError("case order must be unique and contiguous")
    cases = [case for _, case in entries]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case ID")
    return {**metadata, "cases": cases}, specs, blocked, fixtures


def compile_corpus():
    sources = _sources()
    dataset, specs, blocked, fixtures = _assemble(sources)
    return copy.deepcopy((dataset, specs, blocked, fixtures, source_hash(sources)))


def dataset_bytes(dataset):
    return (json.dumps(dataset, indent=2, ensure_ascii=False) + "\n").encode()


def verify_corpus():
    try:
        dataset, specs, blocked, fixtures, digest = compile_corpus()
    except (KeyError, TypeError, UnicodeError) as exc:
        raise ValueError(f"invalid corpus: {exc}") from exc
    try:
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        raw = DATASET.read_bytes()
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read corpus snapshot or lock: {exc}") from exc
    expected = {
        "schema_version": 1,
        "source_sha256": digest,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if lock != expected or dataset_bytes(dataset) != raw:
        raise ValueError(
            "corpus source or snapshot changed; review and run build_dataset.py --write"
        )
    return dataset, specs, blocked, fixtures, digest
