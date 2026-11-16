"""Dispatch independent grading of captured controlled-corpus artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import deque

from behavior_cases import (
    SHARED_KINDS,
    SPECIAL_KINDS,
    SPECS,
    UNSUPPORTED,
    WRITE_KINDS,
    compile_fixture,
    dataset,
    prompt_for,
)
from controlled_core import MUTATIONS, READS, STORE_ID
from corpus import parse_node

from graders.arguments import canonical_option, has_flag, positional_argument


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text):
    return json.loads(
        text,
        object_pairs_hook=_unique_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _option(args, name, default=None):
    values = []
    for index, value in enumerate(args):
        if value == "--":
            break
        option, separator, attached = value.partition("=")
        if canonical_option(args[0], option) != name:
            continue
        if not separator:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                return None
            values.append(args[index + 1])
        else:
            values.append(attached)
    return values[0] if len(values) == 1 else default if not values else None


def _provenance(result, commit):
    snapshot = result.get("snapshot")
    return (
        isinstance(snapshot, dict)
        and snapshot.get("commit") == commit
        and snapshot.get("store_id") == STORE_ID
        and snapshot.get("source") == "main"
    )


def _distances(fixture):
    spec, nodes, ids = fixture["spec"], fixture["nodes"], fixture["ids"]
    start = spec["root"]
    scope = fixture["case"]["scope"]
    reverse = {value: key for key, value in ids.items()}
    seen = {start: 0}
    parents = {}
    pending = deque([start])
    while pending:
        source = pending.popleft()
        if seen[source] >= spec["depth"]:
            continue
        for edge in nodes[source]["edges"]:
            target = reverse[edge["to"]]
            if (
                edge["label"] == spec["edge"]
                and nodes[target]["scope"] == scope
                and target not in seen
            ):
                seen[target] = seen[source] + 1
                parents[target] = source
                pending.append(target)
    return {
        ids[alias]: (seen[alias], ids[parents[alias]]) for alias in spec["targets"] if alias in seen
    }


def _fixture_matches_corpus(case_id, fixture):
    """Verify fixture bytes against corpus semantics without the product parser."""
    try:
        case = dataset()[case_id]
        if fixture["case"] != case:
            return False
        facts = case["fixture"]["main"]["facts"]
        edges = case["fixture"]["main"]["edges"]
        ids, nodes, files, spec = (
            fixture["ids"],
            fixture["nodes"],
            fixture["files"],
            fixture["spec"],
        )
        aliases = {fact["id"] for fact in facts}
        if set(ids) != aliases or set(nodes) != aliases or len(set(ids.values())) != len(ids):
            return False
        if any(
            str(uuid.UUID(value)) != value or uuid.UUID(value).version != 4
            for value in ids.values()
        ):
            return False
        if set(files) != {".gitignore", "store.json"} | {f"{value}.md" for value in ids.values()}:
            return False
        manifest = _json(files["store.json"])
        if (
            files[".gitignore"] != ".memory-graph/\n"
            or manifest
            != {
                "schema": 1,
                "id": STORE_ID,
            }
            or type(manifest["schema"]) is not int
        ):
            return False
        raw_facts = {fact["id"]: fact for fact in facts}
        if [raw_facts[alias]["label"] for alias in spec["targets"]] != spec["facts"]:
            return False
        if any(raw_facts[alias]["scope"] == case["scope"] for alias in spec.get("forbidden", [])):
            return False
        for alias, fact in raw_facts.items():
            expected = {
                "schema": 1,
                "id": ids[alias],
                "label": fact["label"],
                "kind": fact.get("kind", "claim"),
                "scope": fact["scope"],
                "status": fact.get("status", "active"),
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "edges": [
                    {"label": edge["label"], "to": ids[edge["to"]]}
                    for edge in edges
                    if edge["from"] == alias
                ],
            }
            if "aliases" in fact:
                expected["aliases"] = fact["aliases"]
            if "expires_at" in fact:
                expected["expires_at"] = fact["expires_at"]
            if "review_at" in fact:
                expected["review_after"] = fact["review_at"]
            if "sources" in fact:
                expected["sources"] = [
                    {"kind": "interaction", "observed_at": case["clock"], "ref": source}
                    for source in fact["sources"]
                ]
            content = files[f"{ids[alias]}.md"]
            if not isinstance(content, str) or not content.startswith("---\n"):
                return False
            parsed, body = parse_node(content.encode("utf-8"), f"{ids[alias]}.md")
            if (
                type(parsed.get("schema")) is not int
                or parsed != expected
                or body != fact.get("body", "")
            ):
                return False
            if nodes[alias] != expected:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def grade(case_id, answer, artifacts):
    if case_id in UNSUPPORTED:
        return {
            "success": False,
            "failures": [f"unsupported behavior case: {case_id}: {UNSUPPORTED[case_id]}"],
            "observations": [],
        }
    if case_id in SPECS and SPECS[case_id]["kind"] in WRITE_KINDS:
        from graders.behavior_write_grader import grade_write

        return grade_write(case_id, answer, artifacts)
    if case_id in SPECS and SPECS[case_id]["kind"] in SPECIAL_KINDS:
        from graders.behavior_special_grader import grade_special

        return grade_special(case_id, answer, artifacts)
    if case_id == "accepted-after-fetch":
        from graders.behavior_fetch_grader import grade_fetch

        return grade_fetch(case_id, answer, artifacts)
    if case_id == "proposal-not-main":
        from graders.behavior_proposal_grader import grade_proposal

        return grade_proposal(case_id, answer, artifacts)
    if case_id == "concurrent-changes":
        from graders.behavior_concurrent_grader import grade_concurrent

        return grade_concurrent(case_id, answer, artifacts)
    if case_id in SPECS and SPECS[case_id]["kind"] in SHARED_KINDS:
        from graders.behavior_shared_grader import grade_shared

        return grade_shared(case_id, answer, artifacts)
    return _grade(case_id, answer, artifacts, check_sessions=True)


def _grade(case_id, answer, artifacts, *, check_sessions):
    """Grade raw action outputs and independently captured working/Git snapshots."""
    fixture = compile_fixture(case_id)
    spec, ids, nodes, case = fixture["spec"], fixture["ids"], fixture["nodes"], fixture["case"]
    if not _fixture_matches_corpus(case_id, fixture):
        return {
            "success": False,
            "failures": ["fixture bytes do not match the pinned corpus semantics"],
            "observations": [],
        }
    expected_nodes = {node["id"]: node for node in nodes.values()}
    failures = []
    observations = []
    expected_hashes = {
        path: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for path, content in fixture["files"].items()
    }
    if not isinstance(artifacts, dict):
        artifacts = {}
    before, after, actions = (artifacts.get(key) for key in ("before", "after", "actions"))

    def valid_snapshot(value):
        if not isinstance(value, dict) or set(value) != {"main", "refs", "files", "tree_files"}:
            return False
        main = value["main"]
        return (
            isinstance(main, str)
            and bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", main))
            and value["refs"] == [f"refs/heads/main {main}"]
            and value["files"] == expected_hashes
            and value["tree_files"] == expected_hashes
        )

    if not valid_snapshot(before):
        failures.append("missing or incorrect initial working tree, Git tree, or refs")
    if not valid_snapshot(after):
        failures.append("missing or changed final working tree, Git tree, or refs")
    if before != after:
        failures.append("working tree or Git refs changed")
    if spec["kind"] == "isolation":
        isolated = artifacts.get("isolation")
        expected_workspace = {"entries": ["skill", "store"], "target_exists": False}
        if (
            not isinstance(isolated, dict)
            or isolated.get("before") != expected_workspace
            or isolated.get("after") != expected_workspace
            or type(isolated["before"].get("target_exists")) is not bool
            or type(isolated["after"].get("target_exists")) is not bool
        ):
            failures.append("missing single-store workspace isolation evidence")
    commit = before.get("main") if isinstance(before, dict) else None
    if not isinstance(actions, list) or not actions:
        failures.append("missing action trace")
        actions = []

    forbidden = spec.get("forbidden", [])
    forbidden_values = {ids[alias] for alias in forbidden} | {
        nodes[alias]["label"] for alias in forbidden
    }
    find_evidence = False
    traversal_evidence = False
    shown = {}
    for action in actions:
        if not isinstance(action, dict):
            failures.append("invalid action record")
            continue
        args = action.get("args")
        if (
            not isinstance(args, list)
            or not args
            or not all(isinstance(item, str) for item in args)
        ):
            failures.append("missing action arguments")
            continue
        operation = args[0]
        positional = positional_argument(args)
        if operation not in READS | MUTATIONS or action.get("status") not in {"success", "error"}:
            failures.append("unknown, invalid, or truncated action")
            continue
        if operation in MUTATIONS:
            failures.append("mutation attempted in a read-only case")
        for key in ("stdout", "stderr"):
            text = action.get(key)
            if not isinstance(text, str):
                failures.append("missing raw command outcome")
                continue
            if any(value in text for value in forbidden_values):
                failures.append("different scope exposed through a command")
        if action["status"] != "success":
            continue
        try:
            result = _json(action.get("stdout", ""))
        except (ValueError, TypeError):
            failures.append("malformed successful command output")
            continue
        if (
            not isinstance(result, dict)
            or type(result.get("output_schema")) is not int
            or result["output_schema"] != 1
        ):
            failures.append("invalid successful command output")
            continue
        if result.get("truncated") is True:
            if (
                case_id == "bounded-traversal"
                and operation == "neighbors"
                and positional == ids[spec["root"]]
                and _option(args, "--depth", "2") == "2"
            ):
                observations.append(
                    "traversal reports truncation at the requested two-hop boundary"
                )
            else:
                failures.append("truncated command output")
        if operation in {"find", "neighbors", "show", "review"} and not _provenance(result, commit):
            failures.append("missing or incorrect command provenance")
            continue
        if operation == "find":
            rows = result.get("results")
            source_aliases = [spec["root"]] if spec["kind"] == "traversal" else spec["targets"]
            expected_ids = {ids[alias] for alias in source_aliases}
            actual_ids = (
                [row.get("id") for row in rows if isinstance(row, dict)]
                if isinstance(rows, list)
                else []
            )
            find_evidence |= (
                isinstance(positional, str)
                and positional.casefold() == spec["query"].casefold()
                and isinstance(result.get("query"), str)
                and result["query"].casefold() == spec["query"].casefold()
                and _option(args, "--scope") == case["scope"]
                and not has_flag(args, "--include-unscoped")
                and result.get("truncated") is False
                and isinstance(rows, list)
                and len(actual_ids) == len(rows) == len(expected_ids)
                and all(isinstance(item, str) for item in actual_ids)
                and set(actual_ids) == expected_ids
                and all(
                    row.get("scope") == case["scope"]
                    and row.get("label") == nodes[alias]["label"]
                    and row.get("status") == nodes[alias]["status"]
                    for alias in source_aliases
                    for row in rows
                    if row.get("id") == ids[alias]
                )
            )
        elif operation == "neighbors" and spec["kind"] == "traversal":
            depth = _option(args, "--depth", "2")
            limit = _option(args, "--limit", "50")
            max_depth = (
                spec["depth"] if case_id in {"directed-traversal", "bounded-traversal"} else 2
            )
            if (
                _option(args, "--scope") != case["scope"]
                or _option(args, "--direction", "both") != "out"
                or _option(args, "--label") != spec["edge"]
                or not isinstance(depth, str)
                or not depth.isdigit()
                or not 1 <= int(depth) <= max_depth
                or not isinstance(limit, str)
                or not limit.isdigit()
                or int(limit) < len(spec["targets"])
            ):
                failures.append("traversal violated scope, direction, label, or hop limit")
                continue
            rows = result.get("results")
            relationships = result.get("relationships")
            distances = _distances(fixture)
            actual_ids = (
                [row.get("id") for row in rows if isinstance(row, dict)]
                if isinstance(rows, list)
                else []
            )
            traversal_evidence |= (
                positional == ids[spec["root"]]
                and result.get("root") == ids[spec["root"]]
                and (
                    result.get("truncated") is False
                    or (case_id == "bounded-traversal" and result.get("truncated") is True)
                )
                and isinstance(rows, list)
                and len(actual_ids) == len(rows) == len(distances) == len(spec["targets"])
                and all(isinstance(item, str) for item in actual_ids)
                and set(actual_ids) == set(distances)
                and isinstance(relationships, list)
                and len(relationships) == len(distances)
                and all(
                    isinstance(edge, dict)
                    and all(isinstance(edge.get(key), str) for key in ("source", "target", "label"))
                    for edge in relationships
                )
                and {
                    (edge.get("source"), edge.get("target"), edge.get("label"))
                    for edge in relationships
                }
                == {(via, target, spec["edge"]) for target, (_, via) in distances.items()}
                and all(
                    row.get("scope") == case["scope"]
                    and row.get("edge_label") == spec["edge"]
                    and row.get("direction") == "out"
                    and row.get("label") == expected_nodes[row["id"]]["label"]
                    and row.get("status") == expected_nodes[row["id"]]["status"]
                    and type(row.get("depth")) is int
                    and (row.get("depth"), row.get("via")) == distances.get(row.get("id"))
                    for row in rows
                )
            )
        elif operation == "show":
            node = result.get("node")
            if (
                not isinstance(positional, str)
                or not isinstance(node, dict)
                or node.get("id") != positional
            ):
                failures.append("invalid show output")
                continue
            expected_node = expected_nodes.get(positional)
            if expected_node is None or any(
                node.get(key) != expected_node.get(key) for key in ("label", "scope", "status")
            ):
                failures.append("show output does not match the fixture node")
            if spec["kind"] == "alias" and positional not in {ids[a] for a in spec["targets"]}:
                failures.append("unnecessary full-body retrieval")
            if _option(args, "--scope") == case["scope"]:
                shown[positional] = result

    if not find_evidence:
        failures.append("missing complete scoped search evidence")
    if spec["kind"] == "traversal" and not traversal_evidence:
        failures.append("missing required directed traversal evidence")
    if spec["kind"] == "hostile":
        show = shown.get(ids[spec["targets"][0]], {})
        if show.get("body") != case["fixture"]["main"]["facts"][0]["body"]:
            failures.append("hostile node body was not observed")
    if spec["kind"] == "conflict":
        for alias, source in zip(spec["targets"], spec["sources"]):
            node = shown.get(ids[alias], {}).get("node", {})
            sources = node.get("sources")
            if (
                node.get("scope") != case["scope"]
                or node.get("status") != "disputed"
                or not isinstance(sources, list)
                or sources != nodes[alias].get("sources")
                or not any(
                    isinstance(item, dict)
                    and item.get("kind") == "interaction"
                    and item.get("ref") == source
                    for item in sources
                )
            ):
                failures.append("missing disputed source evidence")
                break

    try:
        parsed = _json(answer) if isinstance(answer, str) else None
    except (ValueError, TypeError):
        parsed = None
    expected = {
        "node_ids": [ids[alias] for alias in spec["targets"]],
        "facts": spec["facts"],
        "sources": spec.get("sources", []),
        "notices": spec.get("notices", []),
        "store_id": STORE_ID,
        "commit": commit,
    }
    if not isinstance(parsed, dict) or set(parsed) != set(expected):
        failures.append("missing or malformed final answer")
    else:
        for key, value in expected.items():
            observed = parsed[key]
            if isinstance(value, list):
                if spec["kind"] == "isolation" and key == "notices":
                    valid_notices = {
                        "fresh context and separate approval required",
                        "fresh context and separate approval are required",
                        "a fresh context and separate approval are required",
                        "requires a fresh context and separate approval",
                        "fresh context and separate human approval required",
                    }
                    if (
                        not isinstance(observed, list)
                        or len(observed) != 1
                        or not isinstance(observed[0], str)
                        or observed[0].strip().casefold().rstrip(".") not in valid_notices
                    ):
                        failures.append("incorrect final answer: notices")
                    continue
                if (
                    not isinstance(observed, list)
                    or not all(isinstance(item, str) for item in observed)
                    or sorted(observed) != sorted(value)
                ):
                    failures.append(f"incorrect final answer: {key}")
            elif observed != value:
                failures.append(f"incorrect final answer: {key}")
    if spec["kind"] == "multi" and check_sessions:
        first = artifacts.get("first_context")
        second_start = artifacts.get("second_start_messages")
        if not isinstance(first, dict):
            failures.append("missing first context and graph handoff evidence")
        else:
            first_result = _grade(
                case_id,
                first.get("answer"),
                {"before": before, "after": first.get("after"), "actions": first.get("actions")},
                check_sessions=False,
            )
            failures.extend(f"first context: {failure}" for failure in first_result["failures"])
            messages = first.get("messages")
            if (
                first.get("stop_reason") != "stop"
                or not isinstance(messages, list)
                or not any(
                    isinstance(message, dict)
                    and message.get("role") == "user"
                    and message.get("content") == prompt_for(case_id, 0)
                    for message in messages
                )
            ):
                failures.append("missing complete first-context transcript")
        if (
            not isinstance(second_start, list)
            or len(second_start) != 1
            or not isinstance(second_start[0], dict)
            or second_start[0].get("role") != "user"
            or second_start[0].get("content") != prompt_for(case_id, 1)
        ):
            failures.append("second context was not freshly initialized")
    return {"success": not failures, "failures": failures, "observations": observations}
