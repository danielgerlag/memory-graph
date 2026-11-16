"""Independently grade raw Git/tree/receipt evidence for standalone writes."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from behavior_cases import compile_fixture, retry_request
from controlled_core import MUTATIONS, READS, STORE_ID, command_arguments

from graders.arguments import canonical_option, has_flag, positional_argument

NODE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.md$")


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def _json(raw):
    return json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _node(raw, path):
    from jsonschema import Draft202012Validator, FormatChecker
    from jsonschema.exceptions import SchemaError, ValidationError
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    text = raw.decode("utf-8")
    if not text.startswith("---\n"):
        raise ValueError("missing node frontmatter")
    front, delimiter, body = text[4:].partition("\n---\n")
    if not delimiter:
        raise ValueError("missing node delimiter")
    yaml = YAML(typ="safe")
    yaml.allow_duplicate_keys = False
    try:
        node = yaml.load(front)
        schema_path = Path(__file__).resolve().parents[2] / "memory-graph/assets/node.schema.json"
        validator = Draft202012Validator(
            _json(schema_path.read_bytes()), format_checker=FormatChecker()
        )
        validator.validate(node)
    except (YAMLError, ValidationError, SchemaError) as exc:
        raise ValueError("invalid node metadata") from exc
    if not isinstance(node, dict) or node.get("id") != path[:-3]:
        raise ValueError("node identity mismatch")
    return node, body


def _unpack(snapshot):
    required = {"main", "refs", "files", "tree_files", "tree", "working", "parents", "message"}
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise ValueError("missing complete raw snapshot")
    main = snapshot["main"]
    if (
        not isinstance(main, str)
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", main)
        or snapshot["refs"] != [f"refs/heads/main {main}"]
        or not isinstance(snapshot["parents"], list)
        or not isinstance(snapshot["message"], str)
    ):
        raise ValueError("invalid main/ref/commit evidence")
    decoded = {}
    for field, hashes in (("tree", "tree_files"), ("working", "files")):
        values = snapshot[field]
        if not isinstance(values, dict) or not isinstance(snapshot[hashes], dict):
            raise TypeError("missing artifact map")
        decoded[field] = {
            path: base64.b64decode(value, validate=True) for path, value in values.items()
        }
        if {
            path: hashlib.sha256(value).hexdigest() for path, value in decoded[field].items()
        } != snapshot[hashes]:
            raise ValueError("raw artifact hashes differ")
    return decoded


def _option(args, name):
    args = command_arguments(args)
    if not args:
        return None
    values = []
    for index, value in enumerate(args):
        if value == "--":
            break
        option, separator, attached = value.partition("=")
        if canonical_option(args[0], option) != name:
            continue
        if not separator and index + 1 < len(args):
            values.append(args[index + 1])
        elif separator:
            values.append(attached)
    return values[0] if len(values) == 1 else None


def _request_hash(action, request):
    encoded = json.dumps(
        {"action": action, "request": request},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source(node, reference, clock):
    return node.get("evidence_mode") == "reported" and node.get("sources") == [
        {"kind": "interaction", "observed_at": clock, "ref": reference}
    ]


def _parse_nodes(tree):
    nodes = {}
    for path, raw in tree.items():
        if NODE.fullmatch(path):
            node, body = _node(raw, path)
            nodes[node["id"]] = (node, body)
        elif path not in {".gitignore", "store.json"} and not re.fullmatch(
            r"operations/[0-9a-f-]{36}\.json", path
        ):
            raise ValueError("unexpected committed file")
    if (
        _json(tree["store.json"]) != {"schema": 1, "id": STORE_ID}
        or tree[".gitignore"] != b".memory-graph/\n"
    ):
        raise ValueError("incorrect store identity")
    return nodes


def _valid_receipt(tree, operation_id, action, request, base, changed, preconditions):
    path = f"operations/{operation_id}.json"
    receipt = _json(tree[path])
    return (
        isinstance(receipt, dict)
        and receipt.get("schema") == 1
        and receipt.get("operation_id") == operation_id
        and receipt.get("action") == action
        and receipt.get("request_hash") == _request_hash(action, request)
        and receipt.get("base") == base
        and receipt.get("preconditions") == preconditions
        and receipt.get("node_ids") == sorted(path[:-3] for path in changed)
        and receipt.get("changed_hashes")
        == {path: hashlib.sha256(tree[path]).hexdigest() for path in changed}
    )


def _check_action(action, operation_id, expected_kind):
    args = action.get("args")
    if isinstance(args, list):
        args = command_arguments(args)
    if (
        not isinstance(args, list)
        or not args
        or not all(isinstance(x, str) for x in args)
        or args[0] != expected_kind
        or _option(args, "--operation") != operation_id
        or _option(args, "--input") != "-"
        or action.get("status") != "success"
        or not isinstance(action.get("input_text"), str)
        or not isinstance(action.get("stdout"), str)
        or not isinstance(action.get("stderr"), str)
    ):
        raise ValueError("invalid mutation action evidence")
    request, response = _json(action["input_text"]), _json(action["stdout"])
    if (
        not isinstance(request, dict)
        or not isinstance(response, dict)
        or type(response.get("output_schema")) is not int
        or response["output_schema"] != 1
        or response.get("state") != "accepted"
        or response.get("operation_id") != operation_id
    ):
        raise ValueError("mutation was not accepted")
    return request, response


def _changed(before, after):
    return {path for path in set(before) | set(after) if before.get(path) != after.get(path)}


def _local_effects(before, after, operation_id):
    allowed = {f".memory-graph/{operation_id}.json", ".memory-graph/write.lock"}
    changes = _changed(before, after)
    if not changes <= allowed:
        raise ValueError("unexpected working-tree changes")
    journal_path = f".memory-graph/{operation_id}.json"
    if journal_path not in after or ".memory-graph/write.lock" not in after:
        raise ValueError("missing local journal or lock")
    return _json(after[journal_path])


def _answer(answer, node_ids, facts, sources, notices, commit):
    value = _json(answer)
    expected = {
        "node_ids": node_ids,
        "facts": facts,
        "sources": sources,
        "notices": notices,
        "store_id": STORE_ID,
        "commit": commit,
    }
    if not isinstance(value, dict) or set(value) != set(expected):
        return False
    for key, item in expected.items():
        got = value[key]
        if isinstance(item, list):
            if (
                not isinstance(got, list)
                or not all(isinstance(x, str) for x in got)
                or sorted(got) != sorted(item)
            ):
                return False
        elif got != item:
            return False
    return True


def grade_write(case_id, answer, artifacts):
    fixture = compile_fixture(case_id)
    spec, case, operation_id = fixture["spec"], fixture["case"], fixture["operation_id"]
    failures = []
    try:
        if not isinstance(artifacts, dict):
            raise TypeError("missing mutation artifacts")
        before, after, actions = (
            artifacts.get("before"),
            artifacts.get("after"),
            artifacts.get("actions"),
        )
        raw_before, raw_after = _unpack(before), _unpack(after)
        old_nodes, new_nodes = _parse_nodes(raw_before["tree"]), _parse_nodes(raw_after["tree"])
        if not isinstance(actions, list) or not actions:
            raise ValueError("missing action trace")
        for action in actions:
            if (
                not isinstance(action, dict)
                or not isinstance(action.get("args"), list)
                or not action["args"]
                or not all(isinstance(x, str) for x in action["args"])
                or action.get("status") not in {"success", "error"}
                or not isinstance(action.get("stdout"), str)
                or not isinstance(action.get("stderr"), str)
            ):
                raise ValueError("incomplete action trace")
            args = command_arguments(action["args"])
            if not args:
                raise ValueError("missing action arguments")
            operation = args[0]
            if operation not in READS | MUTATIONS | {"operation"}:
                raise ValueError("forbidden action")
            if operation == "operation" and args[1:] != ["status", operation_id]:
                raise ValueError("wrong operation status request")
            if operation in {"find", "show", "neighbors", "review"} and (
                _option(args, "--scope") != case["scope"] or has_flag(args, "--include-unscoped")
            ):
                raise ValueError("read command omitted or widened the required scope")
        mutations = [
            action for action in actions if command_arguments(action["args"])[0] in MUTATIONS
        ]
        expected_before = {path: text.encode() for path, text in fixture["files"].items()}
        if spec["kind"] != "retry" and (
            raw_before["tree"] != expected_before or raw_before["working"] != expected_before
        ):
            raise ValueError("initial fixture differs from corpus")
        expected_action = {
            "create": "create",
            "reuse": "update",
            "correction": "supersede",
            "retry": "create",
        }[spec["kind"]]
        parsed_mutations = [
            _check_action(action, operation_id, expected_action) for action in mutations
        ]
        if parsed_mutations:
            if spec["kind"] == "retry":
                if any(
                    request != retry_request(case_id)
                    or response.get("replayed") is not True
                    or response.get("receipt")
                    != _json(raw_before["tree"][f"operations/{operation_id}.json"])
                    or response.get("snapshot", {}).get("commit") != before["main"]
                    or response.get("snapshot", {}).get("store_id") != STORE_ID
                    for request, response in parsed_mutations
                ):
                    raise ValueError("retry changed the original request or did not reconcile")
            elif parsed_mutations[0][1].get("replayed") is not False or any(
                request != parsed_mutations[0][0] or response.get("replayed") is not True
                for request, response in parsed_mutations[1:]
            ):
                raise ValueError("mutations did not reuse one operation id and request")
        if spec["kind"] in {"create", "correction"} and not parsed_mutations:
            raise ValueError("missing accepted mutation")

        if spec["kind"] == "retry":
            bootstrap = artifacts.get("bootstrap")
            if not isinstance(bootstrap, dict):
                raise ValueError("missing withheld seed acknowledgment")
            seed = _unpack(bootstrap.get("before"))
            expected_empty = {k: v for k, v in expected_before.items() if not k.endswith(".md")}
            if seed["tree"] != expected_empty or seed["working"] != expected_empty:
                raise ValueError("retry seed did not start from empty fixture")
            request, response = _check_action(bootstrap["action"], operation_id, "create")
            seed_id = fixture["ids"][spec["seed_request"]["id_alias"]]
            if (
                request != retry_request(case_id)
                or response.get("replayed") is not False
                or before["parents"] != [bootstrap["before"]["main"]]
                or set(raw_before["tree"])
                != set(seed["tree"]) | {f"{seed_id}.md", f"operations/{operation_id}.json"}
                or not _valid_receipt(
                    raw_before["tree"],
                    operation_id,
                    "create",
                    request,
                    bootstrap["before"]["main"],
                    {f"{seed_id}.md"},
                    {},
                )
                or response.get("commit") != before["main"]
                or response.get("receipt")
                != _json(raw_before["tree"][f"operations/{operation_id}.json"])
            ):
                raise ValueError("seed acceptance is not independently proven")
            seed_journal = _local_effects(seed["working"], raw_before["working"], operation_id)
            if (
                seed_journal.get("store_id") != STORE_ID
                or seed_journal.get("base") != bootstrap["before"]["main"]
                or seed_journal.get("request_hash") != _request_hash("create", request)
            ):
                raise ValueError("retry seed journal is invalid")
            if len(old_nodes) != 1 or seed_id not in old_nodes:
                raise ValueError("retry seed is missing or duplicated")
            old_node, _ = old_nodes[seed_id]
            if (
                old_node.get("label") != spec["facts"][0]
                or old_node.get("scope") != case["scope"]
                or old_node.get("expires_at") != spec["seed_request"]["expires_at"]
            ):
                raise ValueError("retry seed content or expiration differs")
            if before != after:
                raise ValueError("retry changed the accepted snapshot or local state")
            status_proofs = [
                a
                for a in actions
                if command_arguments(a["args"]) == ["operation", "status", operation_id]
                and a["status"] == "success"
            ]
            valid_status = False
            for action in status_proofs:
                output = _json(action["stdout"])
                if not (
                    isinstance(output, dict)
                    and type(output.get("output_schema")) is int
                    and output["output_schema"] == 1
                    and output.get("state") == "accepted"
                    and output.get("operation_id") == operation_id
                    and output.get("snapshot_valid") is True
                    and type(output.get("diagnostic_error_count")) is int
                    and output["diagnostic_error_count"] == 0
                    and output.get("snapshot", {}).get("commit") == before["main"]
                    and output.get("snapshot", {}).get("store_id") == STORE_ID
                    and output.get("receipt")
                    == _json(raw_before["tree"][f"operations/{operation_id}.json"])
                ):
                    raise ValueError("operation status lacks valid accepted snapshot evidence")
                valid_status = True
            if not valid_status and not parsed_mutations:
                raise ValueError("no accepted operation reconciliation")
            answer_ids = [seed_id]
        elif spec["kind"] == "reuse" and not parsed_mutations:
            target_id = fixture["ids"][spec["targets"][0]]
            if before != after or set(new_nodes) != {target_id}:
                raise ValueError("repeated fact changed or duplicated")
            answer_ids = [target_id]
        else:
            if len(parsed_mutations) < 1:
                raise ValueError("missing mutation")
            request, response = parsed_mutations[0]
            if after["parents"] != [before["main"]] or after["main"] == before["main"]:
                raise ValueError("mutation is not exactly one forward commit")
            if (
                response.get("commit") != after["main"]
                or response.get("checkout_updated") is not False
            ):
                raise ValueError("accepted mutation commit evidence differs")
            if not all(
                raw_after["tree"].get(path) == value
                for path, value in raw_before["tree"].items()
                if not path.endswith(".md")
            ):
                raise ValueError("unrelated committed data changed")
            journal = _local_effects(raw_before["working"], raw_after["working"], operation_id)
            if (
                not isinstance(journal, dict)
                or journal.get("schema") != 1
                or journal.get("store_id") != STORE_ID
                or journal.get("base") != before["main"]
                or journal.get("request_hash") != _request_hash(expected_action, request)
            ):
                raise ValueError("local operation journal differs from request")
            if spec["kind"] == "create":
                if old_nodes or len(new_nodes) != 1:
                    raise ValueError("creation did not produce exactly one node")
                new_id = next(iter(new_nodes))
                node, body = new_nodes[new_id]
                if (
                    node.get("label") != spec["facts"][0]
                    or node.get("scope") != case["scope"]
                    or node.get("status") != "active"
                    or node.get("edges") != []
                    or body.strip() not in {"", spec["facts"][0]}
                    or not _source(node, spec["sources"][0], case["clock"])
                ):
                    raise ValueError("created fact or reported evidence differs")
                changed = {f"{new_id}.md"}
                preconditions = {}
                answer_ids = [new_id]
            elif spec["kind"] == "reuse":
                old_id = fixture["ids"][spec["targets"][0]]
                if set(new_nodes) != {old_id} or request.get("id") != old_id:
                    raise ValueError("repeat update changed identity or duplicated node")
                prior, prior_body = old_nodes[old_id]
                node, body = new_nodes[old_id]
                if (
                    request.get("expected_revision")
                    != hashlib.sha256(raw_before["tree"][f"{old_id}.md"]).hexdigest()
                    or {k: v for k, v in node.items() if k != "updated_at"}
                    != {k: v for k, v in prior.items() if k != "updated_at"}
                    or body != prior_body
                    or set(request.get("patch", {})) - {"aliases", "tags"}
                ):
                    raise ValueError("repeat update changed unrelated knowledge")
                changed = {f"{old_id}.md"}
                preconditions = {
                    old_id: hashlib.sha256(raw_before["tree"][f"{old_id}.md"]).hexdigest()
                }
                answer_ids = [old_id]
            else:
                target_alias = spec["targets"][0]
                old_id = fixture["ids"][target_alias]
                if set(old_nodes) != {old_id} or len(new_nodes) != 2 or old_id not in new_nodes:
                    raise ValueError("supersession did not preserve exactly two nodes")
                new_id = next(node_id for node_id in new_nodes if node_id != old_id)
                previous, previous_body = old_nodes[old_id]
                old, old_body = new_nodes[old_id]
                replacement, replacement_body = new_nodes[new_id]
                if (
                    request.get("id") != old_id
                    or request.get("expected_revision")
                    != hashlib.sha256(raw_before["tree"][f"{old_id}.md"]).hexdigest()
                    or previous.get("label") != fixture["nodes"][target_alias]["label"]
                    or previous.get("scope") != case["scope"]
                    or previous.get("status") != "active"
                    or previous.get("sources") != fixture["nodes"][target_alias].get("sources")
                    or old.get("status") != "superseded"
                    or old.get("superseded_by") != new_id
                    or {
                        k: v
                        for k, v in old.items()
                        if k not in {"status", "superseded_by", "updated_at"}
                    }
                    != {
                        k: v
                        for k, v in previous.items()
                        if k not in {"status", "superseded_by", "updated_at"}
                    }
                    or old_body != previous_body
                    or replacement.get("label") != spec["facts"][0]
                    or replacement.get("scope") != case["scope"]
                    or replacement.get("status") != "active"
                    or replacement.get("edges") != []
                    or replacement_body.strip() not in {"", spec["facts"][0]}
                    or not _source(replacement, spec["sources"][0], case["clock"])
                ):
                    raise ValueError("supersession lost history or evidence")
                changed = {f"{old_id}.md", f"{new_id}.md"}
                preconditions = {
                    old_id: hashlib.sha256(raw_before["tree"][f"{old_id}.md"]).hexdigest()
                }
                answer_ids = [new_id]
            if (
                _changed(raw_before["tree"], raw_after["tree"])
                != changed | {f"operations/{operation_id}.json"}
                or not _valid_receipt(
                    raw_after["tree"],
                    operation_id,
                    expected_action,
                    request,
                    before["main"],
                    changed,
                    preconditions,
                )
                or response.get("receipt")
                != _json(raw_after["tree"][f"operations/{operation_id}.json"])
            ):
                raise ValueError("commit or receipt contains unexpected effects")
            for trailer in (
                "Memory-Commit-Schema: 1",
                f"Memory-Operation: {expected_action}",
                f"Memory-Operation-Id: {operation_id}",
                f"Memory-Store: {STORE_ID}",
            ):
                if trailer not in after["message"].splitlines():
                    raise ValueError("missing commit provenance trailer")

        if spec["kind"] != "retry":
            finds = [
                a
                for a in actions
                if command_arguments(a["args"])[0] == "find" and a["status"] == "success"
            ]
            if spec["kind"] == "reuse":
                first_mutation = actions.index(mutations[0]) if mutations else len(actions)
                found = False
                for action in actions[:first_mutation]:
                    if (
                        command_arguments(action["args"])[0] != "find"
                        or action["status"] != "success"
                    ):
                        continue
                    output = _json(action["stdout"])
                    rows = output.get("results", []) if isinstance(output, dict) else []
                    positional = positional_argument(action["args"])
                    found |= (
                        isinstance(positional, str)
                        and positional.casefold() == spec["query"].casefold()
                        and isinstance(output, dict)
                        and type(output.get("output_schema")) is int
                        and output["output_schema"] == 1
                        and output.get("query", "").casefold() == spec["query"].casefold()
                        and output.get("truncated") is False
                        and output.get("snapshot", {}).get("commit") == before["main"]
                        and output.get("snapshot", {}).get("store_id") == STORE_ID
                        and output.get("snapshot", {}).get("source") == "main"
                        and [row.get("id") for row in rows] == answer_ids
                        and all(
                            row.get("label") == spec["facts"][0]
                            and row.get("scope") == case["scope"]
                            and row.get("status") == "active"
                            for row in rows
                        )
                    )
                if not found:
                    raise ValueError("repeat did not search the existing fact before reuse")
            if spec["kind"] == "correction":
                mutation_index = actions.index(mutations[0])
                finds = [
                    a
                    for a in actions[mutation_index + 1 :]
                    if command_arguments(a["args"])[0] == "find" and a["status"] == "success"
                ]
                found = False
                for action in finds:
                    output = _json(action["stdout"])
                    rows = output.get("results", []) if isinstance(output, dict) else []
                    positional = positional_argument(action["args"])
                    found |= (
                        isinstance(output, dict)
                        and isinstance(positional, str)
                        and type(output.get("output_schema")) is int
                        and output["output_schema"] == 1
                        and output.get("query", "").casefold() == positional.casefold()
                        and output.get("truncated") is False
                        and output.get("snapshot", {}).get("commit") == after["main"]
                        and output.get("snapshot", {}).get("store_id") == STORE_ID
                        and output.get("snapshot", {}).get("source") == "main"
                        and [row.get("id") for row in rows] == answer_ids
                        and all(
                            row.get("label") == spec["facts"][0]
                            and row.get("scope") == case["scope"]
                            and row.get("status") == "active"
                            for row in rows
                        )
                    )
                if not found:
                    raise ValueError("current recall did not show only the replacement")
        if not _answer(
            answer,
            answer_ids,
            spec["facts"],
            spec.get("sources", []),
            spec.get("notices", []),
            after["main"],
        ):
            raise ValueError("answer lacks correct outcome or provenance")
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError, ImportError) as exc:
        failures.append(str(exc) or type(exc).__name__)
    return {"success": not failures, "failures": failures, "observations": []}
