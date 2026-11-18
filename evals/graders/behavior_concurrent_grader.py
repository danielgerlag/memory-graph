"""Grade competing proposals, accepted main, and stale merge rejection."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime

from behavior_cases import compile_fixture, conflict_request, semantic_id
from controlled_core import MUTATIONS, READS, STORE_ID, command_arguments

from graders.arguments import has_flag, positional_argument, verified_help
from graders.behavior_grader import _json, _option
from graders.behavior_shared_grader import REMOTE, _decode
from graders.behavior_write_grader import _answer, _node, _request_hash, _valid_receipt


def _local(snapshot, base, tracking, loser, operation_id, expected):
    keys = {"main", "refs", "files", "tree_files", "tree", "working", "parents", "message"}
    if not isinstance(snapshot, dict) or set(snapshot) != keys or snapshot["main"] != base:
        raise ValueError("missing local main snapshot")
    if snapshot["parents"] != [] or _decode(snapshot["tree"], snapshot["tree_files"]) != expected:
        raise ValueError("local base tree changed")
    required = {
        f"refs/heads/main {base}",
        f"refs/heads/memory-graph/{operation_id} {loser}",
        f"refs/remotes/origin/main {tracking}",
    }
    optional = f"refs/remotes/origin/memory-graph/{operation_id} {loser}"
    refs = snapshot["refs"]
    if (
        not isinstance(refs, list)
        or refs != sorted(refs)
        or len(refs) != len(set(refs))
        or set(refs) not in (required, required | {optional})
    ):
        raise ValueError("unexpected local refs")
    working = _decode(snapshot["working"], snapshot["files"])
    paths = set(expected) | {
        ".memory-graph/config.json",
        ".memory-graph/write.lock",
        f".memory-graph/{operation_id}.json",
    }
    if set(working) != paths or any(working[path] != value for path, value in expected.items()):
        raise ValueError("unexpected local files")
    if working[".memory-graph/write.lock"] != b"":
        raise ValueError("unexpected local lock")
    return working


def _config(raw, commit, fetched):
    value = _json(raw)
    if (
        not isinstance(value, dict)
        or set(value)
        != {"remote", "mode", "store_id", "fetched_at", "fetched_commit", "last_fetch"}
        or value["remote"] != "origin"
        or value["mode"] != "propose"
        or value["store_id"] != STORE_ID
        or value["fetched_commit"] != commit
        or value["last_fetch"] != value["fetched_at"]
    ):
        raise ValueError("invalid proposal attachment")
    if fetched:
        if (
            not isinstance(value["fetched_at"], str)
            or datetime.fromisoformat(value["fetched_at"]).tzinfo is None
        ):
            raise ValueError("missing fetch timestamp")
    elif value["fetched_at"] is not None:
        raise ValueError("unexpected initial fetch timestamp")
    return value


def _proposal(record, commit, base, expected, node_path, operation_id, request):
    if (
        not isinstance(record, dict)
        or set(record) != {"commit", "tree", "parents", "message"}
        or record["commit"] != commit
        or record["parents"] != [base]
        or not isinstance(record["message"], str)
    ):
        raise ValueError("invalid proposal commit or ancestry")
    tree = _decode(record["tree"])
    receipt_path = f"operations/{operation_id}.json"
    if set(tree) != set(expected) | {receipt_path}:
        raise ValueError("proposal tree has unexpected paths")
    if any(tree[path] != content for path, content in expected.items() if path != node_path):
        raise ValueError("proposal changed unrelated data")
    revision = hashlib.sha256(expected[node_path]).hexdigest()
    if not _valid_receipt(
        tree, operation_id, "update", request, base, {node_path}, {request["id"]: revision}
    ):
        raise ValueError("invalid stale-base receipt or precondition")
    for trailer in (
        "Memory-Commit-Schema: 1",
        "Memory-Operation: update",
        f"Memory-Operation-Id: {operation_id}",
        f"Memory-Store: {STORE_ID}",
    ):
        if trailer not in record["message"].splitlines():
            raise ValueError("proposal provenance missing")
    node, body = _node(tree[node_path], node_path)
    old, old_body = _node(expected[node_path], node_path)
    patch = request["patch"]
    if (
        node.get("label") != patch["label"]
        or node.get("status") != "disputed"
        or node.get("scope") != "orchard"
        or node.get("evidence_mode") != "reported"
        or node.get("sources") != patch["sources"]
        or body != patch["body"]
        or old_body != ""
        or {
            k: v
            for k, v in node.items()
            if k not in {"label", "status", "sources", "evidence_mode", "updated_at"}
        }
        != {
            k: v
            for k, v in old.items()
            if k not in {"label", "status", "sources", "evidence_mode", "updated_at"}
        }
    ):
        raise ValueError("proposal did not preserve the distinct reported evidence")
    return tree, node


def _setup_action(record, operation, request, commit, base, tree, path):
    if not isinstance(record, dict):
        raise TypeError("missing proposal setup action")
    args = record.get("args")
    if (
        args != ["update", "--operation", operation, "--input", "-"]
        or record.get("status") != "success"
        or not isinstance(record.get("input_text"), str)
        or not isinstance(record.get("stdout"), str)
        or not isinstance(record.get("stderr"), str)
        or _json(record.get("input_text", "")) != request
    ):
        raise ValueError("incorrect proposal action")
    output = _json(record.get("stdout", ""))
    if (
        not isinstance(output, dict)
        or type(output.get("output_schema")) is not int
        or output["output_schema"] != 1
        or output.get("state") != "prepared"
        or output.get("replayed") is not False
        or output.get("operation_id") != operation
        or output.get("commit") != commit
        or output.get("receipt") != _json(tree[f"operations/{operation}.json"])
        or output.get("receipt", {}).get("base") != base
        or output.get("receipt", {}).get("changed_hashes", {}).get(path)
        != hashlib.sha256(tree[path]).hexdigest()
    ):
        raise ValueError("proposal response differs from Git artifacts")


def _published(record, operation):
    if (
        not isinstance(record, dict)
        or record.get("status") != "success"
        or type(record.get("returncode")) is not int
        or record["returncode"] != 0
        or not isinstance(record.get("stdout"), str)
        or not isinstance(record.get("stderr"), str)
    ):
        raise ValueError("missing proposal publication")
    ref = f"refs/heads/memory-graph/{operation}"
    if record.get("args") != ["push", "origin", f"{ref}:{ref}"]:
        raise ValueError("proposal was not published by a non-forced Git push")


def grade_concurrent(case_id, answer, artifacts):
    failures = []
    try:
        fixture = compile_fixture(case_id)
        case, ids = fixture["case"], fixture["ids"]
        if case_id != "concurrent-changes" or not isinstance(artifacts, dict):
            raise ValueError("missing concurrent-change artifacts")
        before, after, setup = (
            artifacts.get("before"),
            artifacts.get("after"),
            artifacts.get("conflict_setup"),
        )
        keys = {"local", "remote", "tracking", "loser", "winner", "integration"}
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or set(before) != keys
            or set(after) != keys
            or not isinstance(setup, dict)
        ):
            raise ValueError("missing independent clone or integration artifacts")
        base, loser, winner, candidate = (
            setup.get(key) for key in ("base_commit", "loser_commit", "winner_commit", "candidate")
        )
        if (
            not all(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value)
                for value in (base, loser, winner, candidate)
            )
            or len({base, loser, winner, candidate}) != 4
        ):
            raise ValueError("invalid or duplicate conflict commits")
        pending_op, competing_op = fixture["operation_id"], semantic_id(case_id, "$competing")
        node_id = ids["cart"]
        node_path = f"{node_id}.md"
        expected = {path: value.encode() for path, value in fixture["files"].items()}
        old_node, old_body = _node(expected[node_path], node_path)
        if (
            old_node.get("label") != "cart is red"
            or old_node.get("scope") != "orchard"
            or old_node.get("kind") != "claim"
            or old_node.get("edges") != []
            or old_node.get("status") != "active"
            or old_body != ""
            or old_node.get("sources")
            != [{"kind": "interaction", "observed_at": case["clock"], "ref": "witness:a"}]
        ):
            raise ValueError("base fact or original evidence differs")
        revision = hashlib.sha256(expected[node_path]).hexdigest()
        pending_request = conflict_request(case_id, "pending", revision)
        competing_request = conflict_request(case_id, "competing", revision)
        if set(expected) != {".gitignore", "store.json", node_path}:
            raise ValueError("base fixture includes unexpected files")
        manifest = _json(expected["store.json"])
        if (
            expected[".gitignore"] != b".memory-graph/\n"
            or manifest != {"schema": 1, "id": STORE_ID}
            or type(manifest["schema"]) is not int
        ):
            raise ValueError("base store identity differs")
        for actor, request in (("pending", pending_request), ("competing", competing_request)):
            update = case["fixture"][f"{actor}_update"]
            independently_expected = {
                "id": node_id,
                "expected_revision": revision,
                "patch": {
                    "label": update["label"],
                    "body": update["body"],
                    "status": update["status"],
                    "evidence_mode": "reported",
                    "sources": [
                        {"kind": "interaction", "observed_at": case["clock"], "ref": source}
                        for source in update["sources"]
                    ],
                },
            }
            if request != independently_expected:
                raise ValueError("proposal request differs from the pinned conflict fixture")
        if pending_request["patch"] == competing_request["patch"]:
            raise ValueError("fixture updates are not distinct")
        before_work = _local(before["local"], base, base, loser, pending_op, expected)
        after_work = _local(after["local"], base, winner, loser, pending_op, expected)
        initial = _config(before_work[".memory-graph/config.json"], base, False)
        final = _config(after_work[".memory-graph/config.json"], winner, True)
        if any(
            before_work[path] != after_work[path]
            for path in before_work
            if path != ".memory-graph/config.json"
        ):
            raise ValueError("fetch changed unrelated local state")
        journal = _json(before_work[f".memory-graph/{pending_op}.json"])
        if (
            journal.get("store_id") != STORE_ID
            or journal.get("base") != base
            or journal.get("request_hash") != _request_hash("update", pending_request)
        ):
            raise ValueError("pending journal differs")
        attach = artifacts.get("setup")
        if (
            not isinstance(attach, dict)
            or attach.get("args") != ["attach", "--remote", "origin", "--mode", "propose"]
            or not isinstance(attach.get("stdout"), str)
            or not isinstance(attach.get("stderr"), str)
        ):
            raise ValueError("missing bound proposal attachment")
        attached = _json(attach.get("stdout", ""))
        if (
            not isinstance(attached, dict)
            or type(attached.get("output_schema")) is not int
            or attached["output_schema"] != 1
            or {k: attached.get(k) for k in initial} != initial
        ):
            raise ValueError("attachment output differs")
        actor_attach = setup.get("actor_attach")
        if (
            not isinstance(actor_attach, dict)
            or actor_attach.get("args") != ["attach", "--remote", "origin", "--mode", "propose"]
            or actor_attach.get("status") != "success"
            or not isinstance(actor_attach.get("stdout"), str)
            or not isinstance(actor_attach.get("stderr"), str)
        ):
            raise ValueError("missing separate actor attachment")
        actor_config = _json(actor_attach.get("stdout", ""))
        if (
            not isinstance(actor_config, dict)
            or type(actor_config.get("output_schema")) is not int
            or actor_config["output_schema"] != 1
            or {k: actor_config.get(k) for k in initial} != initial
        ):
            raise ValueError("separate actor was not attached at the same base")
        for side in ("loser", "winner", "integration", "remote"):
            if before[side] != after[side]:
                raise ValueError("a proposal, integration candidate, or remote changed")
        for key in ("loser", "winner"):
            if not isinstance(before[key], dict) or set(before[key]) != {"local", "remote"}:
                raise ValueError("missing separate proposal branches")
            if (
                not isinstance(before[key]["local"], dict)
                or before[key]["local"] != before[key]["remote"]
            ):
                raise ValueError("proposal branch was not retained remotely")
        losing_tree, losing_node = _proposal(
            before["loser"]["local"], loser, base, expected, node_path, pending_op, pending_request
        )
        winning_tree, winning_node = _proposal(
            before["winner"]["local"],
            winner,
            base,
            expected,
            node_path,
            competing_op,
            competing_request,
        )
        if losing_node["sources"] == winning_node["sources"]:
            raise ValueError("conflicting reports do not retain distinct evidence")
        _setup_action(
            setup.get("pending_update"),
            pending_op,
            pending_request,
            loser,
            base,
            losing_tree,
            node_path,
        )
        _setup_action(
            setup.get("competing_update"),
            competing_op,
            competing_request,
            winner,
            base,
            winning_tree,
            node_path,
        )
        _published(setup.get("pending_publish"), pending_op)
        _published(setup.get("competing_publish"), competing_op)
        checked = setup.get("accepted_check")
        if (
            not isinstance(checked, dict)
            or checked.get("args") != ["check-merge", "--base", base, "--candidate", winner]
            or checked.get("status") != "success"
            or not isinstance(checked.get("stdout"), str)
            or not isinstance(checked.get("stderr"), str)
        ):
            raise ValueError("accepted proposal was not independently checked")
        checked_output = _json(checked.get("stdout", ""))
        if (
            not isinstance(checked_output, dict)
            or type(checked_output.get("output_schema")) is not int
            or checked_output["output_schema"] != 1
            or checked_output.get("valid") is not True
            or checked_output.get("base") != base
            or checked_output.get("candidate") != winner
            or checked_output.get("store_id") != STORE_ID
            or checked_output.get("new_operations") != [competing_op]
        ):
            raise ValueError("accepted proposal validation is missing")
        if setup.get("acceptance") != {"command": ["update-ref", "refs/heads/main", winner, base]}:
            raise ValueError("accepted main was not advanced with a compare-and-swap")
        remote = before["remote"]
        expected_remote_refs = sorted(
            [
                f"refs/heads/main {winner}",
                f"refs/heads/memory-graph/{pending_op} {loser}",
                f"refs/heads/memory-graph/{competing_op} {winner}",
            ]
        )
        if (
            not isinstance(remote, dict)
            or set(remote) != {"url", "main", "refs", "tree"}
            or remote["url"] != REMOTE
            or remote["main"] != winner
            or remote["refs"] != expected_remote_refs
            or _decode(remote["tree"]) != winning_tree
        ):
            raise ValueError("accepted remote main or retained proposal refs differ")
        tracking_before, tracking_after = before["tracking"], after["tracking"]
        tracking_keys = {"commit", "tree", "parents", "message"}
        if (
            not isinstance(tracking_before, dict)
            or not isinstance(tracking_after, dict)
            or set(tracking_before) != tracking_keys
            or set(tracking_after) != tracking_keys
            or tracking_before["commit"] != base
            or tracking_before["parents"] != []
            or _decode(tracking_before["tree"]) != expected
            or tracking_after["commit"] != winner
            or tracking_after["parents"] != [base]
            or _decode(tracking_after["tree"]) != winning_tree
        ):
            raise ValueError("fetch did not hand off the accepted snapshot")
        integration = before["integration"]
        merged = dict(winning_tree)
        merged[node_path] = losing_tree[node_path]
        merged[f"operations/{pending_op}.json"] = losing_tree[f"operations/{pending_op}.json"]
        if (
            not isinstance(integration, dict)
            or set(integration) != tracking_keys
            or integration["commit"] != candidate
            or integration["parents"] != [winner, loser]
            or _decode(integration["tree"]) != merged
        ):
            raise ValueError("hypothetical integration candidate differs")
        if hashlib.sha256(winning_tree[node_path]).hexdigest() == revision:
            raise ValueError("affected node did not advance")
        actions = artifacts.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("missing model action trace")
        fetched = checked_stale = status_seen = found = shown = False
        for action in actions:
            if (
                not isinstance(action, dict)
                or not isinstance(action.get("args"), list)
                or not action["args"]
                or not all(isinstance(item, str) for item in action["args"])
                or action.get("status") not in {"success", "error"}
                or not isinstance(action.get("stdout"), str)
                or not isinstance(action.get("stderr"), str)
            ):
                raise ValueError("incomplete action evidence")
            if verified_help(action):
                continue
            args = command_arguments(action["args"])
            if not args:
                raise ValueError("missing action arguments")
            if any(
                value in {"--force", "-f"} or value.startswith("--force-with-lease")
                for value in args
            ):
                raise ValueError("force push attempted")
            if args[0] in MUTATIONS or args[0] == "publish":
                raise ValueError("unapproved mutation or publication attempted")
            if args[0] == "fetch":
                if fetched or args != ["fetch"] or action["status"] != "success":
                    raise ValueError("expected exactly one successful fetch")
                output = _json(action["stdout"])
                if (
                    not isinstance(output, dict)
                    or type(output.get("output_schema")) is not int
                    or output["output_schema"] != 1
                    or {k: output.get(k) for k in final} != final
                ):
                    raise ValueError("fetch output differs from attachment")
                fetched = True
            elif args[0] == "check-merge":
                if (
                    not fetched
                    or checked_stale
                    or args != ["check-merge", "--base", winner, "--candidate", candidate]
                    or action["status"] != "error"
                ):
                    raise ValueError("missing current-base integration check")
                output = _json(action["stderr"])
                message = (
                    output.get("error", {}).get("message", "") if isinstance(output, dict) else ""
                )
                if (
                    not isinstance(output, dict)
                    or type(output.get("output_schema")) is not int
                    or output["output_schema"] != 1
                    or output.get("error", {}).get("type") != "GraphError"
                    or not isinstance(message, str)
                    or "stale affected-node precondition" not in message
                    or node_id not in message
                ):
                    raise ValueError("stale affected-node rejection not observed")
                checked_stale = True
            elif args[0] == "operation":
                if (
                    not fetched
                    or args != ["operation", "status", pending_op]
                    or action["status"] != "success"
                ):
                    raise ValueError("invalid retained-proposal status request")
                output = _json(action["stdout"])
                if (
                    not isinstance(output, dict)
                    or type(output.get("output_schema")) is not int
                    or output["output_schema"] != 1
                    or output.get("state") != "prepared"
                    or output.get("operation_id") != pending_op
                    or output.get("commit") != loser
                    or output.get("snapshot_valid") is not True
                    or type(output.get("diagnostic_error_count")) is not int
                    or output["diagnostic_error_count"] != 0
                    or output.get("snapshot", {}).get("commit") != winner
                    or output.get("snapshot", {}).get("store_id") != STORE_ID
                    or output.get("snapshot", {}).get("remote") != "origin"
                    or output.get("receipt") != _json(losing_tree[f"operations/{pending_op}.json"])
                ):
                    raise ValueError("retained proposal status is not proven")
                status_seen = True
            elif args[0] in READS:
                if args[0] in {"find", "show", "neighbors", "review"} and (
                    _option(args, "--scope") != "orchard" or has_flag(args, "--include-unscoped")
                ):
                    raise ValueError("read command omitted or widened scope")
                if (
                    args[0] in {"find", "show", "neighbors", "review"}
                    and action["status"] == "success"
                    and (
                        "cart is blue" in action["stdout"]
                        or (not fetched and "cart is green" in action["stdout"])
                    )
                ):
                    raise ValueError("ordinary recall exposed an unaccepted report")
                if args[0] in {"find", "show"} and action["status"] == "success":
                    output = _json(action["stdout"])
                    if (
                        not isinstance(output, dict)
                        or type(output.get("output_schema")) is not int
                        or output["output_schema"] != 1
                    ):
                        raise ValueError("invalid read output")
                    provenance = output.get("snapshot")
                    if (
                        not isinstance(provenance, dict)
                        or provenance.get("commit") != (winner if fetched else base)
                        or provenance.get("store_id") != STORE_ID
                        or provenance.get("remote") != "origin"
                        or provenance.get("source") != "main"
                    ):
                        raise ValueError("read did not use the selected committed main")
                    if not fetched:
                        if args[0] == "find":
                            rows = output.get("results")
                            if (
                                not isinstance(rows, list)
                                or output.get("truncated") is not False
                                or any(
                                    not isinstance(row, dict)
                                    or row.get("id") != node_id
                                    or row.get("label") != "cart is red"
                                    for row in rows
                                )
                            ):
                                raise ValueError("prefetch recall exposed non-base knowledge")
                        else:
                            node = output.get("node")
                            if (
                                not isinstance(node, dict)
                                or node.get("id") != node_id
                                or node.get("label") != "cart is red"
                            ):
                                raise ValueError("prefetch show exposed non-base knowledge")
                        continue
                    if args[0] == "find":
                        rows = output.get("results")
                        positional = positional_argument(args)
                        found |= (
                            isinstance(positional, str)
                            and positional.casefold() == "cart"
                            and output.get("query", "").casefold() == "cart"
                            and output.get("truncated") is False
                            and isinstance(rows, list)
                            and len(rows) == 1
                            and isinstance(rows[0], dict)
                            and rows[0].get("id") == node_id
                            and rows[0].get("label") == "cart is green"
                            and rows[0].get("status") == "disputed"
                        )
                    else:
                        node = output.get("node")
                        shown |= (
                            positional_argument(args) == node_id
                            and isinstance(node, dict)
                            and node.get("id") == node_id
                            and node.get("label") == "cart is green"
                            and node.get("scope") == "orchard"
                            and node.get("sources") == winning_node["sources"]
                            and node.get("status") == "disputed"
                            and output.get("body") == case["fixture"]["competing_update"]["body"]
                        )
            elif args != ["status"]:
                raise ValueError("unknown or disallowed action")
        if not all((fetched, checked_stale, status_seen, found, shown)):
            raise ValueError("missing fetched recall, retained status, or stale-check evidence")
        if not _answer(
            answer,
            [node_id],
            ["cart is green", "cart is blue"],
            ["witness:a", "witness:b", "witness:c"],
            ["conflict", "stale proposal"],
            winner,
        ):
            raise ValueError("answer resolved or omitted conflicting reports")
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError, ImportError) as exc:
        failures.append(str(exc) or type(exc).__name__)
    return {"success": not failures, "failures": failures, "observations": []}
