"""Grade a proposal branch staying outside ordinary main recall."""

from __future__ import annotations

import json
import re

from behavior_cases import compile_fixture
from controlled_core import MUTATIONS, READS, STORE_ID, command_arguments

from graders.arguments import has_flag, positional_argument, verified_help
from graders.behavior_grader import _json, _option
from graders.behavior_shared_grader import REMOTE, _decode
from graders.behavior_write_grader import (
    _answer,
    _local_effects,
    _node,
    _request_hash,
    _valid_receipt,
)

POLICY_SHA256 = "e0c20d8b588b6663b29c4e49f82f290ace08abf5effb9ef2bfded9be2cab4ea9"


def _local(value, refs, expected):
    keys = {"main", "refs", "files", "tree_files", "tree", "working", "parents", "message"}
    if not isinstance(value, dict) or set(value) != keys or value["refs"] != refs:
        raise ValueError("invalid local snapshot")
    if value["main"] != refs[0].split(" ", 1)[1] or value["parents"] != []:
        raise ValueError("local main moved or has an invalid base")
    if _decode(value["tree"], value["tree_files"]) != expected:
        raise ValueError("local main tree changed")
    return _decode(value["working"], value["files"])


def grade_proposal(case_id, answer, artifacts):
    failures = []
    try:
        fixture = compile_fixture(case_id)
        if case_id != "proposal-not-main" or fixture["case"]["fixture"]["main"] != {
            "facts": [],
            "edges": [],
        }:
            raise ValueError("incorrect proposal fixture")
        if not isinstance(artifacts, dict):
            raise TypeError("missing proposal artifacts")
        before, after = artifacts.get("before"), artifacts.get("after")
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or set(before) != {"local", "remote", "proposal", "policy"}
            or set(after) != set(before)
        ):
            raise ValueError("missing local, remote, or proposal artifacts")
        base = before["local"]["main"]
        if not isinstance(base, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", base):
            raise ValueError("invalid main commit")
        if (
            before["policy"] != {"sha256": POLICY_SHA256, "executable": True}
            or before["policy"]["executable"] is not True
            or after["policy"] != before["policy"]
        ):
            raise ValueError("synthetic remote main-protection policy is missing or changed")
        probe = artifacts.get("policy_test")
        if (
            not isinstance(probe, dict)
            or type(probe.get("returncode")) is not int
            or probe["returncode"] == 0
            or not isinstance(probe.get("stdout"), str)
            or not isinstance(probe.get("stderr"), str)
            or "pre-receive hook declined" not in (probe["stdout"] + probe["stderr"])
        ):
            raise ValueError("direct unauthorized main push was not demonstrably rejected")
        ancestry = probe.get("candidate_ancestry")
        if (
            not isinstance(ancestry, list)
            or len(ancestry) != 2
            or ancestry[1] != base
            or not isinstance(ancestry[0], str)
            or ancestry[0] == base
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", ancestry[0])
        ):
            raise ValueError("unauthorized test commit is not a distinct child of main")
        operation_id = fixture["operation_id"]
        ref = f"refs/heads/memory-graph/{operation_id}"
        initial_refs = [f"refs/heads/main {base}", f"refs/remotes/origin/main {base}"]
        expected = {name: content.encode() for name, content in fixture["files"].items()}
        if expected != {
            ".gitignore": b".memory-graph/\n",
            "store.json": (
                json.dumps({"schema": 1, "id": STORE_ID}, sort_keys=True) + "\n"
            ).encode(),
        }:
            raise ValueError("proposal base fixture differs")
        before_work = _local(before["local"], initial_refs, expected)
        if (
            set(before_work)
            != set(expected) | {".memory-graph/config.json", ".memory-graph/write.lock"}
            or before_work[".memory-graph/write.lock"] != b""
        ):
            raise ValueError("unexpected initial working files")
        if before["proposal"] != {"local": None, "remote": None}:
            raise ValueError("proposal already existed")
        remote_before, remote_after = before["remote"], after["remote"]
        if (
            not isinstance(remote_before, dict)
            or not isinstance(remote_after, dict)
            or set(remote_before) != {"url", "main", "refs", "tree"}
            or set(remote_after) != set(remote_before)
            or remote_before["url"] != REMOTE
            or remote_after["url"] != REMOTE
            or remote_before["main"] != base
            or remote_after["main"] != base
            or remote_before["refs"] != [f"refs/heads/main {base}"]
            or _decode(remote_before["tree"]) != expected
            or _decode(remote_after["tree"]) != expected
        ):
            raise ValueError("remote main differs or changed")
        proposal = after["proposal"]
        if (
            not isinstance(proposal, dict)
            or set(proposal) != {"local", "remote"}
            or not isinstance(proposal["local"], dict)
            or proposal["local"] != proposal["remote"]
        ):
            raise ValueError("local and published proposal branches differ")
        candidate = proposal["local"]
        if set(candidate) != {"commit", "tree", "parents", "message"} or candidate["parents"] != [
            base
        ]:
            raise ValueError("proposal is not one commit on the fetched base")
        commit = candidate["commit"]
        if (
            not isinstance(commit, str)
            or len(commit) != len(base)
            or commit == base
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)
        ):
            raise ValueError("invalid proposal commit")
        if remote_after["refs"] != [f"refs/heads/main {base}", f"{ref} {commit}"]:
            raise ValueError("unexpected published branches")
        required_refs = [
            f"refs/heads/main {base}",
            f"{ref} {commit}",
            f"refs/remotes/origin/main {base}",
        ]
        optional_tracking = f"refs/remotes/origin/memory-graph/{operation_id} {commit}"
        actual_refs = after["local"].get("refs")
        if (
            not isinstance(actual_refs, list)
            or actual_refs != sorted(actual_refs)
            or len(actual_refs) != len(set(actual_refs))
            or set(actual_refs)
            not in (set(required_refs), set(required_refs) | {optional_tracking})
        ):
            raise ValueError("unexpected local proposal refs")
        after_refs = actual_refs
        after_work = _local(after["local"], after_refs, expected)
        config = _json(before_work[".memory-graph/config.json"])
        if config != {
            "remote": "origin",
            "mode": "propose",
            "store_id": STORE_ID,
            "fetched_at": None,
            "fetched_commit": base,
            "last_fetch": None,
        }:
            raise ValueError("incorrect proposal attachment")
        if after_work.get(".memory-graph/config.json") != before_work[".memory-graph/config.json"]:
            raise ValueError("attachment configuration changed")
        setup = artifacts.get("setup")
        if not isinstance(setup, dict) or setup.get("args") != [
            "attach",
            "--remote",
            "origin",
            "--mode",
            "propose",
        ]:
            raise ValueError("missing proposal attachment")
        setup_result = _json(setup["stdout"])
        if (
            not isinstance(setup_result, dict)
            or type(setup_result.get("output_schema")) is not int
            or setup_result["output_schema"] != 1
            or {key: setup_result.get(key) for key in config} != config
        ):
            raise ValueError("proposal attachment output differs")
        actions = artifacts.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("missing action trace")
        prepared = published = recalled = False
        request = response = None
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
            if args[0] == "create":
                if (
                    prepared
                    or published
                    or action["status"] != "success"
                    or _option(args, "--operation") != operation_id
                    or _option(args, "--input") != "-"
                ):
                    raise ValueError("invalid proposal creation action")
                request = _json(action.get("input_text", ""))
                response = _json(action["stdout"])
                if (
                    not isinstance(request, dict)
                    or not isinstance(response, dict)
                    or type(response.get("output_schema")) is not int
                    or response["output_schema"] != 1
                    or response.get("state") != "prepared"
                    or response.get("operation_id") != operation_id
                    or response.get("replayed") is not False
                    or response.get("commit") != commit
                ):
                    raise ValueError("proposal was not prepared")
                prepared = True
            elif args[0] == "publish":
                if (
                    not prepared
                    or published
                    or action["status"] != "success"
                    or _option(args, "--operation") != operation_id
                ):
                    raise ValueError("invalid proposal publication action")
                output = _json(action["stdout"])
                if (
                    not isinstance(output, dict)
                    or type(output.get("output_schema")) is not int
                    or output["output_schema"] != 1
                    or output.get("status") != "proposed"
                    or output.get("remote") != "origin"
                    or output.get("ref") != ref
                    or output.get("commit") != commit
                    or output.get("accepted") is not False
                    or output.get("review_created") is not False
                ):
                    raise ValueError("publication did not report an unaccepted proposal")
                published = True
            elif args[0] in MUTATIONS or args[0] not in READS | {"status"}:
                raise ValueError("unexpected mutation or remote action")
            elif args[0] in {"find", "show", "neighbors", "review"}:
                if _option(args, "--scope") != "workshop" or has_flag(args, "--include-unscoped"):
                    raise ValueError("read command omitted or widened scope")
                if args[0] == "find" and action["status"] == "success" and published:
                    output = _json(action["stdout"])
                    provenance = output.get("snapshot", {}) if isinstance(output, dict) else {}
                    positional = positional_argument(args)
                    recalled |= (
                        isinstance(positional, str)
                        and positional.casefold() == "wheel"
                        and type(output.get("output_schema")) is int
                        and output["output_schema"] == 1
                        and output.get("truncated") is False
                        and output.get("results") == []
                        and provenance.get("commit") == base
                        and provenance.get("store_id") == STORE_ID
                        and provenance.get("remote") == "origin"
                        and provenance.get("source") == "main"
                    )
        if not prepared or not published or not recalled:
            raise ValueError("missing prepare, publish, or proposal-exclusion evidence")
        tree = _decode(candidate["tree"])
        node_paths = [path for path in tree if path.endswith(".md")]
        if len(node_paths) != 1:
            raise ValueError("proposal must contain exactly one fact")
        node_path = node_paths[0]
        node, body = _node(tree[node_path], node_path)
        if (
            node.get("label") != "wheel is blue"
            or node.get("scope") != "workshop"
            or node.get("status") != "active"
            or node.get("edges") != []
            or node.get("sources", []) != []
            or body.strip() not in {"", "wheel is blue"}
        ):
            raise ValueError("proposal fact or evidence differs")
        receipt_path = f"operations/{operation_id}.json"
        if (
            set(tree) != set(expected) | {node_path, receipt_path}
            or any(tree[path] != value for path, value in expected.items())
            or not _valid_receipt(tree, operation_id, "create", request, base, {node_path}, {})
            or response.get("receipt") != _json(tree[receipt_path])
        ):
            raise ValueError("proposal receipt or committed changes differ")
        journal = _local_effects(before_work, after_work, operation_id)
        if (
            journal.get("base") != base
            or journal.get("store_id") != STORE_ID
            or journal.get("request_hash") != _request_hash("create", request)
        ):
            raise ValueError("proposal journal differs")
        for trailer in (
            "Memory-Commit-Schema: 1",
            "Memory-Operation: create",
            f"Memory-Operation-Id: {operation_id}",
            f"Memory-Store: {STORE_ID}",
        ):
            if trailer not in candidate["message"].splitlines():
                raise ValueError("proposal provenance missing")
        if not _answer(answer, [node["id"]], ["wheel is blue"], [], ["proposed"], commit):
            raise ValueError("proposal answer misstates publication")
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError, ImportError) as exc:
        failures.append(str(exc) or type(exc).__name__)
    return {"success": not failures, "failures": failures, "observations": []}
