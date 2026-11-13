"""Verify an independently accepted local remote commit and subsequent fetch."""

from __future__ import annotations

import json
import re
from datetime import datetime

from behavior_cases import acceptance_request, compile_fixture, semantic_id
from behavior_grader import _json, _option
from behavior_shared_grader import REMOTE, _decode
from behavior_write_grader import _check_action, _node, _valid_receipt
from controlled_core import MUTATIONS, READS, STORE_ID


def _local(value, refs):
    keys = {"main", "refs", "files", "tree_files", "tree", "working", "parents", "message"}
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["refs"] != refs
        or value["main"] != refs[0].split(" ", 1)[1]
    ):
        raise ValueError("invalid local Git snapshot")
    if value["parents"] != [] or not isinstance(value["message"], str):
        raise ValueError("invalid local base commit")
    return _decode(value["tree"], value["tree_files"]), _decode(value["working"], value["files"])


def _config(raw, expected_commit, *, fetched):
    config = _json(raw)
    if not isinstance(config, dict) or set(config) != {
        "remote",
        "mode",
        "store_id",
        "fetched_at",
        "fetched_commit",
        "last_fetch",
    }:
        raise ValueError("invalid attachment config")
    if (
        config["remote"] != "origin"
        or config["mode"] != "read"
        or config["store_id"] != STORE_ID
        or config["fetched_commit"] != expected_commit
        or config["last_fetch"] != config["fetched_at"]
    ):
        raise ValueError("incorrect attachment or fetched ref")
    if fetched:
        if not isinstance(config["fetched_at"], str):
            raise ValueError("missing fetch time")
        if datetime.fromisoformat(config["fetched_at"]).tzinfo is None:
            raise ValueError("fetch time must have a timezone")
    elif config["fetched_at"] is not None:
        raise ValueError("unexpected initial fetch time")
    return config


def grade_fetch(case_id, answer, artifacts):
    failures = []
    try:
        fixture = compile_fixture(case_id)
        case = fixture["case"]
        if case_id != "accepted-after-fetch" or case["fixture"]["main"] != {
            "facts": [],
            "edges": [],
        }:
            raise ValueError("invalid acceptance fixture")
        if not isinstance(artifacts, dict):
            raise TypeError("missing fetch artifacts")
        before, after = artifacts.get("before"), artifacts.get("after")
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or set(before) != {"local", "remote", "tracking"}
            or set(after) != set(before)
        ):
            raise ValueError("missing local, remote, or tracking artifacts")
        base = before["local"]["main"]
        remote = before["remote"]
        if (
            not isinstance(base, str)
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", base)
            or not isinstance(remote, dict)
            or set(remote) != {"url", "main", "refs", "tree"}
            or remote["url"] != REMOTE
            or not isinstance(remote["main"], str)
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", remote["main"])
            or remote["main"] == base
            or remote["refs"] != [f"refs/heads/main {remote['main']}"]
            or after["remote"] != remote
        ):
            raise ValueError("remote acceptance is missing or changed")
        accepted = remote["main"]
        before_tree, before_work = _local(
            before["local"], [f"refs/heads/main {base}", f"refs/remotes/origin/main {base}"]
        )
        after_tree, after_work = _local(
            after["local"], [f"refs/heads/main {base}", f"refs/remotes/origin/main {accepted}"]
        )
        expected = {name: content.encode() for name, content in fixture["files"].items()}
        if expected != {
            ".gitignore": b".memory-graph/\n",
            "store.json": (
                json.dumps({"schema": 1, "id": STORE_ID}, sort_keys=True) + "\n"
            ).encode(),
        }:
            raise ValueError("base fixture does not match the empty corpus store")
        if before_tree != expected or after_tree != expected:
            raise ValueError("local main was modified")
        work_paths = set(expected) | {".memory-graph/config.json", ".memory-graph/write.lock"}
        if set(before_work) != work_paths or set(after_work) != work_paths:
            raise ValueError("unexpected local working files")
        if any(
            before_work[path] != expected[path] or after_work[path] != expected[path]
            for path in expected
        ):
            raise ValueError("local checkout changed")
        if (
            before_work[".memory-graph/write.lock"] != b""
            or after_work[".memory-graph/write.lock"] != b""
        ):
            raise ValueError("unexpected lock contents")
        initial_config = _config(before_work[".memory-graph/config.json"], base, fetched=False)
        final_config = _config(after_work[".memory-graph/config.json"], accepted, fetched=True)
        setup = artifacts.get("setup")
        if (
            not isinstance(setup, dict)
            or setup.get("args") != ["attach", "--remote", "origin", "--mode", "read"]
            or not isinstance(setup.get("stdout"), str)
            or not isinstance(setup.get("stderr"), str)
        ):
            raise ValueError("missing read-only attachment evidence")
        setup_result = _json(setup["stdout"])
        if (
            not isinstance(setup_result, dict)
            or type(setup_result.get("output_schema")) is not int
            or setup_result["output_schema"] != 1
            or {k: setup_result.get(k) for k in initial_config} != initial_config
        ):
            raise ValueError("attachment result differs from initial configuration")

        actor = artifacts.get("acceptance")
        if not isinstance(actor, dict):
            raise TypeError("missing separate acceptance actor")
        operation_id = semantic_id(case_id, "$acceptance")
        request, response = _check_action(actor, operation_id, "create")
        if (
            request != acceptance_request(case_id)
            or response.get("replayed") is not False
            or response.get("commit") != accepted
        ):
            raise ValueError("acceptance actor request or commit differs")
        remote_tree = _decode(remote["tree"])
        node_id = request["id"]
        node_path = f"{node_id}.md"
        receipt_path = f"operations/{operation_id}.json"
        if (
            set(remote_tree) != set(expected) | {node_path, receipt_path}
            or any(remote_tree[path] != data for path, data in expected.items())
            or not _valid_receipt(
                remote_tree, operation_id, "create", request, base, {node_path}, {}
            )
            or response.get("receipt") != _json(remote_tree[receipt_path])
        ):
            raise ValueError("remote accepted tree or receipt differs")
        node, body = _node(remote_tree[node_path], node_path)
        if (
            node.get("label") != "wheel is blue"
            or node.get("scope") != "workshop"
            or node.get("status") != "active"
            or node.get("edges") != []
            or node.get("sources", []) != []
            or body
        ):
            raise ValueError("accepted fact differs")
        tracking_before, tracking_after = before["tracking"], after["tracking"]
        required = {"commit", "tree", "parents", "message"}
        if (
            not isinstance(tracking_before, dict)
            or not isinstance(tracking_after, dict)
            or set(tracking_before) != required
            or set(tracking_after) != required
            or tracking_before["commit"] != base
            or tracking_before["parents"] != []
            or _decode(tracking_before["tree"]) != expected
            or not isinstance(tracking_before["message"], str)
            or not tracking_before["message"].startswith("Synthetic shared fixture")
            or tracking_after["commit"] != accepted
            or tracking_after["parents"] != [base]
            or _decode(tracking_after["tree"]) != remote_tree
            or not isinstance(tracking_after["message"], str)
        ):
            raise ValueError("tracking ref did not advance to the complete accepted commit")
        for trailer in (
            "Memory-Commit-Schema: 1",
            "Memory-Operation: create",
            f"Memory-Operation-Id: {operation_id}",
            f"Memory-Store: {STORE_ID}",
        ):
            if trailer not in tracking_after["message"].splitlines():
                raise ValueError("missing acceptance provenance")

        actions = artifacts.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("missing action trace")
        fetch_index = None
        found = False
        for index, action in enumerate(actions):
            if (
                not isinstance(action, dict)
                or not isinstance(action.get("args"), list)
                or not action["args"]
                or not all(isinstance(item, str) for item in action["args"])
                or action.get("status") not in {"success", "error"}
                or not isinstance(action.get("stdout"), str)
                or not isinstance(action.get("stderr"), str)
            ):
                raise ValueError("incomplete action trace")
            args = action["args"]
            if args[0] in MUTATIONS or args[0] not in READS | {"fetch", "status"}:
                raise ValueError("unauthorized mutation or publication")
            if args[0] == "fetch":
                if args != ["fetch"] or action["status"] != "success" or fetch_index is not None:
                    raise ValueError("fetch was not one successful explicit action")
                output = _json(action["stdout"])
                if (
                    not isinstance(output, dict)
                    or type(output.get("output_schema")) is not int
                    or output["output_schema"] != 1
                    or {key: output.get(key) for key in final_config} != final_config
                ):
                    raise ValueError("fetch response does not name accepted commit")
                fetch_index = index
            elif args[0] in {"find", "show", "neighbors", "review"}:
                if _option(args, "--scope") != "workshop" or "--include-unscoped" in args:
                    raise ValueError("read command omitted or widened scope")
                if args[0] == "show" and action["status"] == "success" and fetch_index is None:
                    raise ValueError("unfetched memory was visible before fetch")
                if args[0] == "find" and action["status"] == "success":
                    output = _json(action["stdout"])
                    rows = output.get("results", []) if isinstance(output, dict) else []
                    provenance = output.get("snapshot", {}) if isinstance(output, dict) else {}
                    if fetch_index is None:
                        if (
                            not isinstance(output, dict)
                            or rows != []
                            or not isinstance(provenance, dict)
                            or provenance.get("commit") != base
                        ):
                            raise ValueError("unfetched memory was visible before fetch")
                        continue
                    found |= (
                        len(args) > 1
                        and args[1].casefold() == "wheel"
                        and type(output.get("output_schema")) is int
                        and output["output_schema"] == 1
                        and output.get("truncated") is False
                        and provenance.get("store_id") == STORE_ID
                        and provenance.get("remote") == "origin"
                        and provenance.get("commit") == accepted
                        and provenance.get("source") == "main"
                        and [row.get("id") for row in rows] == [node_id]
                        and rows[0].get("label") == "wheel is blue"
                        and rows[0].get("scope") == "workshop"
                    )
        if fetch_index is None or not found:
            raise ValueError("missing fetch followed by accepted-snapshot recall")
        if _json(answer) != {
            "node_ids": [node_id],
            "facts": ["wheel is blue"],
            "sources": [],
            "notices": [],
            "store_id": STORE_ID,
            "commit": accepted,
        }:
            raise ValueError("answer lacks accepted-snapshot provenance")
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError, ImportError) as exc:
        failures.append(str(exc) or type(exc).__name__)
    return {"success": not failures, "failures": failures, "observations": []}
