"""Independently grade read-only attachments to a local bare remote."""

from __future__ import annotations

import base64
import hashlib
import re

from behavior_cases import compile_fixture
from controlled_core import MUTATIONS, READS, STORE_ID, command_arguments

from graders.arguments import has_flag, positional_argument, verified_help
from graders.behavior_grader import (
    _fixture_matches_corpus,
    _json,
    _option,
    query_names_targets,
)

REMOTE = "/workspace/remote.git"


def _decode(values, hashes=None):
    if not isinstance(values, dict):
        raise TypeError("missing raw files")
    result = {key: base64.b64decode(value, validate=True) for key, value in values.items()}
    if (
        hashes is not None
        and {key: hashlib.sha256(value).hexdigest() for key, value in result.items()} != hashes
    ):
        raise ValueError("raw file hashes differ")
    return result


def _snapshot(value, fixture, *, offline):
    keys = {"local", "remote", "remote_available"} if offline else {"local", "remote"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("missing local and bare-remote artifacts")
    if offline and value["remote_available"] is not False:
        raise ValueError("configured remote is not proven unavailable")
    local, remote = value["local"], value["remote"]
    keys = {"main", "refs", "files", "tree_files", "tree", "working", "parents", "message"}
    if not isinstance(local, dict) or set(local) != keys:
        raise ValueError("missing local snapshot")
    if not isinstance(remote, dict) or set(remote) != {"url", "main", "refs", "tree"}:
        raise ValueError("missing remote snapshot")
    main = local["main"]
    if (
        not isinstance(main, str)
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", main)
        or local["refs"] != [f"refs/heads/main {main}", f"refs/remotes/origin/main {main}"]
        or remote["url"] != REMOTE
        or remote["main"] != main
        or remote["refs"] != [f"refs/heads/main {main}"]
        or local["parents"] != []
        or not isinstance(local["message"], str)
        or not local["message"].startswith("Synthetic shared fixture")
    ):
        raise ValueError("local and remote main/tracking refs differ")
    expected = {name: content.encode() for name, content in fixture["files"].items()}
    tree = _decode(local["tree"], local["tree_files"])
    working = _decode(local["working"], local["files"])
    if tree != expected or _decode(remote["tree"]) != expected:
        raise ValueError("local or remote committed tree differs from fixture")
    if set(working) != set(expected) | {".memory-graph/config.json", ".memory-graph/write.lock"}:
        raise ValueError("unexpected local attachment files")
    if (
        any(working[path] != data for path, data in expected.items())
        or working[".memory-graph/write.lock"] != b""
    ):
        raise ValueError("working tree differs from fixture")
    config = _json(working[".memory-graph/config.json"])
    if config != {
        "remote": "origin",
        "mode": "read",
        "store_id": STORE_ID,
        "endpoint_id": hashlib.sha256(REMOTE.encode("utf-8")).hexdigest(),
        "fetched_at": None,
        "fetched_commit": main,
        "last_fetch": None,
    }:
        raise ValueError("local attachment is not the expected read-only store")
    return main, config


def grade_shared(case_id, answer, artifacts):
    failures = []
    try:
        fixture = compile_fixture(case_id)
        if not _fixture_matches_corpus(case_id, fixture):
            raise ValueError("fixture bytes do not match corpus semantics")
        if not isinstance(artifacts, dict):
            raise TypeError("missing shared-store artifacts")
        before, after = artifacts.get("before"), artifacts.get("after")
        offline = fixture["spec"]["kind"] == "offline-stale"
        commit, config = _snapshot(before, fixture, offline=offline)
        _snapshot(after, fixture, offline=offline)
        if before != after:
            raise ValueError("local or remote store changed")
        setup = artifacts.get("setup")
        if (
            not isinstance(setup, dict)
            or setup.get("args") != ["attach", "--remote", "origin", "--mode", "read"]
            or not isinstance(setup.get("stdout"), str)
            or not isinstance(setup.get("stderr"), str)
        ):
            raise ValueError("missing real read-only attachment result")
        setup_result = _json(setup["stdout"])
        if (
            not isinstance(setup_result, dict)
            or type(setup_result.get("output_schema")) is not int
            or setup_result["output_schema"] != 1
        ):
            raise ValueError("invalid attachment result")
        if {key: setup_result.get(key) for key in config} != config:
            raise ValueError("attachment result differs from local configuration")
        actions = artifacts.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("missing action trace")
        node_id = fixture["ids"]["cart"]
        found = False
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
            if args[0] == "fetch":
                if not offline or action["status"] != "error" or args != ["fetch"]:
                    raise ValueError("unexpected successful or unauthorized fetch")
                continue
            if args[0] in MUTATIONS or args[0] not in READS | {"status"}:
                raise ValueError("mutation or publication attempted in read-only attachment")
            if args[0] in {"find", "show", "neighbors", "review"} and (
                _option(args, "--scope") != fixture["case"]["scope"]
                or has_flag(args, "--include-unscoped")
            ):
                raise ValueError("read command omitted or widened scope")
            if args[0] != "find" or action["status"] != "success":
                continue
            output = _json(action["stdout"])
            if not isinstance(output, dict):
                raise TypeError("invalid retrieval output")
            provenance = output.get("snapshot")
            rows = output.get("results")
            positional = positional_argument(args)
            found |= (
                query_names_targets(positional, fixture)
                and type(output.get("output_schema")) is int
                and output["output_schema"] == 1
                and output.get("truncated") is False
                and isinstance(provenance, dict)
                and provenance.get("store_id") == STORE_ID
                and provenance.get("commit") == commit
                and provenance.get("source") == "main"
                and provenance.get("remote") == "origin"
                and (not offline or provenance.get("freshness") == "unknown")
                and isinstance(rows, list)
                and len(rows) == 1
                and isinstance(rows[0], dict)
                and rows[0].get("id") == node_id
                and rows[0].get("label") == "cart is red"
                and rows[0].get("scope") == fixture["case"]["scope"]
            )
        if not found:
            raise ValueError("missing scoped retrieval with remote/commit provenance")
        response = _json(answer)
        notices = ["stale or unknown freshness"] if offline else []
        if response != {
            "node_ids": [node_id],
            "facts": ["cart is red"],
            "sources": [],
            "notices": notices,
            "store_id": STORE_ID,
            "commit": commit,
        }:
            raise ValueError("answer lacks correct shared-store provenance")
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError) as exc:
        failures.append(str(exc) or type(exc).__name__)
    return {"success": not failures, "failures": failures, "observations": []}
