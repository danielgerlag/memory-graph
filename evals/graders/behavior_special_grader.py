"""Grade pinned-clock reads and diagnosis using captured raw artifacts."""

from __future__ import annotations

import hashlib

from behavior_cases import compile_fixture, semantic_id
from controlled_core import MUTATIONS, READS, STORE_ID, command_arguments

from graders.arguments import has_flag, positional_argument
from graders.behavior_grader import _fixture_matches_corpus, _json, _option
from graders.behavior_write_grader import _answer, _unpack


def grade_special(case_id, answer, artifacts):
    failures = []
    try:
        fixture = compile_fixture(case_id)
        case, spec, ids = fixture["case"], fixture["spec"], fixture["ids"]
        if spec["kind"] != "malformed" and not _fixture_matches_corpus(case_id, fixture):
            raise ValueError("fixture differs from corpus semantics")
        if spec["kind"] == "malformed":
            broken = semantic_id(case_id, "broken")
            expected_bad = f'---\nschema: 1\nschema: 1\nid: "{broken}"\n---\n'
            if fixture["files"].get(f"{broken}.md") != expected_bad:
                raise ValueError("malformed fixture differs from corpus")
        if not isinstance(artifacts, dict):
            raise TypeError("missing artifacts")
        before, after = artifacts.get("before"), artifacts.get("after")
        raw_before, raw_after = _unpack(before), _unpack(after)
        expected = {path: value.encode() for path, value in fixture["files"].items()}
        if (
            raw_before["tree"] != expected
            or raw_before["working"] != expected
            or raw_after != raw_before
            or before != after
        ):
            raise ValueError("initial fixture or read-only snapshot differs")
        actions = artifacts.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("missing action trace")
        found = diagnosed = reviewed = False
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
            args = command_arguments(action["args"])
            if not args:
                raise ValueError("missing action arguments")
            positional = positional_argument(args)
            if args[0] in MUTATIONS or args[0] not in READS:
                raise ValueError("mutation or forbidden action in read-only case")
            if args[0] in {"find", "show", "neighbors", "review"} and (
                _option(args, "--scope") != case["scope"] or has_flag(args, "--include-unscoped")
            ):
                raise ValueError("read command omitted or widened scope")
            if spec["kind"] == "malformed":
                if args[0] == "find":
                    if action["status"] != "error":
                        raise ValueError("invalid graph returned a successful recall")
                    error = _json(action["stderr"])
                    message = (
                        error.get("error", {}).get("message", "") if isinstance(error, dict) else ""
                    )
                    found |= (
                        isinstance(positional, str)
                        and positional.casefold() == spec["query"].casefold()
                        and "invalid main snapshot" in message
                    )
                elif args[0] == "diagnose":
                    output = _json(action["stdout"])
                    details = output.get("errors", []) if isinstance(output, dict) else []
                    snapshot = output.get("snapshot", {}) if isinstance(output, dict) else {}
                    path = f"{ids['broken']}.md"
                    diagnosed |= (
                        action["status"] == "error"
                        and type(output.get("output_schema")) is int
                        and output["output_schema"] == 1
                        and output.get("valid") is False
                        and snapshot.get("commit") == before["main"]
                        and snapshot.get("store_id") == STORE_ID
                        and isinstance(details, list)
                        and len(details) == 1
                        and isinstance(details[0], dict)
                        and details[0].get("path") == path
                        and details[0].get("sha256") == hashlib.sha256(expected[path]).hexdigest()
                        and isinstance(details[0].get("error"), str)
                        and bool(details[0]["error"])
                    )
                continue
            if action["status"] != "success":
                continue
            output = _json(action["stdout"])
            if (
                not isinstance(output, dict)
                or type(output.get("output_schema")) is not int
                or output["output_schema"] != 1
            ):
                raise ValueError("invalid command output")
            snapshot = output.get("snapshot", {})
            if (
                not isinstance(snapshot, dict)
                or snapshot.get("commit") != before["main"]
                or snapshot.get("store_id") != STORE_ID
                or snapshot.get("source") != "main"
            ):
                raise ValueError("invalid command provenance")
            if output.get("truncated") is True:
                raise ValueError("truncated command output")
            if args[0] == "find":
                rows = output.get("results")
                target = ids[spec["targets"][0]]
                found |= (
                    isinstance(positional, str)
                    and positional.casefold() == spec["query"].casefold()
                    and output.get("truncated") is False
                    and isinstance(rows, list)
                    and len(rows) == 1
                    and isinstance(rows[0], dict)
                    and rows[0].get("id") == target
                    and rows[0].get("label") == spec["facts"][0]
                    and rows[0].get("scope") == case["scope"]
                    and rows[0].get("expired") is False
                )
            elif args[0] == "review" and spec["kind"] == "timed-review":
                rows = output.get("results")
                if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                    by_id = {row.get("id"): row for row in rows}
                    reviewed |= (
                        output.get("truncated") is False
                        and set(by_id) == {ids["expired"], ids["review"]}
                        and by_id[ids["expired"]].get("expired") is True
                        and by_id[ids["review"]].get("review_due") is True
                        and by_id[ids["review"]].get("expired") is False
                    )
        if not found or (spec["kind"] == "malformed" and not diagnosed):
            raise ValueError("missing honest recall or diagnostic evidence")
        if spec["kind"] == "timed-review" and not reviewed:
            raise ValueError("missing expired and review-due evidence")
        target_ids = [ids[alias] for alias in spec["targets"]]
        notices = (
            [f"review due: {ids['review']}"]
            if spec["kind"] == "timed-review"
            else spec.get("notices", [])
        )
        if not _answer(answer, target_ids, spec["facts"], [], notices, before["main"]):
            raise ValueError("incorrect answer or provenance")
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError) as exc:
        failures.append(str(exc) or type(exc).__name__)
    return {"success": not failures, "failures": failures, "observations": []}
