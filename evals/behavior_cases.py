"""Pinned declarative corpus selection and synthetic fixture loading."""

from __future__ import annotations

import hashlib
from string import Template

from corpus import DATASET, semantic_id, verify_corpus

WRITE_KINDS = {"create", "reuse", "retry", "correction"}
SHARED_KINDS = {
    "shared-read",
    "offline-stale",
    "accepted-after-fetch",
    "proposal-not-main",
    "concurrent-changes",
}
SPECIAL_KINDS = {"timed-review", "timed-read", "malformed"}

_, SPECS, UNSUPPORTED, _, _ = verify_corpus()
DATASET_SHA256 = hashlib.sha256(DATASET.read_bytes()).hexdigest()


def dataset():
    data, specs, blocked, _, _ = verify_corpus()
    if hashlib.sha256(DATASET.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("behavior corpus changed; review and repin adapter expectations")
    if specs != SPECS or blocked != UNSUPPORTED:
        raise ValueError("controlled specifications changed during this process")
    if data.get("schema_version") != 1 or data.get("trial_count") != 3:
        raise ValueError("unsupported behavior corpus schema")
    cases = [case for case in data["cases"] if case["lane"] == "behavior"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("behavior corpus must have unique cases")
    if set(SPECS) & set(UNSUPPORTED) or set(SPECS) | set(UNSUPPORTED) != {
        case["id"] for case in cases
    }:
        raise ValueError("behavior inventory does not match the pinned corpus")
    return {case["id"]: case for case in cases}


def inventory():
    return [
        {
            "case_id": case_id,
            "adapter": "UNSUPPORTED" if case_id in UNSUPPORTED else "SUPPORTED",
            "reason": UNSUPPORTED.get(case_id),
            "result": "NOT RUN",
            "trials": [{"trial": trial, "result": "NOT RUN"} for trial in range(1, 4)],
        }
        for case_id in dataset()
    ]


def compile_fixture(case_id):
    cases = dataset()
    if case_id not in SPECS or case_id in UNSUPPORTED:
        raise ValueError(f"unsupported behavior case: {case_id}")
    case = cases[case_id]
    expected_prompts = 2 if SPECS[case_id]["kind"] == "multi" else 1
    expected_mutations = {
        "create": {"node:create", "receipt:create"},
        "reuse": {"node:update", "receipt:create"},
        "retry": {"receipt:reconcile"},
        "correction": {"node:update", "node:create", "receipt:create"},
        "accepted-after-fetch": {"ref:fetch"},
        "proposal-not-main": {"proposal:create", "receipt:create"},
        "concurrent-changes": {"ref:fetch"},
    }.get(SPECS[case_id]["kind"], set())
    if (
        len(case["prompts"]) != expected_prompts
        or set(case["allowed_mutations"]) != expected_mutations
    ):
        raise ValueError("supported case contract changed")
    _, _, _, fixtures, _ = verify_corpus()
    fixture = fixtures[case_id]
    return {
        "case": case,
        "spec": SPECS[case_id],
        "ids": fixture["ids"],
        "nodes": fixture["nodes"],
        "files": {name: content.decode("utf-8") for name, content in fixture["files"].items()},
        "operation_id": semantic_id(case_id, "$operation"),
    }


def prompt_for(case_id, session):
    case = dataset()[case_id]
    spec = SPECS[case_id]
    variables = {"operation_id": semantic_id(case_id, "$operation"), "clock": case["clock"]}
    if spec.get("sources"):
        variables["source"] = spec["sources"][0]
    if spec.get("seed_request"):
        seed = spec["seed_request"]
        variables["seed_node_id"] = semantic_id(case_id, seed["id_alias"])
        variables.update({f"seed_{key}": value for key, value in seed.items() if key != "id_alias"})
    extra = Template(spec["instruction"]).substitute(variables)
    facts_instruction = (
        "In facts use exact node labels or accurately state the relationship using the "
        "node and edge labels. Node IDs may include the starting node and related nodes. "
        "Use recorded source refs in sources. "
        if spec["kind"] == "traversal"
        else "Use exact node labels in facts and source refs in sources. "
    )
    return (
        f"{case['prompts'][session]}\n"
        f"Use exact scope {case['scope']}. {extra} "
        "Use the memory CLI. Return only JSON with node_ids, facts, sources, notices "
        "(all arrays of strings), store_id, and commit from the retrieved snapshot. "
        f"{facts_instruction}Use the notice "
        "'unknown' when no matching fact is available, not merely because a found "
        "fact lacks sources; an empty sources array already states that. Use the "
        "notice 'disagreement' for conflicting evidence, and empty arrays when "
        "not applicable."
    )


def retry_request(case_id):
    spec = SPECS.get(case_id, {})
    if spec.get("kind") != "retry" or "seed_request" not in spec:
        raise ValueError("case has no retry seed")
    request = dict(spec["seed_request"])
    request["id"] = semantic_id(case_id, request.pop("id_alias"))
    return request


def acceptance_request(case_id):
    if case_id != "accepted-after-fetch":
        raise ValueError("case has no accepted remote seed")
    fact = dataset()[case_id]["fixture"]["accepted_fact"]
    return {
        "id": semantic_id(case_id, fact["id"]),
        "label": fact["label"],
        "kind": "claim",
        "scope": fact["scope"],
    }


def conflict_request(case_id, actor, revision):
    if case_id != "concurrent-changes" or actor not in {"pending", "competing"}:
        raise ValueError("case has no such conflicting update")
    case = dataset()[case_id]
    update = case["fixture"][f"{actor}_update"]
    return {
        "id": semantic_id(case_id, update["id"]),
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
