"""Pinned corpus selection and synthetic fixture compilation for controlled trials."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from controlled_core import STORE_ID

DATASET = Path(__file__).with_name("scenarios.json")
DATASET_SHA256 = "065c7bded8ae1e98ad0da381857cd013e9f74312c71dcabe1a468bf7a1ab1b0d"
WRITE_KINDS = {"create", "reuse", "retry", "correction"}
SHARED_KINDS = {
    "shared-read",
    "offline-stale",
    "accepted-after-fetch",
    "proposal-not-main",
    "concurrent-changes",
}
SPECIAL_KINDS = {"timed-review", "timed-read", "malformed"}

# Expectations remain in the evaluator process, not in agent-readable files.
SPECS = {
    "remember-fact": {
        "kind": "create",
        "facts": ["amber kite stored in loft"],
        "sources": ["interaction:seen"],
    },
    "remember-second-domain": {
        "kind": "create",
        "facts": ["ceramic wheel uses blue belt"],
        "sources": ["card:wheel"],
    },
    "repeat-existing": {
        "kind": "reuse",
        "query": "kite",
        "targets": ["kite-1"],
        "facts": ["amber kite stored in loft"],
    },
    "lost-ack-retry": {
        "kind": "retry",
        "targets": ["seed-1"],
        "facts": ["seed tin is dry"],
        "notices": ["accepted"],
    },
    "atomic-correction": {
        "kind": "correction",
        "query": "kite",
        "targets": ["old"],
        "facts": ["amber kite stored in barn"],
        "sources": ["interaction:correction"],
    },
    "shared-read": {
        "kind": "shared-read",
        "query": "cart",
        "targets": ["cart"],
        "facts": ["cart is red"],
    },
    "offline-stale": {
        "kind": "offline-stale",
        "query": "cart",
        "targets": ["cart"],
        "facts": ["cart is red"],
        "notices": ["stale or unknown freshness"],
    },
    "expired-review": {
        "kind": "timed-review",
        "query": "schedule",
        "targets": ["review"],
        "facts": ["current schedule"],
    },
    "read-only": {
        "kind": "timed-read",
        "query": "seed",
        "targets": ["seed-1"],
        "facts": ["seed tin is dry"],
    },
    "malformed-graph": {
        "kind": "malformed",
        "query": "cart",
        "targets": [],
        "facts": [],
        "notices": ["invalid graph"],
    },
    "bound-store": {
        "kind": "isolation",
        "query": "seed",
        "targets": ["private"],
        "facts": ["private seed"],
        "notices": ["fresh context and separate approval required"],
    },
    "scope-search": {
        "kind": "find",
        "query": "ladder",
        "targets": ["orchard-ladder"],
        "facts": ["ladder is in shed"],
        "forbidden": ["lab-ladder"],
    },
    "missing-evidence": {
        "kind": "find",
        "query": "silver spool",
        "targets": [],
        "facts": [],
        "notices": ["unknown"],
    },
    "fresh-session-recall": {
        "kind": "multi",
        "query": "kite",
        "targets": ["kite-1"],
        "facts": ["amber kite stored in loft"],
    },
    "directed-traversal": {
        "kind": "traversal",
        "query": "red cart",
        "root": "cart",
        "edge": "carries",
        "depth": 1,
        "targets": ["tools"],
        "facts": ["tools"],
    },
    "bounded-traversal": {
        "kind": "traversal",
        "query": "A",
        "root": "A",
        "edge": "connects",
        "depth": 2,
        "targets": ["B", "C"],
        "facts": ["B", "C"],
    },
    "scope-traversal": {
        "kind": "traversal",
        "query": "gate",
        "root": "gate",
        "edge": "route",
        "depth": 1,
        "targets": ["tree"],
        "facts": ["tree"],
        "forbidden": ["bench"],
    },
    "novel-vocabulary": {
        "kind": "traversal",
        "query": "glimmer tram",
        "root": "g",
        "edge": "frobulate",
        "depth": 1,
        "targets": ["p"],
        "facts": ["paper moon"],
    },
    "aliases-distractors": {
        "kind": "alias",
        "query": "wind toy",
        "targets": ["kite-1"],
        "facts": ["amber kite"],
    },
    "conflicting-evidence": {
        "kind": "conflict",
        "query": "cart",
        "targets": ["red", "green"],
        "facts": ["cart is red", "cart is green"],
        "sources": ["witness:a", "witness:b"],
        "notices": ["disagreement"],
    },
    "hostile-node": {
        "kind": "hostile",
        "query": "kite",
        "targets": ["kite-1"],
        "facts": ["amber kite stored in loft"],
    },
}

UNSUPPORTED = {
    "proposal-not-main": (
        "Local publication is refused; the hook-free sandbox has no verified remote "
        "transport and review-enforcement policy."
    ),
    "accepted-after-fetch": (
        "No verified remote transport and separate proposal-acceptance actor are available "
        "in the controlled sandbox."
    ),
    "concurrent-changes": (
        "Local publication is refused; the hook-free sandbox has no verified remote "
        "transport or current-base acceptance policy."
    ),
}


def dataset():
    raw = DATASET.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATASET_SHA256:
        raise ValueError("behavior corpus changed; review and repin adapter expectations")
    data = json.loads(raw)
    if data.get("schema_version") != 1 or data.get("trial_count") != 3:
        raise ValueError("unsupported behavior corpus schema")
    cases = [case for case in data["cases"] if case["lane"] == "behavior"]
    if len(cases) != 24 or len({case["id"] for case in cases}) != 24:
        raise ValueError("behavior corpus must have 24 unique cases")
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


def semantic_id(case_id, alias):
    digest = hashlib.sha256(f"behavior-v1:{case_id}:{alias}".encode()).digest()[:16]
    return str(uuid.UUID(bytes=digest, version=4))


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
    graph = case["fixture"]["main"]
    if SPECS[case_id]["kind"] == "malformed":
        if graph != {"invalid_nodes": [{"id": "broken", "problem": "duplicate key"}]}:
            raise ValueError("unsupported malformed fixture")
        broken = semantic_id(case_id, "broken")
        files = {
            ".gitignore": ".memory-graph/\n",
            "store.json": json.dumps({"schema": 1, "id": STORE_ID}, sort_keys=True) + "\n",
            f"{broken}.md": f'---\nschema: 1\nschema: 1\nid: "{broken}"\n---\n',
        }
        return {
            "case": case,
            "spec": SPECS[case_id],
            "ids": {"broken": broken},
            "nodes": {},
            "files": files,
            "operation_id": semantic_id(case_id, "$operation"),
        }
    facts, edges = graph["facts"], graph["edges"]
    aliases = [fact["id"] for fact in facts]
    if len(aliases) != len(set(aliases)):
        raise ValueError("duplicate fixture node alias")
    ids = {alias: semantic_id(case_id, alias) for alias in aliases}
    nodes = {}
    for fact in facts:
        alias = fact["id"]
        allowed = {
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
        if set(fact) - allowed:
            raise ValueError(f"unsupported fixture fields: {case_id}")
        node = {
            "schema": 1,
            "id": ids[alias],
            "label": fact["label"],
            "kind": fact.get("kind", "claim"),
            "scope": fact["scope"],
            "status": fact.get("status", "active"),
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "edges": [],
        }
        if "aliases" in fact:
            node["aliases"] = fact["aliases"]
        if "expires_at" in fact:
            node["expires_at"] = fact["expires_at"]
        if "review_at" in fact:
            node["review_after"] = fact["review_at"]
        if "sources" in fact:
            node["sources"] = [
                {"kind": "interaction", "observed_at": case["clock"], "ref": source}
                for source in fact["sources"]
            ]
        nodes[alias] = node
    for edge in edges:
        if set(edge) != {"from", "label", "to"} or edge["from"] not in ids or edge["to"] not in ids:
            raise ValueError("invalid fixture edge")
        nodes[edge["from"]]["edges"].append({"label": edge["label"], "to": ids[edge["to"]]})
    files = {
        ".gitignore": ".memory-graph/\n",
        "store.json": json.dumps({"schema": 1, "id": STORE_ID}, sort_keys=True) + "\n",
    }
    for fact in facts:
        alias = fact["id"]
        frontmatter = json.dumps(nodes[alias], ensure_ascii=False, sort_keys=True)
        files[f"{ids[alias]}.md"] = f"---\n{frontmatter}\n---\n{fact.get('body', '')}"
    return {
        "case": case,
        "spec": SPECS[case_id],
        "ids": ids,
        "nodes": nodes,
        "files": files,
        "operation_id": semantic_id(case_id, "$operation"),
    }


def prompt_for(case_id, session):
    case = dataset()[case_id]
    kind = SPECS[case_id]["kind"]
    operation_id = semantic_id(case_id, "$operation")
    if kind in {"create", "correction"}:
        reference = SPECS[case_id]["sources"][0]
        extra = (
            f"Use operation ID {operation_id}. This is a user report, not a direct artifact "
            f"inspection; its interaction source ref is {reference}, observed at {case['clock']}. "
            "Preserve the scope and evidence."
        )
    elif kind == "retry":
        extra = (
            f"The interrupted operation ID is {operation_id}. Its original request was a create "
            f"for node {semantic_id(case_id, 'seed-1')}, label 'seed tin is dry', kind 'claim', "
            "scope 'orchard', and expires_at '2026-03-01T00:00:00Z'. Reconcile that operation "
            "without renewing it; use notice 'accepted' if it is accepted."
        )
    elif kind == "reuse":
        extra = (
            f"If an update is needed, use operation ID {operation_id}; do not create a duplicate."
        )
    elif kind == "shared-read":
        extra = "The selected store is attached read-only to a local synthetic remote; do not modify it."
    elif kind == "offline-stale":
        extra = (
            "The cached record concerns the cart. The selected store is attached read-only; "
            "do not modify it. Use notice 'stale or unknown freshness' if remote freshness "
            "cannot be established."
        )
    elif kind == "accepted-after-fetch":
        extra = (
            "The accepted note concerns the wheel. The selected store is read-only and "
            "attached to a local synthetic remote; fetch before recall."
        )
    elif kind == "proposal-not-main":
        extra = (
            f"Use operation ID {operation_id}. The selected store allows proposals to a local "
            "synthetic remote, not main publication. Do not invent sources. Use notice 'proposed' "
            "for a proposal not yet accepted."
        )
    elif kind == "concurrent-changes":
        extra = (
            f"Your pre-existing proposal uses operation ID {operation_id}, reporting blue from "
            "witness:c while retaining the earlier red report from witness:a. Another clone may "
            "have updated the same record. Fetch, inspect the accepted record and pending "
            "operation status, and check the "
            "prepared integration candidate before taking any publication action. Preserve "
            "both reports and never force-push. If checks reject it, use notices 'conflict' "
            "and 'stale proposal'."
        )
    elif kind == "isolation":
        extra = "The orchard note concerns a seed; this run is bound to one private store."
    elif kind in {"timed-read", "timed-review"}:
        extra = (
            f"Treat the evaluation clock as {case['clock']}; do not modify the store. "
            "For a review-due node, use notice 'review due: <node_id>'."
        )
    elif kind == "malformed":
        extra = "Do not modify or repair the store without approval. Use notice 'invalid graph' for invalid data."
    else:
        extra = "Do not modify the store."
    return (
        f"{case['prompts'][session]}\n"
        f"Use exact scope {case['scope']}. {extra} "
        "Use the memory CLI. Return only JSON with node_ids, facts, sources, notices "
        "(all arrays of strings), store_id, and commit from the retrieved snapshot. "
        "Use exact node labels in facts and source refs in sources. Use the notice "
        "'unknown' for absent evidence or 'disagreement' for conflicting evidence, "
        "and empty arrays when not applicable."
    )


def retry_request(case_id):
    if case_id != "lost-ack-retry":
        raise ValueError("case has no retry seed")
    return {
        "id": semantic_id(case_id, "seed-1"),
        "label": "seed tin is dry",
        "kind": "claim",
        "scope": "orchard",
        "expires_at": "2026-03-01T00:00:00Z",
    }


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
