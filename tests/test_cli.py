import json
import os
import shlex
import stat
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
from memory_graph import git_store, service

SCRIPT = Path(__file__).resolve().parents[1] / "memory-graph/scripts/memory_graph.py"


def uid():
    return str(uuid.uuid4())


def run(root, *args, data=None, ok=True):
    command = [sys.executable, str(SCRIPT), "--root", str(root), *args]
    result = subprocess.run(
        command,
        input=json.dumps(data) if data is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if ok:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert result.returncode != 0, result.stdout
    return json.loads(result.stderr) if result.stderr else json.loads(result.stdout)


def mutate(root, action, data, operation=None, ok=True):
    operation = operation or uid()
    result = run(root, action, "--operation", operation, "--input", "-", data=data, ok=ok)
    return result, operation


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "store"
    run(root, "init")
    return root


def configure_promisor_marker(root, tmp_path):
    marker = tmp_path / "remote-helper-ran"
    helper = tmp_path / "fake-upload-pack"
    helper.write_text(f"#!/bin/sh\nprintf ran > {shlex.quote(str(marker))}\nexit 1\n")
    helper.chmod(0o755)
    for key, value in (
        ("core.repositoryformatversion", "1"),
        ("extensions.partialClone", "danger"),
        ("remote.danger.promisor", "true"),
        ("remote.danger.url", "ext::" + str(helper).replace("%", "%%").replace(" ", "% ")),
        ("protocol.ext.allow", "always"),
    ):
        subprocess.run(["git", "-C", str(root), "config", key, value], check=True, timeout=30)
    return marker


def test_retry_patch_and_committed_snapshot(store):
    create = {
        "label": "A snowflake",
        "kind": "concept",
        "body": "Original\n\n---\nBody\n",
        "scope": "one",
    }
    created, op = mutate(store, "create", create)
    node_id = created["receipt"]["node_ids"][0]
    replay, _ = mutate(store, "create", create, op)
    assert replay["replayed"] and replay["state"] == "accepted"
    assert replay["receipt"] == created["receipt"]
    conflict, _ = mutate(store, "create", {**create, "label": "different"}, op, ok=False)
    assert "different request" in conflict["error"]["message"]

    before = run(store, "show", node_id)
    assert before["body"] == create["body"]
    assert before["snapshot"]["commit"] == created["commit"]
    patch = {"id": node_id, "expected_revision": before["revision"], "patch": {"tags": ["winter"]}}
    mutate(store, "update", patch)
    updated = run(store, "show", node_id)
    assert updated["body"] == create["body"]
    assert updated["node"]["scope"] == "one"
    assert updated["node"]["tags"] == ["winter"]
    stale, _ = mutate(store, "update", patch, ok=False)
    assert "stale" in stale["error"]["message"]
    (store / f"{node_id}.md").write_text("dirty and invalid", encoding="utf-8")
    assert run(store, "show", node_id)["body"] == create["body"]
    assert len(run(store, "find", "snowflake")["results"]) == 1


def test_export_view_is_a_separate_immutable_snapshot(store, tmp_path):
    created, _ = mutate(store, "create", {"label": "Temporary", "kind": "item", "ttl_seconds": 0})
    node_id = created["receipt"]["node_ids"][0]
    _, files = git_store.load_snapshot(store)
    (store / f"{node_id}.md").write_text("dirty", encoding="utf-8")
    destination = tmp_path / "view"
    result = run(store, "export-view", "--dest", str(destination))
    assert result["snapshot"]["commit"] == created["commit"]
    assert result["includes_inactive"] and not result["live_store"]
    assert (destination / f"{node_id}.md").read_bytes() == files[f"{node_id}.md"]
    if os.name != "nt":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o700
        assert stat.S_IMODE((destination / f"{node_id}.md").stat().st_mode) == 0o600
    assert not (destination / "operations").exists()
    assert run(destination, "find", "Temporary", ok=False)["error"]["type"] == "GitStoreError"
    assert (
        "already exists"
        in run(store, "export-view", "--dest", str(destination), ok=False)["error"]["message"]
    )
    inside = run(store, "export-view", "--dest", str(store / "view"), ok=False)
    assert "inside the memory store" in inside["error"]["message"]


def test_retry_after_interrupted_preparation_reuses_generated_values(store, monkeypatch):
    op = uid()
    request = {"label": "Temporary", "kind": "claim", "ttl_seconds": 60}
    original = git_store.commit_snapshot

    def interrupted(*args, **kwargs):
        raise OSError("injected interruption")

    monkeypatch.setattr(git_store, "commit_snapshot", interrupted)
    with pytest.raises(OSError, match="injected"):
        service.mutate(store, op, "create", request)
    journal = service.operation_status(store, op)["journal"]
    monkeypatch.setattr(git_store, "commit_snapshot", original)
    result = service.mutate(store, op, "create", request)
    assert result["receipt"]["generated_ids"] == journal["generated_ids"]
    assert result["receipt"]["resolved_at"] == journal["resolved_at"]
    assert result["receipt"]["node_ids"] == [journal["generated_ids"]["create"]]


def test_cooperating_writers_do_not_lose_commits(store):
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                mutate, store, "create", {"label": f"Concurrent {index}", "kind": "item"}
            )
            for index in range(3)
        ]
        results = [future.result()[0] for future in futures]
    assert all(result["state"] == "accepted" for result in results)
    assert len(run(store, "find", "Concurrent")["results"]) == 3


def test_atomic_batch_scope_traversal_and_supersession(store):
    first, second, other = uid(), uid(), uid()
    batch = {
        "changes": [
            {
                "op": "create",
                "node": {
                    "id": first,
                    "label": "Alpha",
                    "kind": "claim",
                    "scope": "s1",
                    "edges": [{"label": "connects", "to": second}],
                },
            },
            {
                "op": "create",
                "node": {
                    "id": second,
                    "label": "Beta",
                    "kind": "concept",
                    "scope": "s1",
                    "edges": [{"label": "connects", "to": other}],
                },
            },
            {
                "op": "create",
                "node": {"id": other, "label": "Gamma", "kind": "concept", "scope": "s2"},
            },
        ]
    }
    mutate(store, "apply", batch)
    result = run(store, "neighbors", first, "--scope", "s1", "--direction", "out")
    assert [item["id"] for item in result["results"]] == [second]
    assert other not in json.dumps(result)
    assert run(store, "find", "Gamma", "--scope", "s1")["results"] == []
    assert run(store, "show", second, "--scope", "s1")["node"]["edges"] == []
    assert run(store, "show", second)["node"]["edges"][0]["to"] == other

    prior = run(store, "show", first)
    result, _ = mutate(
        store,
        "supersede",
        {
            "id": first,
            "expected_revision": prior["revision"],
            "replacement": {"label": "Alpha revised", "kind": "claim", "scope": "s1"},
        },
    )
    assert result["state"] == "accepted"
    assert run(store, "find", "Alpha", "--scope", "s1")["results"][0]["label"] == "Alpha revised"
    old = run(store, "show", first, "--include-inactive")
    assert old["node"]["status"] == "superseded"
    assert old["node"]["superseded_by"] in result["receipt"]["node_ids"]
    assert run(store, "show", first, ok=False)["error"]


def test_neighbors_preserves_distinct_relationships_and_cycles(store):
    a, b = uid(), uid()
    mutate(
        store,
        "apply",
        {
            "changes": [
                {
                    "op": "create",
                    "node": {
                        "id": a,
                        "label": "A",
                        "kind": "item",
                        "edges": [
                            {"label": "first", "to": b, "confidence": 0.7},
                            {"label": "second", "to": b},
                            {"label": "self", "to": a},
                        ],
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": b,
                        "label": "B",
                        "kind": "item",
                        "edges": [{"label": "reverse", "to": a}],
                    },
                },
            ]
        },
    )
    result = run(store, "neighbors", a, "--depth", "1")
    assert [row["id"] for row in result["results"]] == [b]
    assert {
        (edge["source"], edge["label"], edge["target"]) for edge in result["relationships"]
    } == {
        (a, "first", b),
        (a, "second", b),
        (a, "self", a),
        (b, "reverse", a),
    }
    assert (
        next(edge for edge in result["relationships"] if edge["label"] == "first")["confidence"]
        == 0.7
    )
    assert not result["truncated"]


def test_neighbor_limit_bounds_expansion_and_hides_omitted_nodes(store):
    a, b, c = uid(), uid(), uid()
    mutate(
        store,
        "apply",
        {
            "changes": [
                {
                    "op": "create",
                    "node": {
                        "id": a,
                        "label": "A",
                        "kind": "item",
                        "edges": [{"label": "next", "to": b}],
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": b,
                        "label": "B",
                        "kind": "item",
                        "edges": [{"label": "next", "to": c}],
                    },
                },
                {"op": "create", "node": {"id": c, "label": "C", "kind": "item"}},
            ]
        },
    )
    result = run(store, "neighbors", a, "--direction", "out", "--depth", "8", "--limit", "1")
    assert [row["id"] for row in result["results"]] == [b]
    assert c not in json.dumps(result)
    assert result["truncated"] is True
    depth_limited = run(store, "neighbors", a, "--direction", "out", "--depth", "1", "--limit", "5")
    assert [row["id"] for row in depth_limited["results"]] == [b]
    assert c not in json.dumps(depth_limited)
    assert depth_limited["truncated"] is True


def test_review_scope_is_exact(store):
    for scope in ("s1", "s2"):
        mutate(
            store, "create", {"label": scope, "kind": "claim", "scope": scope, "status": "disputed"}
        )
    assert [row["scope"] for row in run(store, "review", "--scope", "s1")["results"]] == ["s1"]
    assert len(run(store, "review")["results"]) == 2


def test_failed_batch_never_exposes_partial_nodes(store):
    node, missing = uid(), uid()
    request = {
        "changes": [
            {
                "op": "create",
                "node": {
                    "id": node,
                    "label": "orphan",
                    "kind": "claim",
                    "edges": [{"label": "x", "to": missing}],
                },
            }
        ]
    }
    before, _ = git_store.load_snapshot(store)
    mutate(store, "apply", request, ok=False)
    after, files = git_store.load_snapshot(store)
    assert before == after
    assert f"{node}.md" not in files


@pytest.mark.parametrize("invalid", [{"id": None}, {"ttl_seconds": None}])
def test_create_rejects_explicit_null_for_nonnullable_inputs(store, invalid):
    before, _ = git_store.load_snapshot(store)
    result, _ = mutate(store, "create", {"label": "Item", "kind": "item", **invalid}, ok=False)
    assert result["error"]
    after, _ = git_store.load_snapshot(store)
    assert after == before


def test_search_rejects_unbounded_query(store):
    error = run(store, "find", "x" * 4097, ok=False)
    assert "4096" in error["error"]["message"]


def test_error_output_is_valid_json_for_unicode_paths(store, tmp_path):
    missing = tmp_path / "λ-missing.json"
    response = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(store),
            "create",
            "--operation",
            uid(),
            "--input",
            str(missing),
        ],
        capture_output=True,
        check=False,
    )
    assert response.returncode != 0
    assert "λ-missing.json" in json.loads(response.stderr)["error"]["message"]


def test_argument_errors_are_structured_json(store):
    response = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(store), "--json", "show"],
        capture_output=True,
        check=False,
    )
    assert response.returncode != 0
    assert "invalid arguments" in json.loads(response.stderr)["error"]["message"]


def test_find_exact_tag_and_kind_filters(store):
    a, b, c = uid(), uid(), uid()
    mutate(
        store,
        "apply",
        {
            "changes": [
                {
                    "op": "create",
                    "node": {
                        "id": a,
                        "label": "Alpha",
                        "kind": "note",
                        "tags": ["Green"],
                        "scope": "one",
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": b,
                        "label": "Beta",
                        "kind": "note",
                        "tags": ["Green"],
                        "scope": "two",
                    },
                },
                {
                    "op": "create",
                    "node": {
                        "id": c,
                        "label": "Gamma",
                        "kind": "claim",
                        "tags": ["green"],
                        "scope": "one",
                    },
                },
            ]
        },
    )
    scoped = run(store, "find", "--tag", "Green", "--kind", "note", "--scope", "one")
    assert [row["id"] for row in scoped["results"]] == [a]
    assert "tag_filter" in scoped["results"][0]["match_reasons"]
    assert run(store, "find", "Gamma", "--tag", "Green")["results"] == []
    assert [row["id"] for row in run(store, "find", "--tag", "green")["results"]] == [c]
    assert "provide a nonempty" in run(store, "find", ok=False)["error"]["message"]


def test_summary_outputs_enforce_actual_utf8_byte_budget(store):
    root_id = uid()
    targets = [uid() for _ in range(25)]
    changes = [
        {
            "op": "create",
            "node": {
                "id": root_id,
                "label": "root",
                "kind": "item",
                "edges": [
                    {"label": f"edge-{index}-" + "r" * 1200, "to": target}
                    for index, target in enumerate(targets)
                ],
            },
        }
    ]
    changes.extend(
        {
            "op": "create",
            "node": {"id": target, "label": "λ" * 1250 + str(index), "kind": "item"},
        }
        for index, target in enumerate(targets)
    )
    mutate(store, "apply", {"changes": changes})
    for args in (["find", "λ", "--limit", "50"], ["neighbors", root_id, "--depth", "1"]):
        response = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(store), *args],
            capture_output=True,
            check=True,
        )
        assert len(response.stdout) <= 65536
        assert json.loads(response.stdout)["truncated"] is True


def test_neighbor_relationship_budget_reports_truncation(store):
    source, target = uid(), uid()
    mutate(
        store,
        "apply",
        {
            "changes": [
                {
                    "op": "create",
                    "node": {
                        "id": source,
                        "label": "source",
                        "kind": "item",
                        "edges": [
                            {"label": "r" * 3000 + str(index), "to": target} for index in range(20)
                        ],
                    },
                },
                {
                    "op": "create",
                    "node": {"id": target, "label": "λ" * 2000, "kind": "item"},
                },
            ]
        },
    )
    response = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(store), "neighbors", source, "--depth", "1"],
        capture_output=True,
        check=True,
    )
    data = json.loads(response.stdout)
    assert len(response.stdout) <= 65536
    assert len(data["results"]) == 1
    assert len(data["relationships"]) < 20
    assert data["truncated"] is True


def test_identical_link_is_idempotent_without_rewriting_node(store):
    a, b = uid(), uid()
    mutate(
        store,
        "apply",
        {
            "changes": [
                {"op": "create", "node": {"id": a, "label": "A", "kind": "object"}},
                {"op": "create", "node": {"id": b, "label": "B", "kind": "object"}},
            ]
        },
    )
    edge = {"label": "relates", "to": b}
    before = run(store, "show", a)
    mutate(store, "link", {"id": a, "expected_revision": before["revision"], "edge": edge})
    linked = run(store, "show", a)
    mutate(store, "link", {"id": a, "expected_revision": linked["revision"], "edge": edge})
    again = run(store, "show", a)
    assert again["revision"] == linked["revision"]
    assert again["node"]["edges"] == [edge]
    mutate(store, "unlink", {"id": a, "expected_revision": again["revision"], "edge": edge})
    assert run(store, "show", a)["node"]["edges"] == []


def test_expiry_does_not_renew_and_null_clears(store):
    result, _ = mutate(store, "create", {"label": "Temporary", "kind": "claim", "ttl_seconds": 0})
    node_id = result["receipt"]["node_ids"][0]
    assert run(store, "find", "Temporary")["results"] == []
    node = run(store, "show", node_id, "--include-inactive")
    expiry = node["node"]["expires_at"]
    assert run(store, "review")["results"][0]["expired"]
    mutate(
        store,
        "update",
        {"id": node_id, "expected_revision": node["revision"], "patch": {"tags": ["checked"]}},
    )
    node = run(store, "show", node_id, "--include-inactive")
    assert node["node"]["expires_at"] == expiry
    mutate(
        store,
        "update",
        {"id": node_id, "expected_revision": node["revision"], "patch": {"expires_at": None}},
    )
    assert len(run(store, "find", "Temporary")["results"]) == 1


def test_submicrosecond_expiry_does_not_expire_early(store, monkeypatch):
    service.mutate(
        store,
        uid(),
        "create",
        {
            "label": "Precise expiry",
            "kind": "item",
            "expires_at": "2030-01-01T00:00:00.0000005Z",
        },
    )
    monkeypatch.setattr(service, "now_utc", lambda: datetime(2030, 1, 1, tzinfo=UTC))
    assert len(service.find(store, "Precise expiry", None, False, 20)["results"]) == 1
    monkeypatch.setattr(service, "now_utc", lambda: datetime(2030, 1, 1, microsecond=1, tzinfo=UTC))
    assert service.find(store, "Precise expiry", None, False, 20)["results"] == []


def test_submicrosecond_ttl_does_not_expire_early(store, monkeypatch):
    start = datetime(2030, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(service, "now_utc", lambda: start)
    operation_id = uid()
    request = {"label": "Precise TTL", "kind": "item", "ttl_seconds": 0.0000005}
    created = service.mutate(store, operation_id, "create", request)
    assert len(service.find(store, "Precise TTL", None, False, 20)["results"]) == 1
    assert service.mutate(store, operation_id, "create", request)["receipt"] == created["receipt"]
    monkeypatch.setattr(service, "now_utc", lambda: datetime(2030, 1, 1, microsecond=1, tzinfo=UTC))
    assert service.find(store, "Precise TTL", None, False, 20)["results"] == []


def test_leap_second_expiry_is_readable(store):
    created, _ = mutate(
        store,
        "create",
        {"label": "Leap expiry", "kind": "item", "expires_at": "2016-12-31T23:59:60Z"},
    )
    node_id = created["receipt"]["node_ids"][0]
    assert run(store, "find", "Leap expiry")["results"] == []
    assert run(store, "show", node_id, "--include-inactive")["lifecycle"]["expired"]


def test_show_hides_edges_to_expired_nodes_unless_explicitly_requested(store):
    source, target = uid(), uid()
    mutate(
        store,
        "apply",
        {
            "changes": [
                {
                    "op": "create",
                    "node": {
                        "id": source,
                        "label": "Source",
                        "kind": "item",
                        "edges": [{"label": "points", "to": target}],
                    },
                },
                {
                    "op": "create",
                    "node": {"id": target, "label": "Target", "kind": "item", "ttl_seconds": 0},
                },
            ]
        },
    )
    assert run(store, "show", source)["node"]["edges"] == []
    assert run(store, "show", source, "--include-inactive")["node"]["edges"][0]["to"] == target


def test_corrupt_main_fails_recall_but_diagnose_reports_hash(store):
    _, prior_op = mutate(store, "create", {"label": "Known", "kind": "item"})
    base, _ = git_store.load_snapshot(store)
    path = f"{uid()}.md"
    invalid_commit = git_store.commit_snapshot(store, base, {path: b"not frontmatter"}, "corrupt")
    assert "invalid main snapshot" in run(store, "find", "anything", ok=False)["error"]["message"]
    diagnostic = run(store, "diagnose", ok=False)
    assert not diagnostic["valid"]
    assert diagnostic["errors"][0]["path"] == path
    assert diagnostic["errors"][0]["sha256"]
    status = run(store, "operation", "status", prior_op)
    assert status["state"] == "accepted"
    assert not status["snapshot_valid"]
    assert status["diagnostic_error_count"] > 0
    repair, op = mutate(
        store,
        "repair",
        {
            "changes": [
                {
                    "path": path,
                    "action": "quarantine",
                    "expected_sha256": diagnostic["errors"][0]["sha256"],
                }
            ]
        },
    )
    assert repair["state"] == "accepted"
    _, files = git_store.load_snapshot(store)
    assert path not in files
    assert files[f"quarantine/{op}/{path}"] == b"not frontmatter"
    assert run(store, "check")["valid"]
    assert run(store, "check-merge", "--base", invalid_commit, "--candidate", repair["commit"])[
        "valid"
    ]


def test_repair_does_not_downgrade_future_schema(store):
    created, _ = mutate(store, "create", {"label": "Versioned", "kind": "claim"})
    node_id = created["receipt"]["node_ids"][0]
    base, files = git_store.load_snapshot(store)
    path = f"{node_id}.md"
    assert b"schema: 1" in files[path]
    future = files[path].replace(b"schema: 1", b"schema: 2", 1)
    git_store.commit_snapshot(store, base, {path: future}, "future schema")
    diagnostic = run(store, "diagnose", ok=False)
    attempt, _ = mutate(
        store,
        "repair",
        {
            "changes": [
                {
                    "path": path,
                    "action": "replace",
                    "expected_sha256": diagnostic["errors"][0]["sha256"],
                    "content": files[path].decode("utf-8"),
                }
            ]
        },
        ok=False,
    )
    assert "unsupported future schema" in attempt["error"]["message"]


def test_damaged_receipts_block_recall_and_automatic_repair(store):
    created, operation_id = mutate(store, "create", {"label": "Item", "kind": "claim"})
    node_id = created["receipt"]["node_ids"][0]
    base, files = git_store.load_snapshot(store)
    receipt_path = f"operations/{operation_id}.json"
    receipt = json.loads(files[receipt_path])
    del receipt["base"]
    git_store.commit_snapshot(store, base, {receipt_path: json.dumps(receipt).encode()}, "damage")
    assert "invalid operation receipt" in run(store, "find", "Item", ok=False)["error"]["message"]
    diagnostic = run(store, "diagnose", ok=False)
    assert diagnostic["errors"][0]["path"] == receipt_path
    assert (
        "receipt history"
        in run(store, "operation", "status", operation_id, ok=False)["error"]["message"]
    )
    error, _ = mutate(
        store,
        "repair",
        {
            "changes": [
                {
                    "path": f"{node_id}.md",
                    "action": "quarantine",
                    "expected_sha256": created["receipt"]["changed_hashes"][f"{node_id}.md"],
                }
            ]
        },
        ok=False,
    )
    assert "receipt history" in error["error"]["message"]


def test_history_and_forward_revert_preserve_retry_receipts(store):
    created, _ = mutate(store, "create", {"label": "Before", "kind": "claim", "body": "original"})
    node_id = created["receipt"]["node_ids"][0]
    original = run(store, "show", node_id)
    request = {
        "id": node_id,
        "expected_revision": original["revision"],
        "patch": {"label": "After"},
    }
    updated, op = mutate(store, "update", request)
    assert run(store, "show", node_id)["node"]["label"] == "After"
    historical = run(store, "show", node_id, "--commit", created["commit"])
    assert historical["node"]["label"] == "Before"
    assert historical["snapshot"]["source"] == "history"
    assert updated["commit"] in run(store, "history")["commits"]
    reverted, _ = mutate(store, "revert", {"target_operation": op})
    assert reverted["state"] == "accepted"
    assert run(store, "show", node_id)["node"]["label"] == "Before"
    replay, _ = mutate(store, "update", request, op)
    assert replay["replayed"]
    assert replay["reverted_by"] == [reverted["operation_id"]]
    assert run(store, "operation", "status", op)["reverted_by"] == [reverted["operation_id"]]
    assert run(store, "show", node_id)["node"]["label"] == "Before"
    _, files = git_store.load_snapshot(store)
    assert f"operations/{op}.json" in files
    assert run(store, "show", node_id, "--commit", "0" * 40, ok=False)["error"]


def test_revert_rejects_later_changes(store):
    created, _ = mutate(store, "create", {"label": "First", "kind": "claim"})
    node_id = created["receipt"]["node_ids"][0]
    current = run(store, "show", node_id)
    _, op = mutate(
        store,
        "update",
        {"id": node_id, "expected_revision": current["revision"], "patch": {"label": "Second"}},
    )
    current = run(store, "show", node_id)
    mutate(
        store,
        "update",
        {"id": node_id, "expected_revision": current["revision"], "patch": {"label": "Third"}},
    )
    error, _ = mutate(store, "revert", {"target_operation": op}, ok=False)
    assert "modified since" in error["error"]["message"]
    assert run(store, "show", node_id)["node"]["label"] == "Third"


def test_merge_gate_validates_base_preconditions_and_receipt_integrity(store):
    created, _ = mutate(store, "create", {"label": "Before", "kind": "claim"})
    node_id = created["receipt"]["node_ids"][0]
    original = run(store, "show", node_id)
    updated, op = mutate(
        store,
        "update",
        {"id": node_id, "expected_revision": original["revision"], "patch": {"label": "After"}},
    )
    valid = run(store, "check-merge", "--base", created["commit"], "--candidate", updated["commit"])
    assert valid["valid"] and valid["new_operations"] == [op]

    _, files = git_store.load_snapshot(store)
    fake = dict(updated["receipt"])
    fake_op = uid()
    fake["operation_id"] = fake_op
    fake["request_hash"] = "0" * 64
    fake["base"] = created["commit"]
    # The receipt refers to the old revision even though main has changed.
    candidate = git_store.commit_snapshot(
        store,
        updated["commit"],
        {f"operations/{fake_op}.json": json.dumps(fake).encode()},
        "candidate",
        ref=f"refs/heads/test/{fake_op}",
        expected_ref="0" * len(updated["commit"]),
    )
    failure = run(
        store, "check-merge", "--base", updated["commit"], "--candidate", candidate, ok=False
    )
    assert "stale affected-node" in failure["error"]["message"]
    assert files[f"operations/{op}.json"]


def test_memory_evidence_must_cite_an_existing_main_history_version(store):
    source, _ = mutate(store, "create", {"label": "Evidence", "kind": "claim"})
    source_id = source["receipt"]["node_ids"][0]
    citation = {
        "kind": "memory",
        "node_id": source_id,
        "revision": source["commit"],
        "observed_at": "2026-01-01T00:00:00Z",
    }
    mutate(store, "create", {"label": "Derived", "kind": "claim", "sources": [citation]})
    assert len(run(store, "find", "Derived")["results"]) == 1
    missing, _ = mutate(
        store,
        "create",
        {"label": "Bad evidence", "kind": "claim", "sources": [{**citation, "node_id": uid()}]},
        ok=False,
    )
    assert "missing node" in missing["error"]["message"]
    base, _ = git_store.load_snapshot(store)
    foreign_branch = git_store.commit_snapshot(
        store, base, {}, "unaccepted", ref=f"refs/heads/test/{uid()}", expected_ref="0" * len(base)
    )
    rejected, _ = mutate(
        store,
        "create",
        {
            "label": "Unpublished evidence",
            "kind": "claim",
            "sources": [{**citation, "revision": foreign_branch}],
        },
        ok=False,
    )
    assert "not in the selected main history" in rejected["error"]["message"]


@pytest.mark.skipif(os.name == "nt", reason="executable marker requires POSIX")
@pytest.mark.parametrize("operation", ["evidence", "historical", "history"])
def test_local_history_and_evidence_never_lazily_execute_helper(
    store, tmp_path, monkeypatch, operation
):
    first, _ = mutate(store, "create", {"label": "Source", "kind": "claim"})
    node_id = first["receipt"]["node_ids"][0]
    request = {"label": "Next", "kind": "claim"}
    if operation == "evidence":
        request["sources"] = [
            {
                "kind": "memory",
                "node_id": node_id,
                "revision": first["commit"],
                "observed_at": "2026-01-01T00:00:00Z",
            }
        ]
    current, _ = mutate(store, "create", request)
    missing = first["commit"]
    (store / ".git" / "objects" / missing[:2] / missing[2:]).unlink()
    marker = configure_promisor_marker(store, tmp_path)
    env = git_store._environment()
    env.pop("GIT_ALLOW_PROTOCOL")
    env.pop("GIT_NO_LAZY_FETCH")
    control = ["merge-base", "--is-ancestor", missing, current["commit"]]
    subprocess.run(
        ["git", "-C", str(store), *control],
        capture_output=True,
        env=env,
        check=False,
        timeout=30,
    )
    assert marker.exists(), "positive control must trigger Git's lazy fetch"
    marker.unlink()
    monkeypatch.setenv("GIT_NO_LAZY_FETCH", "0")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "ext")
    if operation == "evidence":
        error = run(store, "find", "Next", ok=False)
    elif operation == "historical":
        error = run(store, "show", node_id, "--commit", missing, ok=False)
    else:
        error = run(store, "history", ok=False)
    assert error["error"]["type"] == "GraphError"
    assert "history" in error["error"]["message"] or "historical" in error["error"]["message"]
    assert not marker.exists()


def test_local_proposal_requires_external_acceptance_and_explicit_fetch(store, tmp_path):
    remote = tmp_path / "remote.git"
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", str(remote)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(store), "push", str(remote), "refs/heads/main:refs/heads/main"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "clone", "--branch", "main", str(remote), str(clone)],
        check=True,
        capture_output=True,
    )
    run(clone, "attach", "--remote", "origin", "--mode", "propose")
    prepared, op = mutate(clone, "create", {"label": "Shared note", "kind": "claim"})
    assert prepared["state"] == "prepared"
    assert run(clone, "find", "Shared note")["results"] == []
    blocked = run(clone, "publish", "--operation", op, ok=False)
    assert "Local publication is unavailable" in blocked["error"]["message"]
    assert run(clone, "find", "Shared note")["results"] == []
    base = prepared["receipt"]["base"]
    assert run(clone, "check-merge", "--base", base, "--candidate", prepared["commit"])["valid"]
    subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "push",
            "--no-follow-tags",
            str(remote),
            f"{prepared['commit']}:refs/heads/main",
        ],
        check=True,
        capture_output=True,
    )
    assert run(clone, "find", "Shared note")["results"] == []
    run(clone, "fetch")
    found = run(clone, "find", "Shared note")
    assert len(found["results"]) == 1
    assert found["snapshot"]["commit"] == prepared["commit"]
    assert run(clone, "operation", "status", op)["state"] == "accepted"
