import os
import sqlite3
import uuid

import pytest
from memory_graph import cache, git_store, schema, service


def test_cache_matches_parser_and_rejects_corruption(tmp_path, monkeypatch):
    root = tmp_path / "store"
    git_store.init_store(root)
    node_id = str(uuid.uuid4())
    node = {
        "schema": 1,
        "id": node_id,
        "label": "cache example",
        "kind": "item",
        "status": "active",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "edges": [],
    }
    data = schema.dump_node(node, b"body\n")
    path = f"{node_id}.md"
    expected = schema.parse_node(data, path)
    with cache.ParseCache(root) as parse_cache:
        assert parse_cache.parse(data, path) == expected

    def unexpected_parse(*args, **kwargs):
        raise AssertionError("cache entry was not used")

    monkeypatch.setattr(schema, "parse_node", unexpected_parse)
    with cache.ParseCache(root) as parse_cache:
        assert parse_cache.parse(data, path) == expected
    with sqlite3.connect(root / ".memory-graph" / "parse-cache.sqlite3") as connection:
        connection.execute("UPDATE nodes SET payload = 'corrupt'")
    with cache.ParseCache(root) as parse_cache, pytest.raises(cache.CacheError, match="checksum"):
        parse_cache.parse(data, path)


def test_cache_is_only_disposable_and_main_stays_authoritative(tmp_path):
    root = tmp_path / "store"
    git_store.init_store(root)
    result = service.mutate(root, str(uuid.uuid4()), "create", {"label": "cached", "kind": "fact"})
    assert service.find(root, "cached", None, False, 20)["results"]
    cache_file = root / ".memory-graph" / "parse-cache.sqlite3"
    cache_file.unlink()
    assert (
        service.find(root, "cached", None, False, 20)["results"][0]["id"]
        in result["receipt"]["node_ids"]
    )


def test_parallel_priming_preserves_parser_errors(tmp_path, monkeypatch):
    root = tmp_path / "store"
    git_store.init_store(root)
    monkeypatch.setattr(cache, "PARALLEL_THRESHOLD", 4)
    entries = []
    for index in range(4):
        node_id = str(uuid.uuid4())
        node = {
            "schema": 1,
            "id": node_id,
            "label": f"Node {index}",
            "kind": "item",
            "status": "active",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "edges": [],
        }
        entries.append((f"{node_id}.md", schema.dump_node(node, b"body")))
    bad_path, _ = entries[-1]
    entries[-1] = (bad_path, b"invalid")
    with cache.ParseCache(root) as parse_cache:
        parse_cache.prime(entries)
        for path, raw in entries[:-1]:
            assert parse_cache.parse(raw, path) == schema.parse_node(raw, path)
        with pytest.raises(ValueError):
            parse_cache.parse(entries[-1][1], bad_path)


def test_receipt_cache_revalidates_changed_content(tmp_path):
    root = tmp_path / "store"
    git_store.init_store(root)
    calls = []

    def validate(raw, path):
        calls.append((raw, path))
        return {"content": raw.decode()}

    with cache.ParseCache(root) as parse_cache:
        assert parse_cache.parse_receipt(b"one", "operations/example", validate) == {
            "content": "one"
        }
        assert parse_cache.parse_receipt(b"one", "operations/example", validate) == {
            "content": "one"
        }
        assert parse_cache.parse_receipt(b"two", "operations/example", validate) == {
            "content": "two"
        }
    assert calls == [(b"one", "operations/example"), (b"two", "operations/example")]


@pytest.mark.skipif(os.name == "nt", reason="Windows permissions use ACLs")
def test_cache_refuses_nonprivate_permissions(tmp_path):
    root = tmp_path / "store"
    git_store.init_store(root)
    with cache.ParseCache(root):
        pass
    (root / ".memory-graph" / "parse-cache.sqlite3").chmod(0o644)
    with pytest.raises(cache.CacheError, match="not private"):
        cache.ParseCache(root)
