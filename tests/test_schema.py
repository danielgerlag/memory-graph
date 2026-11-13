import copy
from datetime import UTC, datetime, timedelta, timezone, tzinfo

import pytest
from memory_graph.schema import (
    MAX_DEPTH,
    MAX_FRONTMATTER_BYTES,
    MAX_NODE_BYTES,
    UnsupportedSchema,
    dump_node,
    parse_node,
    timestamp_reached,
    validate_graph,
)

A = "12345678-1234-4123-8123-123456789abc"
B = "12345678-1234-4123-8123-123456789abd"
C = "12345678-1234-4123-8123-123456789abe"
STAMP = "2026-11-12T18:00:00Z"
REV = "a" * 40


def node(id=A, **kwargs):
    return {
        "schema": 1,
        "id": id,
        "label": "Some memory",
        "kind": "arbitrary",
        "status": "active",
        "created_at": STAMP,
        "updated_at": STAMP,
        "edges": [],
        **kwargs,
    }


def parse(data, id=A):
    return parse_node(data, f"{id}.md")


def test_block_yaml_and_canonical_roundtrip_preserve_body():
    data = (
        f"""---
schema: 1
id: "{A}"
label: "This: is quoted"
kind: topic
status: active
created_at: "2026-11-12T20:30:00+02:30"
updated_at: '{STAMP}'
properties:
  arbitrary: [yes, false, 3, null]
edges:
  - label: references
    to: "{B}"
sources:
  - kind: web
    uri: https://example.com/doc
    observed_at: "{STAMP}"
    mutable: true
---
""".encode()
        + b"# Body\r\n\r\n"
    )
    meta, body = parse(data)
    assert meta["properties"]["arbitrary"] == ["yes", False, 3, None]
    assert body == b"# Body\r\n\r\n"
    output = dump_node(meta, body)
    assert b'created_at: "2026-11-12T18:00:00Z"\n' in output
    assert b'  - {"label": "references", "to":' in output
    assert b'  - {"kind": "web", "mutable": true,' in output
    parsed, reparsed_body = parse(output)
    assert reparsed_body == body
    assert dump_node(parsed, reparsed_body) == output


def test_empty_body_and_crlf_delimiters():
    encoded = dump_node(node(), b"")
    assert parse(encoded)[1] == b""
    assert parse(encoded.replace(b"\n", b"\r\n"))[1] == b""
    assert parse(encoded[:-1])[1] == b""


@pytest.mark.parametrize(
    "extra, message",
    [
        ("label: again\n", "duplicate key"),
        ("unknown: true\n", "Additional properties"),
        ("properties: &anchor {}\n", "anchors"),
        ("properties: {first: &x 1, second: *x}\n", "anchors"),
        ("properties: {<<: {x: 1}}\n", "merge"),
        ("properties: !custom {x: 1}\n", "tag"),
        ("properties: {bad: .inf}\n", "finite"),
        ("properties: {1: value}\n", "non-string"),
        ("properties: {date: 2026-11-12}\n", "JSON-compatible"),
    ],
)
def test_reject_invalid_yaml_and_values(extra, message):
    valid = dump_node(node(), b"")
    data = valid.replace(b"edges: []\n", ("edges: []\n" + extra).encode())
    with pytest.raises(ValueError, match=message):
        parse(data)


def test_unquoted_timestamp_and_unquoted_merge_literal():
    valid = dump_node(node(), b"")
    with pytest.raises(ValueError, match="quoted"):
        parse(valid.replace(f'"{STAMP}"'.encode(), STAMP.encode(), 1))
    quoted_key = valid.replace(b"edges: []\n", b'edges: []\nproperties: {"<<": "ordinary"}\n')
    assert parse(quoted_key)[0]["properties"] == {"<<": "ordinary"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema": 2},
        {"schema": 1.0},
        {"id": A.upper()},
        {"label": ""},
        {"confidence": True},
        {"confidence": 1.1},
        {"status": "other"},
        {"status": "superseded"},
        {"superseded_by": B},
        {"created_at": "2026-02-30T00:00:00Z"},
        {"updated_at": "2026-11-12"},
        {"updated_at": "2026-11-12T00:00:60Z"},
        {"expires_at": "2026-11-12T01:00:00+24:00"},
        {"properties": {"x": float("nan")}},
    ],
)
def test_invalid_metadata(overrides):
    with pytest.raises(ValueError):
        dump_node(node(**overrides), b"")


def test_timestamps_preserve_fraction_and_convert_offsets_without_mutation():
    meta = node(created_at="2026-11-12t00:00:00.123456789-03:30")
    serialized = dump_node(meta, b"")
    assert b'created_at: "2026-11-12T03:30:00.123456789Z"' in serialized
    assert meta["created_at"] == "2026-11-12t00:00:00.123456789-03:30"
    assert b'"0001-01-01T00:00:00Z"' in dump_node(node(created_at="0001-01-01T00:00:00Z"), b"")
    assert b'"2016-12-31T23:59:60Z"' in dump_node(node(created_at="2017-01-01T00:59:60+01:00"), b"")


def test_timestamp_reached_orders_leap_second_at_minute_boundary():
    before = datetime(2016, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
    after = datetime(2017, 1, 1, tzinfo=UTC)
    for leap in ("2016-12-31T23:59:60Z", "2017-01-01T05:29:60.999999999+05:30"):
        assert not timestamp_reached(leap, before)
        assert timestamp_reached(leap, after)
    assert timestamp_reached("2016-12-31T23:59:59.999999Z", before)
    assert not timestamp_reached("2017-01-01T00:00:00Z", before)
    assert timestamp_reached("2017-01-01T00:00:00Z", after)


def test_timestamp_reached_compares_fractions_without_rounding():
    now = datetime(2026, 1, 1, microsecond=123456, tzinfo=UTC)
    assert timestamp_reached("2026-01-01T00:00:00.123455999999999Z", now)
    assert timestamp_reached("2026-01-01T00:00:00.123456000000000Z", now)
    assert not timestamp_reached("2026-01-01T00:00:00.123456000000001Z", now)
    tiny = "2026-01-01T00:00:00." + "0" * 5000 + "1Z"
    assert not timestamp_reached(tiny, now.replace(microsecond=0))
    assert timestamp_reached(tiny, now.replace(microsecond=1))


def test_timestamp_reached_normalizes_offsets_on_both_inputs():
    now = datetime(2026, 1, 1, 5, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert timestamp_reached("2025-12-31T20:00:00-04:00", now)
    assert not timestamp_reached("2025-12-31T20:00:00.0000001-04:00", now)
    assert timestamp_reached("2026-01-01T00:30:00+01:00", now)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "2026-01-01",
        "2026-02-30T00:00:00Z",
        "2026-01-01T00:00:60Z",
        "2026-01-01T00:00:00.١Z",
        None,
    ],
)
def test_timestamp_reached_rejects_invalid_timestamp(value):
    with pytest.raises(ValueError):
        timestamp_reached(value, datetime(2026, 1, 1, tzinfo=UTC))


def test_timestamp_reached_rejects_naive_or_invalid_now():
    class NoOffset(tzinfo):
        def utcoffset(self, dt):
            return None

    for now in (datetime(2026, 1, 1), datetime(2026, 1, 1, tzinfo=NoOffset()), None):
        with pytest.raises(ValueError, match="aware datetime"):
            timestamp_reached(STAMP, now)


def test_unsupported_schema_is_distinct_in_all_entry_points():
    future = node(schema=2)
    with pytest.raises(UnsupportedSchema, match="unsupported schema version 2"):
        dump_node(future, b"")
    with pytest.raises(UnsupportedSchema, match="unsupported schema version 2"):
        validate_graph({A: future})
    data = dump_node(node(), b"").replace(b"schema: 1", b"schema: 2")
    with pytest.raises(UnsupportedSchema, match=f"{A}.md.*unsupported schema version 2"):
        parse(data)
    data = data.replace(b"edges: []", b"edges: []\nnew_field: !future value")
    data = data.replace(f'"{STAMP}"'.encode(), STAMP.encode(), 1)
    with pytest.raises(UnsupportedSchema, match="unsupported schema version 2"):
        parse(data)


def test_yaml_11_directive_is_rejected():
    valid = dump_node(node(), b"")
    with pytest.raises(ValueError):
        parse(valid.replace(b"---\n", b"---\n%YAML 1.1\n---\n", 1))


def test_filename_and_invalid_utf8_and_boundaries():
    valid = dump_node(node(), b"body")
    with pytest.raises(ValueError, match="filename"):
        parse_node(valid, f"{B}.md")
    with pytest.raises(ValueError, match="UTF-8"):
        parse(valid + b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        dump_node(node(), b"\xff")
    with pytest.raises(ValueError, match="frontmatter must start"):
        parse(b"\n" + valid)
    with pytest.raises(ValueError, match="missing closing"):
        parse(b"---\nschema: 1\n")
    with pytest.raises(ValueError, match="node exceeds"):
        parse(b"a" * (MAX_NODE_BYTES + 1))
    with pytest.raises(ValueError, match="frontmatter exceeds"):
        parse(b"---\n#" + b"x" * MAX_FRONTMATTER_BYTES + b"\n---\n")
    with pytest.raises(ValueError, match="node exceeds"):
        dump_node(node(), b"x" * MAX_NODE_BYTES)


def test_depth_and_cycles_are_bounded_on_input_and_output():
    values = []
    current = values
    for _ in range(MAX_DEPTH):
        child = []
        current.append(child)
        current = child
    with pytest.raises(ValueError, match="depth"):
        dump_node(node(properties={"nested": values}), b"")
    current.append(values)
    with pytest.raises(ValueError, match="depth|cycle"):
        dump_node(node(properties={"nested": values}), b"")
    valid = dump_node(node(), b"")
    nested = "[" * MAX_DEPTH + "0" + "]" * MAX_DEPTH
    with pytest.raises(ValueError, match="depth"):
        parse(
            valid.replace(b"edges: []\n", f"edges: []\nproperties: {{nested: {nested}}}\n".encode())
        )


def test_typed_sources_and_locators():
    sources = [
        {
            "kind": "repository_file",
            "observed_at": STAMP,
            "uri": "https://example.com/repo",
            "revision": REV,
            "path": "docs/x.md",
            "section": "Intro",
            "line": 2,
        },
        {
            "kind": "web",
            "observed_at": STAMP,
            "uri": "https://example.com/doc",
            "content_hash": "sha256:abc",
        },
        {"kind": "web", "observed_at": STAMP, "uri": "https://example.com/live", "mutable": True},
        {"kind": "interaction", "observed_at": STAMP, "ref": "opaque"},
        {"kind": "memory", "observed_at": STAMP, "node_id": B, "revision": REV},
    ]
    assert parse(dump_node(node(sources=sources), b""))[0]["sources"] == sources
    validate_graph({A: node(sources=[sources[-1]])})
    for index, key, value in [
        (0, "revision", "short"),
        (0, "path", "../x"),
        (0, "uri", "relative/path"),
        (1, "content_hash", None),
        (2, "mutable", False),
        (3, "ref", ""),
        (4, "node_id", "not-a-uuid"),
    ]:
        broken = copy.deepcopy(sources)
        broken[index][key] = value
        with pytest.raises(ValueError):
            dump_node(node(sources=broken), b"")


def test_edges_uniqueness_and_graph_cycles():
    a = node(edges=[{"label": "rel", "to": B}, {"label": "other", "to": B}])
    b = node(B, edges=[{"label": "back", "to": A}])
    validate_graph({A: a, B: b})
    a["edges"].append({"label": "rel", "to": B, "confidence": 0.9})
    with pytest.raises(ValueError, match="duplicate edge"):
        validate_graph({A: a, B: b})
    with pytest.raises(ValueError, match="duplicate edge"):
        parse(
            dump_node(node(), b"").replace(
                b"edges: []",
                f'edges: [{{"label": "x", "to": "{B}"}}, {{"label": "x", "to": "{B}"}}]'.encode(),
            )
        )


def test_graph_references_identity_and_supersession():
    validate_graph({})
    with pytest.raises(ValueError, match="does not match"):
        validate_graph({B: node()})
    with pytest.raises(ValueError, match="missing node"):
        validate_graph({A: node(edges=[{"label": "rel", "to": B}])})
    with pytest.raises(ValueError, match="missing node"):
        validate_graph({A: node(status="superseded", superseded_by=B)})
    with pytest.raises(ValueError, match="itself"):
        validate_graph({A: node(status="superseded", superseded_by=A)})
    with pytest.raises(ValueError, match="same scope"):
        validate_graph(
            {A: node(status="superseded", superseded_by=B, scope="one"), B: node(B, scope="two")}
        )
    with pytest.raises(ValueError, match="cycle"):
        validate_graph(
            {
                A: node(status="superseded", superseded_by=B),
                B: node(B, status="superseded", superseded_by=C),
                C: node(C, status="superseded", superseded_by=A),
            }
        )
    validate_graph(
        {
            A: node(status="superseded", superseded_by=B),
            B: node(B, status="superseded", superseded_by=C),
            C: node(C),
        }
    )
