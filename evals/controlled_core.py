"""Model-independent configuration and scoring for the controlled smoke task."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "inspect_scope_store.json"
FIXTURE_SHA256 = "7c35d076635f953a38fe4de6eb2d6a6e5d21c11a91156fc0e514305335cdb35d"
EXPECTED_ID = "71e813da-9a91-478a-911b-e21fa8099a01"
FORBIDDEN_ID = "71e813da-9a91-478a-911b-e21fa8099a02"
STORE_ID = "52bd8904-9a68-4391-9c0c-03af22bf2211"
MUTATIONS = {"create", "update", "apply", "supersede", "link", "unlink", "repair", "revert"}
READS = {"find", "show", "neighbors", "review", "check", "diagnose"}


def fixture_files() -> dict[str, str]:
    raw = FIXTURE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError("synthetic fixture hash mismatch")
    data = json.loads(raw)
    if (
        not isinstance(data, dict)
        or set(data) != {"store.json", ".gitignore", f"{EXPECTED_ID}.md", f"{FORBIDDEN_ID}.md"}
        or not all(isinstance(value, str) for value in data.values())
    ):
        raise ValueError("invalid synthetic fixture")
    return data


def validate_options(args, env):
    """Reject implicit models, credentials, unpinned images and unbounded runs."""
    if not args.run:
        raise ValueError("live execution requires --run")
    if not isinstance(args.model, str) or not re.fullmatch(
        r"(openai|anthropic)/[A-Za-z0-9._:-]+", args.model
    ):
        raise ValueError("--model must explicitly name an openai/ or anthropic/ model")
    expected_env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[
        args.model.split("/", 1)[0]
    ]
    if args.credential_env != expected_env or not env.get(expected_env):
        raise ValueError(f"--credential-env must explicitly select a nonempty {expected_env}")
    if not isinstance(args.image, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", args.image
    ):
        raise ValueError("--image must be an immutable image reference ending in @sha256:<digest>")
    positive = ("budget_usd", "input_price", "output_price", "cpus")
    for name in positive:
        value = getattr(args, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive finite number")
    caps = {
        "time_limit": 1800,
        "wall_timeout": 3600,
        "tool_timeout": 120,
        "turn_limit": 100,
        "tool_limit": 100,
        "token_limit": 1_000_000,
        "max_tokens": 32768,
        "memory_mb": 8192,
    }
    for name, cap in caps.items():
        value = getattr(args, name)
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError(f"--{name.replace('_', '-')} must be between 1 and {cap}")
    if (
        args.cpus > 8
        or args.tool_timeout > args.time_limit
        or args.max_tokens > args.token_limit
        or args.wall_timeout < args.time_limit
    ):
        raise ValueError("invalid relative resource limits")
    if not args.log_dir:
        raise ValueError("--log-dir is required")
    if args.docker_host and not args.docker_host.startswith("unix:///"):
        raise ValueError("--docker-host must be an explicit unix:/// socket")


def grade(answer, before, after, calls):
    """Score only concrete observations; missing evidence is never a pass."""
    failures = []
    expected_files = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name, value in fixture_files().items()
    }

    def valid_snapshot(value):
        if not isinstance(value, dict) or set(value) != {"main", "refs", "files"}:
            return False
        main, refs, files = value["main"], value["refs"], value["files"]
        return (
            isinstance(main, str)
            and bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", main))
            and refs == [f"refs/heads/main {main}"]
            and isinstance(files, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
        )

    if not valid_snapshot(before) or before.get("files") != expected_files:
        failures.append("missing initial snapshot")
    if not valid_snapshot(after):
        failures.append("missing final snapshot")
    if before != after:
        failures.append("store or refs changed")
    if not isinstance(calls, list) or not calls:
        failures.append("no tool-call evidence")
        calls = []
    found = False
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("args"), list) or not call["args"]:
            failures.append("invalid tool-call evidence")
            continue
        args = call["args"]
        if not all(isinstance(item, str) for item in args) or args[0] not in READS | MUTATIONS:
            failures.append("unknown or invalid tool call")
            continue
        if args[0] in MUTATIONS:
            failures.append("mutation attempted for a read-only request")
        for field in ("stdout", "stderr"):
            text = call.get(field, "")
            if isinstance(text, str) and (FORBIDDEN_ID in text or "ladder is in lab" in text):
                failures.append("different scope exposed through a tool")
        if call.get("status") not in {"success", "error"}:
            failures.append("missing command outcome")
            continue
        if call["status"] == "success":
            try:
                result = json.loads(call["stdout"])
            except (KeyError, TypeError, json.JSONDecodeError):
                failures.append("missing or malformed successful tool output")
                continue
            if not isinstance(result, dict):
                failures.append("invalid tool output")
                continue
            if result.get("truncated") is True:
                failures.append("truncated tool output")
            snapshot = result.get("snapshot")
            if args[0] == "find" and isinstance(snapshot, dict):
                rows = result.get("results")
                if (
                    isinstance(rows, list)
                    and result.get("truncated") is False
                    and snapshot.get("commit") == (before or {}).get("main")
                    and snapshot.get("store_id") == STORE_ID
                    and any(
                        isinstance(row, dict)
                        and row.get("id") == EXPECTED_ID
                        and row.get("scope") == "orchard"
                        and row.get("label") == "ladder is in shed"
                        for row in rows
                    )
                ):
                    found = True
    if not found:
        failures.append("no complete, provenanced retrieval of the expected node")
    try:
        parsed = json.loads(answer) if isinstance(answer, str) else None
    except json.JSONDecodeError:
        parsed = None
    expected = {
        "node_ids": [EXPECTED_ID],
        "facts": ["ladder is in shed"],
        "store_id": STORE_ID,
        "commit": before.get("main") if isinstance(before, dict) else None,
    }
    if parsed != expected:
        failures.append("answer is missing, invalid, unprovenanced, or incorrect")
    return {"success": not failures, "failures": failures}
