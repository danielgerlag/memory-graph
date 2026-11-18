"""Model-independent configuration and scoring for the controlled smoke task."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

FIXTURE = Path(__file__).parent / "fixtures" / "smoke-scope-search"
FIXTURE_SHA256 = "ac68df6fb2ba945652886949211da264bfc9802f479f19f56aaf0c930150c8c1"
EXPECTED_ID = "71e813da-9a91-478a-911b-e21fa8099a01"
FORBIDDEN_ID = "71e813da-9a91-478a-911b-e21fa8099a02"
STORE_ID = "52bd8904-9a68-4391-9c0c-03af22bf2211"
MUTATIONS = {"create", "update", "apply", "supersede", "link", "unlink", "repair", "revert"}
READS = {"find", "show", "neighbors", "review", "check", "diagnose"}
FOUNDRY_MODEL = re.compile(r"openai/azure/[A-Za-z0-9._:-]+")
FOUNDRY_HOST = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\."
    r"(?:openai\.azure\.com|cognitiveservices\.azure\.com|services\.ai\.azure\.com)"
)


def command_arguments(args):
    """Strip only leading, harmless global JSON flags for command inspection."""
    index = 0
    while index < len(args) and args[index] == "--json":
        index += 1
    return args[index:]


def is_safe_help(args):
    command = command_arguments(args)
    return command in (
        ["--help"],
        ["-h"],
        ["fetch", "--help"],
        ["fetch", "-h"],
    )


def fixture_files() -> dict[str, str]:
    if FIXTURE.is_symlink() or not FIXTURE.is_dir():
        raise ValueError("invalid synthetic fixture directory")
    paths = sorted(FIXTURE.iterdir())
    if {path.name for path in paths} != {
        "store.json",
        ".gitignore",
        f"{EXPECTED_ID}.md",
        f"{FORBIDDEN_ID}.md",
    } or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("invalid synthetic fixture")
    digest = hashlib.sha256()
    files = {}
    for path in paths:
        raw = path.read_bytes()
        digest.update(path.name.encode("utf-8") + b"\0" + hashlib.sha256(raw).digest())
        files[path.name] = raw.decode("utf-8")
    if digest.hexdigest() != FIXTURE_SHA256:
        raise ValueError("synthetic fixture hash mismatch")
    return files


def validate_options(args, env):
    """Reject implicit models, credentials, unpinned images and unbounded runs."""
    if not args.run:
        raise ValueError("live execution requires --run")
    model = args.model
    foundry = isinstance(model, str) and FOUNDRY_MODEL.fullmatch(model) is not None
    azure_auth = getattr(args, "azure_auth", None)
    if foundry:
        expected_envs = {"AZUREAI_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"}
        if azure_auth not in (None, "api-key", "default"):
            raise ValueError("--azure-auth must be api-key or default")
        endpoint = getattr(args, "azure_endpoint", None)
        try:
            parts = urlsplit(endpoint if isinstance(endpoint, str) else "")
            valid_endpoint = (
                parts.scheme == "https"
                and parts.hostname is not None
                and FOUNDRY_HOST.fullmatch(parts.hostname) is not None
                and parts.username is None
                and parts.password is None
                and parts.port in (None, 443)
                and parts.path in ("", "/")
                and not parts.query
                and not parts.fragment
                and "?" not in endpoint
                and "#" not in endpoint
            )
        except ValueError:
            valid_endpoint = False
        if not valid_endpoint:
            raise ValueError("--azure-endpoint must be a standard HTTPS Foundry resource endpoint")
        version = getattr(args, "azure_api_version", None)
        if version is not None and (
            not isinstance(version, str)
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:-preview)?", version) is None
        ):
            raise ValueError("--azure-api-version must be an Azure API version")
        family = getattr(args, "model_family", None)
        if family is not None and (
            not isinstance(family, str) or re.fullmatch(r"[A-Za-z0-9._:-]+", family) is None
        ):
            raise ValueError("--model-family must name the underlying model")
    elif isinstance(model, str) and re.fullmatch(r"(openai|anthropic)/[A-Za-z0-9._:-]+", model):
        if (
            getattr(args, "azure_endpoint", None)
            or getattr(args, "azure_api_version", None)
            or azure_auth is not None
            or getattr(args, "model_family", None)
        ):
            raise ValueError("Foundry options require an openai/azure/ model")
        expected_envs = {
            {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[model.split("/", 1)[0]]
        }
    else:
        raise ValueError(
            "--model must explicitly name an openai/, anthropic/, or openai/azure/ model"
        )
    if foundry and azure_auth == "default":
        if args.credential_env is not None and args.credential_env not in expected_envs:
            raise ValueError("--credential-env is not valid for a Foundry configuration")
    elif args.credential_env not in expected_envs or not env.get(args.credential_env):
        names = " or ".join(sorted(expected_envs))
        raise ValueError(f"--credential-env must explicitly select a nonempty {names}")
    if not isinstance(args.image, str) or not re.fullmatch(
        r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}", args.image
    ):
        raise ValueError(
            "--image must be an immutable image ID or reference ending in @sha256:<digest>"
        )
    if (getattr(args, "input_price", None) is None) != (
        getattr(args, "output_price", None) is None
    ):
        raise ValueError("--input-price and --output-price must be supplied together")
    if getattr(args, "budget_usd", None) is not None and not pricing_configured(args):
        raise ValueError(
            "--budget-usd requires both prices; omit the budget to run without a dollar cap"
        )
    for name in ("budget_usd", "input_price", "output_price", "cpus"):
        value = getattr(args, name, None)
        if value is None and name != "cpus":
            continue
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


def pricing_configured(args):
    return (
        getattr(args, "input_price", None) is not None
        and getattr(args, "output_price", None) is not None
    )


def effective_budget(args):
    return getattr(args, "budget_usd", None) if pricing_configured(args) else None


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
        if not all(isinstance(item, str) for item in args):
            failures.append("unknown or invalid tool call")
            continue
        args = command_arguments(args)
        if not args or args[0] not in READS | MUTATIONS:
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
