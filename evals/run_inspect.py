#!/usr/bin/env python3
"""Explicit opt-in entrypoint for one controlled Inspect AI smoke trial."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from controlled_core import (
    FIXTURE_SHA256,
    FOUNDRY_MODEL,
    effective_budget,
    pricing_configured,
    validate_options,
)
from profiles import apply_profiles

AZURE_SCOPE = "https://cognitiveservices.azure.com/.default"
# Only the host model client receives these; the sandbox has no host mounts.
AZURE_IDENTITY_ENV = (
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_CLIENT_CERTIFICATE_PATH",
    "AZURE_CLIENT_CERTIFICATE_PASSWORD",
    "AZURE_CLIENT_SEND_CERTIFICATE_CHAIN",
    "AZURE_USERNAME",
    "AZURE_PASSWORD",
    "AZURE_AUTHORITY_HOST",
    "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_TOKEN_CREDENTIALS",
    "AZURE_IDENTITY_DISABLE_MULTITENANTAUTH",
    "AZURE_REGIONAL_AUTHORITY_NAME",
    "AZURE_POD_IDENTITY_AUTHORITY_HOST",
    "AZURE_KUBERNETES_SNI_NAME",
    "AZURE_KUBERNETES_TOKEN_PROXY",
    "AZURE_KUBERNETES_CA_FILE",
    "AZURE_KUBERNETES_CA_DATA",
    "IDENTITY_ENDPOINT",
    "IDENTITY_HEADER",
    "IDENTITY_SERVER_THUMBPRINT",
    "IMDS_ENDPOINT",
    "MSI_ENDPOINT",
    "MSI_SECRET",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", help="YAML agent profile")
    p.add_argument("--harness", help="YAML harness profile")
    p.add_argument("--config", help="combined controlled-run YAML configuration")
    p.add_argument("--run", action="store_true", help="explicitly authorize one model trial")
    p.add_argument("--model")
    p.add_argument("--credential-env", help="explicit provider API-key variable name")
    p.add_argument("--azure-endpoint", help="HTTPS Foundry resource endpoint")
    p.add_argument("--azure-api-version", help="Azure OpenAI API version")
    p.add_argument("--azure-auth", choices=("api-key", "default"), help="Foundry authentication")
    p.add_argument(
        "--model-family", help="underlying model when a Foundry deployment uses an alias"
    )
    p.add_argument("--image", help="locally available image pinned by sha256 digest")
    p.add_argument("--log-dir", help="new directory for retained logs and isolated HOME")
    p.add_argument("--docker-host", help="optional explicit unix:/// Docker socket")
    p.add_argument("--budget-usd", type=float)
    p.add_argument("--input-price", type=float, help="USD per million input tokens")
    p.add_argument("--output-price", type=float, help="USD per million output tokens")
    p.add_argument("--time-limit", type=int, help="sample wall-clock seconds")
    p.add_argument(
        "--wall-timeout", type=int, help="outer run wall-clock seconds, including startup"
    )
    p.add_argument("--tool-timeout", type=int, help="seconds per sandbox command")
    p.add_argument("--turn-limit", type=int)
    p.add_argument("--tool-limit", type=int)
    p.add_argument("--token-limit", type=int, help="total sample tokens")
    p.add_argument("--max-tokens", type=int, help="per-generation output token cap")
    p.add_argument("--cpus", type=float)
    p.add_argument("--memory-mb", type=int)
    return p


def isolate_environment(args, root):
    """Keep the model and Docker host environment separate from the sandbox."""
    default_auth = getattr(args, "azure_auth", None) == "default"
    credential = None if default_auth else os.environ[args.credential_env]
    path = os.environ.get("PATH", os.defpath)
    original_home = Path(os.environ["HOME"]).expanduser() if os.environ.get("HOME") else Path.home()
    compose = original_home / ".docker" / "cli-plugins" / "docker-compose"
    home = root / "home"
    home.mkdir(mode=0o700)
    if compose.is_file() and os.access(compose, os.X_OK):
        plugin_dir = home / ".docker" / "cli-plugins"
        plugin_dir.mkdir(mode=0o700, parents=True)
        (plugin_dir / "docker-compose").symlink_to(compose.resolve(strict=True))
    env = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "TMPDIR": str(root),
        "PATH": path,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if default_auth:
        env.update({name: os.environ[name] for name in AZURE_IDENTITY_ENV if os.environ.get(name)})
        for name, fallback in (
            ("AZURE_CONFIG_DIR", original_home / ".azure"),
            ("AZD_CONFIG_DIR", original_home / ".azd"),
        ):
            env[name] = str(Path(os.environ.get(name) or fallback).expanduser().resolve())
    else:
        env[args.credential_env] = credential
    if args.docker_host:
        env["DOCKER_HOST"] = args.docker_host
    if FOUNDRY_MODEL.fullmatch(args.model):
        env["AZUREAI_OPENAI_BASE_URL"] = args.azure_endpoint
        if args.azure_api_version:
            env["AZUREAI_OPENAI_API_VERSION"] = args.azure_api_version
    os.environ.clear()
    os.environ.update(env)


def preflight_default_credentials(args):
    if getattr(args, "azure_auth", None) != "default":
        return
    try:
        from azure.core.exceptions import ClientAuthenticationError
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise ImportError("Azure credentials require `uv sync --locked --group eval`") from exc
    try:
        with DefaultAzureCredential() as credential:
            credential.get_token(AZURE_SCOPE)
    except (ClientAuthenticationError, ValueError) as exc:
        raise ValueError(
            "Azure default credentials unavailable; run `az login` or `azd auth login` "
            "and check the selected identity"
        ) from exc


@contextmanager
def azure_credential_log_guard(args):
    if getattr(args, "azure_auth", None) != "default":
        yield
        return
    logger = logging.getLogger("azure.identity")
    previous_level = logger.level
    logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


def default_log_dir(args, label):
    if args.run and args.log_dir is None:
        args.log_dir = str(Path("eval-runs") / f"{label}-{uuid.uuid4().hex}")


def model_args(args):
    if FOUNDRY_MODEL.fullmatch(args.model):
        from openai import DefaultAsyncHttpxClient

        return {
            "http_client": DefaultAsyncHttpxClient(follow_redirects=False, trust_env=False),
            "max_retries": 0,
        }
    return {}


def pricing_config(args):
    from inspect_ai.model import ModelCost, ModelInfo, get_model_info, set_model_info

    cost = None
    if pricing_configured(args):
        cache_price = max(args.input_price, args.output_price)
        cost = ModelCost(
            input=args.input_price,
            output=args.output_price,
            input_cache_read=cache_price,
            input_cache_write=cache_price,
        )
    family = getattr(args, "model_family", None)
    info = get_model_info(args.model)
    if info is None and (cost is not None or family is not None):
        set_model_info(args.model, ModelInfo(cost=cost, family=family))
    elif info is not None and family is not None:
        set_model_info(args.model, info.model_copy(update={"family": family}))
    if family is not None and FOUNDRY_MODEL.fullmatch(args.model):
        set_model_info(args.model.split("/", 1)[1], ModelInfo(cost=cost, family=family))
    return {args.model: cost} if cost is not None else None


def cost_control(args):
    configured = pricing_configured(args)
    budget = effective_budget(args)
    if not configured:
        reason = "No prices configured; cost is unknown and no dollar budget is enforced."
    elif budget is None:
        reason = "No dollar budget configured; estimated cost is not capped."
    else:
        reason = "The configured dollar budget is a soft limit."
    return {
        "pricing": "CONFIGURED" if configured else "UNKNOWN",
        "budget": "ENABLED" if budget is not None else "DISABLED",
        "reason": reason,
    }


def warn_if_unbudgeted(args):
    control = cost_control(args)
    if control["budget"] == "DISABLED":
        print(
            f"warning: {control['reason']} Provider usage can still be billed.",
            file=sys.stderr,
        )


def usage_summary(sample, *, pricing, budget):
    usages = sample.model_usage
    if not usages or not all(
        type(usage.total_tokens) is int and usage.total_tokens > 0 for usage in usages.values()
    ):
        return {}, ["missing model token usage"]
    if pricing and any(
        type(usage.total_cost) not in (int, float)
        or not math.isfinite(usage.total_cost)
        or usage.total_cost < 0
        for usage in usages.values()
    ):
        return {}, ["missing model cost evidence"]
    report = {
        name: {"tokens": usage.total_tokens, "cost_usd": usage.total_cost if pricing else None}
        for name, usage in usages.items()
    }
    if (
        pricing
        and budget is not None
        and sum(usage.total_cost for usage in usages.values()) > budget
    ):
        return report, ["reported cost exceeds the budget"]
    return report, []


def main(argv=None):
    args = parser().parse_args(argv)
    root = None
    created = False
    attempted = False
    try:
        apply_profiles(args, runner="controlled")
        default_log_dir(args, "inspect")
        validate_options(args, os.environ)
        warn_if_unbudgeted(args)
        if not hasattr(signal, "SIGALRM"):
            raise ValueError("the outer wall timeout requires SIGALRM (Unix)")
        root = Path(args.log_dir).resolve()
        if not root.is_relative_to(Path.cwd().resolve()):
            raise ValueError("--log-dir must be inside the current working directory")
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        created = True
        isolate_environment(args, root)
        from inspect_ai import eval as inspect_eval
        from inspect_task import build_task

        task = build_task(args)
        prices = pricing_config(args)

        def wall_expired(signum, frame):
            raise TimeoutError("outer evaluation timeout")

        old_handler = signal.signal(signal.SIGALRM, wall_expired)
        signal.alarm(args.wall_timeout)
        try:
            with azure_credential_log_guard(args):
                preflight_default_credentials(args)
                attempted = True
                logs = inspect_eval(
                    task,
                    model=args.model,
                    model_args=model_args(args),
                    log_dir=str(root / "inspect"),
                    log_samples=True,
                    sandbox_prebuilt=True,
                    sandbox_cleanup=True,
                    max_samples=1,
                    max_sandboxes=1,
                    max_tasks=1,
                    max_connections=1,
                    parallel_tool_calls=False,
                    max_retries=0,
                    retry_on_error=0,
                    task_retry_attempts=0,
                    fail_on_error=True,
                    score_on_error=False,
                    max_tokens=args.max_tokens,
                    timeout=args.time_limit,
                    cost_limit=effective_budget(args),
                    model_cost_config=prices,
                    display="plain",
                    log_model_api=False,
                    ctl_server=False,
                )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        problems = []
        if len(logs) != 1 or logs[0].status != "success":
            problems.append("Inspect run did not complete successfully")
        samples = logs[0].samples if len(logs) == 1 else None
        usage_report = {}
        if not samples or len(samples) != 1:
            problems.append("missing sample evidence")
        else:
            sample = samples[0]
            if sample.id != "scope-search" or sample.epoch != 1 or sample.completed_at is None:
                problems.append("incomplete or unexpected sample identity")
            if sample.error is not None or sample.limit is not None:
                problems.append("sample error or limit exhaustion")
            if sample.output is None or sample.output.stop_reason != "stop":
                problems.append("missing or truncated model completion")
            if (
                not sample.scores
                or len(sample.scores) != 1
                or next(iter(sample.scores.values())).value != "C"
            ):
                problems.append("missing or failed deterministic score")
            usage_report, usage_problems = usage_summary(
                sample, pricing=pricing_configured(args), budget=effective_budget(args)
            )
            problems.extend(usage_problems)
        report = {
            "status": "PASS" if not problems else "FAIL",
            "problems": problems,
            "lane": "controlled Inspect smoke; explicitly loaded skill",
            "release": "BLOCKED",
            "native_discovery": "NOT RUN",
            "fixture_sha256": FIXTURE_SHA256,
            "model": args.model,
            "model_family": args.model_family,
            "azure_auth": (args.azure_auth or "api-key")
            if FOUNDRY_MODEL.fullmatch(args.model)
            else None,
            "azure_endpoint": args.azure_endpoint,
            "azure_api_version": args.azure_api_version,
            "image": args.image,
            "output_dir": str(root),
            "skill_sha256": task.metadata["skill_sha256"],
            "limits": {
                "budget_usd": effective_budget(args),
                "requested_budget_usd": args.budget_usd,
                "time_seconds": args.time_limit,
                "wall_timeout_seconds": args.wall_timeout,
                "tool_timeout_seconds": args.tool_timeout,
                "turns": args.turn_limit,
                "tools": args.tool_limit,
                "tokens": args.token_limit,
                "max_tokens": args.max_tokens,
                "cpus": args.cpus,
                "memory_mb": args.memory_mb,
            },
            "input_usd_per_million": args.input_price,
            "output_usd_per_million": args.output_price,
            "usage": usage_report or None,
            "cost_control": cost_control(args),
            "inspect_log": logs[0].location if len(logs) == 1 else None,
        }
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if not problems else 1
    except Exception as exc:  # noqa: BLE001 - record an attempted run without leaking credentials
        message = f"controlled eval {'failed' if attempted else 'not run'}: {type(exc).__name__}"
        if not attempted and isinstance(exc, (ValueError, OSError, ImportError)):
            message += f": {exc}"
        report_written = False
        if created and root is not None and root.is_dir():
            try:
                (root / "report.json").write_text(
                    json.dumps(
                        {
                            "status": "ERROR" if attempted else "NOT RUN",
                            "reason": type(exc).__name__,
                            "cost_control": cost_control(args),
                            "budget_usd": effective_budget(args),
                            "requested_budget_usd": args.budget_usd,
                            "release": "BLOCKED",
                            "native_discovery": "NOT RUN",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                report_written = True
            except OSError as report_error:
                print(
                    f"could not write diagnostic report: {type(report_error).__name__}",
                    file=sys.stderr,
                )
        print(message, file=sys.stderr)
        if report_written:
            print(f"report: {root / 'report.json'}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
