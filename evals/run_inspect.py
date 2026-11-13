#!/usr/bin/env python3
"""Explicit opt-in entrypoint for one controlled Inspect AI smoke trial."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
from pathlib import Path

from controlled_core import FIXTURE_SHA256, validate_options


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="explicitly authorize one model trial")
    p.add_argument("--model")
    p.add_argument("--credential-env", help="explicit provider API-key variable name")
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
    """Expose only the chosen API key and a clean, newly-created HOME."""
    credential = os.environ[args.credential_env]
    path = os.environ.get("PATH", os.defpath)
    home = root / "home"
    home.mkdir(mode=0o700)
    env = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "TMPDIR": str(root),
        "PATH": path,
        args.credential_env: credential,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if args.docker_host:
        env["DOCKER_HOST"] = args.docker_host
    os.environ.clear()
    os.environ.update(env)


def main(argv=None):
    args = parser().parse_args(argv)
    root = None
    created = False
    attempted = False
    try:
        validate_options(args, os.environ)
        if not hasattr(signal, "SIGALRM"):
            raise ValueError("the outer wall timeout requires SIGALRM (Unix)")
        root = Path(args.log_dir).resolve()
        if not root.is_relative_to(Path.cwd().resolve()):
            raise ValueError("--log-dir must be inside the current working directory")
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        created = True
        isolate_environment(args, root)
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelCost
        from inspect_task import build_task

        task = build_task(args)
        cache_price = max(args.input_price, args.output_price)
        prices = {
            args.model: ModelCost(
                input=args.input_price,
                output=args.output_price,
                input_cache_read=cache_price,
                input_cache_write=cache_price,
            )
        }
        attempted = True

        def wall_expired(signum, frame):
            raise TimeoutError("outer evaluation timeout")

        old_handler = signal.signal(signal.SIGALRM, wall_expired)
        signal.alarm(args.wall_timeout)
        try:
            logs = inspect_eval(
                task,
                model=args.model,
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
                cost_limit=args.budget_usd,
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
            if not sample.model_usage or not all(
                usage.total_tokens is not None and usage.total_tokens > 0
                for usage in sample.model_usage.values()
            ):
                problems.append("missing model token usage")
            elif any(
                usage.total_cost is None or not math.isfinite(usage.total_cost)
                for usage in sample.model_usage.values()
            ):
                problems.append("missing model cost evidence")
            else:
                usage_report = {
                    name: {"tokens": usage.total_tokens, "cost_usd": usage.total_cost}
                    for name, usage in sample.model_usage.items()
                }
                if sum(usage.total_cost for usage in sample.model_usage.values()) > args.budget_usd:
                    problems.append("reported cost exceeds the budget")
        report = {
            "status": "PASS" if not problems else "FAIL",
            "problems": problems,
            "lane": "controlled Inspect smoke; explicitly loaded skill",
            "release": "BLOCKED",
            "native_discovery": "NOT RUN",
            "fixture_sha256": FIXTURE_SHA256,
            "model": args.model,
            "image": args.image,
            "skill_sha256": task.metadata["skill_sha256"],
            "limits": {
                "budget_usd": args.budget_usd,
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
            "inspect_log": logs[0].location if len(logs) == 1 else None,
        }
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if not problems else 1
    except Exception as exc:  # noqa: BLE001 - record an attempted run without leaking credentials
        message = f"controlled eval {'failed' if attempted else 'not run'}: {type(exc).__name__}"
        if not attempted and isinstance(exc, (ValueError, OSError, ImportError)):
            message += f": {exc}"
        if created and root is not None and root.is_dir():
            try:
                (root / "report.json").write_text(
                    json.dumps(
                        {
                            "status": "ERROR" if attempted else "NOT RUN",
                            "reason": type(exc).__name__,
                            "release": "BLOCKED",
                            "native_discovery": "NOT RUN",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
