#!/usr/bin/env python3
"""Opt-in controlled execution of one independently scored corpus trial."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import run_inspect
from behavior_cases import DATASET_SHA256, SPECS, UNSUPPORTED, inventory
from behavior_grader import grade
from controlled_core import validate_options

ADAPTER_FILES = (
    "behavior_cases.py",
    "behavior_capture.py",
    "behavior_grader.py",
    "behavior_inspect.py",
    "behavior_write_capture.py",
    "behavior_write_tool.py",
    "behavior_write_grader.py",
    "behavior_write_inspect.py",
    "behavior_shared_capture.py",
    "behavior_shared_grader.py",
    "behavior_shared_inspect.py",
    "behavior_fetch_grader.py",
    "behavior_concurrent_capture.py",
    "behavior_concurrent_tool.py",
    "behavior_concurrent_grader.py",
    "behavior_proposal_grader.py",
    "behavior_special_grader.py",
    "controlled_core.py",
    "inspect_task.py",
    "run_inspect.py",
    "run_behavior.py",
)


def adapter_hashes():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ADAPTER_FILES
    }


def parser():
    p = run_inspect.parser()
    p.description = __doc__
    p.add_argument(
        "--inventory", action="store_true", help="list case implementations without running"
    )
    p.add_argument("--case-id", help="select exactly one implemented behavior case")
    p.add_argument("--trial", type=int, help="trial number 1, 2, or 3")
    return p


def result_problems(logs, case_id, budget):
    problems = []
    usage_report = {}
    score_detail = None
    observations = []
    if len(logs) != 1 or logs[0].status != "success":
        problems.append("Inspect run did not complete successfully")
    samples = logs[0].samples if len(logs) == 1 else None
    if not samples or len(samples) != 1:
        problems.append("missing sample evidence")
        return problems, usage_report, score_detail, observations
    sample = samples[0]
    if sample.id != case_id or sample.epoch != 1 or sample.completed_at is None:
        problems.append("incomplete or unexpected sample identity")
    if sample.error is not None or sample.limit is not None:
        problems.append("sample error or limit exhaustion")
    if sample.output is None or sample.output.stop_reason != "stop":
        problems.append("missing or truncated model completion")
    if not sample.scores or len(sample.scores) != 1:
        problems.append("missing deterministic score")
    else:
        score = next(iter(sample.scores.values()))
        score_detail = score.explanation
        if score.value != "C":
            problems.append("deterministic behavior checks failed")
        metadata = score.metadata
        if not isinstance(metadata, dict) or not isinstance(metadata.get("artifacts"), dict):
            problems.append("missing captured grading artifacts")
        elif sample.output is None:
            problems.append("missing model output for captured grading artifacts")
        else:
            checked = grade(case_id, sample.output.completion, metadata["artifacts"])
            observations = checked["observations"]
            if checked["success"] != (score.value == "C"):
                problems.append("inconsistent captured grading artifacts")
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
        if sum(usage.total_cost for usage in sample.model_usage.values()) > budget:
            problems.append("reported cost exceeds the budget")
    return problems, usage_report, score_detail, observations


def trial_coverage(cases, case_id, trial, status):
    return [
        {
            **case,
            "result": "INCOMPLETE" if case["case_id"] == case_id else "NOT RUN",
            "trials": [
                {
                    "trial": index,
                    "result": status
                    if case["case_id"] == case_id and index == trial
                    else "NOT RUN",
                }
                for index in range(1, 4)
            ],
        }
        for case in cases
    ]


def valid_log_location(location, log_dir):
    if not isinstance(location, str) or not location:
        return False
    if "://" in location:
        url = urlsplit(location)
        if url.scheme != "file" or url.netloc not in {"", "localhost"}:
            return False
        location = unquote(url.path)
    path = Path(location).resolve()
    return path.is_relative_to(Path(log_dir).resolve()) and path.is_file()


def main(argv=None):
    args = parser().parse_args(argv)
    root = None
    created = False
    attempted = False
    try:
        cases = inventory()
        if args.inventory:
            print(json.dumps({"status": "NOT RUN", "cases": cases}, indent=2))
            return 0
        if args.case_id not in SPECS or args.case_id in UNSUPPORTED:
            reason = UNSUPPORTED.get(args.case_id, "unknown or unselected case")
            print(json.dumps({"status": "BLOCKED", "case_id": args.case_id, "reason": reason}))
            return 2
        if type(args.trial) is not int or not 1 <= args.trial <= 3:
            raise ValueError("--trial must explicitly be 1, 2, or 3")
        validate_options(args, os.environ)
        source_hashes = adapter_hashes()
        if not hasattr(signal, "SIGALRM"):
            raise ValueError("the outer wall timeout requires SIGALRM (Unix)")
        root = Path(args.log_dir).resolve()
        if not root.is_relative_to(Path.cwd().resolve()):
            raise ValueError("--log-dir must be inside the current working directory")
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        created = True
        run_inspect.isolate_environment(args, root)
        from behavior_inspect import build_task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelCost
        from inspect_task import bundle_hash, skill_files

        task = build_task(args)
        rate = max(args.input_price, args.output_price)
        prices = {
            args.model: ModelCost(
                input=args.input_price,
                output=args.output_price,
                input_cache_read=rate,
                input_cache_write=rate,
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
        problems, usage, score_detail, observations = result_problems(
            logs, args.case_id, args.budget_usd
        )
        if len(logs) != 1 or not valid_log_location(logs[0].location, root / "inspect"):
            problems.append("missing or unsafe retained Inspect log")
        if adapter_hashes() != source_hashes:
            problems.append("evaluation adapter changed during the trial")
        if bundle_hash(skill_files()) != task.metadata["skill_sha256"]:
            problems.append("skill or schema changed during the trial")
        status = (
            "PASS"
            if not problems
            else "FAIL"
            if problems == ["deterministic behavior checks failed"]
            else "ERROR"
        )
        report = {
            "status": status,
            "case_id": args.case_id,
            "trial": args.trial,
            "problems": problems,
            "score_detail": score_detail,
            "observations": observations,
            "dataset_sha256": DATASET_SHA256,
            "adapter_hashes": source_hashes,
            "grader_sha256": hashlib.sha256(
                Path(__file__).with_name("behavior_grader.py").read_bytes()
            ).hexdigest(),
            "case_adapter_sha256": hashlib.sha256(
                Path(__file__).with_name("behavior_cases.py").read_bytes()
            ).hexdigest(),
            "capture_sha256": hashlib.sha256(
                Path(__file__).with_name("behavior_capture.py").read_bytes()
            ).hexdigest(),
            "skill_sha256": task.metadata["skill_sha256"],
            "model": args.model,
            "image": args.image,
            "usage": usage or None,
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
            "inspect_log": logs[0].location if len(logs) == 1 else None,
            "coverage": trial_coverage(cases, args.case_id, args.trial, status),
            "release": "BLOCKED",
            "native_discovery": "NOT RUN",
            "comparison": "NOT RUN",
        }
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if not problems else 1
    except Exception as exc:  # noqa: BLE001 - prevent accidental credential disclosure
        if created and root is not None:
            try:
                (root / "report.json").write_text(
                    json.dumps(
                        {
                            "status": "ERROR" if attempted else "NOT RUN",
                            "case_id": args.case_id,
                            "trial": args.trial,
                            "reason": type(exc).__name__,
                            "release": "BLOCKED",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        message = (
            f"controlled corpus trial {'failed' if attempted else 'not run'}: {type(exc).__name__}"
        )
        if not attempted and isinstance(exc, (ValueError, OSError, ImportError)):
            message += f": {exc}"
        print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
