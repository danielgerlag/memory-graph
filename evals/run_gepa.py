#!/usr/bin/env python3
"""Optimize skill instructions against bounded, explicitly authorized controlled trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import run_behavior
from behavior_cases import SPECS, UNSUPPORTED
from controlled_core import MUTATIONS, READS, command_arguments, validate_options
from corpus import verify_corpus
from inspect_task import bundle_hash, skill_files
from profiles import apply_profiles

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CASES = ("scope-search", "directed-traversal", "conflicting-evidence")
VALIDATION_CASES = ("aliases-distractors", "missing-evidence", "novel-vocabulary")
MAX_INSTRUCTION_BYTES = 16384


class OptimizationError(RuntimeError):
    """A trial, reflection call, or optimization prerequisite failed."""


def split_skill(text):
    if not text.startswith("---\n"):
        raise ValueError("skill is missing its frontmatter")
    head, separator, body = text[4:].partition("\n---\n")
    if not separator or not body.strip():
        raise ValueError("skill frontmatter or instructions are missing")
    return "---\n" + head + "\n---\n", body


def valid_body(body):
    return (
        isinstance(body, str)
        and bool(body.strip())
        and len(body.encode("utf-8")) <= MAX_INSTRUCTION_BYTES
        and not body.lstrip().startswith("---")
    )


def write_private(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(text)


def action_summary(report):
    from inspect_ai.log import read_eval_log

    log = read_eval_log(report["inspect_log"])
    if not log.samples or not log.samples[0].scores:
        raise OptimizationError("saved trial has no score metadata")
    score = next(iter(log.samples[0].scores.values()))
    metadata = score.metadata
    actions = metadata.get("artifacts", {}).get("actions") if isinstance(metadata, dict) else None
    if not isinstance(actions, list):
        raise OptimizationError("saved trial has no action evidence")
    summary = []
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("args"), list):
            continue
        args = command_arguments(action["args"])
        if args and args[0] in READS | MUTATIONS | {"fetch", "status"}:
            status = action.get("status")
            summary.append(
                {
                    "operation": args[0],
                    "status": status if status in {"success", "error"} else "invalid",
                }
            )
    return summary


class TrialAdapter:
    propose_new_texts = None

    def __init__(self, *, root, config, prefix, bundle, max_trials, timeout):
        self.root = root
        self.config = config
        self.prefix = prefix
        self.bundle = bundle
        self.max_trials = max_trials
        self.timeout = timeout
        self.adapter_hashes = run_behavior.adapter_hashes()
        self.corpus_hash = verify_corpus()[4]
        self.trials = 0
        self.metric_calls = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        if self.metric_calls + len(batch) > self.max_trials:
            raise OptimizationError("evaluation budget exhausted")
        self.metric_calls += len(batch)
        body = candidate.get("instructions")
        if not valid_body(body):
            feedback = {
                "status": "INVALID",
                "checks": "Supply only bounded skill instructions, without frontmatter.",
                "actions": [],
            }
            return EvaluationBatch(
                outputs=[feedback] * len(batch),
                scores=[0.0] * len(batch),
                trajectories=[feedback] * len(batch) if capture_traces else None,
            )
        text = self.prefix + body
        candidate_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        candidate_path = self.root / "candidates" / f"{candidate_sha}.md"
        if not candidate_path.exists():
            write_private(candidate_path, text)
        elif candidate_path.read_text(encoding="utf-8") != text:
            raise OptimizationError("candidate file changed")
        expected_bundle = bundle_hash({**self.bundle, "SKILL.md": text.encode("utf-8")})
        outputs, scores, traces = [], [], []
        for case_id in batch:
            if case_id not in SPECS or case_id in UNSUPPORTED:
                raise OptimizationError("unsupported optimization case")
            self.trials += 1
            trial_dir = self.root / "trials" / uuid.uuid4().hex
            command = [
                sys.executable,
                str(ROOT / "evals/run_behavior.py"),
                "--run",
                "--config",
                str(self.config),
                "--case-id",
                case_id,
                "--trial",
                "1",
                "--skill-instructions",
                str(candidate_path),
                "--log-dir",
                str(trial_dir),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise OptimizationError("controlled trial exceeded its outer timeout") from exc
            path = trial_dir / "report.json"
            if not path.is_file():
                raise OptimizationError(f"controlled trial produced no report: {trial_dir}")
            report = json.loads(path.read_text(encoding="utf-8"))
            status = report.get("status")
            if (
                status not in {"PASS", "FAIL"}
                or report.get("case_id") != case_id
                or report.get("skill_sha256") != expected_bundle
                or report.get("adapter_hashes") != self.adapter_hashes
                or report.get("corpus_source_sha256") != self.corpus_hash
                or completed.returncode != (0 if status == "PASS" else 1)
            ):
                raise OptimizationError(
                    f"controlled trial failed or produced invalid evidence: {path}"
                )
            feedback = {
                "status": status,
                "checks": report.get("score_detail") or "",
                "actions": action_summary(report),
                "report": str(path.relative_to(ROOT)),
            }
            outputs.append(feedback)
            scores.append(float(status == "PASS"))
            if capture_traces:
                traces.append(feedback)
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traces or None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if eval_batch.trajectories is None:
            raise OptimizationError("reflection requires captured trial evidence")
        records = [
            {
                "Feedback": {
                    "status": item["status"],
                    "checks": item["checks"],
                    "actions": item["actions"],
                }
            }
            for item in eval_batch.trajectories
        ]
        return {component: records for component in components_to_update}


class Reflector:
    def __init__(self, *, root, config, limit, timeout, max_tokens):
        self.root = root
        self.config = config
        self.limit = limit
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.calls = 0
        self.usage = []
        self.failure = None

    def __call__(self, prompt):
        if self.failure:
            raise OptimizationError(self.failure)
        if self.calls >= self.limit:
            raise OptimizationError("reflection budget exhausted")
        if not isinstance(prompt, str):
            raise OptimizationError("multimodal reflection is not supported")
        prompt += (
            "\n\nPreserve all safety and privacy requirements. Write reusable instructions only; "
            "do not include scenario-specific answers, names, identifiers, or examples. "
            "Do not include or modify YAML frontmatter."
        )
        self.calls += 1
        path = self.root / "reflections" / uuid.uuid4().hex
        command = [
            sys.executable,
            str(ROOT / "evals/gepa_reflection.py"),
            "--run",
            "--config",
            str(self.config),
            "--log-dir",
            str(path),
            "--max-tokens",
            str(self.max_tokens),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.failure = "reflection exceeded its timeout"
            raise OptimizationError(self.failure) from exc
        if completed.returncode != 0:
            match = re.fullmatch(
                r"reflection failed: ([A-Za-z_][A-Za-z_0-9]*)\s*", completed.stderr
            )
            kind = match.group(1) if match else "unknown error"
            self.failure = f"reflection failed: {kind}"
            raise OptimizationError(self.failure)
        try:
            result = json.loads(completed.stdout)
            text = result["completion"]
            if not isinstance(text, str) or not text:
                raise ValueError("empty reflection")
        except (ValueError, KeyError, TypeError) as exc:
            self.failure = "invalid reflection response"
            raise OptimizationError(self.failure) from exc
        self.usage.append(
            {
                "input_tokens": result.get("input_tokens"),
                "output_tokens": result.get("output_tokens"),
            }
        )
        return text


class QuietLogger:
    def __init__(self):
        self.failures = []

    def log(self, message):
        if message.startswith("Per-task reflection failed:") or re.match(
            r"Iteration \d+: Exception (?:building reflective dataset|during optimization):",
            message,
        ):
            self.failures.append("GEPA could not complete a reflection or evaluation")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="authorize model calls")
    p.add_argument("--config", required=True, help="trusted controlled-run profile")
    p.add_argument("--max-trials", type=int, default=18)
    p.add_argument("--max-reflections", type=int, default=3)
    p.add_argument("--reflection-max-tokens", type=int, default=4096)
    p.add_argument("--max-seconds", type=int, default=3600)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    root = None
    adapter = reflector = None
    try:
        if not args.run:
            raise ValueError("--run is required to authorize model calls")
        if not 8 <= args.max_trials <= 100 or not 1 <= args.max_reflections <= 20:
            raise ValueError("trial or reflection limit is outside the permitted range")
        if not 1 <= args.reflection_max_tokens <= 8192 or not 1 <= args.max_seconds <= 7200:
            raise ValueError("reflection token or time limit is outside the permitted range")
        config = Path(args.config).absolute()
        trial_args = run_behavior.parser().parse_args(["--run", "--config", str(config)])
        apply_profiles(trial_args, runner="controlled")
        trial_args.log_dir = "eval-runs/optimization-preflight"
        validate_options(trial_args, os.environ)
        if args.reflection_max_tokens > trial_args.token_limit:
            raise ValueError("reflection token limit exceeds the configured trial token limit")
        if set(TRAIN_CASES) & set(VALIDATION_CASES):
            raise ValueError("training and validation cases must be disjoint")
        bundle = skill_files()
        prefix, body = split_skill(bundle["SKILL.md"].decode("utf-8"))
        root = ROOT / "eval-runs" / f"gepa-{uuid.uuid4().hex}"
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name in ("candidates", "trials", "reflections"):
            (root / name).mkdir(mode=0o700)
        write_private(root / "seed.md", prefix + body)
        adapter = TrialAdapter(
            root=root,
            config=config,
            prefix=prefix,
            bundle=bundle,
            max_trials=args.max_trials,
            timeout=trial_args.wall_timeout + 30,
        )
        reflector = Reflector(
            root=root,
            config=config,
            limit=args.max_reflections,
            timeout=trial_args.wall_timeout + 30,
            max_tokens=args.reflection_max_tokens,
        )
        deadline = time.monotonic() + args.max_seconds

        def stop(_state):
            return (
                adapter.metric_calls + 2 + len(VALIDATION_CASES) > args.max_trials
                or reflector.calls >= args.max_reflections
                or time.monotonic() >= deadline
            )

        import gepa

        logger = QuietLogger()
        result = gepa.optimize(
            seed_candidate={"instructions": body},
            trainset=list(TRAIN_CASES),
            valset=list(VALIDATION_CASES),
            adapter=adapter,
            reflection_lm=reflector,
            reflection_minibatch_size=1,
            max_metric_calls=args.max_trials,
            stop_callbacks=[stop],
            run_dir=str(root / "state"),
            logger=logger,
            display_progress_bar=False,
            use_wandb=False,
            use_mlflow=False,
            raise_on_exception=True,
        )
        if reflector.failure:
            raise OptimizationError(reflector.failure)
        if logger.failures:
            raise OptimizationError(logger.failures[0])
        best = result.best_candidate["instructions"]
        if not valid_body(best):
            raise OptimizationError("optimizer returned invalid skill instructions")
        best_path = root / "best.md"
        write_private(best_path, prefix + best)
        if (
            bundle_hash(skill_files()) != bundle_hash(bundle)
            or run_behavior.adapter_hashes() != adapter.adapter_hashes
            or verify_corpus()[4] != adapter.corpus_hash
        ):
            raise OptimizationError("skill, graders, or corpus changed during optimization")
        from gepa_holdout import CASE_IDS, run_holdout

        if set(CASE_IDS) & (set(TRAIN_CASES) | set(VALIDATION_CASES)):
            raise OptimizationError("holdout cases overlap with optimization cases")

        holdout = run_holdout(
            root,
            config,
            prefix + body,
            prefix + best,
            trial_args.wall_timeout + 30,
        )
        summary = {
            "status": "COMPLETE",
            "training_cases": TRAIN_CASES,
            "validation_cases": VALIDATION_CASES,
            "controlled_trials": adapter.trials,
            "metric_calls": adapter.metric_calls,
            "reflection_calls": reflector.calls,
            "reflection_usage": reflector.usage,
            "best_candidate": str(best_path.relative_to(ROOT)),
            "best_validation_score": result.val_aggregate_scores[result.best_idx],
            "baseline_validation_score": result.val_aggregate_scores[0],
            "candidates": [
                {
                    "validation_score": score,
                    "candidate": str(
                        (
                            root
                            / "candidates"
                            / f"{hashlib.sha256((prefix + candidate['instructions']).encode('utf-8')).hexdigest()}.md"
                        ).relative_to(ROOT)
                    ),
                }
                for candidate, score in zip(
                    result.candidates, result.val_aggregate_scores, strict=True
                )
            ],
            "skill_changed": best != body,
            "holdout": holdout,
            "release": "BLOCKED",
        }
        write_private(root / "summary.json", json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - prevent accidental credential disclosure
        message = (
            str(exc) if isinstance(exc, (ValueError, OptimizationError)) else type(exc).__name__
        )
        if root is not None:
            summary = {
                "status": "ERROR",
                "reason": message,
                "controlled_trials": adapter.trials if adapter else 0,
                "reflection_calls": reflector.calls if reflector else 0,
                "release": "BLOCKED",
            }
            write_private(root / "summary.json", json.dumps(summary, indent=2) + "\n")
        print(f"optimization failed: {message}", file=sys.stderr)
        if root is not None:
            print(f"output directory: {root}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
