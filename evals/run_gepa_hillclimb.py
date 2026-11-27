#!/usr/bin/env python3
"""Hill-climb the skill instructions with GEPA across several models, then test on fresh holdouts.

Subcommands:
  optimize  GEPA search; each candidate is scored by controlled trials on every scoring model.
  holdout   compare candidate instruction files on fresh-vocabulary variants of withheld cases.

Optimization trains on one set of cases and selects on a disjoint validation set. Cases in
HOLDOUT_CASES (as fresh variants) and the cases in neither set are never seen during the search.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import run_behavior
from behavior_cases import SPECS, UNSUPPORTED, dataset
from controlled_core import command_arguments
from corpus import verify_corpus
from gepa_holdout import EXTENDED_CASE_IDS as HOLDOUT_CASES
from gepa_holdout import classify_error, run_holdout_matrix
from inspect_task import bundle_hash, skill_files
from run_gepa import OptimizationError, Reflector, split_skill, valid_body, write_private

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CASES = (
    "remember-fact",
    "repeat-existing",
    "atomic-correction",
    "directed-traversal",
    "scope-search",
    "conflicting-evidence",
    "expired-review",
    "malformed-graph",
)
VALIDATION_CASES = (
    "lost-ack-retry",
    "bounded-traversal",
    "aliases-distractors",
    "missing-evidence",
    "read-only",
    "shared-read",
    "bound-store",
)
INFRA_RETRIES = 2
MAX_COMMANDS_SHOWN = 14


def trial_facts(report):
    """Commands, final answer, and any limit that stopped the trial, from the Inspect log."""
    from inspect_ai.log import read_eval_log

    if not report.get("inspect_log"):
        return [], "", None
    log = read_eval_log(report["inspect_log"])
    sample = log.samples[0] if log.samples else None
    if sample is None:
        return [], "", None
    limit = getattr(sample.limit, "type", None) if sample.limit else None
    score = next(iter(sample.scores.values())) if sample.scores else None
    metadata = score.metadata if score else None
    actions = ((metadata or {}).get("artifacts") or {}).get("actions") or []
    commands = []
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("args"), list):
            continue
        args = command_arguments([str(item)[:80] for item in action["args"]])
        status = action.get("status")
        entry = {"command": " ".join(args[:8]), "result": "ok" if status == "success" else None}
        if status == "invalid":
            entry["result"] = "REJECTED by the tool (never ran)"
        elif status == "error":
            message = ""
            try:
                message = json.loads(action.get("stderr") or "{}")["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = (action.get("stderr") or "")[:160]
            entry["result"] = f"error: {message[:200]}"
        if entry["result"] != "ok" and action.get("input_text"):
            entry["input"] = str(action["input_text"])[:240]
        commands.append(entry)
    answer = (sample.output.completion or "")[:300] if sample.output else ""
    return commands[:MAX_COMMANDS_SHOWN], answer, limit


# Plain-language reading of the grader's terse failure messages, shown to the reflection model.
EXPLANATIONS = (
    (
        ("incomplete action trace", "incomplete action evidence", "unknown, invalid, or truncated"),
        (
            "A command was rejected by the tool before it ran (for example an invented "
            "subcommand or an unsupported option), so the trial is invalid. Every command must "
            "be a real CLI operation with valid arguments."
        ),
    ),
    (
        ("omitted or widened",),
        (
            "A read command (find, show, neighbors, review) lacked the requested --scope or "
            "widened it. Every read must carry the requested scope."
        ),
    ),
    (
        ("missing honest recall or diagnostic evidence",),
        (
            "The commands did not show the required evidence: a scoped search that finds the "
            "target and, for an invalid store, a failed search followed by a `diagnose` run "
            "that identifies the damaged file."
        ),
    ),
    (
        ("invalid mutation action evidence", "missing accepted mutation"),
        (
            "A mutation did not match the requested change: wrong command for the change, an "
            "operation ID reused for a different request, input not passed as `--input -`, or "
            "the change was not accepted."
        ),
    ),
    (
        ("supersession lost history",),
        (
            "A replacement must keep the old node (marked superseded and linked to the new one) "
            "and give the new node only evidence that supports it, in one change."
        ),
    ),
    (
        ("created fact or reported evidence",),
        (
            "The created node must state the requested fact in the requested scope with the "
            "caller's reported interaction source exactly as supplied. "
        ),
    ),
    (
        ("incorrect answer or provenance", "incorrect final answer", "answer lacks"),
        (
            "The final JSON must list exactly the relevant current nodes (no expired, "
            "superseded, or other-scope nodes), their exact labels, the store ID and commit "
            "from the retrieved snapshot, and only the notices the situation calls for."
        ),
    ),
    (
        ("missing complete scoped search", "missing scoped retrieval", "repeat did not search"),
        (
            "The run must show a successful scoped search that returns the relevant node before "
            "the answer or any change."
        ),
    ),
)


def explain(checks):
    lowered = checks.lower()
    for needles, text in EXPLANATIONS:
        if any(needle in lowered for needle in needles):
            return f"{checks} ({text})"
    return checks


class MultiModelAdapter:
    """Score a candidate on each case with several models; the case score is the mean."""

    propose_new_texts = None

    def __init__(self, *, root, config, prefix, bundle, models, workers, max_trials, timeout, env):
        self.root, self.config, self.prefix, self.bundle = root, config, prefix, bundle
        self.models, self.workers, self.max_trials = list(models), workers, max_trials
        self.timeout, self.env = timeout, env
        self.adapter_hashes = run_behavior.adapter_hashes()
        self.corpus_hash = verify_corpus()[4]
        self.prompts = {c["id"]: " / ".join(c["prompts"]) for c in dataset().values()}
        self.lock = threading.Lock()
        self.trials = 0
        self.metric_calls = 0
        self.outcomes = {}

    def _record(self, row):
        with self.lock:
            self.outcomes[row["outcome"]] = self.outcomes.get(row["outcome"], 0) + 1
            with (self.root / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")

    def run_trial(self, case_id, model, path, expected_bundle, candidate_sha, traces):
        last = {"outcome": "INFRA", "checks": "trial could not be scored", "commands": []}
        for attempt in range(1 + INFRA_RETRIES):
            with self.lock:
                if self.trials >= self.max_trials:
                    raise OptimizationError("trial budget exhausted")
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
                "--model",
                f"openai/azure/{model}",
                "--skill-instructions",
                str(path),
                "--log-dir",
                str(trial_dir),
            ]
            started = time.monotonic()
            try:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                continue
            report_path = trial_dir / "report.json"
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            status = report.get("status")
            limit = None
            if status == "ERROR":
                kind, limit = classify_error(report)
                if kind == "infra":
                    continue  # a crash or an API error is not evidence about the skill
            elif status not in {"PASS", "FAIL"}:
                continue
            if (
                report.get("case_id") != case_id
                or report.get("skill_sha256") != expected_bundle
                or report.get("adapter_hashes") != self.adapter_hashes
                or report.get("corpus_source_sha256") != self.corpus_hash
            ):
                raise OptimizationError(f"trial produced invalid evidence: {report_path}")
            commands, answer, _ = trial_facts(report)
            outcome = status if status in {"PASS", "FAIL"} else "BUDGET"
            checks = report.get("score_detail") or ""
            if outcome == "BUDGET":
                checks = f"ran out of its {limit} budget before finishing"
            result = {"outcome": outcome, "checks": checks, "commands": commands, "answer": answer}
            self._record(
                {
                    "candidate": candidate_sha[:12],
                    "case": case_id,
                    "model": model,
                    "outcome": outcome,
                    "checks": checks[:200],
                    "seconds": round(time.monotonic() - started, 1),
                    "traces": traces,
                    "report": str(report_path.relative_to(ROOT)),
                    "attempts": attempt + 1,
                }
            )
            return result
        self._record(
            {
                "candidate": candidate_sha[:12],
                "case": case_id,
                "model": model,
                "outcome": "INFRA",
                "traces": traces,
            }
        )
        return last

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        with self.lock:
            self.metric_calls += len(batch)
        body = candidate.get("instructions")
        if not valid_body(body):
            bad = {"outcome": "INVALID", "checks": "Supply bounded skill instructions only."}
            return EvaluationBatch(
                outputs=[{m: bad for m in self.models}] * len(batch),
                scores=[0.0] * len(batch),
                trajectories=None,
            )
        text = self.prefix + body
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        path = self.root / "candidates" / f"{sha}.md"
        if not path.exists():
            write_private(path, text)
        expected = bundle_hash({**self.bundle, "SKILL.md": text.encode("utf-8")})
        jobs = []
        for case_id in batch:
            if case_id not in SPECS or case_id in UNSUPPORTED:
                raise OptimizationError("unsupported optimization case")
            jobs.extend((case_id, model) for model in self.models)
        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            results = list(
                pool.map(
                    lambda job: self.run_trial(*job, path, expected, sha, capture_traces), jobs
                )
            )
        outputs, scores = [], []
        for index, case_id in enumerate(batch):
            mine = results[index * len(self.models) : (index + 1) * len(self.models)]
            outputs.append({"case": case_id, "models": dict(zip(self.models, mine, strict=True))})
            scores.append(sum(r["outcome"] == "PASS" for r in mine) / len(mine))
        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=outputs if capture_traces else None
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if eval_batch.trajectories is None:
            raise OptimizationError("reflection requires captured trial evidence")
        records = []
        for item in eval_batch.trajectories:
            lines, shown = [], {}
            for model, result in item["models"].items():
                verdict = {"PASS": "passed", "FAIL": "FAILED", "BUDGET": "ran out of budget"}.get(
                    result["outcome"], result["outcome"]
                )
                lines.append(f"{model}: {verdict}. {explain(result['checks'])}".strip())
                shown[model] = {
                    "commands": result.get("commands", []),
                    "final_answer": result.get("answer", ""),
                }
            records.append(
                {
                    "Inputs": {"user_request": self.prompts[item["case"]]},
                    "Generated Outputs": shown,
                    "Feedback": "\n".join(lines),
                }
            )
        return {component: records for component in components_to_update}


TARGETED_EDITS = (
    "\n\nEdit conservatively. Keep every existing sentence and the document's structure unless "
    "the feedback shows that specific text is wrong or missing. Change or add only what the "
    "failures justify, usually a few sentences, and leave working guidance word for word. "
    "Return the complete updated instructions, not a diff and not a summary."
)


class RetryingReflector:
    def __init__(self, inner, attempts=2, suffix=""):
        self.inner, self.attempts, self.suffix = inner, attempts, suffix

    def __call__(self, prompt):
        prompt += self.suffix
        for attempt in range(self.attempts):
            try:
                return self.inner(prompt)
            except OptimizationError:
                if attempt + 1 >= self.attempts:
                    raise
                self.inner.failure = None
                time.sleep(5)
        raise OptimizationError("reflection failed")


class FileLogger:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def log(self, message):
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")


def environment(docker_bin):
    env = dict(os.environ)
    if docker_bin:
        env["PATH"] = f"{docker_bin}{os.pathsep}{env['PATH']}"
    return env


def run_options(args, seed_text):
    return {
        "models": list(args.models),
        "reflection_model": args.reflection_model,
        "targeted_edits": args.targeted_edits,
        "minibatch": args.minibatch,
        "seed_sha256": hashlib.sha256(seed_text.encode("utf-8")).hexdigest(),
    }


def prepare_run(root, args, seed_text):
    """Create a run directory, or validate a resume before any model call is made.

    Returns the trials and reflections already spent so budgets stay cumulative.
    """
    options = run_options(args, seed_text)
    if not root.exists():
        for name in ("candidates", "trials", "reflections"):
            (root / name).mkdir(mode=0o700, parents=True)
        write_private(root / "seed.md", seed_text)
        write_private(root / "options.json", json.dumps(options, indent=2) + "\n")
        return 0, 0
    if not args.resume:
        raise ValueError(f"{root} exists; pass --resume to continue it")
    if (root / "summary.json").exists():
        raise ValueError(f"{root} already finished; start a new --name instead of resuming it")
    saved = root / "options.json"
    if not saved.is_file() or json.loads(saved.read_text(encoding="utf-8")) != options:
        raise ValueError("models, reflection model, seed, or settings differ from the original run")
    evaluations = root / "evaluations.jsonl"
    trials = (
        len(evaluations.read_text(encoding="utf-8").splitlines()) if evaluations.exists() else 0
    )
    return trials, len(list((root / "reflections").iterdir()))


def optimize(args):
    if set(TRAIN_CASES) & set(VALIDATION_CASES) or set(HOLDOUT_CASES) & (
        set(TRAIN_CASES) | set(VALIDATION_CASES)
    ):
        raise ValueError("training, validation, and holdout cases must be disjoint")
    config = Path(args.config).absolute()
    trial_args = run_behavior.parser().parse_args(["--run", "--config", str(config)])
    from profiles import apply_profiles

    apply_profiles(trial_args, runner="controlled")
    bundle = skill_files()
    prefix, body = split_skill(Path(args.seed).read_text(encoding="utf-8"))
    root = ROOT / "eval-runs" / args.name
    spent_trials, spent_reflections = prepare_run(root, args, prefix + body)
    adapter = MultiModelAdapter(
        root=root,
        config=config,
        prefix=prefix,
        bundle=bundle,
        models=args.models,
        workers=args.workers,
        max_trials=args.max_trials,
        timeout=trial_args.wall_timeout + 60,
        env=environment(args.docker_bin),
    )
    adapter.trials = spent_trials
    reflector = Reflector(
        root=root,
        config=config,
        limit=args.max_reflections,
        timeout=trial_args.wall_timeout + 60,
        max_tokens=args.reflection_max_tokens,
        model=f"openai/azure/{args.reflection_model}",
    )
    reflector.calls = spent_reflections
    deadline = time.monotonic() + args.max_seconds
    # Stop before the hard trial cap can be hit in the middle of a batch (which would lose the
    # best candidate): leave room for one validation pass with every retry.
    headroom = len(args.models) * max(len(VALIDATION_CASES), args.minibatch) * (1 + INFRA_RETRIES)

    def stop(_state):
        return (
            reflector.calls >= args.max_reflections
            or time.monotonic() >= deadline
            or adapter.trials + headroom > args.max_trials
        )

    import gepa

    logger = FileLogger(root / "gepa.log")
    result = gepa.optimize(
        seed_candidate={"instructions": body},
        trainset=list(TRAIN_CASES),
        valset=list(VALIDATION_CASES),
        adapter=adapter,
        reflection_lm=RetryingReflector(
            reflector, suffix=TARGETED_EDITS if args.targeted_edits else ""
        ),
        reflection_minibatch_size=args.minibatch,
        candidate_selection_strategy="pareto",
        max_metric_calls=args.max_metric_calls,
        stop_callbacks=[stop],
        run_dir=str(root / "state"),
        logger=logger,
        display_progress_bar=False,
        use_wandb=False,
        use_mlflow=False,
        raise_on_exception=True,
        seed=args.seed_value,
    )
    best = result.best_candidate["instructions"]
    if not valid_body(best):
        raise OptimizationError("optimizer returned invalid skill instructions")
    write_private(root / "best.md", prefix + best)
    if (
        bundle_hash(skill_files()) != bundle_hash(bundle)
        or run_behavior.adapter_hashes() != adapter.adapter_hashes
        or verify_corpus()[4] != adapter.corpus_hash
    ):
        raise OptimizationError("skill, graders, or corpus changed during optimization")
    candidates = []
    for index, (candidate, score) in enumerate(
        zip(result.candidates, result.val_aggregate_scores, strict=True)
    ):
        text = prefix + candidate["instructions"]
        candidates.append(
            {
                "index": index,
                "validation_score": score,
                "parents": (result.parents[index] if getattr(result, "parents", None) else None),
                "sha": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "bytes": len(text.encode("utf-8")),
            }
        )
    summary = {
        "status": "COMPLETE",
        "training_cases": TRAIN_CASES,
        "validation_cases": VALIDATION_CASES,
        "holdout_cases": HOLDOUT_CASES,
        "scoring_models": args.models,
        "reflection_model": args.reflection_model,
        "targeted_edits": args.targeted_edits,
        "controlled_trials": adapter.trials,
        "metric_calls": adapter.metric_calls,
        "trial_outcomes": adapter.outcomes,
        "reflection_calls": reflector.calls,
        "reflection_usage": reflector.usage,
        "best_index": result.best_idx,
        "best_validation_score": result.val_aggregate_scores[result.best_idx],
        "seed_validation_score": result.val_aggregate_scores[0],
        "candidates": candidates,
        "skill_changed": best != body,
    }
    write_private(root / "summary.json", json.dumps(summary, indent=2, default=str) + "\n")
    with contextlib.suppress(Exception):  # optional extra detail only
        write_private(root / "result.json", json.dumps(result.to_dict(), indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


def candidate_texts(paths):
    """Map each candidate file to its text; two files with one name would silently collapse."""
    texts = {}
    for path in paths:
        name = Path(path).stem
        if name in texts:
            raise ValueError(f"candidate name {name!r} is used twice; rename one of the files")
        texts[name] = Path(path).read_text(encoding="utf-8")
    return texts


def holdout(args):
    if not 1 <= args.trials <= 3:
        raise ValueError("--trials must be 1 to 3 (the corpus defines three trials per case)")
    config = Path(args.config).absolute()
    candidates = candidate_texts(args.candidates)
    root = ROOT / "eval-runs" / args.name
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    trial_args = run_behavior.parser().parse_args(["--run", "--config", str(config)])
    from profiles import apply_profiles

    apply_profiles(trial_args, runner="controlled")
    result = run_holdout_matrix(
        root,
        config,
        candidates,
        args.models,
        HOLDOUT_CASES,
        args.trials,
        args.workers,
        trial_args.wall_timeout + 60,
        env=environment(args.docker_bin),
    )
    write_private(root / "holdout.json", json.dumps(result, indent=2, default=str) + "\n")
    print(str(root / "holdout.json"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    common.add_argument("--name", required=True, help="output folder name under eval-runs/")
    common.add_argument("--models", nargs="+", required=True)
    common.add_argument("--workers", type=int, default=6)
    common.add_argument(
        "--docker-bin", help="directory with a real docker CLI to put first on PATH"
    )
    o = sub.add_parser("optimize", parents=[common])
    o.add_argument("--seed", required=True, help="SKILL.md to start from (inside the repository)")
    o.add_argument("--reflection-model", required=True)
    o.add_argument("--max-metric-calls", type=int, default=160, help="case evaluations")
    o.add_argument("--max-trials", type=int, default=800, help="hard cap on model trials")
    o.add_argument("--max-reflections", type=int, default=14)
    o.add_argument("--reflection-max-tokens", type=int, default=8192)
    o.add_argument("--max-seconds", type=int, default=7200)
    o.add_argument("--minibatch", type=int, default=3)
    o.add_argument("--seed-value", type=int, default=0)
    o.add_argument("--resume", action="store_true")
    o.add_argument(
        "--targeted-edits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="tell the reflection model to make small edits instead of rewriting the document",
    )
    h = sub.add_parser("holdout", parents=[common])
    h.add_argument("--candidates", nargs="+", required=True)
    h.add_argument("--trials", type=int, default=3)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        {"optimize": optimize, "holdout": holdout}[args.command](args)
        return 0
    except Exception as exc:  # noqa: BLE001 - keep provider details out of the console
        message = (
            str(exc) if isinstance(exc, (ValueError, OptimizationError)) else type(exc).__name__
        )
        print(f"{args.command} failed: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
