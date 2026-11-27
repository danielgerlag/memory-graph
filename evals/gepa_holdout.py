"""Freeze fresh synthetic case variants and compare the baseline and candidate."""

from __future__ import annotations

import io
import json
import re
import secrets
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from inspect_task import bundle_hash, skill_files
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ("hostile-node", "fresh-session-recall", "scope-traversal")
EXTENDED_CASE_IDS = (*CASE_IDS, "remember-second-domain", "novel-vocabulary")
ALIAS_KEYS = {"id", "root", "targets", "forbidden", "from", "to"}


def _rewrite(value, replacements, key=None):
    if key in ALIAS_KEYS:
        return value
    if isinstance(value, dict):
        return {name: _rewrite(item, replacements, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, replacements) for item in value]
    if isinstance(value, str):
        for old, new in replacements:
            value = re.sub(
                rf"(?<![\w-]){re.escape(old)}(?![\w-])",
                lambda _, replacement=new: replacement,
                value,
            )
    return value


def _yaml():
    parser = YAML(typ="safe", pure=True)
    parser.default_flow_style = False
    return parser


def _dump(value):
    output = io.StringIO()
    _yaml().dump(value, output)
    return output.getvalue()


def _rewrite_case(path, replacements):
    case = _yaml().load(path.read_text(encoding="utf-8"))
    checks = case["checks"]
    case = _rewrite(case, replacements)
    for original, rewritten in zip(checks, case["checks"]):
        if original.get("path") == "/answer/node_ids":
            rewritten["value"] = original["value"]
    path.write_text(_dump(case), encoding="utf-8")


def _rewrite_node(path, replacements):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("holdout node has no frontmatter")
    front, separator, body = text[4:].partition("\n---\n")
    if not separator:
        raise ValueError("holdout node has no closing frontmatter")
    node = _rewrite(_yaml().load(front), replacements)
    path.write_text(
        "---\n" + _dump(node) + "---\n" + _rewrite(body, replacements), encoding="utf-8"
    )


def prepare_suite(root, case_ids=CASE_IDS):
    suite = Path(root).resolve(strict=True) / "holdout-suite"
    suite.mkdir(mode=0o700)
    shutil.copytree(
        ROOT / "evals",
        suite / "evals",
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.local.yaml"),
    )
    skill_dir = suite / "memory-graph"
    for name, content in skill_files().items():
        target = skill_dir / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(content)
    transforms = _yaml().load((ROOT / "evals" / "gepa_holdout.yaml").read_text(encoding="utf-8"))
    if not isinstance(transforms, dict) or not set(case_ids) <= set(transforms):
        raise ValueError("invalid holdout transformations")
    nonce = secrets.token_hex(4)
    for case_id in case_ids:
        mapping = transforms[case_id]
        if not isinstance(mapping, dict) or not all(
            isinstance(key, str) and key and isinstance(value, str) and "{nonce}" in value
            for key, value in mapping.items()
        ):
            raise ValueError("invalid holdout substitution")
        replacements = sorted(
            ((key, value.replace("{nonce}", nonce)) for key, value in mapping.items()),
            key=lambda item: -len(item[0]),
        )
        _rewrite_case(suite / "evals" / "cases" / "behavior" / f"{case_id}.yaml", replacements)
        for node in (suite / "evals" / "fixtures" / "graphs" / case_id).glob("*.md"):
            _rewrite_node(node, replacements)
    result = subprocess.run(
        [sys.executable, str(suite / "evals" / "build_dataset.py"), "--write"],
        cwd=suite,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        raise ValueError(f"could not compile the fresh holdout corpus (exit {result.returncode})")
    return suite


def run_holdout(root, config, baseline, candidate, timeout):
    root = Path(root).resolve(strict=True)
    config = Path(config).resolve(strict=True)
    suite = prepare_suite(root)
    candidates = {"baseline": baseline}
    if candidate != baseline:
        candidates["candidate"] = candidate
    candidate_dir = suite / "candidate-instructions"
    candidate_dir.mkdir(mode=0o700)
    results = {}
    for name, text in candidates.items():
        expected_bundle = bundle_hash({**skill_files(), "SKILL.md": text.encode("utf-8")})
        candidate_path = candidate_dir / f"{name}.md"
        candidate_path.write_text(text, encoding="utf-8")
        candidate_path.chmod(0o600)
        outcomes = []
        for case_id in CASE_IDS:
            trial_dir = suite / "eval-runs" / f"holdout-{uuid.uuid4().hex}"
            command = [
                sys.executable,
                str(suite / "evals" / "run_behavior.py"),
                "--run",
                "--config",
                str(config),
                "--case-id",
                case_id,
                "--trial",
                "1",
                "--skill-instructions",
                str(candidate_path),
                "--log-dir",
                str(trial_dir),
            ]
            result = subprocess.run(
                command, cwd=suite, capture_output=True, text=True, timeout=timeout, check=False
            )
            path = trial_dir / "report.json"
            if not path.is_file():
                raise ValueError(f"holdout trial produced no report: {trial_dir}")
            report = json.loads(path.read_text(encoding="utf-8"))
            status = report.get("status")
            if (
                status not in {"PASS", "FAIL"}
                or report.get("case_id") != case_id
                or report.get("skill_sha256") != expected_bundle
                or result.returncode != (0 if status == "PASS" else 1)
            ):
                raise ValueError(f"holdout trial failed or produced invalid evidence: {path}")
            outcomes.append(
                {
                    "case_id": case_id,
                    "status": status,
                    "score_detail": report.get("score_detail"),
                    "observations": report.get("observations", []),
                    "report": str(path.relative_to(ROOT)),
                }
            )
        results[name] = outcomes
    return {
        "status": "COMPLETE",
        "description": "Withheld cases with fresh randomized vocabulary, one trial each.",
        "results": results,
    }


def run_holdout_matrix(
    root, config, candidates, models, case_ids, trials, workers, timeout, env=None, infra_retries=2
):
    """Run every candidate on fresh-vocabulary variants for several models, in parallel.

    Infrastructure ERRORs are retried; a trial that still cannot be scored is recorded as ERROR
    instead of aborting the comparison.
    """
    import concurrent.futures

    root = Path(root).resolve(strict=True)
    config = Path(config).resolve(strict=True)
    suite = prepare_suite(root, case_ids)
    candidate_dir = suite / "candidate-instructions"
    candidate_dir.mkdir(mode=0o700)
    paths = {}
    for name, text in candidates.items():
        paths[name] = candidate_dir / f"{name}.md"
        paths[name].write_text(text, encoding="utf-8")
        paths[name].chmod(0o600)

    def one(name, model, case_id, trial):
        row = {"candidate": name, "model": model, "case": case_id, "trial": trial}
        for _attempt in range(1 + infra_retries):
            trial_dir = suite / "eval-runs" / f"holdout-{uuid.uuid4().hex}"
            command = [
                sys.executable,
                str(suite / "evals" / "run_behavior.py"),
                "--run",
                "--config",
                str(config),
                "--case-id",
                case_id,
                "--trial",
                str(trial),
                "--model",
                f"openai/azure/{model}",
                "--skill-instructions",
                str(paths[name]),
                "--log-dir",
                str(trial_dir),
            ]
            try:
                subprocess.run(
                    command,
                    cwd=suite,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                continue
            path = trial_dir / "report.json"
            if not path.is_file():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            row.update(
                status=report.get("status"),
                detail=report.get("score_detail") or "; ".join(report.get("problems") or []),
                observations=report.get("observations", []),
                report=str(path.relative_to(root)),
            )
            if report.get("status") in {"PASS", "FAIL"} or "limit exhaustion" in " ".join(
                report.get("problems") or []
            ):
                return row
        row.setdefault("status", "ERROR")
        return row

    jobs = [
        (name, model, case_id, trial)
        for name in candidates
        for model in models
        for case_id in case_ids
        for trial in range(1, trials + 1)
    ]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(lambda job: one(*job), jobs))
    return {
        "status": "COMPLETE",
        "description": "Withheld cases with fresh randomized vocabulary.",
        "suite": str(suite.relative_to(root)),
        "rows": rows,
    }
