#!/usr/bin/env python3
"""Re-grade saved controlled trials offline with the current graders.

Only grading is re-run, against the saved Inspect transcript; no model is called. ERROR
trials keep their status. The result is cached in ``<run>/regrade.json`` together with the
digest of the graders that produced it, so a grader change invalidates the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

CACHE_FORMAT = 2
IDENTITY_FIELDS = ("skill_sha256", "dataset_sha256", "corpus_source_sha256", "image", "limits")


class RegradeError(Exception):
    pass


def grader_files():
    from run_behavior import adapter_hashes

    hashes = adapter_hashes()
    return {name: hashes[name] for name in sorted(hashes) if name.startswith("graders/")}


def digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def _score_artifacts(trial):
    from inspect_ai.log import read_eval_log

    logs = sorted((trial / "inspect").glob("*.eval"))
    if len(logs) != 1:
        raise RegradeError(f"expected one Inspect log, found {len(logs)}")
    sample = read_eval_log(str(logs[0])).samples[0]
    score = next(iter(sample.scores.values()))
    return sample.output.completion, score.metadata["artifacts"]


def _grade_trial(trial, report):
    from graders import behavior_grader

    verdict, detail = report["status"], report.get("score_detail") or ""
    if verdict in ("PASS", "FAIL"):
        try:
            answer, artifacts = _score_artifacts(trial)
            graded = behavior_grader.grade(report["case_id"], answer, artifacts)
        except Exception as exc:
            raise RegradeError(
                f"{trial.name}: cannot re-grade: {type(exc).__name__}: {exc}"
            ) from exc
        verdict = "PASS" if graded["success"] else "FAIL"
        detail = "; ".join(graded["failures"])
    return verdict, detail


def _identity(reports):
    identity = {}
    for field in IDENTITY_FIELDS:
        values = {json.dumps(report.get(field), sort_keys=True) for report in reports}
        if len(values) != 1:
            raise RegradeError(f"run mixes several values of {field}: {sorted(values)}")
        identity[field] = json.loads(next(iter(values)))
    return identity


def regrade(directory, *, refresh=False):
    """Return ``{"grader_sha256", "identity", "trials"}`` for a run directory."""
    root = Path(directory)
    files = grader_files()
    current = digest(files)
    cache = root / "regrade.json"
    if cache.exists() and not refresh:
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
        except ValueError:
            saved = None
        if (
            isinstance(saved, dict)
            and saved.get("format") == CACHE_FORMAT
            and saved.get("grader_sha256") == current
        ):
            return saved
    reports = sorted(root.glob("*/*/report.json"))
    if not reports:
        raise RegradeError(
            f"no trial reports under {root} (expected <model>/<case>-t<N>/report.json)"
        )
    loaded = [(path.parent, json.loads(path.read_text(encoding="utf-8"))) for path in reports]
    trials = []
    for trial, report in loaded:
        verdict, detail = _grade_trial(trial, report)
        trials.append(
            {
                "model": report["model"],
                "case": report["case_id"],
                "trial": report["trial"],
                "original": report["status"],
                "regraded": verdict,
                "detail": detail,
            }
        )
    keys = [(t["model"], t["case"], t["trial"]) for t in trials]
    if len(set(keys)) != len(keys):
        raise RegradeError("run contains several reports for the same model, case and trial")
    result = {
        "format": CACHE_FORMAT,
        "grader_sha256": current,
        "grader_files": files,
        "identity": _identity([report for _, report in loaded]),
        "trials": trials,
    }
    fd, name = tempfile.mkstemp(dir=root, prefix=".regrade-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=1)
    os.replace(name, cache)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory: <model>/<case>-t<N>/report.json")
    parser.add_argument("--refresh", action="store_true", help="ignore any cached re-grade")
    args = parser.parse_args(argv)
    try:
        result = regrade(args.run, refresh=args.refresh)
    except RegradeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    trials = result["trials"]
    changed = sum(t["original"] != t["regraded"] for t in trials)
    print(f"{len(trials)} trials re-graded, {changed} verdicts differ from the recorded ones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
