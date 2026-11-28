#!/usr/bin/env python3
"""Record compact evaluation baselines and test a candidate run against one.

A baseline is a small JSON file of per-model, per-case pass counts plus the identity of
what produced them (skill, dataset, corpus, graders, limits). It holds no transcripts.

  baseline.py record RUN --name NAME     summarize a saved run directory (re-graded offline)
  baseline.py compare BASELINE CANDIDATE paired comparison; CANDIDATE is a run directory or
                                         another baseline file

Trials are scored per (model, case) cell. The overall delta is the mean of per-case deltas,
and its interval comes from a bootstrap that resamples cases, because the 3 trials of one
case are not independent evidence. Models and cases are never pooled into a single trial
count. ERROR trials are excluded from rates and reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

FORMAT = 1
BASELINES = Path(__file__).resolve().parent / "baselines"
MARGIN = 0.05
CASE_ALPHA = 0.01
MAX_ERROR_RATE = 0.05
RESAMPLES = 10000
SEED = 20260101
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
INCOMPATIBLE = ("dataset_sha256", "corpus_source_sha256", "grader_sha256")
ADVISORY = ("limits", "image")


class BaselineError(Exception):
    pass


def summarize(result, *, name, note=""):
    counts = {}
    for trial in result["trials"]:
        cell = counts.setdefault(trial["model"], {}).setdefault(
            trial["case"], {"pass": 0, "fail": 0, "error": 0}
        )
        verdict = trial["regraded"]
        cell["pass" if verdict == "PASS" else "fail" if verdict == "FAIL" else "error"] += 1
    return {
        "format": FORMAT,
        "name": name,
        "note": note,
        "recorded": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "identity": {**result["identity"], "grader_sha256": result["grader_sha256"]},
        "results": {
            model: {case: counts[model][case] for case in sorted(counts[model])}
            for model in sorted(counts)
        },
    }


def validate(baseline, source):
    try:
        if baseline["format"] != FORMAT:
            raise BaselineError(f"{source}: unsupported baseline format {baseline['format']!r}")
        identity = baseline["identity"]
        for field in (*INCOMPATIBLE, "skill_sha256"):
            if not isinstance(identity[field], str):
                raise BaselineError(f"{source}: identity.{field} is missing")
        for model, cases in baseline["results"].items():
            for case, cell in cases.items():
                if set(cell) != {"pass", "fail", "error"} or any(
                    not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in cell.values()
                ):
                    raise BaselineError(f"{source}: malformed counts for {model} / {case}")
    except (KeyError, TypeError, AttributeError) as exc:
        raise BaselineError(f"{source}: malformed baseline ({exc!r})") from exc
    return baseline


def load(reference, *, refresh=False):
    """Load a baseline name or file, or summarize a run directory with the current graders."""
    path = Path(reference)
    if not path.exists() and not path.suffix and NAME.match(reference):
        path = BASELINES / f"{reference}.json"
    if path.is_dir():
        import regrade

        try:
            result = regrade.regrade(path, refresh=refresh)
        except regrade.RegradeError as exc:
            raise BaselineError(str(exc)) from exc
        return validate(summarize(result, name=path.name), str(path))
    if not path.is_file():
        raise BaselineError(f"{reference}: not a baseline file, baseline name, or run directory")
    try:
        return validate(json.loads(path.read_text(encoding="utf-8")), str(path))
    except ValueError as exc:
        raise BaselineError(f"{path}: not valid JSON") from exc


def cells(baseline):
    return {
        (model, case): (cell["pass"], cell["fail"], cell["error"])
        for model, cases in baseline["results"].items()
        for case, cell in cases.items()
    }


def percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap(per_case, rng, resamples):
    """95% interval of the mean of per-case values, resampling cases with replacement."""
    values = list(per_case)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(resamples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def fisher_drop(base_pass, base_n, cand_pass, cand_n):
    """One-sided p-value that the candidate's pass rate is lower than the baseline's."""
    total, passes = base_n + cand_n, base_pass + cand_pass
    return sum(
        math.comb(passes, k) * math.comb(total - passes, base_n - k) / math.comb(total, base_n)
        for k in range(base_pass, min(base_n, passes) + 1)
    )


def mean(values):
    return sum(values) / len(values) if values else 0.0


def compare(base, cand, *, margin=MARGIN, case_alpha=CASE_ALPHA, resamples=RESAMPLES, seed=SEED):
    """Return a structured comparison; raises BaselineError when the runs are incomparable."""
    problems = [
        f"{field} differs (baseline {str(base['identity'][field])[:12]}, "
        f"candidate {str(cand['identity'][field])[:12]})"
        for field in INCOMPATIBLE
        if base["identity"][field] != cand["identity"][field]
    ]
    if problems:
        hint = ""
        if any(p.startswith("grader_sha256") for p in problems):
            hint = (
                "; re-record the baseline from its saved run directory with the current "
                "graders, or re-run it"
            )
        raise BaselineError("runs are not comparable: " + "; ".join(problems) + hint)

    notes = []
    for field in ADVISORY:
        before, after = base["identity"].get(field), cand["identity"].get(field)
        if before == after:
            continue
        if isinstance(before, dict) and isinstance(after, dict):
            keys = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
            notes.append(f"{field} differ ({', '.join(keys)}); results may not be comparable")
        else:
            notes.append(f"{field} differs between the runs")
    left, right = cells(base), cells(cand)
    models = sorted({m for m, _ in left} & {m for m, _ in right})
    cases = sorted({c for _, c in left} & {c for _, c in right})
    notes += [
        f"{kind} {name} exists in only one run and is excluded"
        for kind, a, b in (
            ("model", {m for m, _ in left}, {m for m, _ in right}),
            ("case", {c for _, c in left}, {c for _, c in right}),
        )
        for name in sorted(a ^ b)
    ]
    if not models or not cases:
        raise BaselineError("runs share no model and case")

    paired = {}
    for model in models:
        for case in cases:
            if (model, case) in left and (model, case) in right:
                (bp, bf, _), (cp, cf, _) = left[(model, case)], right[(model, case)]
                if bp + bf and cp + cf:
                    paired[(model, case)] = (bp, bp + bf, cp, cp + cf)
    if not paired:
        raise BaselineError("no (model, case) cell has a valid trial in both runs")

    def delta(cell):
        bp, bn, cp, cn = cell
        return cp / cn - bp / bn

    rng = random.Random(seed)
    per_case = {
        case: mean([delta(paired[(m, case)]) for m in models if (m, case) in paired])
        for case in cases
        if any((m, case) in paired for m in models)
    }
    lo, hi = bootstrap(per_case.values(), rng, resamples)
    overall = {"delta": mean(per_case.values()), "low": lo, "high": hi}

    by_model = {}
    for model in models:
        values = {case: delta(paired[(model, case)]) for case in cases if (model, case) in paired}
        if values:
            mlo, mhi = bootstrap(values.values(), rng, resamples)
            by_model[model] = {"delta": mean(values.values()), "low": mlo, "high": mhi}

    case_rows = []
    for case in per_case:
        selected = [paired[(m, case)] for m in models if (m, case) in paired]
        bp, bn = sum(c[0] for c in selected), sum(c[1] for c in selected)
        cp, cn = sum(c[2] for c in selected), sum(c[3] for c in selected)
        case_rows.append(
            {
                "case": case,
                "baseline": (bp, bn),
                "candidate": (cp, cn),
                "p_drop": fisher_drop(bp, bn, cp, cn),
                "p_gain": fisher_drop(cp, cn, bp, bn),
            }
        )

    cand_errors = sum(e for _, _, e in right.values())
    cand_trials = sum(sum(v) for v in right.values())
    error_rate = cand_errors / cand_trials if cand_trials else 0.0
    lost = [r["case"] for r in case_rows if r["p_drop"] <= case_alpha]
    reasons = []
    if error_rate > MAX_ERROR_RATE:
        reasons.append(f"{error_rate:.0%} of candidate trials errored (limit {MAX_ERROR_RATE:.0%})")
    if lost:
        reasons.append("cases that lost passes: " + ", ".join(lost))
    if overall["high"] < 0:
        verdict = "REGRESSION"
    elif overall["low"] >= -margin and not reasons:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
        if overall["low"] < -margin:
            reasons.insert(
                0, f"lower bound {overall['low']:+.1%} is below the {-margin:+.0%} margin"
            )
    return {
        "verdict": verdict,
        "reasons": reasons,
        "notes": notes,
        "margin": margin,
        "overall": overall,
        "models": by_model,
        "cases": case_rows,
        "case_alpha": case_alpha,
        "baseline": base,
        "candidate": cand,
        "error_rate": error_rate,
    }


def _rate(passes, n):
    return f"{passes}/{n} = {passes / n:.0%}" if n else "n/a"


def render(result):
    base, cand = result["baseline"], result["candidate"]
    overall = result["overall"]
    lines = [
        f"## Baseline comparison: {cand['name']} vs {base['name']}",
        "",
        f"**Verdict: {result['verdict']}**",
        "",
        (
            f"Candidate minus baseline pass rate: {overall['delta']:+.1%} "
            f"(95% interval {overall['low']:+.1%} to {overall['high']:+.1%}, "
            f"non-inferiority margin {-result['margin']:+.0%})."
        ),
    ]
    for reason in result["reasons"]:
        lines.append(f"- {reason}")
    same_skill = base["identity"]["skill_sha256"] == cand["identity"]["skill_sha256"]
    lines += [
        "",
        f"Same skill text in both runs: {'yes (this measures noise)' if same_skill else 'no'}.",
    ]
    for note in result["notes"]:
        lines.append(f"- Note: {note}.")
    lines += [
        "",
        "### By model",
        "",
        "| Model | Delta | 95% interval |",
        "|---|---:|---|",
    ]
    for model, row in result["models"].items():
        lines.append(f"| {model} | {row['delta']:+.1%} | {row['low']:+.1%} to {row['high']:+.1%} |")
    lines += [
        "",
        "### By case (passes pooled over the shared models)",
        "",
        "| Case | Baseline | Candidate | Change |",
        "|---|---:|---:|---|",
    ]
    for row in result["cases"]:
        flag = ""
        if row["p_drop"] <= result["case_alpha"]:
            flag = f"LOST (p={row['p_drop']:.3f})"
        elif row["p_gain"] <= result["case_alpha"]:
            flag = f"gained (p={row['p_gain']:.3f})"
        lines.append(
            f"| {row['case']} | {_rate(*row['baseline'])} | {_rate(*row['candidate'])} | {flag} |"
        )
    lines += [
        "",
        (
            "Intervals resample cases, so they include trial noise. A case is marked only at "
            f"p <= {result['case_alpha']} (one-sided Fisher exact), because many cases are tested."
        ),
    ]
    return "\n".join(lines) + "\n"


def write_new(path, value, *, force):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise BaselineError(f"{path} exists; pass --force to replace it")
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".baseline-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=1, sort_keys=False)
        stream.write("\n")
    os.replace(name, path)


def current_skill_digest():
    from native_core import skill_bundle, skill_digest

    return skill_digest(skill_bundle())


def run_record(args):
    if not NAME.match(args.name):
        raise BaselineError("--name must be 1-64 letters, digits, '.', '_' or '-'")
    baseline = load(str(args.run), refresh=args.refresh)
    baseline["name"], baseline["note"] = args.name, args.note
    output = args.output or BASELINES / f"{args.name}.json"
    write_new(output, baseline, force=args.force)
    counts = [sum(c.values()) for cases in baseline["results"].values() for c in cases.values()]
    errors = sum(c["error"] for cases in baseline["results"].values() for c in cases.values())
    print(
        f"wrote {output} ({len(baseline['results'])} models, {sum(counts)} trials, {errors} errors)"
    )
    if baseline["identity"]["skill_sha256"] != current_skill_digest():
        print("note: the recorded skill text differs from the skill in this checkout")
    return 0


def run_compare(args):
    result = compare(
        load(args.baseline),
        load(args.candidate, refresh=args.refresh),
        margin=args.margin,
        case_alpha=args.case_alpha,
        resamples=args.resamples,
    )
    sys.stdout.write(render(result))
    return 0 if result["verdict"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record", help="summarize a run directory into a baseline file")
    record.add_argument("run", type=Path, help="<model>/<case>-t<N>/report.json run directory")
    record.add_argument("--name", required=True)
    record.add_argument("--note", default="", help="free text, e.g. what was run and why")
    record.add_argument("--output", type=Path, help="default: evals/baselines/NAME.json")
    record.add_argument("--force", action="store_true", help="replace an existing file")
    record.add_argument("--refresh", action="store_true", help="ignore the cached re-grade")
    record.set_defaults(handler=run_record)
    check = sub.add_parser("compare", help="compare a candidate against a baseline")
    check.add_argument("baseline", help="baseline file or name under evals/baselines/")
    check.add_argument("candidate", help="run directory (re-graded) or baseline file")
    check.add_argument("--margin", type=float, default=MARGIN, help="non-inferiority margin")
    check.add_argument("--case-alpha", type=float, default=CASE_ALPHA)
    check.add_argument("--resamples", type=int, default=RESAMPLES)
    check.add_argument("--refresh", action="store_true", help="ignore the cached re-grade")
    check.set_defaults(handler=run_compare)
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except BaselineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
