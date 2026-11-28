"""Offline tests for compact evaluation baselines and paired comparison."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import baseline
import regrade

MODELS = [f"m{i}" for i in range(5)]
CASES = [f"case-{i:02d}" for i in range(22)]
IDENTITY = {
    "skill_sha256": "s" * 64,
    "dataset_sha256": "d" * 64,
    "corpus_source_sha256": "c" * 64,
    "grader_sha256": "g" * 64,
    "image": "sha256:image",
    "limits": {"turns": 16},
}


def make(name, passes=None, *, identity=None, errors=0):
    """A baseline with 3 trials per cell; passes maps (model, case) to a pass count."""
    passes = passes or {}
    results = {}
    for model in MODELS:
        for case in CASES:
            k = passes.get((model, case), 2)
            results.setdefault(model, {})[case] = {
                "pass": k,
                "fail": 3 - k - errors,
                "error": errors,
            }
    return {
        "format": baseline.FORMAT,
        "name": name,
        "note": "",
        "recorded": "2026-01-01T00:00:00Z",
        "identity": {**IDENTITY, **(identity or {})},
        "results": results,
    }


def run(base, cand, **options):
    return baseline.compare(base, cand, resamples=500, **options)


def test_identical_runs_pass():
    result = run(make("a"), make("b"))
    assert result["verdict"] == "PASS"
    assert result["overall"]["delta"] == 0


def test_clear_drop_is_a_regression():
    drop = {(m, c): 0 for m in MODELS for c in CASES}
    result = run(make("a"), make("b", drop))
    assert result["verdict"] == "REGRESSION"
    assert result["overall"]["high"] < 0


def test_small_drop_is_inconclusive_not_a_pass():
    some = {(m, c): 1 for m in MODELS for c in CASES[:8]}
    result = run(make("a"), make("b", some))
    assert result["verdict"] != "PASS"


def test_lost_case_blocks_a_pass():
    lost = {(m, CASES[0]): 0 for m in MODELS}
    base = make("a", {(m, CASES[0]): 3 for m in MODELS})
    result = run(base, make("b", lost))
    assert result["verdict"] != "PASS"
    assert any(CASES[0] in reason for reason in result["reasons"])
    assert "LOST" in baseline.render(result)


def test_errors_cap_the_verdict_and_are_excluded_from_rates():
    result = run(make("a"), make("b", {(m, c): 1 for m in MODELS for c in CASES}, errors=1))
    assert result["error_rate"] > baseline.MAX_ERROR_RATE
    assert result["verdict"] != "PASS"
    cell = result["candidate"]["results"][MODELS[0]][CASES[0]]
    assert cell == {"pass": 1, "fail": 1, "error": 1}
    assert result["cases"][0]["candidate"][1] == len(MODELS) * 2


@pytest.mark.parametrize("field", ["dataset_sha256", "corpus_source_sha256", "grader_sha256"])
def test_changed_identity_is_not_comparable(field):
    with pytest.raises(baseline.BaselineError, match=field):
        run(make("a"), make("b", identity={field: "x" * 64}))


def test_grader_mismatch_explains_how_to_recover():
    with pytest.raises(baseline.BaselineError, match="re-record"):
        run(make("a"), make("b", identity={"grader_sha256": "x" * 64}))


def test_limit_differences_are_reported_not_fatal():
    result = run(make("a"), make("b", identity={"limits": {"turns": 8}}))
    assert any("limits differ (turns)" in note for note in result["notes"])


def test_models_and_cases_in_only_one_run_are_excluded():
    other = make("b")
    del other["results"][MODELS[0]]
    del other["results"][MODELS[1]][CASES[0]]
    result = run(make("a"), other)
    assert f"model {MODELS[0]} exists in only one run" in " ".join(result["notes"])
    assert MODELS[0] not in result["models"]
    first = next(row for row in result["cases"] if row["case"] == CASES[0])
    assert first["baseline"][1] == 3 * (len(MODELS) - 2)


def test_fisher_drop_matches_exact_value():
    assert baseline.fisher_drop(3, 3, 0, 3) == pytest.approx(1 / 20)
    assert baseline.fisher_drop(1, 3, 2, 3) > 0.5


def test_validate_rejects_malformed_files():
    broken = make("a")
    broken["results"][MODELS[0]][CASES[0]]["pass"] = -1
    with pytest.raises(baseline.BaselineError, match="malformed counts"):
        baseline.validate(broken, "x.json")
    del broken["identity"]
    with pytest.raises(baseline.BaselineError, match="malformed"):
        baseline.validate(broken, "x.json")


def test_load_resolves_names_files_and_rejects_unknown(tmp_path):
    (tmp_path / "v1.json").write_text(json.dumps(make("v1")), encoding="utf-8")
    with patch.object(baseline, "BASELINES", tmp_path):
        assert baseline.load("v1")["name"] == "v1"
        with pytest.raises(baseline.BaselineError, match="not a baseline"):
            baseline.load("missing")
    assert baseline.load(str(tmp_path / "v1.json"))["name"] == "v1"
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(baseline.BaselineError, match="not valid JSON"):
        baseline.load(str(tmp_path / "bad.json"))


def regraded(*verdicts):
    trials = [
        {
            "model": "m0",
            "case": "case-00",
            "trial": number,
            "original": "FAIL",
            "regraded": verdict,
            "detail": "",
        }
        for number, verdict in enumerate(verdicts, 1)
    ]
    return {"grader_sha256": "g" * 64, "identity": {**IDENTITY}, "trials": trials}


def test_record_writes_counts_and_refuses_to_overwrite(tmp_path):
    output = tmp_path / "new.json"
    args = baseline.argparse.Namespace(
        run=tmp_path, name="v9", note="why", output=output, force=False, refresh=False
    )
    with (
        patch.object(regrade, "regrade", return_value=regraded("PASS", "FAIL", "ERROR")),
        patch.object(baseline, "current_skill_digest", return_value=IDENTITY["skill_sha256"]),
    ):
        assert baseline.run_record(args) == 0
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert saved["results"]["m0"]["case-00"] == {"pass": 1, "fail": 1, "error": 1}
        assert (saved["name"], saved["note"]) == ("v9", "why")
        with pytest.raises(baseline.BaselineError, match="--force"):
            baseline.run_record(args)
        args.name = "../escape"
        with pytest.raises(baseline.BaselineError, match="--name"):
            baseline.run_record(args)


def trial_report(root, folder, case, trial, **fields):
    directory = root / folder / f"{case}-t{trial}"
    directory.mkdir(parents=True)
    report = {
        "status": "ERROR",
        "case_id": case,
        "trial": trial,
        "model": f"p/{folder}",
        "skill_sha256": "s",
        "dataset_sha256": "d",
        "corpus_source_sha256": "c",
        "image": "i",
        "limits": {},
        **fields,
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_regrade_cache_is_tied_to_the_grader_digest(tmp_path):
    trial_report(tmp_path, "m", "a", 1)
    files = {"graders/x.py": "1"}
    with patch.object(regrade, "grader_files", return_value=files):
        first = regrade.regrade(tmp_path)
        (tmp_path / "m" / "a-t1" / "report.json").unlink()
        assert regrade.regrade(tmp_path) == first
    with (
        patch.object(regrade, "grader_files", return_value={"graders/x.py": "2"}),
        pytest.raises(regrade.RegradeError, match="no trial reports"),
    ):
        regrade.regrade(tmp_path)


def test_regrade_rejects_mixed_runs_and_duplicates(tmp_path):
    trial_report(tmp_path, "m", "a", 1)
    trial_report(tmp_path, "m", "a", 2, skill_sha256="other")
    with (
        patch.object(regrade, "grader_files", return_value={}),
        pytest.raises(regrade.RegradeError, match="skill_sha256"),
    ):
        regrade.regrade(tmp_path)
    duplicate = tmp_path / "dup"
    trial_report(duplicate, "m", "a", 1)
    trial_report(duplicate, "m2", "a", 1, model="p/m")
    with (
        patch.object(regrade, "grader_files", return_value={}),
        pytest.raises(regrade.RegradeError, match="several reports"),
    ):
        regrade.regrade(duplicate)


def test_committed_baselines_are_well_formed():
    files = sorted(baseline.BASELINES.glob("*.json"))
    assert files
    for path in files:
        loaded = baseline.validate(json.loads(path.read_text(encoding="utf-8")), str(path))
        assert loaded["name"] == path.stem
