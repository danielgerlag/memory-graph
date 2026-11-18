"""Tests for a point-in-time, current-skill case/trial snapshot."""

import csv
import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import snapshot

DATASET = {
    "trial_count": 2,
    "cases": [
        {"id": "a", "lane": "behavior", "prompts": ["Find a fact"]},
        {"id": "blocked", "lane": "behavior", "prompts": ["Review a proposal"]},
        {"id": "native", "lane": "native", "prompts": ["Use the skill"], "should_activate": False},
    ],
}
ADAPTERS = {"graders/behavior_grader.py": "current", "capture.py": "same"}


def save(path, report, *, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    os.utime(path, ns=(mtime, mtime))
    return path


def controlled(case, trial, status, *, skill="skill", dataset="dataset", corpus="corpus"):
    return {
        "status": status,
        "case_id": case,
        "trial": trial,
        "model": "provider/model",
        "skill_sha256": skill,
        "dataset_sha256": dataset,
        "corpus_source_sha256": corpus,
        "grader_sha256": "current",
        "adapter_hashes": ADAPTERS,
        "release": "BLOCKED",
        "score_detail": "passed" if status == "PASS" else "failed",
    }


def build(root, *, cutoff_ns=None):
    return snapshot.build_rows(
        root,
        DATASET,
        {"blocked": "requires review"},
        "skill",
        "dataset",
        "corpus",
        ADAPTERS,
        "now",
        cutoff_ns=cutoff_ns,
    )


def test_snapshot_selects_latest_matching_real_trial_and_separates_other_lanes(tmp_path):
    root = tmp_path / "runs"
    save(root / "old/report.json", controlled("a", 1, "FAIL"), mtime=100)
    save(root / "new/report.json", controlled("a", 1, "PASS"), mtime=200)
    save(root / "other-skill/report.json", controlled("a", 1, "FAIL", skill="other"), mtime=300)
    save(root / "holdout/report.json", controlled("a", 1, "FAIL", dataset="other"), mtime=400)
    save(root / "other-corpus/report.json", controlled("a", 1, "FAIL", corpus="other"), mtime=500)
    save(root / "unrun/report.json", controlled("a", 1, "NOT RUN"), mtime=600)
    imported = {
        "status": "BLOCKED",
        "evidence_kind": "imported assertions",
        "dataset_sha256": "dataset",
        "corpus_source_sha256": "corpus",
        "run": {"skill_sha256": "skill", "condition": "with-skill"},
        "results": [
            {"lane": "behavior", "case_id": "a", "trial": 1, "status": "SCORED", "success": False}
        ],
    }
    save(root / "import-report.json", imported, mtime=700)
    control = json.loads(json.dumps(imported))
    control["run"]["condition"] = "no-skill"
    control["results"][0]["success"] = True
    save(root / "control-report.json", control, mtime=800)
    rows = {(row["lane"], row["case_id"], row["trial"]): row for row in build(root)}
    assert len(rows) == 6
    result = rows[("behavior", "a", 1)]
    assert result["status"] == result["recorded_status"] == "PASS"
    assert result["report_path"] == "new/report.json"
    assert result["evaluator_match"] == "MATCH"
    assert result["imported_status"] == "FAIL"
    assert rows[("behavior", "a", 2)]["status"] == "NOT RUN"
    assert rows[("behavior", "blocked", 1)]["status"] == "BLOCKED"
    assert rows[("behavior", "blocked", 1)]["details"] == "requires review"
    assert rows[("native", "native", 1)]["status"] == "NOT RUN"
    assert rows[("native", "native", 1)]["availability"] == "UNVERIFIED"
    assert rows[("native", "native", 1)]["expected_activation"] is False


def test_latest_error_is_not_hidden_by_an_older_pass(tmp_path):
    root = tmp_path / "runs"
    save(root / "pass/report.json", controlled("a", 1, "PASS"), mtime=100)
    save(root / "error/report.json", controlled("a", 1, "ERROR"), mtime=200)
    row = next(item for item in build(root) if item["case_id"] == "a")
    assert row["status"] == "ERROR"
    assert row["report_path"] == "error/report.json"


def test_native_capture_remains_unverified(tmp_path):
    root = tmp_path / "runs"
    save(
        root / "native/report.json",
        {
            "status": "BLOCKED",
            "native_support": "BLOCKED",
            "activation": "UNVERIFIED",
            "skill_sha256": "skill",
            "dataset_sha256": "dataset",
            "corpus_source_sha256": "corpus",
            "results": [
                {
                    "case_id": "native",
                    "trial": 1,
                    "status": "BLOCKED",
                    "sessions": [{"activation": {"status": "UNVERIFIED"}}],
                }
            ],
        },
        mtime=100,
    )
    row = next(item for item in build(root) if item["lane"] == "native" and item["trial"] == 1)
    assert row["status"] == "BLOCKED"
    assert row["availability"] == "UNVERIFIED"
    assert row["activation_status"] == "UNVERIFIED"


def test_native_activation_result_does_not_certify_host_support(tmp_path):
    root = tmp_path / "runs"
    save(
        root / "native/report.json",
        {
            "status": "PASS",
            "native_support": "BLOCKED",
            "release": "BLOCKED",
            "activation": "VERIFIED",
            "skill_sha256": "skill",
            "dataset_sha256": "dataset",
            "corpus_source_sha256": "corpus",
            "results": [
                {
                    "case_id": "native",
                    "trial": 1,
                    "status": "PASS",
                    "observed_activation": False,
                    "sessions": [{"activation": {"status": "VERIFIED"}}],
                }
            ],
        },
        mtime=100,
    )
    row = next(item for item in build(root) if item["lane"] == "native" and item["trial"] == 1)
    assert row["status"] == "PASS"
    assert row["availability"] == "UNVERIFIED"
    assert row["activation_status"] == "VERIFIED"
    assert row["expected_activation"] is False
    assert row["observed_activation"] is False
    assert row["release_status"] == "BLOCKED"


def test_snapshot_excludes_reports_written_after_the_cutoff(tmp_path):
    root = tmp_path / "runs"
    save(root / "before/report.json", controlled("a", 1, "FAIL"), mtime=100)
    save(root / "after/report.json", controlled("a", 1, "PASS"), mtime=300)
    row = next(item for item in build(root, cutoff_ns=200) if item["case_id"] == "a")
    assert row["status"] == "FAIL" and row["report_path"] == "before/report.json"


def test_matching_saved_regrade_preserves_original_status_and_provenance(tmp_path):
    root = tmp_path / "runs"
    report = controlled("a", 1, "FAIL")
    report["grader_sha256"] = "old"
    report["adapter_hashes"] = {"graders/behavior_grader.py": "old", "capture.py": "same"}
    source = save(root / "trial/report.json", report, mtime=100)
    fields = [
        "case_id",
        "trial",
        "saved_status",
        "regraded_status",
        "observations",
        "failed_checks",
        "source_report",
        "original_grader_sha256",
        "current_grader_sha256",
        "skill_sha256",
    ]
    path = root / "regraded-test.csv"
    record = dict.fromkeys(fields, "")
    record.update(
        case_id="a",
        trial="1",
        saved_status="FAIL",
        regraded_status="PASS",
        observations="schema warning",
        source_report=str(source),
        original_grader_sha256="old",
        current_grader_sha256="current",
        skill_sha256="skill",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(record)
    os.utime(path, ns=(200, 200))
    row = next(item for item in build(root) if item["case_id"] == "a" and item["trial"] == 1)
    assert row["status"] == "PASS"
    assert row["recorded_status"] == "FAIL"
    assert row["result_basis"] == "SAVED REGRADING"
    assert row["evaluator_match"] == "REGRADED"
    assert row["grader_sha256"] == "current" and row["recorded_grader_sha256"] == "old"
    assert "schema warning" in row["regrade_notes"]
    assert "NOT HASH-BOUND" in row["regrade_provenance"]

    row = next(
        item for item in build(root, cutoff_ns=150) if item["case_id"] == "a" and item["trial"] == 1
    )
    assert row["status"] == "FAIL" and row["result_basis"] == "SAVED TRIAL"

    os.utime(source, ns=(300, 300))
    row = next(item for item in build(root) if item["case_id"] == "a" and item["trial"] == 1)
    assert row["status"] == "FAIL"
    assert row["result_basis"] == "SAVED TRIAL"
    assert row["evaluator_match"] == "DIFFERENT"


def test_export_is_private_immutable_and_spreadsheet_safe(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    dataset = {
        "trial_count": 1,
        "cases": [{"id": "=case", "lane": "behavior", "prompts": ["=formula"]}],
    }
    output = root / "snapshot.csv"
    with (
        patch.object(snapshot, "verify_corpus", return_value=(dataset, {}, {}, {}, "corpus")),
        patch.object(snapshot, "skill_bundle", return_value={}),
        patch.object(snapshot, "skill_digest", return_value="skill"),
        patch.object(snapshot, "adapter_hashes", return_value=ADAPTERS),
        patch.object(snapshot, "DATASET_SHA256", "dataset"),
    ):
        rows = snapshot.export(root, output)
        assert len(rows) == 1 and rows[0]["status"] == "NOT RUN"
        with output.open(encoding="utf-8", newline="") as file:
            row = next(csv.DictReader(file))
        assert row["case_id"] == "'=case" and row["description"] == "'=formula"
        with pytest.raises(snapshot.ReportError, match="cannot create output"):
            snapshot.export(root, output)
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_invalid_or_symlinked_reports_do_not_create_output(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "report.json").write_text('{"status":"PASS","status":"FAIL"}')
    with pytest.raises(snapshot.ReportError, match="duplicate JSON key"):
        build(root)
    (root / "report.json").unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    try:
        (root / "report.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(snapshot.ReportError, match="symlink"):
        build(root)
