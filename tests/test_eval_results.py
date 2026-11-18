"""Offline tests for tabular exports of saved evaluation reports."""

import csv
import hashlib
import json
import os
import stat

import pytest

from evals import results


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def controlled(case, trial, status, *, tokens=100, cost=None):
    return {
        "status": status,
        "case_id": case,
        "trial": trial,
        "model": "provider/model",
        "skill_sha256": "skill",
        "dataset_sha256": "dataset",
        "corpus_source_sha256": "corpus",
        "grader_sha256": "grader",
        "adapter_hashes": {"b.py": "b", "a.py": "a"},
        "release": "BLOCKED",
        "score_detail": "passed" if status == "PASS" else "incorrect answer",
        "problems": [] if status == "PASS" else ["deterministic behavior checks failed"],
        "usage": {"provider/model": {"tokens": tokens, "cost_usd": cost}},
        "cost_control": {
            "pricing": "UNKNOWN" if cost is None else "CONFIGURED",
            "budget": "DISABLED",
        },
        "limits": {"tokens": 15000, "budget_usd": None},
        "coverage": [{"case_id": case, "trials": [{"trial": 1, "result": "NOT RUN"}]}],
    }


def test_controlled_runs_export_one_row_each_with_unknown_cost(tmp_path):
    root = tmp_path / "runs"
    first_report = controlled("case-a", 1, "PASS")
    first_report["observations"] = ["empty source placeholder removed; answer schema differs"]
    first = save(root / "first/report.json", first_report)
    save(root / "second/report.json", controlled("case-a", 2, "FAIL", tokens=250, cost=0.25))
    output = root / "comparison.csv"
    reports, count = results.export(root, [first], output)
    rows = read_csv(output)
    assert (reports, count, len(rows)) == (2, 2, 2)
    by_run = {row["run"]: row for row in rows}
    assert by_run["first"]["status"] == "PASS"
    assert by_run["first"]["tokens"] == "100"
    assert by_run["first"]["cost_usd"] == ""
    assert by_run["first"]["token_limit"] == "15000"
    assert by_run["first"]["corpus_source_sha256"] == "corpus"
    expected_adapter = hashlib.sha256(b'{"a.py":"a","b.py":"b"}').hexdigest()
    assert by_run["first"]["adapter_sha256"] == expected_adapter
    assert by_run["first"]["pricing_status"] == "UNKNOWN"
    assert by_run["first"]["budget_status"] == "DISABLED"
    assert by_run["first"]["report_status"] == "PASS"
    assert by_run["first"]["release_status"] == "BLOCKED"
    assert "answer schema differs" in by_run["first"]["details"]
    assert by_run["second"]["status"] == "FAIL"
    assert by_run["second"]["cost_usd"] == "0.25"
    assert "incorrect answer" in by_run["second"]["details"]
    if os.name != "nt":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_imported_results_keep_missing_distinct_from_failed(tmp_path):
    root = tmp_path / "runs"
    save(
        root / "import-report.json",
        {
            "status": "BLOCKED",
            "evidence_kind": "imported assertions",
            "lane": "all",
            "gates": {"release": "BLOCKED"},
            "run": {
                "run_id": "import-1",
                "condition": "candidate",
                "model": {"id": "test/model", "configuration": {"mode": "synthetic"}},
                "host": {"name": "test-host", "version": "1"},
            },
            "usage": {"tokens": None, "cost": None, "reason": "NOT RUN"},
            "results": [
                {
                    "case_id": "a",
                    "trial": 1,
                    "lane": "behavior",
                    "status": "SCORED",
                    "success": True,
                },
                {
                    "case_id": "a",
                    "trial": 2,
                    "lane": "behavior",
                    "status": "SCORED",
                    "success": False,
                    "checks": [{"pass": False, "path": "/answer/facts", "op": "equals"}],
                },
                {
                    "case_id": "a",
                    "trial": 3,
                    "lane": "behavior",
                    "status": "MISSING",
                    "success": False,
                },
                {
                    "case_id": "b",
                    "trial": 1,
                    "lane": "behavior",
                    "status": "ERROR",
                    "error": "bad evidence",
                },
            ],
        },
    )
    output = root / "comparison.csv"
    assert results.export(root, [], output) == (1, 4)
    rows = read_csv(output)
    assert [row["status"] for row in rows] == ["PASS", "FAIL", "MISSING", "ERROR"]
    assert all(row["report_type"] == "imported" for row in rows)
    assert all(
        row["report_status"] == "BLOCKED" and row["release_status"] == "BLOCKED" for row in rows
    )
    assert all(row["tokens"] == "" and row["cost_usd"] == "" for row in rows)
    assert all(row["run"] == "import-1" and row["host"] == "test-host/1" for row in rows)
    assert rows[-1]["details"] == "bad evidence"
    assert rows[1]["details"] == "failed check: /answer/facts (equals)"


def test_native_capture_is_not_reported_as_a_pass(tmp_path):
    root = tmp_path / "runs"
    save(
        root / "native/report.json",
        {
            "status": "BLOCKED",
            "native_support": "BLOCKED",
            "activation": "UNVERIFIED",
            "model": "copilot-model",
            "results": [
                {
                    "case_id": "native-positive-02",
                    "trial": 1,
                    "status": "BLOCKED",
                    "expected_activation": True,
                    "sessions": [{"activation": {"status": "UNVERIFIED"}}],
                    "problems": ["host skill-load event mapping is not validated"],
                },
                {
                    "case_id": "native-negative-01",
                    "trial": 1,
                    "status": "NOT RUN",
                    "expected_activation": False,
                    "sessions": [],
                    "problems": ["run timeout"],
                },
            ],
        },
    )
    output = root / "comparison.csv"
    assert results.export(root, [], output) == (1, 2)
    rows = {row["case_id"]: row for row in read_csv(output)}
    assert rows["native-positive-02"]["status"] == "BLOCKED"
    assert rows["native-positive-02"]["activation_status"] == "UNVERIFIED"
    assert rows["native-positive-02"]["expected_activation"] == "True"
    assert rows["native-positive-02"]["observed_activation"] == ""
    assert rows["native-negative-01"]["status"] == "NOT RUN"


def test_verified_native_activation_does_not_certify_release(tmp_path):
    root = tmp_path / "runs"
    save(
        root / "native/report.json",
        {
            "status": "PASS",
            "native_support": "BLOCKED",
            "release": "BLOCKED",
            "activation": "VERIFIED",
            "results": [
                {
                    "case_id": "native-positive-02",
                    "trial": 1,
                    "status": "PASS",
                    "expected_activation": True,
                    "observed_activation": True,
                    "sessions": [{"activation": {"status": "VERIFIED"}}],
                }
            ],
        },
    )
    output = root / "comparison.csv"
    results.export(root, [], output)
    row = read_csv(output)[0]
    assert row["status"] == "PASS"
    assert row["activation_status"] == "VERIFIED"
    assert row["observed_activation"] == "True"
    assert row["release_status"] == "BLOCKED"


def test_smoke_report_with_case_metadata_stays_a_distinct_lane(tmp_path):
    root = tmp_path / "runs"
    save(
        root / "smoke/report.json",
        {
            "status": "PASS",
            "lane": "controlled Inspect smoke; explicitly loaded skill",
            "case_id": "scope-search",
            "trial": 1,
            "release": "BLOCKED",
            "usage": {"model": {"tokens": 100, "cost_usd": None}},
        },
    )
    output = root / "comparison.csv"
    results.export(root, [], output)
    row = read_csv(output)[0]
    assert row["report_type"] == "controlled-smoke"
    assert row["case_id"] == "scope-search" and row["trial"] == "1"
    assert row["release_status"] == "BLOCKED"


def test_unrun_and_error_reports_are_visible_without_inventing_metadata(tmp_path):
    root = tmp_path / "runs"
    save(
        root / "offline-report.json",
        {"status": "NOT RUN", "evidence_kind": "imported assertions", "lane": "all", "results": []},
    )
    save(
        root / "failed/report.json",
        {"status": "ERROR", "case_id": "a", "trial": 1, "reason": "TimeoutError"},
    )
    save(root / "interrupted/report.json", {"status": "INTERRUPTED", "release": "BLOCKED"})
    output = root / "comparison.csv"
    assert results.export(root, [], output) == (3, 3)
    rows = {row["status"]: row for row in read_csv(output)}
    assert rows["NOT RUN"]["case_id"] == ""
    assert rows["ERROR"]["case_id"] == "a"
    assert rows["ERROR"]["model"] == "" and rows["ERROR"]["tokens"] == ""
    assert rows["ERROR"]["details"] == "TimeoutError"
    assert rows["INTERRUPTED"]["case_id"] == ""


def test_neutralizes_spreadsheet_formulas_and_omits_raw_metadata(tmp_path):
    root = tmp_path / "runs"
    report = controlled("=malicious", 1, "FAIL")
    report["score_detail"] = '  =HYPERLINK("https://example.invalid")'
    report["azure_endpoint"] = "not-for-export"
    save(root / "trial/report.json", report)
    output = root / "comparison.csv"
    results.export(root, [], output)
    row = read_csv(output)[0]
    assert row["case_id"] == "'=malicious"
    assert row["details"].startswith("'  =HYPERLINK")
    assert "not-for-export" not in output.read_text()


def test_invalid_reports_fail_without_creating_output(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "report.json").write_text('{"status":"PASS","status":"FAIL"}')
    output = root / "comparison.csv"
    with pytest.raises(results.ReportError, match="duplicate JSON key"):
        results.export(root, [], output)
    assert not output.exists()
    (root / "report.json").write_text('{"status":"PASS"}')
    with pytest.raises(results.ReportError, match="unrecognized"):
        results.export(root, [], output)
    assert not output.exists()


def test_symlinked_reports_are_rejected(tmp_path):
    root = tmp_path / "runs"
    actual = save(tmp_path / "actual.json", controlled("a", 1, "PASS"))
    link = root / "trial/report.json"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(actual)
    except OSError:
        pytest.skip("symlinks are unavailable")
    output = root / "comparison.csv"
    with pytest.raises(results.ReportError, match="symlink"):
        results.export(root, [], output)
    assert not output.exists()


def test_cli_refuses_to_overwrite_and_uses_extra_reports(tmp_path, capsys):
    root = tmp_path / "runs"
    save(root / "trial/report.json", controlled("a", 1, "PASS"))
    extra = save(tmp_path / "custom.json", controlled("b", 1, "FAIL"))
    output = root / "comparison.csv"
    assert results.main(["--root", str(root), "--report", str(extra), "--output", str(output)]) == 0
    assert len(read_csv(output)) == 2
    original = output.read_bytes()
    assert results.main(["--root", str(root), "--output", str(output)]) == 2
    assert output.read_bytes() == original
    assert "cannot create output" in capsys.readouterr().err
