#!/usr/bin/env python3
"""Freeze the latest verifiable case/trial results for the current skill."""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from behavior_cases import DATASET_SHA256
from corpus import verify_corpus
from native_core import skill_bundle, skill_digest
from results import (
    MAX_REPORT_BYTES,
    ReportError,
    _adapter_digest,
    _csv_value,
    discover_reports,
    read_report,
    rows_from_report,
)
from run_behavior import adapter_hashes

FIELDS = (
    "case_id",
    "description",
    "lane",
    "trial",
    "status",
    "recorded_status",
    "result_basis",
    "availability",
    "evaluator_match",
    "model",
    "report_modified_utc",
    "details",
    "regrade_notes",
    "regrade_provenance",
    "tokens",
    "cost_usd",
    "release_status",
    "report_path",
    "regrade_path",
    "imported_status",
    "imported_modified_utc",
    "imported_report_path",
    "activation_status",
    "expected_activation",
    "observed_activation",
    "snapshot_utc",
    "skill_sha256",
    "dataset_sha256",
    "corpus_source_sha256",
    "grader_sha256",
    "adapter_sha256",
    "recorded_grader_sha256",
    "recorded_adapter_sha256",
)


def _read_stable(path):
    if path.is_symlink():
        raise ReportError(f"report must not be a symlink: {path}")
    before = path.stat()
    report = read_report(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ReportError(f"report changed while reading: {path}")
    return report, after.st_mtime_ns


def _regrade_files(root):
    found = []

    def fail(error):
        raise ReportError(f"cannot scan regrades: {error}") from error

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        dirs[:] = sorted(name for name in dirs if not (Path(directory) / name).is_symlink())
        for name in sorted(names):
            if name.startswith("regraded-") and name.endswith(".csv"):
                path = Path(directory) / name
                if path.is_symlink() or not path.is_file():
                    raise ReportError(f"regrade must be a regular, non-symlink file: {path}")
                found.append(path)
    return found


def _read_regrades(root, selected, current_adapters, skill_hash, cutoff_ns):
    expected_headers = {
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
    }
    main_grader = "graders/behavior_grader.py"
    chosen = {}
    for path in _regrade_files(root):
        before = path.stat()
        if cutoff_ns is not None and before.st_mtime_ns > cutoff_ns:
            continue
        if before.st_size > MAX_REPORT_BYTES:
            raise ReportError(f"regrade exceeds size limit: {path}")
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or set(reader.fieldnames) != expected_headers:
                    raise ReportError(f"invalid regrade columns: {path}")
                if len(reader.fieldnames) != len(expected_headers):
                    raise ReportError(f"duplicate regrade columns: {path}")
                records = list(reader)
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ReportError(f"cannot read regrade: {path}") from exc
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ReportError(f"regrade changed while reading: {path}")
        for record in records:
            if None in record or any(value is None for value in record.values()):
                raise ReportError(f"invalid regrade row: {path}")
            try:
                trial = int(record["trial"])
            except ValueError as exc:
                raise ReportError(f"invalid regrade trial: {path}") from exc
            key = ("behavior", record["case_id"], trial)
            source = selected.get(key)
            if record["regraded_status"] not in {"PASS", "FAIL"}:
                raise ReportError(f"invalid regrade status: {path}")
            if source is None:
                continue
            row, report, report_path, modified = source
            saved_adapters = report.get("adapter_hashes")
            if (
                not isinstance(saved_adapters, dict)
                or set(saved_adapters) != set(current_adapters)
                or any(
                    saved_adapters[name] != current_adapters[name]
                    for name in current_adapters
                    if name != main_grader
                )
                or record["original_grader_sha256"] != report.get("grader_sha256")
                or record["current_grader_sha256"] != current_adapters[main_grader]
                or record["skill_sha256"] != skill_hash
                or record["saved_status"] != row["status"]
                or Path(record["source_report"]).resolve() != report_path.resolve()
                or modified > after.st_mtime_ns
            ):
                continue
            rank = (after.st_mtime_ns, str(path))
            if key not in chosen or rank > chosen[key][0]:
                chosen[key] = (rank, record, path)
    return chosen


def build_rows(
    root,
    dataset,
    unsupported,
    skill_hash,
    dataset_hash,
    corpus_hash,
    adapters,
    now,
    *,
    cutoff_ns=None,
):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ReportError(f"report directory must exist and not be a symlink: {root}")
    cases = {(case["lane"], case["id"]): case for case in dataset["cases"]}
    selected = {}
    imported = {}
    for path in discover_reports(root):
        if path.is_symlink():
            raise ReportError(f"report must not be a symlink: {path}")
        if cutoff_ns is not None and path.stat().st_mtime_ns > cutoff_ns:
            continue
        report, modified = _read_stable(path)
        if cutoff_ns is not None and modified > cutoff_ns:
            raise ReportError(f"report changed after snapshot start: {path}")
        run = report.get("run") or {}
        if not isinstance(run, dict):
            raise ReportError(f"invalid run metadata: {path}")
        if (
            report.get("skill_sha256", run.get("skill_sha256")) != skill_hash
            or report.get("dataset_sha256") != dataset_hash
            or report.get("corpus_source_sha256") != corpus_hash
            or run.get("condition") == "no-skill"
        ):
            continue
        seen = set()
        for row in rows_from_report(report, path, root):
            kind = row.get("report_type")
            if kind not in {"controlled-behavior", "native-capture", "imported"}:
                continue
            key = (row.get("lane"), row.get("case_id"), row.get("trial"))
            identity = (kind, *key)
            if identity in seen:
                raise ReportError(f"duplicate case and trial in report: {path}")
            seen.add(identity)
            if (
                key[:2] not in cases
                or type(key[2]) is not int
                or not 1 <= key[2] <= dataset["trial_count"]
                or row.get("status") in {"NOT RUN", "MISSING"}
            ):
                continue
            if kind == "controlled-behavior" and key[0] != "behavior":
                continue
            if kind == "native-capture" and key[0] != "native":
                continue
            target = imported if kind == "imported" else selected
            candidate = (row, report, path, modified)
            previous = target.get(key)
            if previous is None or (modified, str(path)) > (previous[3], str(previous[2])):
                target[key] = candidate
    regrades = _read_regrades(root, selected, adapters, skill_hash, cutoff_ns)
    current_adapter = _adapter_digest({"adapter_hashes": adapters})
    rows = []
    for case in dataset["cases"]:
        lane, case_id = case["lane"], case["id"]
        for trial in range(1, dataset["trial_count"] + 1):
            key = (lane, case_id, trial)
            blocked = lane == "behavior" and case_id in unsupported
            row = {
                "case_id": case_id,
                "description": " / ".join(case.get("prompts", [])),
                "lane": lane,
                "trial": trial,
                "status": "BLOCKED" if blocked else "NOT RUN",
                "availability": "BLOCKED"
                if blocked
                else "UNVERIFIED"
                if lane == "native"
                else "SUPPORTED",
                "details": unsupported.get(case_id, "") if blocked else "",
                "snapshot_utc": now,
                "skill_sha256": skill_hash,
                "dataset_sha256": dataset_hash,
                "corpus_source_sha256": corpus_hash,
            }
            if lane == "native":
                row["expected_activation"] = case.get("should_activate", "")
            if key in selected and not blocked:
                saved, report, _, _ = selected[key]
                row.update(
                    {
                        name: saved.get(name, "")
                        for name in (
                            "model",
                            "report_modified_utc",
                            "details",
                            "tokens",
                            "cost_usd",
                            "release_status",
                            "report_path",
                            "activation_status",
                            "observed_activation",
                            "grader_sha256",
                            "adapter_sha256",
                        )
                    }
                )
                row["recorded_status"] = saved["status"]
                if "expected_activation" in saved and saved["expected_activation"] != "":
                    row["expected_activation"] = saved["expected_activation"]
                row["recorded_grader_sha256"] = saved.get("grader_sha256", "")
                row["recorded_adapter_sha256"] = saved.get("adapter_sha256", "")
                row["status"] = saved["status"]
                row["result_basis"] = "SAVED TRIAL"
                row["evaluator_match"] = (
                    "MATCH"
                    if saved.get("adapter_sha256") == current_adapter
                    else "DIFFERENT"
                    if saved.get("adapter_sha256")
                    else "UNKNOWN"
                )
                if key in regrades:
                    _, regrade, regrade_path = regrades[key]
                    row["status"] = regrade["regraded_status"]
                    row["result_basis"] = "SAVED REGRADING"
                    row["evaluator_match"] = "REGRADED"
                    row["grader_sha256"] = adapters["graders/behavior_grader.py"]
                    row["adapter_sha256"] = current_adapter
                    row["regrade_provenance"] = "PATH AND VERSION ONLY; SOURCE NOT HASH-BOUND"
                    row["regrade_notes"] = " | ".join(
                        part for part in (regrade["observations"], regrade["failed_checks"]) if part
                    )
                    row["regrade_path"] = str(regrade_path.relative_to(root))
            if key in imported:
                evidence = imported[key][0]
                row["imported_status"] = evidence["status"]
                row["imported_modified_utc"] = evidence["report_modified_utc"]
                row["imported_report_path"] = evidence["report_path"]
            rows.append(row)
    return rows


def export(root, output):
    dataset, _, unsupported, _, corpus_hash = verify_corpus()
    skill_hash = skill_digest(skill_bundle())
    adapters = adapter_hashes()
    cutoff_ns = time.time_ns()
    now = datetime.fromtimestamp(cutoff_ns / 1_000_000_000, UTC).isoformat()
    rows = build_rows(
        root,
        dataset,
        unsupported,
        skill_hash,
        DATASET_SHA256,
        corpus_hash,
        adapters,
        now,
        cutoff_ns=cutoff_ns,
    )
    current_dataset, _, current_unsupported, _, current_corpus_hash = verify_corpus()
    if (
        skill_digest(skill_bundle()) != skill_hash
        or current_dataset != dataset
        or current_unsupported != unsupported
        or current_corpus_hash != corpus_hash
        or adapter_hashes() != adapters
    ):
        raise ReportError("skill, corpus, or evaluators changed while building the snapshot")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_value(row.get(name, "")) for name in FIELDS})
    output = Path(output)
    if not output.parent.is_dir():
        raise ReportError(f"output directory does not exist: {output.parent}")
    try:
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(buffer.getvalue())
    except OSError as exc:
        raise ReportError(f"cannot create output {output}: {exc}") from exc
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval-runs"))
    parser.add_argument("--output", type=Path, help="new CSV file; never overwritten")
    args = parser.parse_args(argv)
    output = (
        args.output
        or args.root / f"snapshot-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.csv"
    )
    try:
        rows = export(args.root, output)
    except (OSError, ValueError, ReportError) as exc:
        print(f"snapshot error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} case/trial rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
