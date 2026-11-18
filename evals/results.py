#!/usr/bin/env python3
"""Export saved evaluation reports as one CSV row per case and trial."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

FIELDS = (
    "run",
    "report_type",
    "lane",
    "case_id",
    "trial",
    "status",
    "report_status",
    "release_status",
    "model",
    "report_modified_utc",
    "tokens",
    "cost_usd",
    "pricing_status",
    "budget_usd",
    "budget_status",
    "token_limit",
    "expected_activation",
    "observed_activation",
    "activation_status",
    "condition",
    "host",
    "source_revision",
    "image",
    "skill_sha256",
    "dataset_sha256",
    "corpus_source_sha256",
    "grader_sha256",
    "adapter_sha256",
    "details",
    "report_path",
)
MAX_REPORT_BYTES = 16 * 1024 * 1024


class ReportError(ValueError):
    """A saved report cannot be safely or accurately represented."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReportError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_report(path):
    if path.is_symlink() or not path.is_file():
        raise ReportError(f"report must be a regular, non-symlink file: {path}")
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_REPORT_BYTES + 1)
        if len(raw) > MAX_REPORT_BYTES:
            raise ReportError(f"report exceeds size limit: {path}")
        report = json.loads(
            raw,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ReportError(f"invalid {value}")),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReportError(f"cannot read report {path}: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(report.get("status"), str):
        raise ReportError(f"invalid report envelope: {path}")
    return report


def discover_reports(root):
    if root.is_symlink() or not root.is_dir():
        raise ReportError(f"report directory must exist and not be a symlink: {root}")
    found = []

    def fail(error):
        raise ReportError(f"cannot scan reports: {error}") from error

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        dirs[:] = sorted(name for name in dirs if not (Path(directory) / name).is_symlink())
        for name in sorted(names):
            if name == "report.json" or name.endswith("-report.json"):
                found.append(Path(directory) / name)
    return found


def _text(value, label):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ReportError(f"{label} must be a string")
    return value


def _number(value, label):
    if value is None:
        return ""
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ReportError(f"{label} must be a nonnegative finite number")
    return value


def _usage(report):
    usage = report.get("usage")
    if usage is None:
        return "", ""
    if not isinstance(usage, dict):
        raise ReportError("usage must be an object")
    # Imported reports explicitly state that usage is not integrated.
    if set(usage) == {"tokens", "cost", "reason"}:
        return _number(usage["tokens"], "tokens"), _number(usage["cost"], "cost")
    tokens = []
    costs = []
    for value in usage.values():
        if not isinstance(value, dict):
            raise ReportError("model usage must be an object")
        tokens.append(_number(value.get("tokens"), "tokens"))
        costs.append(_number(value.get("cost_usd"), "cost_usd"))
    return (
        _number(sum(tokens), "total tokens")
        if tokens and all(value != "" for value in tokens)
        else "",
        _number(sum(costs), "total cost") if costs and all(value != "" for value in costs) else "",
    )


def _adapter_digest(report):
    hashes = report.get("adapter_hashes")
    if hashes is None or hashes == {}:
        return ""
    if not isinstance(hashes, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in hashes.items()
    ):
        raise ReportError("adapter hashes must be an object of strings")
    canonical = json.dumps(hashes, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _details(*values):
    result = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value and value not in result:
                result.append(value)
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            for item in value:
                if item and item not in result:
                    result.append(item)
        else:
            raise ReportError("report details must be strings")
    return "; ".join(result)


def _boolean(value, label):
    if value is None:
        return ""
    if type(value) is not bool:
        raise ReportError(f"{label} must be a boolean")
    return value


def _failed_checks(result):
    checks = result.get("checks", [])
    if not isinstance(checks, list):
        raise ReportError("checks must be a list")
    failures = []
    for check in checks:
        if not isinstance(check, dict) or type(check.get("pass")) is not bool:
            raise ReportError("invalid scored check")
        if check["pass"] is False:
            path = _text(check.get("path"), "check path")
            operation = _text(check.get("op"), "check operation")
            failures.append(f"failed check: {path}" + (f" ({operation})" if operation else ""))
    return failures


def rows_from_report(report, path, root):
    path = Path(path)
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(path.resolve())
    run = report.get("run") or {}
    if not isinstance(run, dict):
        raise ReportError(f"invalid run metadata: {path}")
    model = report.get("model", run.get("model"))
    if isinstance(model, dict):
        model = model.get("id")
    host = run.get("host")
    if isinstance(host, dict):
        host = "/".join(str(host[key]) for key in ("name", "version") if host.get(key))
    if host is None and report.get("host_version_requested"):
        host = (
            "Copilot CLI (requested "
            + _text(report["host_version_requested"], "host version")
            + ")"
        )
    limits = report.get("limits", run.get("limits")) or {}
    if not isinstance(limits, dict):
        raise ReportError(f"invalid limits: {path}")
    tokens, cost = _usage(report)
    control = report.get("cost_control") or {}
    if not isinstance(control, dict):
        raise ReportError(f"invalid cost control: {path}")
    gates = report.get("gates") or {}
    if not isinstance(gates, dict):
        raise ReportError(f"invalid gates: {path}")
    base = {
        "run": _text(run.get("run_id"), "run_id")
        or (str(Path(relative).parent) if path.name == "report.json" else relative),
        "model": _text(model, "model"),
        "report_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        "tokens": tokens,
        "cost_usd": cost,
        "pricing_status": _text(control.get("pricing"), "pricing status"),
        "budget_usd": _number(limits.get("budget_usd"), "budget_usd"),
        "budget_status": _text(control.get("budget"), "budget status"),
        "token_limit": _number(limits.get("tokens"), "token_limit"),
        "condition": _text(run.get("condition"), "condition"),
        "host": _text(host, "host"),
        "source_revision": _text(run.get("source_revision"), "source_revision"),
        "image": _text(report.get("image", run.get("sandbox_image")), "image"),
        "report_status": report["status"],
        "release_status": _text(report.get("release", gates.get("release")), "release status"),
        "skill_sha256": _text(report.get("skill_sha256", run.get("skill_sha256")), "skill_sha256"),
        "dataset_sha256": _text(report.get("dataset_sha256"), "dataset_sha256"),
        "corpus_source_sha256": _text(report.get("corpus_source_sha256"), "corpus_source_sha256"),
        "grader_sha256": _text(report.get("grader_sha256"), "grader_sha256"),
        "adapter_sha256": _adapter_digest(report),
        "report_path": relative,
    }
    if report.get("lane") == "controlled Inspect smoke; explicitly loaded skill":
        return [
            {
                **base,
                "report_type": "controlled-smoke",
                "lane": "behavior",
                "case_id": _text(report.get("case_id"), "case_id"),
                "trial": _number(report.get("trial"), "trial"),
                "status": report["status"],
                "details": _details(report.get("problems"), report.get("reason")),
            }
        ]
    if "case_id" in report:
        if not isinstance(report["case_id"], str) or type(report.get("trial")) is not int:
            raise ReportError(f"invalid controlled case or trial: {path}")
        return [
            {
                **base,
                "report_type": "controlled-behavior",
                "lane": "behavior",
                "case_id": report["case_id"],
                "trial": report["trial"],
                "status": report["status"],
                "details": _details(
                    report.get("score_detail"),
                    report.get("problems"),
                    report.get("reason"),
                    report.get("observations"),
                ),
            }
        ]
    imported = isinstance(report.get("evidence_kind"), str)
    native = report.get("native_support") == "BLOCKED" and "activation" in report
    if imported or native:
        results = report.get("results")
        if not isinstance(results, list):
            raise ReportError(f"missing result list: {path}")
        kind = "imported" if imported else "native-capture"
        if not results:
            return [
                {
                    **base,
                    "report_type": kind,
                    "lane": _text(report.get("lane"), "lane") if imported else "native",
                    "status": report["status"],
                    "details": _details(report.get("reason")),
                }
            ]
        rows = []
        for result in results:
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("case_id"), str)
                or type(result.get("trial")) is not int
                or not isinstance(result.get("status"), str)
            ):
                raise ReportError(f"invalid trial result: {path}")
            status = result["status"]
            if imported and status == "SCORED":
                if type(result.get("success")) is not bool:
                    raise ReportError(f"scored result missing success: {path}")
                status = "PASS" if result["success"] else "FAIL"
            sessions = result.get("sessions", [])
            if not isinstance(sessions, list) or not all(
                isinstance(item, dict) for item in sessions
            ):
                raise ReportError(f"invalid session list: {path}")
            activation_status = ""
            if native:
                statuses = [
                    session.get("activation", {}).get("status")
                    for session in sessions
                    if isinstance(session.get("activation"), dict)
                ]
                activation_status = (
                    "; ".join(str(value) for value in statuses if value) or "UNVERIFIED"
                )
            rows.append(
                {
                    **base,
                    "report_type": kind,
                    "lane": _text(result.get("lane"), "lane") if imported else "native",
                    "case_id": result["case_id"],
                    "trial": result["trial"],
                    "status": status,
                    # Run-level usage is not attributable to individual imported trials.
                    "tokens": "",
                    "cost_usd": "",
                    "expected_activation": _boolean(
                        result.get("expected_activation"), "expected_activation"
                    ),
                    "observed_activation": _boolean(
                        result.get("observed_activation"), "observed_activation"
                    ),
                    "activation_status": activation_status,
                    "details": _details(
                        result.get("error"),
                        result.get("problems"),
                        _failed_checks(result) if imported and result["status"] == "SCORED" else [],
                    ),
                }
            )
        return rows
    if report["status"] in {"ERROR", "NOT RUN", "INTERRUPTED"}:
        return [
            {
                **base,
                "report_type": "unclassified",
                "status": report["status"],
                "details": _details(report.get("reason")),
            }
        ]
    raise ReportError(f"unrecognized evaluation report: {path}")


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.lstrip()
        if value.startswith(("\t", "\r", "\n")) or stripped.startswith(("=", "+", "-", "@")):
            return "'" + value
    return value


def export(root, extra_reports, output):
    root = Path(root)
    paths = discover_reports(root) + [Path(path) for path in extra_reports]
    unique = {}
    for path in paths:
        if path.is_symlink():
            raise ReportError(f"report must not be a symlink: {path}")
        unique.setdefault(path.resolve(), path)
    if not unique:
        raise ReportError("no saved evaluation reports found")
    rows = []
    for path in sorted(unique.values(), key=str):
        rows.extend(rows_from_report(read_report(path), path, root))
    rows.sort(
        key=lambda row: (
            row.get("case_id", ""),
            row["report_modified_utc"],
            row["run"],
            row.get("trial") if isinstance(row.get("trial"), int) else -1,
        )
    )
    output = Path(output)
    if not output.parent.is_dir():
        raise ReportError(f"output directory does not exist: {output.parent}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key, "")) for key in FIELDS})
    try:
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(buffer.getvalue())
    except OSError as exc:
        raise ReportError(f"cannot create output {output}: {exc}") from exc
    return len(unique), len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("eval-runs"), help="saved report directory"
    )
    parser.add_argument(
        "--report", type=Path, action="append", default=[], help="additional report file"
    )
    parser.add_argument(
        "--output", type=Path, help="new CSV file; existing files are never overwritten"
    )
    args = parser.parse_args(argv)
    output = (
        args.output
        or args.root / f"results-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.csv"
    )
    try:
        reports, rows = export(args.root, args.report, output)
    except (OSError, ReportError) as exc:
        print(f"results error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {rows} rows from {reports} reports to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
