#!/usr/bin/env python3
"""Report an offline evaluation or grade a single imported run."""

import argparse
import json
import sys
from pathlib import Path

from grader import EvidenceError, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).with_name("scenarios.json"))
    parser.add_argument(
        "--evidence", type=Path, help="import manifest; artifacts are relative to its directory"
    )
    parser.add_argument("--lane", choices=("all", "behavior", "native"), default="all")
    parser.add_argument("--output", type=Path, help="write JSON report instead of stdout")
    args = parser.parse_args()
    try:
        report = run(args.dataset, args.evidence, args.lane)
        output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
    except (EvidenceError, OSError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
