#!/usr/bin/env python3
"""Report an offline evaluation or grade a single imported run."""

import argparse
import json
import sys
from pathlib import Path

from corpus import DATASET, verify_corpus
from graders.imported import EvidenceError, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--evidence", type=Path, help="import manifest; artifacts are relative to its directory"
    )
    parser.add_argument("--lane", choices=("all", "behavior", "native"), default="all")
    parser.add_argument("--output", type=Path, help="write JSON report instead of stdout")
    args = parser.parse_args()
    try:
        corpus_sha = verify_corpus()[4] if args.dataset.resolve() == DATASET.resolve() else None
        report = run(args.dataset, args.evidence, args.lane)
        if corpus_sha is not None:
            report["corpus_source_sha256"] = corpus_sha
        output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
    except (EvidenceError, OSError, ValueError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
