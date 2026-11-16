#!/usr/bin/env python3
"""Check or explicitly regenerate the JSON compatibility snapshot and corpus lock."""

import argparse
import hashlib
import json
import sys

from corpus import DATASET, LOCK, compile_corpus, dataset_bytes, verify_corpus
from graders.imported import validate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate after reviewing changes")
    args = parser.parse_args(argv)
    try:
        dataset, _, _, _, source_sha = compile_corpus()
        validate_dataset(dataset)
        if args.write:
            raw = dataset_bytes(dataset)
            DATASET.write_bytes(raw)
            LOCK.write_bytes(
                (
                    json.dumps(
                        {
                            "schema_version": 1,
                            "source_sha256": source_sha,
                            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                        },
                        indent=2,
                    )
                    + "\n"
                ).encode("utf-8")
            )
        else:
            verify_corpus()
        print(f"corpus valid: {len(dataset['cases'])} cases; source sha256 {source_sha}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"corpus error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
