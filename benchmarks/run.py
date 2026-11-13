"""Reproducible, disposable Git-backed search benchmark; no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

try:
    import resource
except ImportError:
    resource = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory-graph" / "scripts"))

from memory_graph import git_store, schema, service


def node_id(index: int) -> str:
    return str(uuid.UUID(int=index + 1, version=4))


def import_fixture(root: Path, initial: str, changes: dict[str, bytes]) -> None:
    # This bulk importer is only used for disposable benchmark fixtures.
    stream = bytearray()
    entries = []
    for mark, (path, data) in enumerate(sorted(changes.items()), 1):
        stream.extend(f"blob\nmark :{mark}\ndata {len(data)}\n".encode())
        stream.extend(data)
        stream.extend(b"\n")
        entries.append(f"M 100644 :{mark} {path}\n")
    message = b"synthetic benchmark fixture"
    stream.extend(
        b"commit refs/heads/main\ncommitter Benchmark <benchmark@example.invalid> 1700000000 +0000\n"
        + f"data {len(message)}\n".encode()
        + message
        + f"\nfrom {initial}\n".encode()
        + "".join(entries).encode()
        + b"\ndone\n"
    )
    subprocess.run(
        git_store._command(root, ["fast-import", "--quiet"]),
        input=bytes(stream),
        capture_output=True,
        check=True,
        timeout=180,
        env=git_store._environment(),
    )


def run(
    size: int,
    iterations: int,
    receipt_count: int,
    cold_iterations: int = 1,
    workload: str = "find",
) -> dict:
    if not 4 <= size <= 10000 or not 1 <= iterations <= 100 or not 0 <= receipt_count <= 10000:
        raise ValueError("nodes must be 4..10000, iterations 1..100, receipts 0..10000")
    if not 1 <= cold_iterations <= iterations:
        raise ValueError("cold iterations must be between 1 and total iterations")
    if workload not in {"find", "tag", "neighbors"}:
        raise ValueError("workload must be find, tag, or neighbors")
    with tempfile.TemporaryDirectory(prefix="memory-graph-benchmark-") as directory:
        root = Path(directory) / "store"
        initial = git_store.init_store(root)
        changes: dict[str, bytes] = {}
        for index in range(size):
            ident = node_id(index)
            node = {
                "schema": 1,
                "id": ident,
                "label": f"Fixture node {index:05}",
                "kind": "benchmark",
                "status": "active",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "tags": ["fixture"],
                "aliases": [],
                "edges": [
                    {"label": "related", "to": node_id((index + offset) % size)}
                    for offset in (1, 2, 3)
                ],
            }
            changes[f"{ident}.md"] = schema.dump_node(node, b"x" * 1024 + b"\n")
        for index in range(receipt_count):
            ident = node_id(100_000 + index)
            receipt = {
                "schema": 1,
                "operation_id": ident,
                "action": "create",
                "request_hash": "0" * 64,
                "base": "0" * 40,
                "preconditions": {},
                "node_ids": [],
                "resolved_at": "2026-01-01T00:00:00Z",
                "generated_ids": {},
                "changed_hashes": {},
            }
            changes[f"operations/{ident}.json"] = service.canonical(receipt) + b"\n"
        corpus_hash = hashlib.sha256(
            b"".join(
                path.encode() + b"\0" + hashlib.sha256(data).digest()
                for path, data in sorted(changes.items())
            )
        ).hexdigest()
        import_fixture(root, initial, changes)
        samples = []
        process_counts = []
        response_sizes = []
        original_popen = subprocess.Popen
        cache_path = root / ".memory-graph" / "parse-cache.sqlite3"
        for index in range(iterations):
            if index < cold_iterations and cache_path.exists():
                cache_path.unlink()
            with patch("subprocess.Popen", wraps=original_popen) as popen:
                start = time.perf_counter()
                if workload == "find":
                    response = service.find(root, "fixture", None, False, 20)
                elif workload == "tag":
                    response = service.find(root, "", None, False, 20, tag="fixture")
                else:
                    response = service.neighbors(root, node_id(0), "out", None, None, False, 2, 50)
                samples.append(time.perf_counter() - start)
                process_counts.append(popen.call_count)
            response_sizes.append(len(service.canonical(response)))
        cold = sorted(samples[:cold_iterations])
        warm = sorted(samples[cold_iterations:])
        rss_multiplier = 1 if sys.platform == "darwin" else 1024
        peak_rss = (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_multiplier)
            if resource is not None
            else None
        )
        return {
            "status": "measured_not_release_gated",
            "nodes": size,
            "receipts": receipt_count,
            "iterations": iterations,
            "workload": workload,
            "corpus_sha256": corpus_hash,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "git": subprocess.run(
                ["git", "--version"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "seconds": {
                "empty_cache": samples[0],
                "cold_p50": statistics.median(cold),
                "cold_p95": cold[math.ceil(0.95 * len(cold)) - 1],
                "warm_p50": statistics.median(warm) if warm else None,
                "warm_p95": warm[math.ceil(0.95 * len(warm)) - 1] if warm else None,
                "samples": samples,
            },
            "git_processes_per_query": process_counts,
            "cold_parse_workers": min(4, os.cpu_count() or 1) if size >= 2000 else 0,
            "response_bytes": response_sizes,
            "fixture_bytes": sum(len(data) for data in changes.values()),
            "parse_cache_bytes": cache_path.stat().st_size,
            "process_peak_rss_bytes_including_fixture_setup": peak_rss,
            "cache": f"{cold_iterations} empty parse-cache sample(s), then warm samples; operating-system cache is uncontrolled",
            "limitations": f"Single {workload} workload; not the pinned release runner or a complete performance gate.",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--receipts", type=int, default=0)
    parser.add_argument("--cold-iterations", type=int, default=1)
    parser.add_argument("--workload", choices=("find", "tag", "neighbors"), default="find")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.nodes, args.iterations, args.receipts, args.cold_iterations, args.workload),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
