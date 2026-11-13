"""Build and capture a synthetic integration candidate in an isolated Git clone."""

from __future__ import annotations

import json

from behavior_shared_capture import PROPOSAL_PROBE, REMOTE, capture_fetch_snapshot
from behavior_write_capture import RAW_PROBE
from inspect_task import STORE, WORK, checked_exec

ACTOR = f"{WORK}/competing-actor"
VALIDATOR = f"{WORK}/conflict-validator"

CANDIDATE_PROBE = r"""
import os, re, subprocess, sys
root, winner, loser, node_path, receipt_path = sys.argv[1:]
if not all(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid) for oid in (winner, loser)):
    raise ValueError("invalid input commit")
if not re.fullmatch(r"[0-9a-f-]{36}\.md", node_path):
    raise ValueError("invalid node path")
if not re.fullmatch(r"operations/[0-9a-f-]{36}\.json", receipt_path):
    raise ValueError("invalid receipt path")
base = ["git", "-C", root, "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false"]
env = dict(os.environ)
index = os.path.join(root, ".git", "conflict-probe.index")
if os.path.lexists(index):
    raise ValueError("probe index already exists")
env["GIT_INDEX_FILE"] = index
def run(args, data=None):
    return subprocess.run(base + list(args), input=data, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True, timeout=15, env=env).stdout
try:
    run(["read-tree", winner])
    for path in (node_path, receipt_path):
        oid = run(["rev-parse", loser + ":" + path]).decode("ascii").strip()
        run(["update-index", "--add", "--cacheinfo", "100644," + oid + "," + path])
    tree = run(["write-tree"]).decode("ascii").strip()
    commit = run(["commit-tree", tree, "-p", winner, "-p", loser],
                 b"Synthetic conflict integration candidate\n").decode("ascii").strip()
    print(commit)
finally:
    if os.path.exists(index):
        os.unlink(index)
"""


async def capture_conflict_snapshot(loser_op, winner_op, candidate, timeout):
    observed = await capture_fetch_snapshot(timeout)
    loser_ref = f"refs/heads/memory-graph/{loser_op}"
    winner_ref = f"refs/heads/memory-graph/{winner_op}"
    loser_raw = await checked_exec(
        ["python3", "-c", PROPOSAL_PROBE, STORE, REMOTE, loser_ref], timeout=timeout
    )
    winner_raw = await checked_exec(
        ["python3", "-c", PROPOSAL_PROBE, ACTOR, REMOTE, winner_ref], timeout=timeout
    )
    candidate_raw = await checked_exec(
        ["python3", "-c", RAW_PROBE, VALIDATOR, candidate], timeout=timeout
    )
    loser, winner, integration = (
        json.loads(loser_raw),
        json.loads(winner_raw),
        json.loads(candidate_raw),
    )
    if not isinstance(loser, dict) or set(loser) != {"local", "remote"}:
        raise ValueError("missing retained proposal artifacts")
    if not isinstance(winner, dict) or set(winner) != {"local", "remote"}:
        raise ValueError("missing competing proposal artifacts")
    if not isinstance(integration, dict) or set(integration) != {
        "tree",
        "working",
        "parents",
        "message",
    }:
        raise ValueError("missing integration candidate artifacts")
    observed["loser"] = loser
    observed["winner"] = winner
    observed["integration"] = {
        "commit": candidate,
        "tree": integration["tree"],
        "parents": integration["parents"],
        "message": integration["message"],
    }
    return observed
