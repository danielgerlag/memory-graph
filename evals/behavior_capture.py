"""Capture raw sandbox and Git artifacts independently of behavioral grading."""

from __future__ import annotations

import json

from inspect_task import STORE, WORK, checked_exec, snapshot

TREE_PROBE = r"""
import hashlib, json, subprocess, sys
root, commit = sys.argv[1:]
base = ["git", "-C", root, "-c", "core.hooksPath=/dev/null"]
listing = subprocess.run(base + ["ls-tree", "-rz", "--full-tree", commit],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=15).stdout
entries = listing.split(b"\0")
files = {}
for entry in entries:
    if not entry:
        continue
    header, path = entry.split(b"\t", 1)
    mode, kind, oid = header.split(b" ", 2)
    name = path.decode("utf-8")
    if kind != b"blob" or mode not in (b"100644", b"100755") or name in files:
        raise ValueError("unexpected Git tree entry")
    data = subprocess.run(base + ["cat-file", "blob", oid.decode("ascii")],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=15).stdout
    if len(data) > 262144:
        raise ValueError("oversized Git tree entry")
    files[name] = hashlib.sha256(data).hexdigest()
    if len(files) > 128:
        raise ValueError("too many Git tree entries")
print(json.dumps(files, sort_keys=True))
"""

WORKSPACE_PROBE = r"""
import json, os, sys
root = sys.argv[1]
entries = sorted(os.listdir(root))
if any(os.path.islink(os.path.join(root, name)) for name in entries):
    raise ValueError("symlink in isolated workspace")
print(json.dumps({"entries": entries, "target_exists": os.path.lexists(os.path.join(root, "shared-workshop"))}))
"""


async def capture_snapshot(tool_timeout):
    observed = await snapshot(tool_timeout)
    raw = await checked_exec(
        ["python3", "-c", TREE_PROBE, STORE, observed["main"]], timeout=tool_timeout
    )
    tree = json.loads(raw)
    if not isinstance(tree, dict) or not all(
        isinstance(path, str) and isinstance(digest, str) for path, digest in tree.items()
    ):
        raise ValueError("invalid Git tree evidence")
    return {**observed, "tree_files": tree}


def capture_trace(state):
    calls = state.store.get("memory_calls")
    if not isinstance(calls, list):
        raise TypeError("missing action trace")
    return json.loads(json.dumps(calls, allow_nan=False))


async def capture_workspace(tool_timeout):
    raw = await checked_exec(["python3", "-c", WORKSPACE_PROBE, WORK], timeout=tool_timeout)
    observed = json.loads(raw)
    if not isinstance(observed, dict) or set(observed) != {"entries", "target_exists"}:
        raise ValueError("missing workspace isolation evidence")
    return observed
