"""Capture raw committed and local artifacts for standalone mutation grading."""

from __future__ import annotations

import base64
import hashlib
import json

from behavior_capture import capture_snapshot
from inspect_task import STORE, checked_exec

RAW_PROBE = r"""
import base64, json, os, subprocess, sys
root, commit = sys.argv[1:]
base = ["git", "-C", root, "-c", "core.hooksPath=/dev/null"]
def git(*args):
    return subprocess.run(base + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True, timeout=15).stdout
items = git("ls-tree", "-rz", "--full-tree", commit).split(b"\0")
tree = {}
for item in items:
    if not item:
        continue
    header, path = item.split(b"\t", 1)
    mode, kind, oid = header.split(b" ", 2)
    path = path.decode("utf-8")
    if kind != b"blob" or mode not in (b"100644", b"100755") or path in tree:
        raise ValueError("unexpected Git object")
    value = git("cat-file", "blob", oid.decode("ascii"))
    if len(value) > 262144 or len(tree) >= 128:
        raise ValueError("Git artifact limits exceeded")
    tree[path] = base64.b64encode(value).decode("ascii")
parents = git("rev-list", "--parents", "-n", "1", commit).decode("ascii").split()
if not parents or parents[0] != commit:
    raise ValueError("invalid commit ancestry")
message = git("show", "-s", "--format=%B", commit).decode("utf-8")
working = {}
ignored = {os.path.join(".memory-graph", "parse-cache.sqlite3" + suffix)
           for suffix in ("", "-journal", "-wal", "-shm")}
for directory, dirs, names in os.walk(root, followlinks=False):
    for name in dirs + names:
        if os.path.islink(os.path.join(directory, name)):
            raise ValueError("symlink in store")
    dirs[:] = sorted(d for d in dirs if not (directory == root and d == ".git"))
    for name in sorted(names):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise ValueError("nonregular store file")
        relative = os.path.relpath(path, root)
        if relative in ignored:
            continue
        with open(path, "rb") as stream:
            value = stream.read(262145)
        if len(value) > 262144 or len(working) >= 128:
            raise ValueError("working artifact limits exceeded")
        working[relative] = base64.b64encode(value).decode("ascii")
print(json.dumps({"tree": tree, "working": working, "parents": parents[1:], "message": message},
                 sort_keys=True))
"""


async def capture_full_snapshot(tool_timeout):
    observed = await capture_snapshot(tool_timeout)
    raw = await checked_exec(
        ["python3", "-c", RAW_PROBE, STORE, observed["main"]], timeout=tool_timeout
    )
    details = json.loads(raw)
    if not isinstance(details, dict) or set(details) != {"tree", "working", "parents", "message"}:
        raise ValueError("invalid raw mutation evidence")
    for key, hashes in (("tree", observed["tree_files"]), ("working", observed["files"])):
        values = details[key]
        if not isinstance(values, dict):
            raise TypeError("missing raw artifact contents")
        try:
            computed = {
                path: hashlib.sha256(base64.b64decode(value, validate=True)).hexdigest()
                for path, value in values.items()
            }
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid raw artifact encoding") from exc
        if computed != hashes:
            raise ValueError("raw artifact content does not match independently captured hashes")
    if not isinstance(details["parents"], list) or not isinstance(details["message"], str):
        raise TypeError("missing Git commit metadata")
    return {**observed, **details}
