"""Capture local and bare-remote artifacts without provider or network access."""

from __future__ import annotations

import json

from behavior_write_capture import RAW_PROBE, capture_full_snapshot
from inspect_task import STORE, WORK, checked_exec

REMOTE = f"{WORK}/remote.git"
OFFLINE_REMOTE = f"{WORK}/offline-remote.git"
REMOTE_PROBE = r"""
import base64, json, subprocess, sys
store, remote = sys.argv[1:]
def run(args):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True, timeout=15).stdout
local = ["git", "-C", store]
bare = ["git", "--git-dir", remote]
url = run(local + ["config", "--local", "--get", "remote.origin.url"]).decode().strip()
main = run(bare + ["rev-parse", "refs/heads/main"]).decode("ascii").strip()
refs = run(bare + ["for-each-ref", "--format=%(refname) %(objectname)"]).decode("ascii").splitlines()
listing = run(bare + ["ls-tree", "-rz", "--full-tree", main])
files = {}
for entry in listing.split(b"\0"):
    if not entry:
        continue
    header, path = entry.split(b"\t", 1)
    mode, kind, oid = header.split(b" ", 2)
    path = path.decode("utf-8")
    if kind != b"blob" or mode not in (b"100644", b"100755") or path in files:
        raise ValueError("unexpected remote tree entry")
    value = run(bare + ["cat-file", "blob", oid.decode("ascii")])
    if len(value) > 262144 or len(files) >= 128:
        raise ValueError("remote artifact limits exceeded")
    files[path] = base64.b64encode(value).decode("ascii")
print(json.dumps({"url": url, "main": main, "refs": refs, "tree": files}, sort_keys=True))
"""

AVAILABILITY_PROBE = r"""
import json, os, sys
print(json.dumps({"available": os.path.lexists(sys.argv[1])}))
"""

PROPOSAL_PROBE = r"""
import base64, json, subprocess, sys
store, remote, ref = sys.argv[1:]
def run(args):
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True, timeout=15).stdout
def snapshot(base):
    lines = run(base + ["for-each-ref", "--format=%(refname) %(objectname)", ref]).decode("ascii").splitlines()
    if not lines:
        return None
    if len(lines) != 1 or not lines[0].startswith(ref + " "):
        raise ValueError("unexpected proposal ref")
    commit = lines[0].split(" ", 1)[1]
    listing = run(base + ["ls-tree", "-rz", "--full-tree", commit])
    files = {}
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, kind, oid = header.split(b" ", 2)
        path = path.decode("utf-8")
        if kind != b"blob" or mode not in (b"100644", b"100755") or path in files:
            raise ValueError("unexpected proposal tree entry")
        value = run(base + ["cat-file", "blob", oid.decode("ascii")])
        if len(value) > 262144 or len(files) >= 128:
            raise ValueError("proposal artifact limits exceeded")
        files[path] = base64.b64encode(value).decode("ascii")
    parents = run(base + ["rev-list", "--parents", "-n", "1", commit]).decode("ascii").split()
    message = run(base + ["show", "-s", "--format=%B", commit]).decode("utf-8")
    if not parents or parents[0] != commit:
        raise ValueError("invalid proposal ancestry")
    return {"commit": commit, "tree": files, "parents": parents[1:], "message": message}
local = ["git", "-C", store]
bare = ["git", "--git-dir", remote]
print(json.dumps({"local": snapshot(local), "remote": snapshot(bare)}, sort_keys=True))
"""


async def capture_shared_snapshot(tool_timeout):
    local = await capture_full_snapshot(tool_timeout)
    raw = await checked_exec(["python3", "-c", REMOTE_PROBE, STORE, REMOTE], timeout=tool_timeout)
    remote = json.loads(raw)
    if not isinstance(remote, dict) or set(remote) != {"url", "main", "refs", "tree"}:
        raise ValueError("missing local bare-remote evidence")
    return {"local": local, "remote": remote}


async def capture_offline_snapshot(tool_timeout):
    local = await capture_full_snapshot(tool_timeout)
    remote_raw = await checked_exec(
        ["python3", "-c", REMOTE_PROBE, STORE, OFFLINE_REMOTE], timeout=tool_timeout
    )
    availability_raw = await checked_exec(
        ["python3", "-c", AVAILABILITY_PROBE, REMOTE], timeout=tool_timeout
    )
    remote, availability = json.loads(remote_raw), json.loads(availability_raw)
    if not isinstance(remote, dict) or set(remote) != {"url", "main", "refs", "tree"}:
        raise ValueError("missing retained remote evidence")
    if not isinstance(availability, dict) or type(availability.get("available")) is not bool:
        raise ValueError("missing remote availability evidence")
    return {"local": local, "remote": remote, "remote_available": availability["available"]}


async def capture_fetch_snapshot(tool_timeout):
    observed = await capture_shared_snapshot(tool_timeout)
    tracking = (
        await checked_exec(
            ["git", "-C", STORE, "rev-parse", "refs/remotes/origin/main"],
            timeout=tool_timeout,
        )
    ).strip()
    raw = await checked_exec(["python3", "-c", RAW_PROBE, STORE, tracking], timeout=tool_timeout)
    details = json.loads(raw)
    if not isinstance(details, dict) or set(details) != {"tree", "working", "parents", "message"}:
        raise ValueError("missing tracking-tree evidence")
    if details["working"] != observed["local"]["working"]:
        raise ValueError("working tree changed during tracking capture")
    observed["tracking"] = {
        "commit": tracking,
        "tree": details["tree"],
        "parents": details["parents"],
        "message": details["message"],
    }
    return observed
