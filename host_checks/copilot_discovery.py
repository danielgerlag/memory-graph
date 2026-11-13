"""Opt-in, network-denied Copilot CLI skill-discovery check for macOS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "memory-graph"
sys.path.insert(0, str(SKILL / "scripts"))

from memory_graph.cli import _validate_skill_bundle

SANDBOX = "/usr/bin/sandbox-exec"
POLICY = "(version 1)(allow default)(deny network*)"
NETWORK_PROBE = """
import errno, socket, sys
try:
    sock = socket.socket()
    sock.connect(("127.0.0.1", 9))
except OSError as exc:
    sys.exit(0 if exc.errno in (errno.EPERM, errno.EACCES) else 1)
sys.exit(1)
"""


def _run(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _found(stdout: str) -> bool:
    return bool(re.search(r"(?m)^\s*memory-graph\s+-\s+", stdout))


def check(copilot: Path) -> dict:
    if sys.platform != "darwin" or not Path(SANDBOX).is_file():
        raise ValueError("this check requires macOS and /usr/bin/sandbox-exec")
    if not copilot.is_absolute() or not copilot.is_file() or not os.access(copilot, os.X_OK):
        raise ValueError("--copilot must name an absolute executable path")
    _validate_skill_bundle(SKILL)
    with tempfile.TemporaryDirectory(prefix="memory-graph-discovery-") as directory:
        root = Path(directory)
        project, home = root / "project", root / "home"
        project.mkdir()
        home.mkdir()
        env = {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(home),
            "COPILOT_HOME": str(home / ".copilot"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "TMPDIR": str(root),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "NO_COLOR": "1",
            "CI": "1",
        }
        sandbox = [SANDBOX, "-p", POLICY]
        probe = _run([*sandbox, sys.executable, "-c", NETWORK_PROBE], project, env)
        if probe.returncode:
            raise RuntimeError("network-denial probe failed; Copilot was not launched")
        git = shutil.which("git", path=env["PATH"])
        if git is None:
            raise RuntimeError("Git was not found on the isolated PATH")
        initialized = _run(
            [*sandbox, git, "init", "--initial-branch=main", "--template="], project, env
        )
        if initialized.returncode:
            raise RuntimeError("cannot initialize the synthetic project")
        dest = project / ".github" / "skills" / "memory-graph"
        dest.parent.mkdir(parents=True)
        shutil.copytree(SKILL, dest, ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc"))
        digest = hashlib.sha256()
        for path in sorted(p for p in dest.rglob("*") if p.is_file()):
            digest.update(path.relative_to(dest).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        version = _run([*sandbox, str(copilot), "--version"], project, env)
        if version.returncode:
            raise RuntimeError("cannot determine Copilot CLI version in the sandbox")
        result = _run([*sandbox, str(copilot), "skill", "list"], project, env)
        if result.returncode:
            raise RuntimeError(f"Copilot skill listing failed with exit status {result.returncode}")
        discovered = _found(result.stdout)
        return {
            "host": "Copilot CLI",
            "version": version.stdout.strip(),
            "skill_sha256": digest.hexdigest(),
            "network_denied": True,
            "discovered": discovered,
            "activation": "NOT RUN",
            "live_behavior": "NOT RUN",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="explicitly run the isolated check")
    parser.add_argument("--copilot", required=True, type=Path, help="absolute Copilot CLI path")
    args = parser.parse_args()
    try:
        if not args.run:
            raise ValueError("pass --run to start the isolated discovery check")
        result = check(args.copilot)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["discovered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
