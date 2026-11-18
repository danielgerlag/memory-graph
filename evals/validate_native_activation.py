#!/usr/bin/env python3
"""Validate native skill-event detection with an offline scripted model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import native_core
import run_native

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "/eval/probe/native_core.py": ROOT / "evals/native_core.py",
    "/eval/probe/native_activation_probe.py": ROOT / "evals/native_activation_probe.py",
    "/eval/probe/SKILL.md": ROOT / "memory-graph/SKILL.md",
}
HOST_INPUTS = (ROOT / "evals/run_native.py", ROOT / "evals/validate_native_activation.py")
COPY_FILE = (
    "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
    "f=p.open('xb'); f.write(sys.stdin.buffer.read()); f.close()"
)


def validate_results(payload, *, deny_read=False):
    if not isinstance(payload, dict) or set(payload) != {"results"}:
        raise ValueError("invalid probe output")
    results = payload["results"]
    if not isinstance(results, list) or len(results) != 2:
        raise ValueError("probe must produce positive and negative observations")
    for result, expected in zip(results, (False, True), strict=True):
        if (
            not isinstance(result, dict)
            or result.get("expected") is not expected
            or result.get("native_launcher") is not True
            or result.get("usage_captured") is not True
            or result.get("read_tools_denied") is not deny_read
            or result.get("skill_offered") is not True
            or type(result.get("model_requests")) is not int
            or result["model_requests"] < (2 if expected else 1)
            or not isinstance(result.get("activation"), dict)
            or result["activation"].get("status") != "VERIFIED"
            or result["activation"].get("detected") is not expected
            or result["activation"].get("observation_complete") is not True
            or result["activation"].get("source") != "session_events"
        ):
            raise ValueError("probe activation result is incomplete or incorrect")
        lines = result["activation"].get("event_lines")
        if (
            not isinstance(lines, list)
            or any(type(line) is not int or line < 1 for line in lines)
            or len(set(lines)) != len(lines)
            or bool(lines) is not expected
        ):
            raise ValueError("probe activation event references are invalid")
        if expected and (
            not isinstance(result.get("tampered"), dict)
            or result["tampered"].get("status") != "UNVERIFIED"
            or "detected" in result["tampered"]
        ):
            raise ValueError("probe failed to reject tampered events")
    return results


def probe_container(image):
    options = SimpleNamespace(image=image, cpus=1, memory_mb=512, run_timeout=180)
    command = native_core.docker_create(options)
    command[command.index("--network=bridge")] = "--network=none"
    return command


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="authorize the local Docker probe")
    p.add_argument(
        "--image", required=True, help="trusted locally available image ID or name@sha256"
    )
    p.add_argument("--output-dir", type=Path, help="new output directory under this workspace")
    p.add_argument("--docker-host", help="optional explicit unix:/// Docker socket")
    p.add_argument("--deny-read", action="store_true", help="also deny agent file-read tools")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    container = None
    root = None
    status = 2
    failure = None
    report = None
    try:
        if not args.run:
            raise ValueError("--run is required")
        if not re.fullmatch(r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}", args.image):
            raise ValueError("--image must be an immutable image ID or name@sha256 reference")
        if args.docker_host and not args.docker_host.startswith("unix:///"):
            raise ValueError("--docker-host must be an explicit unix:/// socket")
        root = (
            args.output_dir or Path("eval-runs") / f"native-activation-probe-{uuid.uuid4().hex}"
        ).resolve()
        if not root.is_relative_to(Path.cwd().resolve()):
            raise ValueError("--output-dir must be inside the current working directory")
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        (root / "docker-home").mkdir(mode=0o700)
        env = run_native.environment(root, args.docker_host)
        deadline = time.monotonic() + 180
        container = (
            run_native.admin(probe_container(args.image), env, deadline).decode("ascii").strip()
        )
        if not re.fullmatch(r"[0-9a-f]{64}", container):
            raise ValueError("invalid container identifier")
        run_native.admin(["docker", "start", container], env, deadline)
        run_native.admin(
            run_native.clean_exec(container, "mkdir", "-p", "/eval/home", "/eval/probe"),
            env,
            deadline,
        )
        version = run_native.host_version(container, env, deadline)
        if not run_native.version_matches(version, native_core.DOCUMENTED_BUILD):
            raise ValueError("host version does not match the validated build")
        hashes = {}
        for destination, path in INPUTS.items():
            if path.is_symlink() or not path.is_file():
                raise ValueError("probe input must be a regular, non-symlink file")
            data = path.read_bytes()
            if len(data) > 1_048_576:
                raise ValueError("probe input exceeds the size limit")
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
            run_native.admin(
                run_native.clean_exec(
                    container, "python3", "-c", COPY_FILE, destination, stdin=True
                ),
                env,
                deadline,
                data=data,
            )
        for path in HOST_INPUTS:
            if path.is_symlink() or not path.is_file():
                raise ValueError("probe input must be a regular, non-symlink file")
            data = path.read_bytes()
            if len(data) > 1_048_576:
                raise ValueError("probe input exceeds the size limit")
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        command = run_native.clean_exec(
            container,
            "python3",
            "/eval/probe/native_activation_probe.py",
            "/eval/probe/native_core.py",
            "/eval/probe/SKILL.md",
            *(["--deny-read"] if args.deny_read else []),
        )
        captured = run_native.captured_exec(
            command, env, deadline, 120, 1_048_576, secrets.token_bytes(32)
        )
        if captured["outcome"] != "complete" or captured["returncode"] != 0:
            if captured["stderr"]:
                diagnostic = root / "probe.stderr"
                fd = os.open(diagnostic, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as file:
                    file.write(captured["stderr"])
            raise RuntimeError(
                f"local host probe failed: {captured['outcome']}, exit {captured['returncode']}"
            )
        payload = native_core.read_json_bytes(captured["stdout"])
        results = validate_results(payload, deny_read=args.deny_read)
        if any(
            hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(ROOT))]
            for path in (*INPUTS.values(), *HOST_INPUTS)
        ):
            raise RuntimeError("probe inputs changed during validation")
        report = {
            "status": "PASS",
            "scope": "offline activation-event mapping only",
            "host_version": native_core.DOCUMENTED_BUILD,
            "image": args.image,
            "read_tools_denied": args.deny_read,
            "inputs_sha256": hashes,
            "results": results,
            "native_support": "UNVERIFIED",
            "release": "BLOCKED",
        }
        status = 0
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError, RuntimeError) as exc:
        failure = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
    finally:
        if container and re.fullmatch(r"[0-9a-f]{64}", container):
            try:
                cleaned = subprocess.run(
                    ["docker", "rm", "-f", container],
                    env=run_native.environment(root, args.docker_host),
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                if cleaned.returncode:
                    failure = "container cleanup was not confirmed"
                    status = 2
            except (OSError, subprocess.SubprocessError):
                failure = "container cleanup failed"
                status = 2
    if failure:
        print(f"activation probe failed: {failure}", file=sys.stderr)
        if root is not None:
            print(f"output directory: {root}", file=sys.stderr)
        return 2
    if report is None or root is None:
        print("activation probe failed: missing report", file=sys.stderr)
        return 2
    path = root / "validation.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"Validated positive, negative, and tampered-event checks: {path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
