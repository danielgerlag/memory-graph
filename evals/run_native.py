#!/usr/bin/env python3
"""Opt-in native Copilot activation trials; full native support remains unverified."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import selectors
import subprocess
import sys
import time
from pathlib import Path

from corpus import verify_corpus
from native_core import (
    CONTAINER_STORE,
    CONTAINER_WORKSPACE,
    activation_status,
    copilot_command,
    docker_create,
    fixture_archive,
    fixture_store,
    load_trials,
    parse_jsonl,
    read_json_bytes,
    skill_bundle,
    skill_digest,
    validate_options,
)
from profiles import apply_profiles

SNAPSHOT = r"""
import hashlib, json, os, subprocess, sys
root = sys.argv[1]
files = {}
for base, dirs, names in os.walk(root, followlinks=False):
    for name in dirs + names:
        if os.path.islink(os.path.join(base, name)):
            raise RuntimeError("symlink in store")
    dirs[:] = sorted(d for d in dirs if not (base == root and d == ".git"))
    for name in sorted(names):
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            raise RuntimeError("nonregular store file")
        with open(path, "rb") as stream:
            files[os.path.relpath(path, root)] = hashlib.sha256(stream.read()).hexdigest()
base = ["git", "-C", root, "-c", "core.hooksPath=/dev/null"]
main = subprocess.check_output(base + ["rev-parse", "refs/heads/main"], text=True).strip()
refs = subprocess.check_output(base + ["for-each-ref", "--format=%(refname) %(objectname)"], text=True).splitlines()
print(json.dumps({"files": files, "main": main, "refs": refs}, sort_keys=True))
"""

READ_USAGE = r"""
import os, stat, sys
with os.fdopen(os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or info.st_size > 1048576:
        raise ValueError("invalid usage file")
    sys.stdout.buffer.write(stream.read(1048577))
"""

READ_SESSION_EVENTS = r"""
import os, stat, sys
root = "/eval/home/.copilot/session-state"
with os.scandir(root) as entries:
    directories = [entry.path for entry in entries if entry.is_dir(follow_symlinks=False)]
if len(directories) != 1:
    raise ValueError("ambiguous session state")
path = os.path.join(directories[0], "events.jsonl")
with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
    info = os.fstat(stream.fileno())
    limit = int(sys.argv[1])
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("invalid session event file")
    sys.stdout.buffer.write(stream.read(limit + 1))
"""

SKILL_HASHES = r"""
import hashlib, json, os, sys
root = sys.argv[1]
exclude = sys.argv[2] if len(sys.argv) > 2 else None
files = {}
for base, dirs, names in os.walk(root, followlinks=False):
    for name in dirs + names:
        if os.path.islink(os.path.join(base, name)):
            raise RuntimeError("symlink in skill bundle")
    dirs[:] = sorted(d for d in dirs if os.path.relpath(os.path.join(base, d), root) != exclude)
    for name in sorted(names):
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            raise RuntimeError("nonregular skill file")
        with open(path, "rb") as stream:
            files[os.path.relpath(path, root)] = hashlib.sha256(stream.read()).hexdigest()
print(json.dumps(files, sort_keys=True))
"""

GIT_ENV = [
    "GIT_CONFIG_NOSYSTEM=1",
    "GIT_CONFIG_GLOBAL=/dev/null",
    "GIT_TERMINAL_PROMPT=0",
    "GIT_AUTHOR_NAME=Synthetic Eval",
    "GIT_AUTHOR_EMAIL=eval@localhost",
    "GIT_COMMITTER_NAME=Synthetic Eval",
    "GIT_COMMITTER_EMAIL=eval@localhost",
    "GIT_AUTHOR_DATE=2026-01-01T00:00:00Z",
    "GIT_COMMITTER_DATE=2026-01-01T00:00:00Z",
]

UNPACK_FIXTURE = r"""
import pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1])
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
    for member in archive:
        name = pathlib.PurePosixPath(member.name)
        if (name.is_absolute() or any(part in ("", ".", "..") for part in member.name.split("/"))
                or not (member.isdir() or member.isfile())):
            raise ValueError("invalid fixture archive entry")
        archive.extract(member, path=root)
"""


def clean_exec(container, *command, stdin=False):
    return [
        "docker",
        "exec",
        *(["-i"] if stdin else []),
        container,
        "/usr/bin/env",
        "-i",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "HOME=/eval/home",
        *GIT_ENV,
        *command,
    ]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true")
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--agent", help="YAML agent profile")
    p.add_argument("--harness", help="YAML harness profile")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image")
    p.add_argument("--host-version")
    p.add_argument("--model")
    p.add_argument("--credential-env")
    p.add_argument("--session-ai-credits", type=int)
    p.add_argument("--run-ai-credits", type=int)
    p.add_argument("--session-timeout", type=int)
    p.add_argument("--run-timeout", type=int)
    p.add_argument("--cpus", type=float)
    p.add_argument("--memory-mb", type=int)
    p.add_argument("--max-output-mb", type=int)
    p.add_argument("--docker-host")
    return p


def environment(root, docker_host, token=None):
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(root / "docker-home"),
        "DOCKER_CONFIG": str(root / "docker-home" / "config"),
    }
    if docker_host:
        env["DOCKER_HOST"] = docker_host
        env["CONTAINER_HOST"] = docker_host
    if token is not None:
        env["COPILOT_GITHUB_TOKEN"] = token
    return env


def remaining(deadline, ceiling):
    seconds = min(ceiling, deadline - time.monotonic())
    if seconds <= 0:
        raise TimeoutError("run wall timeout")
    return seconds


def admin(command, env, deadline, *, data=None, limit=2_097_152):
    input_options = {"input": data} if data is not None else {"stdin": subprocess.DEVNULL}
    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        timeout=remaining(deadline, 30),
        check=False,
        **input_options,
    )
    if len(result.stdout) > limit or len(result.stderr) > limit:
        raise RuntimeError("administrative command exceeded output limit")
    if result.returncode:
        raise RuntimeError(f"administrative command failed: {command[1]}")
    return result.stdout


def captured_exec(command, env, deadline, session_timeout, max_bytes, secret):
    """Capture bounded raw streams without ever writing an observed token to disk."""
    process = subprocess.Popen(
        command, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    selector = selectors.DefaultSelector()
    streams = {"stdout": bytearray(), "stderr": bytearray()}
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    stop_at = min(deadline, time.monotonic() + session_timeout)
    outcome = "complete"
    try:
        while selector.get_map():
            timeout = stop_at - time.monotonic()
            if timeout <= 0:
                outcome = "timeout"
                break
            for key, _ in selector.select(timeout):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buf = streams[key.data]
                buf.extend(chunk)
                if secret in buf:
                    outcome = "credential-leak"
                    break
                if sum(len(item) for item in streams.values()) > max_bytes:
                    outcome = "output-limit"
                    break
            if outcome != "complete":
                break
        if outcome != "complete":
            process.kill()
        try:
            returncode = process.wait(timeout=max(1, min(5, stop_at - time.monotonic())))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            outcome, returncode = "timeout", None
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    if outcome == "credential-leak":
        return {"outcome": outcome, "returncode": returncode, "stdout": b"", "stderr": b""}
    return {
        "outcome": outcome,
        "returncode": returncode,
        "stdout": bytes(streams["stdout"]),
        "stderr": bytes(streams["stderr"]),
    }


def artifact(directory, name, data):
    path = directory / name
    path.write_bytes(data)
    return {
        "path": f"{directory.name}/{name}",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def exec_snapshot(container, env, deadline):
    raw = admin(clean_exec(container, "python3", "-c", SNAPSHOT, CONTAINER_STORE), env, deadline)
    value = read_json_bytes(raw)
    if not isinstance(value, dict) or set(value) != {"files", "main", "refs"}:
        raise ValueError("invalid graph snapshot")
    if (
        not isinstance(value["main"], str)
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value["main"])
        or value["refs"] != [f"refs/heads/main {value['main']}"]
        or not isinstance(value["files"], dict)
    ):
        raise ValueError("incomplete graph snapshot")
    return value, raw


def initialize(container, archive, env, deadline):
    admin(clean_exec(container, "mkdir", "-p", CONTAINER_WORKSPACE, "/eval/home"), env, deadline)
    admin(
        clean_exec(container, "python3", "-c", UNPACK_FIXTURE, CONTAINER_WORKSPACE, stdin=True),
        env,
        deadline,
        data=archive,
    )
    base = clean_exec(
        container,
        "git",
        "-C",
        CONTAINER_STORE,
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "commit.gpgSign=false",
    )
    admin(base + ["init", "--initial-branch=main", "--template="], env, deadline)
    admin(base + ["add", "--all"], env, deadline)
    admin(base + ["commit", "-m", "Synthetic native evaluation fixture"], env, deadline)


def host_version(container, env, deadline):
    command = [
        "docker",
        "exec",
        container,
        "/usr/bin/env",
        "-i",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "HOME=/eval/home",
        "COPILOT_HOME=/eval/home/.copilot",
        "COPILOT_CACHE_HOME=/cli-tmp/cache",
        "TMPDIR=/cli-tmp",
        "copilot",
        "--no-auto-update",
        "--version",
    ]
    return admin(command, env, deadline)


def version_matches(raw, expected):
    text = raw.decode("utf-8").strip()
    return bool(
        re.search(
            r"(?<![A-Za-z0-9.])v?" + re.escape(expected) + r"(?=$|[\s,;:!?)\]]|\.(?=\s|$))",
            text,
        )
    )


def activation_run_status(results):
    if not results or any(
        result.get("status") not in {"PASS", "FAIL", "ERROR", "BLOCKED", "NOT RUN"}
        for result in results
    ):
        raise ValueError("invalid native activation trial results")
    statuses = {result["status"] for result in results}
    if "ERROR" in statuses:
        return "ERROR"
    if "FAIL" in statuses:
        return "FAIL"
    if statuses != {"PASS"}:
        return "BLOCKED"
    return "PASS"


def collect_trial(args, trial, root, env, secret, archive, bundle, deadline):
    directory = root / f"{trial['case_id']}-trial-{trial['trial']}"
    directory.mkdir(mode=0o700)
    report = {
        "case_id": trial["case_id"],
        "trial": trial["trial"],
        "expected_activation": trial["should_activate"],
        "critical": trial["critical"],
        "status": "BLOCKED",
        "assessment": "native skill activation only",
        "sessions": [],
        "artifacts": {},
        "problems": [],
    }
    container = None
    phase = "container creation"
    try:
        container = admin(docker_create(args), env, deadline).decode("ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", container):
            raise ValueError("invalid Docker container identifier")
        phase = "container startup"
        admin(["docker", "start", container], env, deadline)
        phase = "fixture initialization"
        initialize(container, archive, env, deadline)
        phase = "skill bundle verification"
        skill_raw = admin(
            clean_exec(
                container,
                "python3",
                "-c",
                SKILL_HASHES,
                f"{CONTAINER_WORKSPACE}/.github/skills/memory-graph",
            ),
            env,
            deadline,
        )
        expected_skill = {name: hashlib.sha256(data).hexdigest() for name, data in bundle.items()}
        if read_json_bytes(skill_raw) != expected_skill:
            raise ValueError("sandbox skill bundle hash mismatch")
        report["artifacts"]["skill_hashes"] = artifact(directory, "skill-hashes.json", skill_raw)
        phase = "host version check"
        version = host_version(container, env, deadline)
        report["artifacts"]["host_version"] = artifact(directory, "host-version.txt", version)
        if not version_matches(version, args.host_version):
            report["problems"].append("host version does not match the requested version")
            return report
        phase = "initial snapshot"
        before, before_raw = exec_snapshot(container, env, deadline)
        if before["files"] != {
            name: hashlib.sha256(data).hexdigest() for name, data in fixture_store().items()
        }:
            raise ValueError("sandbox fixture hash mismatch")
        report["artifacts"]["before"] = artifact(directory, "before.json", before_raw)
        workspace_raw = admin(
            clean_exec(container, "python3", "-c", SKILL_HASHES, CONTAINER_WORKSPACE, "store/.git"),
            env,
            deadline,
        )
        workspace = read_json_bytes(workspace_raw)
        expected_workspace = {
            f".github/skills/memory-graph/{name}": digest for name, digest in expected_skill.items()
        }
        expected_workspace.update(
            {f"store/{name}": digest for name, digest in before["files"].items()}
        )
        if workspace != expected_workspace:
            raise ValueError("initial sandbox workspace hash mismatch")
        report["artifacts"]["workspace_before"] = artifact(
            directory, "workspace-before.json", workspace_raw
        )
        for index, prompt in enumerate(trial["prompts"], 1):
            phase = f"native session {index}"
            run = {"session": index, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            command = [
                "docker",
                "exec",
                "--env",
                "COPILOT_GITHUB_TOKEN",
                container,
                *copilot_command(args, prompt, index),
            ]
            captured = captured_exec(
                command,
                environment(root, args.docker_host, secret.decode()),
                deadline,
                args.session_timeout,
                args.max_output_mb * 1024 * 1024,
                secret,
            )
            run.update(outcome=captured["outcome"], exit_status=captured["returncode"])
            run["exit"] = artifact(
                directory,
                f"session-{index}-exit.json",
                (
                    json.dumps({"outcome": run["outcome"], "exit_status": run["exit_status"]})
                    + "\n"
                ).encode(),
            )
            if captured["outcome"] == "credential-leak":
                report["problems"].append("credential appeared in a stream; raw streams withheld")
                report["status"] = "ERROR"
                report["sessions"].append(run)
                return report
            run["transcript"] = artifact(directory, f"session-{index}.jsonl", captured["stdout"])
            run["stderr"] = artifact(directory, f"session-{index}.stderr", captured["stderr"])
            if captured["outcome"] != "complete" or captured["returncode"] != 0:
                report["problems"].append(
                    "native session failed, timed out, or output was truncated"
                )
                report["status"] = "ERROR"
            if captured["outcome"] != "complete":
                report["problems"].append("usage and final snapshot unavailable after interruption")
                report["sessions"].append(run)
                return report
            try:
                session_events = admin(
                    clean_exec(
                        container,
                        "python3",
                        "-c",
                        READ_SESSION_EVENTS,
                        str(args.max_output_mb * 1024 * 1024),
                    ),
                    env,
                    deadline,
                    limit=args.max_output_mb * 1024 * 1024,
                )
                if secret in session_events:
                    run["session_events_status"] = "WITHHELD"
                    report["problems"].append(
                        "credential appeared in session events; evidence withheld"
                    )
                    report["status"] = "ERROR"
                    report["sessions"].append(run)
                    return report
                run["session_events"] = artifact(
                    directory, f"session-{index}-events.jsonl", session_events
                )
                try:
                    parse_jsonl(session_events)
                    run["activation"] = activation_status(
                        args.host_version,
                        captured["stdout"],
                        session_events,
                        outcome=captured["outcome"],
                        exit_status=captured["returncode"],
                        expected_skill=bundle["SKILL.md"],
                    )
                except (ValueError, TypeError, UnicodeError, json.JSONDecodeError):
                    run["session_events_status"] = "CAPTURED_INVALID"
                    run["activation"] = {
                        "status": "UNVERIFIED",
                        "reason": "invalid or incomplete native event evidence",
                    }
                    report["problems"].append("invalid or incomplete native event evidence")
                    report["status"] = "ERROR"
                else:
                    if run["activation"]["status"] == "VERIFIED":
                        run["session_events_status"] = "CAPTURED_VERIFIED"
                    else:
                        run["session_events_status"] = "CAPTURED_UNVERIFIED"
                        report["problems"].append(
                            f"skill activation unverified: {run['activation']['reason']}"
                        )
            except (ValueError, OSError, subprocess.SubprocessError, RuntimeError):
                run["session_events_status"] = "MISSING_OR_UNSAFE"
                run["activation"] = {
                    "status": "UNVERIFIED",
                    "reason": "native session events are missing or unsafe",
                }
                report["problems"].append("missing or unsafe native session events")
                report["status"] = "ERROR"
            try:
                usage = admin(
                    clean_exec(
                        container, "python3", "-c", READ_USAGE, f"/eval/session-{index}-usage.json"
                    ),
                    env,
                    deadline,
                    limit=1_048_576,
                )
                if secret in usage:
                    run["usage_status"] = "WITHHELD"
                    report["problems"].append("credential appeared in usage; evidence withheld")
                    report["status"] = "ERROR"
                    report["sessions"].append(run)
                    return report
                run["usage"] = artifact(directory, f"session-{index}-usage.json", usage)
                parsed = read_json_bytes(usage)
                if not isinstance(parsed, dict) or not parsed:
                    raise ValueError("empty usage")
                run["usage_status"] = "CAPTURED_UNVERIFIED"
            except (ValueError, OSError, subprocess.SubprocessError, RuntimeError):
                run["usage_status"] = "MISSING_OR_INVALID"
                report["problems"].append("missing or invalid usage evidence")
                report["status"] = "ERROR"
            report["sessions"].append(run)
            if captured["outcome"] != "complete" or captured["returncode"] != 0:
                break
        phase = "final snapshot"
        after, after_raw = exec_snapshot(container, env, deadline)
        report["artifacts"]["after"] = artifact(directory, "after.json", after_raw)
        after_workspace_raw = admin(
            clean_exec(container, "python3", "-c", SKILL_HASHES, CONTAINER_WORKSPACE, "store/.git"),
            env,
            deadline,
        )
        report["artifacts"]["workspace_after"] = artifact(
            directory, "workspace-after.json", after_workspace_raw
        )
        after_workspace = read_json_bytes(after_workspace_raw)
        if before != after:
            report["problems"].append("store or Git refs changed in the read-only profile")
            report["status"] = "ERROR"
        if workspace != after_workspace:
            report["problems"].append("workspace files changed in the read-only profile")
            report["status"] = "ERROR"
        if len(report["sessions"]) != len(trial["prompts"]):
            report["problems"].append("missing sessions")
            report["status"] = "ERROR"
        if not report["problems"] and all(
            session.get("activation", {}).get("status") == "VERIFIED"
            for session in report["sessions"]
        ):
            if len(report["sessions"]) == 1:
                report["observed_activation"] = report["sessions"][0]["activation"]["detected"]
            matched = all(
                session["activation"]["detected"] == trial["should_activate"]
                for session in report["sessions"]
            )
            report["status"] = "PASS" if matched else "FAIL"
            if not matched:
                report["problems"].append("skill activation differed from the expected result")
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError, RuntimeError) as exc:
        report["problems"].append(f"{phase} failed: {type(exc).__name__}")
        report["status"] = "ERROR"
    finally:
        if container and re.fullmatch(r"[0-9a-f]{64}", container):
            try:
                cleanup = subprocess.run(
                    ["docker", "rm", "-f", container],
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                    check=False,
                )
                if cleanup.returncode:
                    report["problems"].append("container cleanup was not confirmed")
                    report["status"] = "ERROR"
            except (OSError, subprocess.SubprocessError):
                report["problems"].append("container cleanup failed")
                report["status"] = "ERROR"
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    root = None
    created = False
    try:
        apply_profiles(args, runner="native")
        trials, dataset_hash, manifest_hash = load_trials(args.manifest)
        validate_options(args, os.environ, trials)
        bundle = skill_bundle()
        archive = fixture_archive(bundle)
        secret = os.environ[args.credential_env].encode()
        root = Path(args.output_dir).resolve()
        if not root.is_relative_to(Path.cwd().resolve()):
            raise ValueError("--output-dir must be inside the current working directory")
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        created = True
        (root / "docker-home").mkdir(mode=0o700)
        env = environment(root, args.docker_host)
        deadline = time.monotonic() + args.run_timeout
        results = []
        for trial in trials:
            if time.monotonic() >= deadline:
                results.append(
                    {
                        "case_id": trial["case_id"],
                        "trial": trial["trial"],
                        "status": "NOT RUN",
                        "problems": ["run timeout"],
                    }
                )
            else:
                results.append(
                    collect_trial(args, trial, root, env, secret, archive, bundle, deadline)
                )
        run_status = activation_run_status(results)
        report = {
            "status": run_status,
            "assessment": "native skill activation only",
            "native_support": "BLOCKED",
            "release": "BLOCKED",
            "activation": (
                "VERIFIED"
                if all(trial["status"] in {"PASS", "FAIL"} for trial in results)
                else (
                    "PARTIALLY VERIFIED"
                    if any(
                        session.get("activation", {}).get("status") == "VERIFIED"
                        for trial in results
                        for session in trial.get("sessions", [])
                    )
                    else "UNVERIFIED"
                )
            ),
            "dataset_sha256": dataset_hash,
            "corpus_source_sha256": verify_corpus()[4],
            "manifest_sha256": manifest_hash,
            "skill_sha256": skill_digest(bundle),
            "fixture_archive_sha256": hashlib.sha256(archive).hexdigest(),
            "image": args.image,
            "host_version_requested": args.host_version,
            "model": args.model,
            "limits": {
                "session_ai_credits": args.session_ai_credits,
                "run_ai_credits": args.run_ai_credits,
                "session_timeout": args.session_timeout,
                "run_timeout": args.run_timeout,
                "cpus": args.cpus,
                "memory_mb": args.memory_mb,
                "max_output_mb": args.max_output_mb,
            },
            "results": results,
        }
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if run_status == "PASS" else 1
    except KeyboardInterrupt:
        if created and root is not None:
            (root / "report.json").write_text(
                json.dumps({"status": "INTERRUPTED", "release": "BLOCKED"}) + "\n", encoding="utf-8"
            )
        print("native trial interrupted; release remains blocked", file=sys.stderr)
        return 130
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        if created and root is not None:
            (root / "report.json").write_text(
                json.dumps({"status": "ERROR", "release": "BLOCKED", "reason": type(exc).__name__})
                + "\n",
                encoding="utf-8",
            )
        print(f"native trial not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
