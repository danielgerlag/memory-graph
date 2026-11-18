"""Inspect task implementation. Imported only by the explicit opt-in runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from controlled_core import (
    FIXTURE_SHA256,
    MUTATIONS,
    READS,
    command_arguments,
    effective_budget,
    fixture_files,
    grade,
)
from inspect_ai import Task
from inspect_ai.agent import as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer
from inspect_ai.solver import solver
from inspect_ai.tool import ToolError, tool
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec, sandbox, store

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "memory-graph"
SKILL_INSTRUCTIONS = None
WORK = "/workspace"
STORE = f"{WORK}/store"
SCRIPT = f"{WORK}/skill/scripts/memory_graph.py"
SANDBOX_ENV = {
    "HOME": WORK,
    "TMPDIR": WORK,
    "PYTHONDONTWRITEBYTECODE": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "Synthetic Eval",
    "GIT_AUTHOR_EMAIL": "eval@localhost",
    "GIT_COMMITTER_NAME": "Synthetic Eval",
    "GIT_COMMITTER_EMAIL": "eval@localhost",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}

# Executed only by the sandbox API. Includes ignored and untracked store files,
# except the disposable parse cache and its SQLite sidecars.
SNAPSHOT = r"""
import hashlib, json, os, sys
root = sys.argv[1]
files = {}
ignored = {os.path.join(".memory-graph", "parse-cache.sqlite3" + suffix)
           for suffix in ("", "-journal", "-wal", "-shm")}
for base, dirs, names in os.walk(root, followlinks=False):
    for name in dirs + names:
        path = os.path.join(base, name)
        if os.path.islink(path):
            raise RuntimeError("symlink in store")
    dirs[:] = sorted(d for d in dirs if not (base == root and d == ".git"))
    for name in sorted(names):
        path = os.path.join(base, name)
        if not os.path.isfile(path):
            raise RuntimeError("non-regular file in store")
        relative = os.path.relpath(path, root)
        if relative in ignored:
            continue
        with open(path, "rb") as stream:
            files[relative] = hashlib.sha256(stream.read()).hexdigest()
print(json.dumps(files, sort_keys=True))
"""


def skill_files():
    """Read only the explicit skill bundle, never a venv, HOME or local stores."""
    paths = [SKILL / "SKILL.md", SKILL / "pyproject.toml", SKILL / "uv.lock"]
    for directory, pattern in (("scripts", "*.py"), ("references", "*.md"), ("assets", "*.json")):
        paths.extend((SKILL / directory).rglob(pattern))
    result = {}
    for path in sorted(paths):
        if "__pycache__" in path.parts or path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid skill file: {path}")
        if not path.resolve().is_relative_to(SKILL.resolve()):
            raise ValueError("skill file escapes bundle")
        data = path.read_bytes()
        if len(data) > 1024 * 1024:
            raise ValueError(f"oversized skill file: {path}")
        result[path.relative_to(SKILL).as_posix()] = data
    if SKILL_INSTRUCTIONS is not None:
        from profiles import _read_profile

        result["SKILL.md"] = _read_profile(SKILL_INSTRUCTIONS).encode("utf-8")
    required = {"SKILL.md", "scripts/memory_graph.py", "assets/node.schema.json"}
    if not required.issubset(result):
        raise ValueError("incomplete skill bundle")
    return result


def bundle_hash(files):
    digest = hashlib.sha256()
    for name, data in sorted(files.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


async def checked_exec(command, *, timeout):
    result = await sandbox().exec(command, env=SANDBOX_ENV, timeout=timeout, timeout_retry=False)
    if not result.success:
        raise RuntimeError(
            f"sandbox setup/snapshot command failed: {command[0]}: {result.stderr[:1000]}"
        )
    return result.stdout


async def snapshot(timeout):
    files = json.loads(await checked_exec(["python3", "-c", SNAPSHOT, STORE], timeout=timeout))
    base = ["git", "-C", STORE]
    main = (await checked_exec(base + ["rev-parse", "refs/heads/main"], timeout=timeout)).strip()
    refs = (
        await checked_exec(
            base + ["for-each-ref", "--format=%(refname) %(objectname)"], timeout=timeout
        )
    ).splitlines()
    if not isinstance(files, dict) or not main or not refs:
        raise RuntimeError("incomplete sandbox snapshot")
    return {"main": main, "refs": refs, "files": files}


@solver
def prepare(expected_bundle_hash, tool_timeout):
    async def solve(state, generate):
        bundle = skill_files()
        if bundle_hash(bundle) != expected_bundle_hash:
            raise RuntimeError("skill bundle changed after task construction")
        for name, data in bundle.items():
            await sandbox().write_file(f"{WORK}/skill/{name}", data)
        for name, data in fixture_files().items():
            await sandbox().write_file(f"{STORE}/{name}", data)
        await checked_exec(
            [
                "python3",
                "-c",
                (
                    "import sys, jsonschema, filelock, ruamel.yaml; "
                    "assert sys.version_info >= (3, 12)"
                ),
            ],
            timeout=tool_timeout,
        )
        base = ["git", "-C", STORE, "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false"]
        await checked_exec(
            base + ["init", "--initial-branch=main", "--template="], timeout=tool_timeout
        )
        await checked_exec(base + ["add", "--all"], timeout=tool_timeout)
        await checked_exec(
            base + ["commit", "-m", "Synthetic evaluation fixture"], timeout=tool_timeout
        )
        state.store.set("initial_snapshot", await snapshot(tool_timeout))
        state.store.set("memory_calls", [])
        return state

    return solve


@tool
def memory_cli(tool_limit: int, tool_timeout: int):
    async def execute(args: list[str], input_text: str = "") -> str:
        """Run the bundled memory CLI against the selected synthetic store.

        Args:
            args: CLI arguments, optionally starting with --json, then an operation.
            input_text: Optional JSON request for a mutation using --input -.
        """
        calls = store().get("memory_calls", [])
        if len(calls) >= tool_limit:
            calls.append({"args": ["tool_limit"], "status": "invalid"})
            store().set("memory_calls", calls)
            raise ToolError("Tool call limit reached")
        command = command_arguments(args) if isinstance(args, list) else []
        valid = (
            isinstance(args, list)
            and 1 <= len(args) <= 24
            and all(
                isinstance(item, str) and len(item) <= 1024 and "\x00" not in item for item in args
            )
            and bool(command)
            and command[0] in READS | MUTATIONS
            and not any(item == "--root" or item.startswith("--root=") for item in args)
            and isinstance(input_text, str)
            and len(input_text) <= 65536
        )
        record = {"args": args, "status": "invalid"}
        calls.append(record)
        store().set("memory_calls", calls)
        if not valid:
            raise ToolError("Invalid or disallowed CLI arguments")
        try:
            result = await sandbox().exec(
                ["python3", SCRIPT, "--root", STORE, *args],
                input=input_text,
                env=SANDBOX_ENV,
                timeout=tool_timeout,
                timeout_retry=False,
            )
        except TimeoutError:
            record["error"] = "timeout"
            store().set("memory_calls", calls)
            raise ToolError("Command timed out") from None
        record.update(
            status="success" if result.success else "error",
            stdout=result.stdout,
            stderr=result.stderr,
        )
        store().set("memory_calls", calls)
        if len(result.stdout) > 100_000 or len(result.stderr) > 100_000:
            record["status"] = "invalid"
            store().set("memory_calls", calls)
            raise ToolError("Command output exceeded evaluation limit")
        if not result.success:
            raise ToolError(result.stderr[:4000] or "Command failed")
        return result.stdout

    return execute


@scorer(metrics=[accuracy()])
def controlled_score(tool_timeout: int):
    async def score(state, target):
        before = state.store.get("initial_snapshot")
        after = await snapshot(tool_timeout)
        calls = state.store.get("memory_calls")
        result = grade(state.output.completion, before, after, calls)
        return Score(
            value=CORRECT if result["success"] else INCORRECT,
            answer=state.output.completion,
            explanation="; ".join(result["failures"]) or "passed",
            metadata={"before": before, "after": after, "calls": calls},
        )

    return score


def build_task(args):
    bundle = skill_files()
    instructions = bundle["SKILL.md"].decode("utf-8")
    prompt = (
        "You have the following explicitly loaded skill instructions. The memory_cli tool runs "
        "the bundled CLI with --root already supplied; pass the remaining arguments only.\n\n"
        + instructions
    )
    sample = Sample(
        id="scope-search",
        input="In the orchard scope, where is the ladder? Do not change the store. "
        "Use the memory CLI and respond with only JSON containing node_ids (array), "
        "facts (array of exact labels), store_id, and commit from the retrieved snapshot.",
    )
    config = ComposeConfig(
        services={
            "default": ComposeService(
                image=args.image,
                init=True,
                entrypoint=["tail", "-f", "/dev/null"],
                command=[],
                user="1000:1000",
                working_dir=WORK,
                environment={"HOME": WORK},
                network_mode="none",
                mem_limit=f"{args.memory_mb}m",
                cpus=args.cpus,
                pids_limit=128,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                pull_policy="never",
                read_only=True,
                tmpfs=["/workspace:rw,mode=1777,size=67108864"],
                shm_size="32m",
            )
        }
    )
    task = Task(
        dataset=[sample],
        solver=[
            prepare(bundle_hash(bundle), args.tool_timeout),
            as_solver(
                react(
                    prompt=prompt,
                    tools=[memory_cli(args.tool_limit, args.tool_timeout)],
                    attempts=1,
                    submit=False,
                    retry_refusals=0,
                    compaction=None,
                )
            ),
        ],
        scorer=controlled_score(args.tool_timeout),
        sandbox=SandboxEnvironmentSpec("docker", config),
        time_limit=args.time_limit,
        turn_limit=args.turn_limit,
        token_limit=args.token_limit,
        cost_limit=effective_budget(args),
        message_limit=2 * args.tool_limit + 2 * args.turn_limit + 10,
        metadata={
            "fixture_sha256": FIXTURE_SHA256,
            "skill_sha256": bundle_hash(bundle),
            "case": "scope-search",
            "condition": "explicitly-loaded-skill",
        },
    )
    return task
