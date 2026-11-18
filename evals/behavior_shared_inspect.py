"""Controlled read-only shared-store trials against a hook-free local remote."""

from __future__ import annotations

import json

from behavior_capture import capture_trace
from behavior_cases import (
    DATASET_SHA256,
    acceptance_request,
    compile_fixture,
    prompt_for,
    semantic_id,
)
from behavior_shared_capture import (
    OFFLINE_REMOTE,
    REMOTE,
    capture_fetch_snapshot,
    capture_offline_snapshot,
    capture_shared_snapshot,
)
from behavior_write_tool import graph_cli
from controlled_core import effective_budget
from inspect_ai import Task
from inspect_ai.agent import as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer
from inspect_ai.solver import solver
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec, sandbox
from inspect_task import SANDBOX_ENV, SCRIPT, STORE, WORK, bundle_hash, checked_exec, skill_files

ACTOR = f"{WORK}/acceptance-actor"


@solver
def prepare_shared(case_id, expected_bundle_hash, tool_timeout):
    async def solve(state, generate):
        fixture = compile_fixture(case_id)
        if case_id not in {"shared-read", "offline-stale", "accepted-after-fetch"}:
            raise ValueError("unsupported shared-store adapter")
        bundle = skill_files()
        if bundle_hash(bundle) != expected_bundle_hash:
            raise RuntimeError("skill bundle changed after task construction")
        for name, data in bundle.items():
            await sandbox().write_file(f"{WORK}/skill/{name}", data)
        for name, content in fixture["files"].items():
            await sandbox().write_file(f"{STORE}/{name}", content)
        await checked_exec(
            [
                "python3",
                "-c",
                "import sys, jsonschema, filelock, ruamel.yaml; assert sys.version_info >= (3, 12)",
            ],
            timeout=tool_timeout,
        )
        base = [
            "git",
            "-C",
            STORE,
            "-c",
            "commit.gpgSign=false",
            "-c",
            "protocol.file.allow=always",
        ]
        await checked_exec(
            base + ["init", "--initial-branch=main", "--template="], timeout=tool_timeout
        )
        await checked_exec(base + ["add", "--all"], timeout=tool_timeout)
        await checked_exec(
            base + ["commit", "-m", "Synthetic shared fixture"], timeout=tool_timeout
        )
        base_commit = (
            await checked_exec(base + ["rev-parse", "refs/heads/main"], timeout=tool_timeout)
        ).strip()
        remote_source = STORE
        if case_id == "accepted-after-fetch":
            await checked_exec(
                [
                    "git",
                    "-c",
                    "protocol.file.allow=always",
                    "clone",
                    "--no-local",
                    "--template=",
                    STORE,
                    ACTOR,
                ],
                timeout=tool_timeout,
            )
            operation_id = semantic_id(case_id, "$acceptance")
            request = json.dumps(acceptance_request(case_id), sort_keys=True)
            actor_args = ["create", "--operation", operation_id, "--input", "-"]
            accepted = await sandbox().exec(
                ["python3", SCRIPT, "--root", ACTOR, *actor_args],
                input=request,
                env=SANDBOX_ENV,
                timeout=tool_timeout,
                timeout_retry=False,
            )
            if not accepted.success:
                raise RuntimeError("could not prepare the accepted fixture commit")
            state.store.set(
                "shared_acceptance",
                {
                    "args": actor_args,
                    "input_text": request,
                    "status": "success",
                    "stdout": accepted.stdout,
                    "stderr": accepted.stderr,
                },
            )
            remote_source = ACTOR
        await checked_exec(
            [
                "git",
                "-c",
                "protocol.file.allow=always",
                "clone",
                "--bare",
                "--no-local",
                "--template=",
                remote_source,
                REMOTE,
            ],
            timeout=tool_timeout,
        )
        await checked_exec(base + ["remote", "add", "origin", REMOTE], timeout=tool_timeout)
        initial_ref = base_commit if case_id == "accepted-after-fetch" else "refs/heads/main"
        await checked_exec(
            base + ["fetch", "--no-tags", "origin", f"{initial_ref}:refs/remotes/origin/main"],
            timeout=tool_timeout,
        )
        args = ["attach", "--remote", "origin", "--mode", "read"]
        attached = await sandbox().exec(
            ["python3", SCRIPT, "--root", STORE, *args],
            env={**SANDBOX_ENV, "GIT_ALLOW_PROTOCOL": "file"},
            timeout=tool_timeout,
            timeout_retry=False,
        )
        if not attached.success:
            raise RuntimeError("local read-only attachment failed")
        state.store.set(
            "shared_setup", {"args": args, "stdout": attached.stdout, "stderr": attached.stderr}
        )
        if case_id == "offline-stale":
            await checked_exec(["mv", REMOTE, OFFLINE_REMOTE], timeout=tool_timeout)
            before = await capture_offline_snapshot(tool_timeout)
        elif case_id == "accepted-after-fetch":
            before = await capture_fetch_snapshot(tool_timeout)
        else:
            before = await capture_shared_snapshot(tool_timeout)
        state.store.set("shared_before", before)
        state.store.set("memory_calls", [])
        return state

    return solve


@solver
def capture_shared_after(case_id, tool_timeout):
    async def solve(state, generate):
        if case_id == "offline-stale":
            after = await capture_offline_snapshot(tool_timeout)
        elif case_id == "accepted-after-fetch":
            after = await capture_fetch_snapshot(tool_timeout)
        elif case_id == "shared-read":
            after = await capture_shared_snapshot(tool_timeout)
        else:
            raise ValueError("unsupported shared-store adapter")
        artifacts = {
            "before": state.store.get("shared_before"),
            "after": after,
            "actions": capture_trace(state),
            "setup": state.store.get("shared_setup"),
        }
        if case_id == "accepted-after-fetch":
            artifacts["acceptance"] = state.store.get("shared_acceptance")
        state.store.set("behavior_artifacts", artifacts)
        return state

    return solve


@scorer(metrics=[accuracy()])
def shared_score(case_id):
    async def score(state, target):
        from graders.behavior_grader import grade

        artifacts = state.store.get("behavior_artifacts")
        result = grade(case_id, state.output.completion, artifacts)
        return Score(
            value=CORRECT if result["success"] else INCORRECT,
            answer=state.output.completion,
            explanation="; ".join(result["failures"]) or "passed",
            metadata={"artifacts": artifacts, "failures": result["failures"], "observations": []},
        )

    return score


def build_task(args):
    compile_fixture(args.case_id)
    if args.case_id not in {"shared-read", "offline-stale", "accepted-after-fetch"}:
        raise ValueError("unsupported shared-store adapter")
    bundle = skill_files()
    prompt = (
        "You have the following explicitly loaded skill instructions. The graph_cli tool runs "
        "the bundled CLI with --root already supplied; pass the remaining arguments only.\n\n"
        + bundle["SKILL.md"].decode("utf-8")
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
    return Task(
        dataset=[Sample(id=args.case_id, input=prompt_for(args.case_id, 0))],
        solver=[
            prepare_shared(args.case_id, bundle_hash(bundle), args.tool_timeout),
            as_solver(
                react(
                    prompt=prompt,
                    tools=[graph_cli(args.tool_limit, args.tool_timeout, local_remote=True)],
                    attempts=1,
                    submit=False,
                    retry_refusals=0,
                    compaction=None,
                )
            ),
            capture_shared_after(args.case_id, args.tool_timeout),
        ],
        scorer=shared_score(args.case_id),
        sandbox=SandboxEnvironmentSpec("docker", config),
        time_limit=args.time_limit,
        turn_limit=args.turn_limit,
        token_limit=args.token_limit,
        cost_limit=effective_budget(args),
        message_limit=2 * args.tool_limit + 2 * args.turn_limit + 10,
        metadata={
            "dataset_sha256": DATASET_SHA256,
            "skill_sha256": bundle_hash(bundle),
            "case": args.case_id,
            "trial": args.trial,
            "condition": "explicitly-loaded-skill",
            "profile": "local-bare-remote; not a provider authorization test",
        },
    )
