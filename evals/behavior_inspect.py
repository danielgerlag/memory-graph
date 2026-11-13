"""Controlled Inspect execution; artifact capture and scoring are separate modules."""

from __future__ import annotations

from behavior_capture import capture_snapshot, capture_trace, capture_workspace
from behavior_cases import (
    DATASET_SHA256,
    SHARED_KINDS,
    SPECIAL_KINDS,
    WRITE_KINDS,
    compile_fixture,
    prompt_for,
)
from behavior_grader import grade
from inspect_ai import Task
from inspect_ai.agent import as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer
from inspect_ai.solver import solver
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec, sandbox
from inspect_task import STORE, WORK, bundle_hash, checked_exec, memory_cli, skill_files


@solver
def prepare_case(case_id, expected_bundle_hash, tool_timeout):
    async def solve(state, generate):
        fixture = compile_fixture(case_id)
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
            base + ["commit", "-m", "Synthetic behavior evaluation fixture"], timeout=tool_timeout
        )
        state.store.set("behavior_before", await capture_snapshot(tool_timeout))
        if fixture["spec"]["kind"] == "isolation":
            state.store.set("isolation_before", await capture_workspace(tool_timeout))
        state.store.set("memory_calls", [])
        return state

    return solve


@solver
def fresh_context(case_id, tool_timeout):
    async def solve(state, generate):
        calls = capture_trace(state)
        state.store.set(
            "first_context",
            {
                "after": await capture_snapshot(tool_timeout),
                "actions": calls,
                "answer": state.output.completion,
                "stop_reason": state.output.stop_reason,
                "messages": [message.model_dump(mode="json") for message in state.messages],
            },
        )
        state.store.set("first_action_count", len(calls))
        state.messages = [ChatMessageUser(content=prompt_for(case_id, 1))]
        state.output = ModelOutput()
        state.store.set(
            "second_start_messages", [message.model_dump(mode="json") for message in state.messages]
        )
        return state

    return solve


@solver
def capture_after(case_id, tool_timeout):
    async def solve(state, generate):
        calls = capture_trace(state)
        first = state.store.get("first_context")
        first_count = state.store.get("first_action_count", 0)
        if type(first_count) is not int or not 0 <= first_count <= len(calls):
            raise ValueError("invalid multi-context action boundary")
        artifacts = {
            "before": state.store.get("behavior_before"),
            "after": await capture_snapshot(tool_timeout),
            "actions": calls[first_count:],
        }
        if compile_fixture(case_id)["spec"]["kind"] == "multi":
            artifacts["first_context"] = first
            artifacts["second_start_messages"] = state.store.get("second_start_messages")
        if compile_fixture(case_id)["spec"]["kind"] == "isolation":
            artifacts["isolation"] = {
                "before": state.store.get("isolation_before"),
                "after": await capture_workspace(tool_timeout),
            }
        state.store.set(
            "behavior_artifacts",
            artifacts,
        )
        return state

    return solve


@scorer(metrics=[accuracy()])
def behavior_score(case_id):
    async def score(state, target):
        artifacts = state.store.get("behavior_artifacts")
        result = grade(case_id, state.output.completion, artifacts)
        return Score(
            value=CORRECT if result["success"] else INCORRECT,
            answer=state.output.completion,
            explanation="; ".join(result["failures"] or result["observations"]) or "passed",
            metadata={
                "artifacts": artifacts,
                "failures": result["failures"],
                "observations": result["observations"],
            },
        )

    return score


def build_task(args):
    fixture = compile_fixture(args.case_id)
    if fixture["spec"]["kind"] in WRITE_KINDS | SPECIAL_KINDS:
        from behavior_write_inspect import build_task as build_write_task

        return build_write_task(args)
    if fixture["spec"]["kind"] in SHARED_KINDS:
        from behavior_shared_inspect import build_task as build_shared_task

        return build_shared_task(args)
    bundle = skill_files()
    instructions = bundle["SKILL.md"].decode("utf-8")
    prompt = (
        "You have the following explicitly loaded skill instructions. The memory_cli tool runs "
        "the bundled CLI with --root already supplied; pass the remaining arguments only.\n\n"
        + instructions
    )
    sample = Sample(
        id=args.case_id,
        input=prompt_for(args.case_id, 0),
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

    def agent_solver():
        return as_solver(
            react(
                prompt=prompt,
                tools=[memory_cli(args.tool_limit, args.tool_timeout)],
                attempts=1,
                submit=False,
                retry_refusals=0,
                compaction=None,
            )
        )

    solvers = [prepare_case(args.case_id, bundle_hash(bundle), args.tool_timeout), agent_solver()]
    if fixture["spec"]["kind"] == "multi":
        solvers.extend([fresh_context(args.case_id, args.tool_timeout), agent_solver()])
    solvers.append(capture_after(args.case_id, args.tool_timeout))
    return Task(
        dataset=[sample],
        solver=solvers,
        scorer=behavior_score(args.case_id),
        sandbox=SandboxEnvironmentSpec("docker", config),
        time_limit=args.time_limit,
        turn_limit=args.turn_limit,
        token_limit=args.token_limit,
        cost_limit=args.budget_usd,
        message_limit=2 * args.tool_limit + 2 * args.turn_limit + 10,
        metadata={
            "dataset_sha256": DATASET_SHA256,
            "skill_sha256": bundle_hash(bundle),
            "case": args.case_id,
            "trial": args.trial,
            "condition": "explicitly-loaded-skill",
            "clock_policy": "runtime clock; fixture has no time boundary",
        },
    )
