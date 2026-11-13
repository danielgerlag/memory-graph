"""Sandboxed standalone write/retry execution with withheld setup evidence."""

from __future__ import annotations

import json

from behavior_capture import capture_trace
from behavior_cases import DATASET_SHA256, compile_fixture, prompt_for, retry_request
from behavior_write_capture import capture_full_snapshot
from behavior_write_tool import graph_cli
from inspect_ai import Task
from inspect_ai.agent import as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer
from inspect_ai.solver import solver
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec, sandbox
from inspect_task import SANDBOX_ENV, SCRIPT, STORE, WORK, bundle_hash, checked_exec, skill_files


@solver
def prepare_write(case_id, expected_bundle_hash, tool_timeout):
    async def solve(state, generate):
        fixture = compile_fixture(case_id)
        bundle = skill_files()
        if bundle_hash(bundle) != expected_bundle_hash:
            raise RuntimeError("skill bundle changed after task construction")
        for name, data in bundle.items():
            await sandbox().write_file(f"{WORK}/skill/{name}", data)
        files = fixture["files"]
        if fixture["spec"]["kind"] == "retry":
            files = {path: value for path, value in files.items() if not path.endswith(".md")}
        for name, content in files.items():
            await sandbox().write_file(f"{STORE}/{name}", content)
        await checked_exec(
            [
                "python3",
                "-c",
                "import sys, jsonschema, filelock, ruamel.yaml; assert sys.version_info >= (3, 12)",
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
        bootstrap = None
        if fixture["spec"]["kind"] == "retry":
            initial = await capture_full_snapshot(tool_timeout)
            args = ["create", "--operation", fixture["operation_id"], "--input", "-"]
            input_text = json.dumps(retry_request(case_id), sort_keys=True)
            result = await sandbox().exec(
                ["python3", SCRIPT, "--root", STORE, *args],
                input=input_text,
                env=SANDBOX_ENV,
                timeout=tool_timeout,
                timeout_retry=False,
            )
            if not result.success:
                raise RuntimeError("retry seed operation failed")
            bootstrap = {
                "before": initial,
                "action": {
                    "args": args,
                    "input_text": input_text,
                    "status": "success",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            }
        state.store.set("behavior_before", await capture_full_snapshot(tool_timeout))
        state.store.set("behavior_bootstrap", bootstrap)
        state.store.set("memory_calls", [])
        return state

    return solve


@solver
def capture_write_after(tool_timeout):
    async def solve(state, generate):
        state.store.set(
            "behavior_artifacts",
            {
                "before": state.store.get("behavior_before"),
                "after": await capture_full_snapshot(tool_timeout),
                "actions": capture_trace(state),
                "bootstrap": state.store.get("behavior_bootstrap"),
            },
        )
        return state

    return solve


@scorer(metrics=[accuracy()])
def write_score(case_id):
    async def score(state, target):
        from behavior_grader import grade

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
    fixture = compile_fixture(args.case_id)
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
            prepare_write(args.case_id, bundle_hash(bundle), args.tool_timeout),
            as_solver(
                react(
                    prompt=prompt,
                    tools=[
                        graph_cli(
                            args.tool_limit,
                            args.tool_timeout,
                            clock=fixture["case"]["clock"]
                            if fixture["spec"]["kind"] in {"timed-read", "timed-review"}
                            else None,
                        )
                    ],
                    attempts=1,
                    submit=False,
                    retry_refusals=0,
                    compaction=None,
                )
            ),
            capture_write_after(args.tool_timeout),
        ],
        scorer=write_score(args.case_id),
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
            "clock_policy": (
                "injected service clock"
                if fixture["spec"]["kind"] in {"timed-read", "timed-review"}
                else "runtime clock; retry expiry is absolute"
            ),
        },
    )
