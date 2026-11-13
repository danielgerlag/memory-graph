"""A bounded sandbox-only current-main integration check for the conflict case."""

from __future__ import annotations

from behavior_concurrent_capture import VALIDATOR
from inspect_ai.tool import ToolError, tool
from inspect_ai.util import sandbox, store
from inspect_task import SANDBOX_ENV, SCRIPT, STORE


@tool
def conflict_check(tool_limit: int, tool_timeout: int):
    async def execute() -> str:
        """Check the prepared integration candidate against the fetched main commit."""
        calls = store().get("memory_calls", [])
        if len(calls) >= tool_limit:
            calls.append({"args": ["tool_limit"], "status": "invalid"})
            store().set("memory_calls", calls)
            raise ToolError("Tool call limit reached")
        evidence = store().get("conflict_setup")
        record = {"args": ["check-merge"], "status": "invalid"}
        calls.append(record)
        store().set("memory_calls", calls)
        if not isinstance(evidence, dict):
            raise ToolError("Missing integration candidate")
        tracking = await sandbox().exec(
            ["git", "-C", STORE, "rev-parse", "refs/remotes/origin/main"],
            env=SANDBOX_ENV,
            timeout=tool_timeout,
            timeout_retry=False,
        )
        if not tracking.success or tracking.stdout.strip() != evidence["winner_commit"]:
            raise ToolError("Fetch the current main before checking the candidate")
        args = [
            "check-merge",
            "--base",
            tracking.stdout.strip(),
            "--candidate",
            evidence["candidate"],
        ]
        result = await sandbox().exec(
            ["python3", SCRIPT, "--root", VALIDATOR, *args],
            env={**SANDBOX_ENV, "GIT_ALLOW_PROTOCOL": "file"},
            timeout=tool_timeout,
            timeout_retry=False,
        )
        record.update(
            args=args,
            status="success" if result.success else "error",
            stdout=result.stdout,
            stderr=result.stderr,
        )
        store().set("memory_calls", calls)
        if len(result.stdout) > 100_000 or len(result.stderr) > 100_000:
            record["status"] = "invalid"
            store().set("memory_calls", calls)
            raise ToolError("Check output exceeded evaluation limit")
        return result.stdout if result.success else result.stderr or result.stdout or "Check failed"

    return execute
