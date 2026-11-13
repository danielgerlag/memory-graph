"""A bounded sandbox-only CLI tool with complete mutation request traces."""

from __future__ import annotations

from controlled_core import MUTATIONS, READS
from inspect_ai.tool import ToolError, tool
from inspect_ai.util import sandbox, store
from inspect_task import SANDBOX_ENV, SCRIPT, STORE

CLOCK_BOOTSTRAP = r"""
import datetime, sys
scripts, fixed = sys.argv[1:3]
sys.path.insert(0, scripts)
from memory_graph import cli, service
instant = datetime.datetime.fromisoformat(fixed.replace("Z", "+00:00"))
service.now_utc = lambda: instant
sys.argv = ["memory_graph.py", *sys.argv[3:]]
raise SystemExit(cli.main())
"""


@tool
def graph_cli(
    tool_limit: int,
    tool_timeout: int,
    local_remote: bool = False,
    clock: str | None = None,
    allow_proposal: bool = False,
):
    async def execute(args: list[str], input_text: str = "") -> str:
        """Run the bundled memory CLI against the selected synthetic store.

        Args:
            args: CLI arguments beginning with an operation such as find, create, or operation.
            input_text: Optional JSON request for an operation using --input -.
        """
        calls = store().get("memory_calls", [])
        if len(calls) >= tool_limit:
            calls.append({"args": ["tool_limit"], "status": "invalid"})
            store().set("memory_calls", calls)
            raise ToolError("Tool call limit reached")
        valid = (
            isinstance(args, list)
            and 1 <= len(args) <= 24
            and all(
                isinstance(item, str) and len(item) <= 1024 and "\x00" not in item for item in args
            )
            and (
                args[0] in READS | MUTATIONS
                or (args[0] == "operation" and args[1:2] == ["status"])
                or (local_remote and args in (["status"], ["fetch"]))
                or (local_remote and allow_proposal and args[0] == "publish")
            )
            and not any(item == "--root" or item.startswith("--root=") for item in args)
            and isinstance(input_text, str)
            and len(input_text) <= 65536
        )
        if valid and args[0] in MUTATIONS:
            valid = args.count("--input") == 1 and args.index("--input") + 1 < len(args)
            valid = valid and args[args.index("--input") + 1] == "-"
        record = {"args": args, "input_text": input_text, "status": "invalid"}
        calls.append(record)
        store().set("memory_calls", calls)
        if not valid:
            raise ToolError("Invalid or disallowed CLI arguments")
        command = (
            [
                "python3",
                "-c",
                CLOCK_BOOTSTRAP,
                "/workspace/skill/scripts",
                clock,
                "--root",
                STORE,
                *args,
            ]
            if clock is not None
            else ["python3", SCRIPT, "--root", STORE, *args]
        )
        try:
            result = await sandbox().exec(
                command,
                input=input_text,
                env={**SANDBOX_ENV, **({"GIT_ALLOW_PROTOCOL": "file"} if local_remote else {})},
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
            raise ToolError(result.stderr[:4000] or result.stdout[:4000] or "Command failed")
        return result.stdout

    return execute
