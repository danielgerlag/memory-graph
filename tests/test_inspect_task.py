"""Offline tests only; no model, Docker container, or provider call is launched."""

import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import controlled_core as core
import run_inspect


def options():
    return SimpleNamespace(
        run=True,
        model="openai/example",
        credential_env="OPENAI_API_KEY",
        image="example/image@sha256:" + "a" * 64,
        log_dir="evals/example-output",
        docker_host=None,
        budget_usd=1.0,
        input_price=1.0,
        output_price=2.0,
        cpus=1.0,
        time_limit=60,
        wall_timeout=90,
        tool_timeout=10,
        turn_limit=5,
        tool_limit=5,
        token_limit=4000,
        max_tokens=500,
        memory_mb=512,
    )


def evidence():
    commit = "a" * 40
    files = {
        name: hashlib.sha256(value.encode()).hexdigest()
        for name, value in core.fixture_files().items()
    }
    snapshot = {"main": commit, "refs": [f"refs/heads/main {commit}"], "files": files}
    answer = json.dumps(
        {
            "node_ids": [core.EXPECTED_ID],
            "facts": ["ladder is in shed"],
            "store_id": core.STORE_ID,
            "commit": commit,
        }
    )
    output = {
        "snapshot": {"store_id": core.STORE_ID, "commit": commit},
        "truncated": False,
        "results": [{"id": core.EXPECTED_ID, "label": "ladder is in shed", "scope": "orchard"}],
    }
    calls = [
        {
            "args": ["find", "ladder", "--scope", "orchard"],
            "status": "success",
            "stdout": json.dumps(output),
            "stderr": "",
        }
    ]
    return answer, snapshot, calls


class ControlledCoreTests(unittest.TestCase):
    def test_synthetic_fixture_parses_with_current_skill(self):
        sys.path.insert(0, str(ROOT / "memory-graph/scripts"))
        try:
            from memory_graph import schema
        except ImportError as exc:
            self.skipTest(f"skill dependencies unavailable: {exc}")
        nodes = {}
        for name, value in core.fixture_files().items():
            if name.endswith(".md"):
                node, _ = schema.parse_node(value.encode(), name)
                nodes[node["id"]] = node
        schema.validate_graph(nodes)
        self.assertEqual(set(nodes), {core.EXPECTED_ID, core.FORBIDDEN_ID})

    def test_explicit_and_bounded_options_required(self):
        opts = options()
        env = {"OPENAI_API_KEY": "synthetic-test-value"}
        core.validate_options(opts, env)
        for field, value in [
            ("run", False),
            ("model", None),
            ("model", "openai/"),
            ("credential_env", None),
            ("image", "example:latest"),
            ("budget_usd", 0),
            ("budget_usd", float("nan")),
            ("time_limit", 0),
            ("wall_timeout", 0),
            ("tool_limit", 0),
            ("token_limit", 0),
            ("cpus", 100),
            ("memory_mb", 100000),
            ("docker_host", "tcp://example"),
        ]:
            with self.subTest(field=field, value=value):
                bad = options()
                setattr(bad, field, value)
                with self.assertRaises(ValueError):
                    core.validate_options(bad, env)
        with self.assertRaises(ValueError):
            core.validate_options(opts, {})

    def test_no_arguments_are_offline_and_do_not_need_inspect(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "evals/run_inspect.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --run", result.stderr)

    def test_only_selected_credential_and_clean_home_are_exposed(self):
        opts = options()
        original = {
            "PATH": "/bin",
            "HOME": "/private-home",
            "OPENAI_API_KEY": "test-key",
            "ANTHROPIC_API_KEY": "other-key",
            "AWS_SECRET_ACCESS_KEY": "other-secret",
        }
        destination = ROOT / "evals" / "explicit-run"
        with patch.dict(os.environ, original, clear=True), patch.object(Path, "mkdir"):
            run_inspect.isolate_environment(opts, destination)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "test-key")
            self.assertEqual(os.environ["HOME"], str(destination / "home"))
            self.assertNotIn("ANTHROPIC_API_KEY", os.environ)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", os.environ)

    def test_valid_trace_and_answer(self):
        answer, before, calls = evidence()
        self.assertTrue(core.grade(answer, before, before, calls)["success"])

    def test_missing_answers_and_missing_tool_or_snapshot_fail(self):
        answer, before, calls = evidence()
        self.assertFalse(core.grade("", before, before, calls)["success"])
        self.assertFalse(core.grade(answer, before, before, [])["success"])
        self.assertFalse(core.grade(answer, None, None, calls)["success"])
        self.assertFalse(core.grade(answer, before, None, calls)["success"])

    def test_snapshot_mutation_and_wrong_fixture_fail(self):
        answer, before, calls = evidence()
        changed = json.loads(json.dumps(before))
        changed["files"]["unexpected"] = "b" * 64
        self.assertFalse(core.grade(answer, before, changed, calls)["success"])
        self.assertFalse(core.grade(answer, changed, changed, calls)["success"])
        changed = json.loads(json.dumps(before))
        changed["refs"] = []
        self.assertFalse(core.grade(answer, changed, changed, calls)["success"])

    def test_scope_leak_mutation_and_truncation_fail(self):
        answer, before, calls = evidence()
        mutations = calls + [{"args": ["create"], "status": "error"}]
        self.assertFalse(core.grade(answer, before, before, mutations)["success"])
        leaked = json.loads(json.dumps(calls))
        leaked[0]["stdout"] += core.FORBIDDEN_ID
        self.assertFalse(core.grade(answer, before, before, leaked)["success"])
        truncated = json.loads(json.dumps(calls))
        response = json.loads(truncated[0]["stdout"])
        response["truncated"] = True
        truncated[0]["stdout"] = json.dumps(response)
        self.assertFalse(core.grade(answer, before, before, truncated)["success"])

    def test_wrong_commit_and_wrong_answer_fail(self):
        answer, before, calls = evidence()
        response = json.loads(calls[0]["stdout"])
        response["snapshot"]["commit"] = "b" * 40
        calls[0]["stdout"] = json.dumps(response)
        self.assertFalse(core.grade(answer, before, before, calls)["success"])
        answer, before, calls = evidence()
        altered = json.loads(answer)
        altered["facts"] = ["ladder is in lab"]
        self.assertFalse(core.grade(json.dumps(altered), before, before, calls)["success"])


try:
    import inspect_task
    from inspect_ai.util import ExecResult
except ImportError:
    inspect_task = None


@unittest.skipIf(inspect_task is None, "optional eval dependencies not installed")
class InspectApiTests(unittest.IsolatedAsyncioTestCase):
    def snapshot_result(self, directory):
        return subprocess.run(
            [sys.executable, "-c", inspect_task.SNAPSHOT, str(directory)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_disposable_cache_is_ignored_but_journals_refs_and_mutations_are_not(self):
        answer, before, calls = evidence()
        fixture = ROOT / "tests/fixtures/inspect_snapshot"
        baseline = self.snapshot_result(fixture / "before")
        with_cache = self.snapshot_result(fixture / "cache-only")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(with_cache.returncode, 0, with_cache.stderr)
        baseline_files = json.loads(baseline.stdout)
        cache_files = json.loads(with_cache.stdout)
        self.assertEqual(cache_files, baseline_files)
        added_files = {
            path: value for path, value in cache_files.items() if path not in baseline_files
        }
        after = {**before, "files": {**before["files"], **added_files}}
        self.assertEqual(after, before)
        self.assertTrue(core.grade(answer, before, after, calls)["success"])

        other = self.snapshot_result(fixture / "other-files")
        self.assertEqual(other.returncode, 0, other.stderr)
        other_files = json.loads(other.stdout)
        for path in (
            ".memory-graph/9bb4dc5d-ec02-4c7f-b046-755eed576041.json",
            ".memory-graph/parse-cache.sqlite3-wal.backup",
            ".memory-graph/nested/parse-cache.sqlite3",
            "parse-cache.sqlite3",
        ):
            with self.subTest(path=path):
                self.assertIn(path, other_files)
                changed_snapshot = {**before, "files": {**before["files"], path: other_files[path]}}
                self.assertIn(
                    "store or refs changed",
                    core.grade(answer, before, changed_snapshot, calls)["failures"],
                )

        changed_ref = {**after, "refs": ["refs/heads/main " + "b" * 40]}
        self.assertIn(
            "store or refs changed", core.grade(answer, before, changed_ref, calls)["failures"]
        )
        mutation = calls + [{"args": ["update"], "status": "error"}]
        self.assertIn(
            "mutation attempted for a read-only request",
            core.grade(answer, before, after, mutation)["failures"],
        )

    def test_snapshot_rejects_symlinks_even_at_cache_paths(self):
        fixture = ROOT / "tests/fixtures/inspect_snapshot/symlinks"
        for directory, path in (
            ("cache", ".memory-graph/parse-cache.sqlite3"),
            ("journal", ".memory-graph/parse-cache.sqlite3-journal"),
            ("wal", ".memory-graph/parse-cache.sqlite3-wal"),
            ("shm", ".memory-graph/parse-cache.sqlite3-shm"),
            ("state", ".memory-graph"),
        ):
            with self.subTest(path=path):
                if not (fixture / directory / path).is_symlink():
                    self.skipTest("symlink fixtures are not available on this platform")
                result = self.snapshot_result(fixture / directory)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("symlink in store", result.stderr)

    def test_task_is_pinned_docker_and_network_disabled(self):
        task = inspect_task.build_task(options())
        self.assertEqual(task.sandbox.type, "docker")
        service = task.sandbox.config.services["default"]
        self.assertEqual(service.network_mode, "none")
        self.assertEqual(service.user, "1000:1000")
        self.assertEqual(service.mem_limit, "512m")
        self.assertIsNone(service.volumes)
        self.assertTrue(service.read_only)
        self.assertEqual(service.pull_policy, "never")
        self.assertEqual(service.tmpfs, ["/workspace:rw,mode=1777,size=67108864"])
        self.assertEqual(len(task.dataset), 1)

    async def test_preparation_uses_only_sandbox_methods(self):
        class StateStore:
            def __init__(self):
                self.data = {}

            def set(self, key, value):
                self.data[key] = value

        class FakeSandbox:
            def __init__(self):
                self.writes = []
                self.commands = []

            async def write_file(self, path, value):
                self.writes.append((path, value))

            async def exec(self, command, **kwargs):
                self.commands.append((command, kwargs))
                if command[:2] == ["python3", "-c"] and len(command) == 4:
                    stdout = json.dumps(evidence()[1]["files"])
                elif "rev-parse" in command:
                    stdout = "a" * 40
                elif "for-each-ref" in command:
                    stdout = "refs/heads/main " + "a" * 40
                else:
                    stdout = ""
                return ExecResult(success=True, returncode=0, stdout=stdout, stderr="")

        fake = FakeSandbox()
        state = SimpleNamespace(store=StateStore())
        bundle = {"SKILL.md": b"synthetic instructions"}
        with (
            patch.object(inspect_task, "sandbox", return_value=fake),
            patch.object(inspect_task, "skill_files", return_value=bundle),
        ):
            await inspect_task.prepare(inspect_task.bundle_hash(bundle), 10)(state, None)
        self.assertEqual(len(fake.writes), 1 + len(core.fixture_files()))
        self.assertTrue(all(path.startswith("/workspace/") for path, _ in fake.writes))
        self.assertTrue(all("evals/" not in path for path, _ in fake.writes))
        self.assertTrue(all(kwargs["timeout_retry"] is False for _, kwargs in fake.commands))
        self.assertEqual(state.store.data["initial_snapshot"], evidence()[1])

    async def test_preparation_refuses_changed_bundle_before_writing(self):
        state = SimpleNamespace(store=None)
        with (
            patch.object(inspect_task, "skill_files", return_value={"SKILL.md": b"changed"}),
            patch.object(inspect_task, "sandbox") as sandbox,
        ):
            with self.assertRaisesRegex(RuntimeError, "bundle changed"):
                await inspect_task.prepare("original-hash", 10)(state, None)
            sandbox.assert_not_called()

    async def test_model_tool_routes_only_through_sandbox(self):
        class StateStore:
            def __init__(self):
                self.data = {"memory_calls": []}

            def get(self, key, default=None):
                return self.data.get(key, default)

            def set(self, key, value):
                self.data[key] = value

        class FakeSandbox:
            async def exec(self, command, **kwargs):
                self.command, self.kwargs = command, kwargs
                return ExecResult(success=True, returncode=0, stdout="{}", stderr="")

        state, fake = StateStore(), FakeSandbox()
        with (
            patch.object(inspect_task, "sandbox", return_value=fake),
            patch.object(inspect_task, "store", return_value=state),
        ):
            tool = inspect_task.memory_cli(2, 10)
            self.assertEqual(await tool(["find", "ladder", "--scope", "orchard"]), "{}")
        self.assertEqual(
            fake.command[:4], ["python3", inspect_task.SCRIPT, "--root", inspect_task.STORE]
        )
        self.assertFalse(fake.kwargs["timeout_retry"])
        self.assertEqual(state.data["memory_calls"][0]["status"], "success")

    async def test_root_override_and_tool_limit_are_recorded(self):
        from inspect_ai.tool import ToolError

        class StateStore:
            def __init__(self):
                self.data = {"memory_calls": []}

            def get(self, key, default=None):
                return self.data.get(key, default)

            def set(self, key, value):
                self.data[key] = value

        state = StateStore()
        with (
            patch.object(inspect_task, "sandbox") as sandbox,
            patch.object(inspect_task, "store", return_value=state),
        ):
            tool = inspect_task.memory_cli(1, 10)
            with self.assertRaises(ToolError):
                await tool(["find", "ladder", "--root", "/other"])
            with self.assertRaises(ToolError):
                await tool(["find", "ladder"])
            sandbox.assert_not_called()
        self.assertEqual([x["status"] for x in state.data["memory_calls"]], ["invalid", "invalid"])


if __name__ == "__main__":
    unittest.main()
