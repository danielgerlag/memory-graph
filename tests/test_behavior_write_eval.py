"""Local synthetic Git/CLI end-to-end tests; never contacts a provider."""

import asyncio
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import behavior_capture
import behavior_cases
import behavior_shared_capture
import behavior_shared_inspect
import behavior_write_capture
import behavior_write_tool
import inspect_task
from controlled_core import STORE_ID
from graders import (
    behavior_grader,
    behavior_shared_grader,
    behavior_special_grader,
    behavior_write_grader,
)
from inspect_ai.util import ExecResult

SCRIPT = ROOT / "memory-graph/scripts/memory_graph.py"


class RealLocalGitTests(unittest.TestCase):
    def setUp(self):
        if Path.cwd().resolve() != ROOT:
            self.skipTest("run the isolated local Git tests from the project root")
        self.root = Path("evals") / f".behavior-e2e-{uuid.uuid4()}"
        self.root.mkdir(mode=0o700)
        self.addCleanup(shutil.rmtree, self.root)
        absolute = str(self.root.resolve())
        self.env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": absolute,
            "TMPDIR": absolute,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ALLOW_PROTOCOL": "file",
            "LC_ALL": "C",
            "GIT_AUTHOR_NAME": "Synthetic Eval",
            "GIT_AUTHOR_EMAIL": "eval@localhost",
            "GIT_COMMITTER_NAME": "Synthetic Eval",
            "GIT_COMMITTER_EMAIL": "eval@localhost",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def process(self, command, *, input_text=None):
        result = subprocess.run(
            command,
            input=input_text,
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def git(self, root, *args):
        return self.process(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "commit.gpgSign=false",
                *args,
            ]
        ).strip()

    def create_store(self, case_id, *, empty=False, root=None):
        fixture = behavior_cases.compile_fixture(case_id)
        root = root or self.root / case_id
        root.mkdir()
        for path, content in fixture["files"].items():
            if empty and path.endswith(".md"):
                continue
            (root / path).write_text(content, encoding="utf-8")
        self.git(root, "init", "--initial-branch=main", "--template=")
        self.git(root, "add", "--all")
        message = (
            "Synthetic shared fixture"
            if fixture["spec"]["kind"] in behavior_cases.SHARED_KINDS
            else "Synthetic fixture"
        )
        self.git(root, "commit", "-m", message)
        return root, fixture

    def snapshot(self, root):
        main = self.git(root, "rev-parse", "refs/heads/main")
        refs = self.git(root, "for-each-ref", "--format=%(refname) %(objectname)").splitlines()
        working = json.loads(self.process([sys.executable, "-c", inspect_task.SNAPSHOT, str(root)]))
        tree = json.loads(
            self.process([sys.executable, "-c", behavior_capture.TREE_PROBE, str(root), main])
        )
        raw = json.loads(
            self.process([sys.executable, "-c", behavior_write_capture.RAW_PROBE, str(root), main])
        )
        self.assertEqual(
            tree,
            {
                path: hashlib.sha256(base64.b64decode(value)).hexdigest()
                for path, value in raw["tree"].items()
            },
        )
        self.assertEqual(
            working,
            {
                path: hashlib.sha256(base64.b64decode(value)).hexdigest()
                for path, value in raw["working"].items()
            },
        )
        return {"main": main, "refs": refs, "files": working, "tree_files": tree, **raw}

    def shared_snapshot(self, root, remote, unavailable=None):
        remote_data = json.loads(
            self.process(
                [sys.executable, "-c", behavior_shared_capture.REMOTE_PROBE, str(root), str(remote)]
            )
        )
        result = {"local": self.snapshot(root), "remote": remote_data}
        if unavailable is not None:
            observed = json.loads(
                self.process(
                    [
                        sys.executable,
                        "-c",
                        behavior_shared_capture.AVAILABILITY_PROBE,
                        str(unavailable),
                    ]
                )
            )
            result["remote_available"] = observed["available"]
        return result

    def seed_shared(self, root, name):
        remote = (self.root / name).resolve()
        self.process(
            ["git", "clone", "--bare", "--no-local", "--template=", str(root), str(remote)]
        )
        self.assertFalse(
            any((remote / "hooks").iterdir()) if (remote / "hooks").exists() else False
        )
        self.git(root, "remote", "add", "origin", str(remote))
        self.git(root, "fetch", "--no-tags", "origin", "refs/heads/main:refs/remotes/origin/main")
        return remote

    def cli(self, root, args, request=None):
        input_text = json.dumps(request, ensure_ascii=False) if request is not None else ""
        output = self.process(
            [sys.executable, str(SCRIPT), "--root", str(root), *args], input_text=input_text
        )
        return {
            "args": args,
            "input_text": input_text,
            "status": "success",
            "stdout": output,
            "stderr": "",
        }

    def cli_error(self, root, args):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(root), *args],
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        return {
            "args": args,
            "input_text": "",
            "status": "error",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def cli_clock(self, root, args, clock):
        output = self.process(
            [
                sys.executable,
                "-c",
                behavior_write_tool.CLOCK_BOOTSTRAP,
                str(ROOT / "memory-graph/scripts"),
                clock,
                "--root",
                str(root),
                *args,
            ],
            input_text="",
        )
        return {"args": args, "input_text": "", "status": "success", "stdout": output, "stderr": ""}

    @staticmethod
    def answer(node_id, fact, commit, *, sources=(), notices=()):
        return json.dumps(
            {
                "node_ids": [node_id],
                "facts": [fact],
                "sources": list(sources),
                "notices": list(notices),
                "store_id": STORE_ID,
                "commit": commit,
            }
        )

    def test_independent_parser_preserves_leap_second_and_submicrosecond(self):
        node_id = behavior_cases.semantic_id("remember-fact", "precision")
        node = {
            "schema": 1,
            "id": node_id,
            "label": "precision",
            "kind": "claim",
            "status": "active",
            "created_at": "2016-12-31T23:59:60Z",
            "updated_at": "2017-01-01T00:00:00.123456789Z",
            "edges": [],
        }
        raw = ("---\n" + json.dumps(node) + "\n---\n").encode()
        parsed, _ = behavior_write_grader._node(raw, f"{node_id}.md")
        self.assertEqual(parsed["created_at"], node["created_at"])
        self.assertEqual(parsed["updated_at"], node["updated_at"])

    def test_scoped_source_grader_accepts_global_json_from_real_cli(self):
        root, fixture = self.create_store("conflicting-evidence")
        before_full = self.snapshot(root)
        scope = fixture["case"]["scope"]
        spec = fixture["spec"]
        actions = [self.cli(root, ["--json", "find", "--scope", scope, spec["query"]])]
        actions.extend(
            self.cli(root, ["--json", "show", "--scope", scope, fixture["ids"][alias]])
            for alias in spec["targets"]
        )
        after_full = self.snapshot(root)
        keys = ("main", "refs", "files", "tree_files")
        before = {key: before_full[key] for key in keys}
        after = {key: after_full[key] for key in keys}
        answer = json.dumps(
            {
                "node_ids": [fixture["ids"][alias] for alias in spec["targets"]],
                "facts": spec["facts"],
                "sources": spec["sources"],
                "notices": spec["notices"],
                "store_id": STORE_ID,
                "commit": before["main"],
            }
        )
        result = behavior_grader.grade(
            "conflicting-evidence",
            answer,
            {"before": before, "after": after, "actions": actions},
        )
        self.assertTrue(result["success"], result["failures"])

    def test_create_real_nodes_receipts_and_reject_bad_evidence(self):
        for case_id in ("remember-fact", "remember-second-domain"):
            with self.subTest(case_id=case_id):
                root, fixture = self.create_store(case_id)
                before = self.snapshot(root)
                spec, case = fixture["spec"], fixture["case"]
                request = {
                    "label": spec["facts"][0],
                    "kind": "claim",
                    "scope": case["scope"],
                    "evidence_mode": "reported",
                    "sources": [
                        {
                            "kind": "interaction",
                            "observed_at": case["clock"],
                            "ref": spec["sources"][0],
                        }
                    ],
                }
                action = self.cli(
                    root,
                    ["--json", "create", "--operation", fixture["operation_id"], "--input", "-"],
                    request,
                )
                result = json.loads(action["stdout"])
                after = self.snapshot(root)
                answer = self.answer(
                    result["receipt"]["node_ids"][0],
                    spec["facts"][0],
                    after["main"],
                    sources=spec["sources"],
                )
                artifacts = {
                    "before": before,
                    "after": after,
                    "actions": [action],
                    "bootstrap": None,
                }
                scored = behavior_write_grader.grade_write(case_id, answer, artifacts)
                self.assertTrue(scored["success"], scored["failures"])
                self.assertTrue(behavior_grader.grade(case_id, answer, artifacts)["success"])
                bad = json.loads(json.dumps(artifacts))
                bad["after"]["parents"] = []
                self.assertFalse(behavior_write_grader.grade_write(case_id, answer, bad)["success"])
                bad = json.loads(json.dumps(artifacts))
                bad["actions"][0]["input_text"] = "{}"
                self.assertFalse(behavior_write_grader.grade_write(case_id, answer, bad)["success"])
                bad = json.loads(json.dumps(artifacts))
                bad["after"]["working"]["extra"] = base64.b64encode(b"unrelated").decode()
                bad["after"]["files"]["extra"] = hashlib.sha256(b"unrelated").hexdigest()
                self.assertFalse(behavior_write_grader.grade_write(case_id, answer, bad)["success"])
                bad = json.loads(json.dumps(artifacts))
                node_path = f"{result['receipt']['node_ids'][0]}.md"
                raw = base64.b64decode(bad["after"]["tree"][node_path]).replace(
                    spec["sources"][0].encode(), b"invented-source"
                )
                bad["after"]["tree"][node_path] = base64.b64encode(raw).decode()
                bad["after"]["tree_files"][node_path] = hashlib.sha256(raw).hexdigest()
                self.assertFalse(behavior_write_grader.grade_write(case_id, answer, bad)["success"])

    def test_repeat_reuses_existing_node_without_rewriting(self):
        root, fixture = self.create_store("repeat-existing")
        before = self.snapshot(root)
        find = self.cli(root, ["find", "kite", "--scope", "orchard"])
        after = self.snapshot(root)
        answer = self.answer(fixture["ids"]["kite-1"], fixture["spec"]["facts"][0], after["main"])
        artifacts = {"before": before, "after": after, "actions": [find], "bootstrap": None}
        result = behavior_write_grader.grade_write("repeat-existing", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])
        artifacts["actions"].append(
            {
                "args": ["create"],
                "input_text": "{}",
                "status": "error",
                "stdout": "",
                "stderr": "denied",
            }
        )
        self.assertFalse(
            behavior_write_grader.grade_write("repeat-existing", answer, artifacts)["success"]
        )

    def test_repeat_can_update_without_changing_fact_identity(self):
        root, fixture = self.create_store("repeat-existing")
        before = self.snapshot(root)
        node_id = fixture["ids"]["kite-1"]
        find = self.cli(root, ["find", "kite", "--scope", "orchard"])
        request = {
            "id": node_id,
            "expected_revision": before["tree_files"][f"{node_id}.md"],
            "patch": {},
        }
        update = self.cli(
            root, ["update", "--operation", fixture["operation_id"], "--input", "-"], request
        )
        after = self.snapshot(root)
        answer = self.answer(node_id, fixture["spec"]["facts"][0], after["main"])
        artifacts = {"before": before, "after": after, "actions": [find, update], "bootstrap": None}
        result = behavior_write_grader.grade_write("repeat-existing", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])

    def test_supersession_is_one_commit_with_history_and_receipt(self):
        root, fixture = self.create_store("atomic-correction")
        before = self.snapshot(root)
        old_id = fixture["ids"]["old"]
        revision = before["tree_files"][f"{old_id}.md"]
        request = {
            "id": old_id,
            "expected_revision": revision,
            "replacement": {
                "label": fixture["spec"]["facts"][0],
                "kind": "claim",
                "scope": "orchard",
                "evidence_mode": "reported",
                "sources": [
                    {
                        "kind": "interaction",
                        "observed_at": fixture["case"]["clock"],
                        "ref": "interaction:correction",
                    }
                ],
            },
        }
        action = self.cli(
            root, ["supersede", "--operation", fixture["operation_id"], "--input", "-"], request
        )
        output = json.loads(action["stdout"])
        replacement = next(node for node in output["receipt"]["node_ids"] if node != old_id)
        recall = self.cli(root, ["find", "kite", "--scope", "orchard"])
        after = self.snapshot(root)
        answer = self.answer(
            replacement,
            fixture["spec"]["facts"][0],
            after["main"],
            sources=fixture["spec"]["sources"],
        )
        artifacts = {
            "before": before,
            "after": after,
            "actions": [action, recall],
            "bootstrap": None,
        }
        result = behavior_write_grader.grade_write("atomic-correction", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])
        bad = json.loads(json.dumps(artifacts))
        bad["actions"] = [action]
        self.assertFalse(
            behavior_write_grader.grade_write("atomic-correction", answer, bad)["success"]
        )
        bad = json.loads(json.dumps(artifacts))
        old_path = f"{old_id}.md"
        changed = base64.b64decode(bad["after"]["tree"][old_path]).replace(
            b"interaction:earlier", b"fabricated-evidence"
        )
        bad["after"]["tree"][old_path] = base64.b64encode(changed).decode()
        bad["after"]["tree_files"][old_path] = hashlib.sha256(changed).hexdigest()
        self.assertIn(
            "supersession lost history or evidence",
            behavior_write_grader.grade_write("atomic-correction", answer, bad)["failures"],
        )

    def test_lost_ack_status_and_replay_preserve_one_effect_and_expiry(self):
        root, fixture = self.create_store("lost-ack-retry", empty=True)
        initial = self.snapshot(root)
        seed = self.cli(
            root,
            ["create", "--operation", fixture["operation_id"], "--input", "-"],
            behavior_cases.retry_request("lost-ack-retry"),
        )
        before = self.snapshot(root)
        status = self.cli(root, ["operation", "status", fixture["operation_id"]])
        replay = self.cli(
            root,
            ["create", "--operation", fixture["operation_id"], "--input", "-"],
            behavior_cases.retry_request("lost-ack-retry"),
        )
        after = self.snapshot(root)
        answer = self.answer(
            fixture["ids"]["seed-1"], "seed tin is dry", after["main"], notices=["accepted"]
        )
        artifacts = {
            "before": before,
            "after": after,
            "actions": [status, replay],
            "bootstrap": {"before": initial, "action": seed},
        }
        result = behavior_write_grader.grade_write("lost-ack-retry", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])
        wrong = json.loads(json.dumps(artifacts))
        payload = json.loads(wrong["actions"][0]["stdout"])
        payload["snapshot_valid"] = False
        wrong["actions"][0]["stdout"] = json.dumps(payload)
        self.assertFalse(
            behavior_write_grader.grade_write("lost-ack-retry", answer, wrong)["success"]
        )

    def test_shared_read_uses_local_bare_remote_and_reports_provenance(self):
        root, fixture = self.create_store("shared-read")
        remote = self.seed_shared(root, "shared-remote.git")
        setup = self.cli(root, ["attach", "--remote", "origin", "--mode", "read"])
        before = self.shared_snapshot(root, remote)
        find = self.cli(root, ["find", "cart", "--scope", "orchard"])
        after = self.shared_snapshot(root, remote)
        answer = self.answer(fixture["ids"]["cart"], "cart is red", after["remote"]["main"])
        artifacts = {"before": before, "after": after, "actions": [find], "setup": setup}
        with patch.object(behavior_shared_grader, "REMOTE", str(remote)):
            result = behavior_shared_grader.grade_shared("shared-read", answer, artifacts)
            self.assertTrue(result["success"], result["failures"])
            self.assertTrue(behavior_grader.grade("shared-read", answer, artifacts)["success"])
            bad = json.loads(json.dumps(artifacts))
            bad["after"]["remote"]["refs"] = []
            self.assertFalse(
                behavior_shared_grader.grade_shared("shared-read", answer, bad)["success"]
            )
            bad = json.loads(json.dumps(artifacts))
            for side in ("before", "after"):
                local = bad[side]["local"]
                path = ".memory-graph/config.json"
                config = json.loads(base64.b64decode(local["working"][path]))
                config["endpoint_id"] = "0" * 64
                encoded = json.dumps(config).encode()
                local["working"][path] = base64.b64encode(encoded).decode()
                local["files"][path] = hashlib.sha256(encoded).hexdigest()
            setup_output = json.loads(bad["setup"]["stdout"])
            setup_output["endpoint_id"] = "0" * 64
            bad["setup"]["stdout"] = json.dumps(setup_output)
            self.assertFalse(
                behavior_shared_grader.grade_shared("shared-read", answer, bad)["success"]
            )
            bad = json.loads(json.dumps(artifacts))
            output = json.loads(bad["actions"][0]["stdout"])
            output["snapshot"]["remote"] = "other"
            bad["actions"][0]["stdout"] = json.dumps(output)
            self.assertFalse(
                behavior_shared_grader.grade_shared("shared-read", answer, bad)["success"]
            )
            bad = json.loads(json.dumps(artifacts))
            bad["actions"].append(
                {"args": ["publish"], "status": "error", "stdout": "", "stderr": ""}
            )
            self.assertFalse(
                behavior_shared_grader.grade_shared("shared-read", answer, bad)["success"]
            )

    def test_bound_store_cannot_expose_an_unmounted_target(self):
        workspace = self.root / "isolated"
        workspace.mkdir()
        (workspace / "skill").mkdir()
        root, fixture = self.create_store("bound-store", root=workspace / "store")
        before_full = self.snapshot(root)
        workspace_before = json.loads(
            self.process([sys.executable, "-c", behavior_capture.WORKSPACE_PROBE, str(workspace)])
        )
        find = self.cli(root, ["find", "seed", "--scope", "orchard"])
        after_full = self.snapshot(root)
        workspace_after = json.loads(
            self.process([sys.executable, "-c", behavior_capture.WORKSPACE_PROBE, str(workspace)])
        )
        before = {key: before_full[key] for key in ("main", "refs", "files", "tree_files")}
        after = {key: after_full[key] for key in ("main", "refs", "files", "tree_files")}
        answer = self.answer(
            fixture["ids"]["private"],
            "private seed",
            after["main"],
            notices=["fresh context and separate approval required"],
        )
        artifacts = {
            "before": before,
            "after": after,
            "actions": [find],
            "isolation": {"before": workspace_before, "after": workspace_after},
        }
        result = behavior_grader.grade("bound-store", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])

    def test_unreachable_local_remote_reports_cached_freshness(self):
        root, fixture = self.create_store("offline-stale")
        remote = self.seed_shared(root, "stale-remote.git")
        retained = (self.root / "retained-remote.git").resolve()
        setup = self.cli(root, ["attach", "--remote", "origin", "--mode", "read"])
        remote.rename(retained)
        before = self.shared_snapshot(root, retained, unavailable=remote)
        failed_fetch = self.cli_error(root, ["fetch"])
        find = self.cli(root, ["find", "cart", "--scope", "orchard"])
        after = self.shared_snapshot(root, retained, unavailable=remote)
        answer = self.answer(
            fixture["ids"]["cart"],
            "cart is red",
            after["remote"]["main"],
            notices=["stale or unknown freshness"],
        )
        artifacts = {
            "before": before,
            "after": after,
            "actions": [failed_fetch, find],
            "setup": setup,
        }
        with patch.object(behavior_shared_grader, "REMOTE", str(remote)):
            result = behavior_shared_grader.grade_shared("offline-stale", answer, artifacts)
            self.assertTrue(result["success"], result["failures"])
            self.assertTrue(behavior_grader.grade("offline-stale", answer, artifacts)["success"])
            bad = json.loads(json.dumps(artifacts))
            bad["before"]["remote_available"] = True
            self.assertFalse(
                behavior_shared_grader.grade_shared("offline-stale", answer, bad)["success"]
            )
            bad = json.loads(json.dumps(artifacts))
            output = json.loads(bad["actions"][1]["stdout"])
            output["snapshot"]["freshness"] = "current"
            bad["actions"][1]["stdout"] = json.dumps(output)
            self.assertFalse(
                behavior_shared_grader.grade_shared("offline-stale", answer, bad)["success"]
            )

    def test_attachment_requires_pinned_endpoint_and_explicit_reattach(self):
        root, _ = self.create_store("shared-read")
        remote = self.seed_shared(root, "pinned-remote.git")
        self.cli(root, ["attach", "--remote", "origin", "--mode", "read"])
        config_path = root / ".memory-graph/config.json"
        config = json.loads(config_path.read_text())
        self.assertEqual(config["endpoint_id"], hashlib.sha256(str(remote).encode()).hexdigest())
        del config["endpoint_id"]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.assertIn("reattach", self.cli_error(root, ["status"])["stderr"])
        self.cli(root, ["attach", "--remote", "origin", "--mode", "read"])
        replacement = (self.root / "replacement.git").resolve()
        self.process(
            ["git", "clone", "--bare", "--no-local", "--template=", str(root), str(replacement)]
        )
        self.git(root, "remote", "set-url", "origin", str(replacement))
        self.assertIn("endpoint changed", self.cli_error(root, ["status"])["stderr"])
        self.cli(root, ["attach", "--remote", "origin", "--mode", "read"])
        config = json.loads(config_path.read_text())
        self.assertEqual(
            config["endpoint_id"], hashlib.sha256(str(replacement).encode()).hexdigest()
        )

    def test_local_publication_is_refused_without_changing_remote(self):
        root, fixture = self.create_store("shared-read")
        remote = self.seed_shared(root, "no-publication.git")
        self.cli(root, ["attach", "--remote", "origin", "--mode", "propose"])
        self.cli(
            root,
            ["create", "--operation", fixture["operation_id"], "--input", "-"],
            {"label": "wheel is blue", "kind": "claim", "scope": "workshop"},
        )
        before = self.shared_snapshot(root, remote)["remote"]
        failed = self.cli_error(root, ["publish", "--operation", fixture["operation_id"]])
        self.assertIn("Local publication is unavailable", failed["stderr"])
        self.assertEqual(before, self.shared_snapshot(root, remote)["remote"])

    def test_shared_adapter_runs_real_hook_free_setup_and_capture(self):
        class StateStore:
            def __init__(inner):
                inner.values = {}

            def get(inner, key, default=None):
                return inner.values.get(key, default)

            def set(inner, key, value):
                inner.values[key] = value

        for case_id in ("shared-read", "offline-stale", "accepted-after-fetch"):
            with self.subTest(case_id=case_id):
                workspace = (self.root / case_id).resolve()
                workspace.mkdir()
                commands = []

                def restore_transport(folder=workspace):
                    transport = folder / "store/.memory-graph/transport.git"
                    if transport.exists():
                        transport.chmod(0o700)

                self.addCleanup(restore_transport)

                def mapped(value, workspace=workspace):
                    if value == "/workspace" or value.startswith("/workspace/"):
                        return str(workspace) + value[len("/workspace") :]
                    return value

                class LocalSandbox:
                    def __init__(inner, commands):
                        inner.commands = commands

                    async def write_file(inner, path, value):
                        target = Path(mapped(path))
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if isinstance(value, bytes):
                            target.write_bytes(value)
                        else:
                            target.write_text(value, encoding="utf-8")

                    async def exec(
                        inner, command, *, input=None, env=None, timeout=None, timeout_retry=False
                    ):
                        self.assertFalse(timeout_retry)
                        args = [mapped(value) for value in command]
                        if args[0] == "python3":
                            args[0] = sys.executable
                        inner.commands.append(args)
                        result = await asyncio.to_thread(
                            subprocess.run,
                            args,
                            input=input,
                            env={
                                **self.env,
                                **{key: mapped(value) for key, value in (env or {}).items()},
                            },
                            text=True,
                            capture_output=True,
                            check=False,
                            timeout=timeout,
                        )
                        return ExecResult(
                            success=result.returncode == 0,
                            returncode=result.returncode,
                            stdout=result.stdout,
                            stderr=result.stderr,
                        )

                sandbox = LocalSandbox(commands)
                state = type("State", (), {"store": StateStore()})()
                bundle = behavior_shared_inspect.skill_files()
                with (
                    patch.object(behavior_shared_inspect, "sandbox", return_value=sandbox),
                    patch.object(inspect_task, "sandbox", return_value=sandbox),
                ):
                    asyncio.run(
                        behavior_shared_inspect.prepare_shared(
                            case_id, behavior_shared_inspect.bundle_hash(bundle), 30
                        )(state, None)
                    )
                    root = workspace / "store"
                    actions = []
                    if case_id == "accepted-after-fetch":
                        actions.append(self.cli(root, ["fetch", "--help"]))
                        actions.append(self.cli(root, ["--help"]))
                        actions.append(self.cli(root, ["find", "wheel", "--scope", "workshop"]))
                        actions.append(self.cli(root, ["fetch"]))
                        actions.append(self.cli(root, ["find", "wheel", "--scope", "workshop"]))
                    elif case_id == "offline-stale":
                        actions.append(self.cli_error(root, ["fetch"]))
                    if case_id != "accepted-after-fetch":
                        actions.append(self.cli(root, ["find", "cart", "--scope", "orchard"]))
                    state.store.set("memory_calls", actions)
                    asyncio.run(
                        behavior_shared_inspect.capture_shared_after(case_id, 30)(state, None)
                    )
                remote = workspace / "remote.git"
                fixture = behavior_cases.compile_fixture(case_id)
                if case_id == "accepted-after-fetch":
                    from graders import behavior_fetch_grader

                    accepted = fixture["case"]["fixture"]["accepted_fact"]
                    answer = self.answer(
                        behavior_cases.semantic_id(case_id, accepted["id"]),
                        accepted["label"],
                        state.store.get("shared_before")["remote"]["main"],
                    )
                    grader = behavior_fetch_grader
                else:
                    answer = self.answer(
                        fixture["ids"]["cart"],
                        "cart is red",
                        state.store.get("shared_before")["remote"]["main"],
                        notices=["stale or unknown freshness"]
                        if case_id == "offline-stale"
                        else [],
                    )
                    grader = behavior_shared_grader
                with patch.object(grader, "REMOTE", str(remote)):
                    result = behavior_grader.grade(
                        case_id, answer, state.store.get("behavior_artifacts")
                    )
                    if case_id == "accepted-after-fetch":
                        artifacts = state.store.get("behavior_artifacts")
                        self.assertNotEqual(
                            artifacts["before"]["tracking"]["commit"],
                            artifacts["before"]["remote"]["main"],
                        )
                        self.assertEqual(
                            artifacts["after"]["tracking"]["commit"],
                            artifacts["after"]["remote"]["main"],
                        )
                        self.assertEqual(json.loads(actions[2]["stdout"])["results"], [])
                        tampered = json.loads(json.dumps(artifacts))
                        tampered["actions"] = [*actions[:3], actions[-1]]
                        self.assertFalse(
                            behavior_grader.grade(case_id, answer, tampered)["success"]
                        )
                        tampered = json.loads(json.dumps(artifacts))
                        tampered["actions"][0]["stdout"] = "not valid help"
                        self.assertFalse(
                            behavior_grader.grade(case_id, answer, tampered)["success"]
                        )
                        retried = json.loads(json.dumps(artifacts))
                        rejected = {
                            "args": ["fetch", "--scope", fixture["case"]["scope"]],
                            "input_text": "",
                            "status": "invalid",
                        }
                        retried["actions"].insert(0, rejected)
                        self.assertTrue(behavior_grader.grade(case_id, answer, retried)["success"])
                        rejected["error"] = "timeout"
                        self.assertFalse(behavior_grader.grade(case_id, answer, retried)["success"])
                        rejected.pop("error")
                        rejected["args"] = ["publish", "--scope", fixture["case"]["scope"]]
                        self.assertFalse(behavior_grader.grade(case_id, answer, retried)["success"])
                        alternative = json.loads(json.dumps(artifacts))
                        alternative["actions"][-1] = self.cli(
                            root,
                            ["find", accepted["label"].split()[-1], "--scope", accepted["scope"]],
                        )
                        self.assertTrue(
                            behavior_grader.grade(case_id, answer, alternative)["success"]
                        )
                        response = json.loads(answer)
                        response["sources"] = [[]]
                        result_with_placeholder = behavior_grader.grade(
                            case_id, json.dumps(response), artifacts
                        )
                        self.assertTrue(result_with_placeholder["success"])
                        self.assertIn(
                            "empty source placeholder removed; answer schema differs",
                            result_with_placeholder["observations"],
                        )
                        response["sources"] = [["invented"]]
                        self.assertFalse(
                            behavior_grader.grade(case_id, json.dumps(response), artifacts)[
                                "success"
                            ]
                        )
                        response["sources"] = []
                        response["notices"] = ["unknown"]
                        result_with_uncertainty = behavior_grader.grade(
                            case_id, json.dumps(response), artifacts
                        )
                        self.assertIn(
                            "ambiguous unknown notice on an unsourced retrieved claim",
                            result_with_uncertainty["observations"],
                        )
                        self.assertTrue(result_with_uncertainty["success"])
                        response["facts"] = []
                        self.assertFalse(
                            behavior_grader.grade(case_id, json.dumps(response), artifacts)[
                                "success"
                            ]
                        )
                        wrapped = "Retrieved the matching memory.\n" + answer
                        result_with_prose = behavior_grader.grade(case_id, wrapped, artifacts)
                        self.assertTrue(result_with_prose["success"])
                        self.assertIn(
                            "text outside JSON was not scored; JSON-only requirement not met",
                            result_with_prose["observations"],
                        )
                        self.assertFalse(
                            behavior_grader.grade(case_id, wrapped + "\n" + answer, artifacts)[
                                "success"
                            ]
                        )
                        for key, wrong in (("store_id", "wrong-store"), ("commit", "0" * 40)):
                            with self.subTest(provenance=key):
                                response = json.loads(answer)
                                response[key] = wrong
                                result_with_wrong_provenance = behavior_grader.grade(
                                    case_id, json.dumps(response), artifacts
                                )
                                self.assertIn(
                                    f"incorrect final answer: {key}",
                                    result_with_wrong_provenance["failures"],
                                )
                        tampered = json.loads(json.dumps(artifacts))
                        tampered["acceptance"]["input_text"] = "{}"
                        self.assertFalse(
                            behavior_grader.grade(case_id, answer, tampered)["success"]
                        )
                        tampered = json.loads(json.dumps(artifacts))
                        tampered["after"]["remote"]["main"] = tampered["before"]["local"]["main"]
                        self.assertFalse(
                            behavior_grader.grade(case_id, answer, tampered)["success"]
                        )
                        tampered = json.loads(json.dumps(artifacts))
                        config_path = ".memory-graph/config.json"
                        working = tampered["after"]["local"]["working"]
                        changed_config = json.loads(base64.b64decode(working[config_path]))
                        changed_config["endpoint_id"] = "0" * 64
                        raw = json.dumps(changed_config).encode()
                        working[config_path] = base64.b64encode(raw).decode()
                        tampered["after"]["local"]["files"][config_path] = hashlib.sha256(
                            raw
                        ).hexdigest()
                        self.assertFalse(
                            behavior_grader.grade(case_id, answer, tampered)["success"]
                        )
                        for forbidden in ("create", "publish"):
                            tampered = json.loads(json.dumps(artifacts))
                            tampered["actions"].append(
                                {
                                    "args": [forbidden],
                                    "status": "error",
                                    "stdout": "",
                                    "stderr": "",
                                }
                            )
                            self.assertFalse(
                                behavior_grader.grade(case_id, answer, tampered)["success"]
                            )
                self.assertTrue(result["success"], result["failures"])
                self.assertFalse(any("push" in command for command in commands))
                hook_dir = workspace / (
                    "offline-remote.git/hooks" if case_id == "offline-stale" else "remote.git/hooks"
                )
                self.assertFalse(any(hook_dir.iterdir()) if hook_dir.exists() else False)

    def test_pinned_clock_expiration_review_and_read_only_ttl(self):
        for case_id in ("expired-review", "read-only"):
            with self.subTest(case_id=case_id):
                root, fixture = self.create_store(case_id)
                before = self.snapshot(root)
                clock = fixture["case"]["clock"]
                find = self.cli_clock(
                    root, ["find", fixture["spec"]["query"], "--scope", "orchard"], clock
                )
                actions = [find]
                notices = []
                if case_id == "expired-review":
                    actions.append(self.cli_clock(root, ["review", "--scope", "orchard"], clock))
                    notices = [f"review due: {fixture['ids']['review']}"]
                after = self.snapshot(root)
                node_id = fixture["ids"][fixture["spec"]["targets"][0]]
                answer = self.answer(
                    node_id, fixture["spec"]["facts"][0], after["main"], notices=notices
                )
                artifacts = {"before": before, "after": after, "actions": actions}
                result = behavior_special_grader.grade_special(case_id, answer, artifacts)
                self.assertTrue(result["success"], result["failures"])
                self.assertTrue(behavior_grader.grade(case_id, answer, artifacts)["success"])
                bad = json.loads(json.dumps(artifacts))
                bad["actions"].append(
                    {"args": ["update"], "status": "error", "stdout": "", "stderr": ""}
                )
                self.assertFalse(
                    behavior_special_grader.grade_special(case_id, answer, bad)["success"]
                )
                if case_id == "expired-review":
                    bad = json.loads(json.dumps(artifacts))
                    bad["actions"] = [find]
                    self.assertFalse(
                        behavior_special_grader.grade_special(case_id, answer, bad)["success"]
                    )
                else:
                    bad = json.loads(json.dumps(artifacts))
                    path = f"{node_id}.md"
                    changed = base64.b64decode(bad["after"]["tree"][path]).replace(
                        b"2026-03-01T00:00:00Z", b"2027-03-01T00:00:00Z"
                    )
                    bad["after"]["tree"][path] = base64.b64encode(changed).decode()
                    bad["after"]["tree_files"][path] = hashlib.sha256(changed).hexdigest()
                    self.assertFalse(
                        behavior_special_grader.grade_special(case_id, answer, bad)["success"]
                    )

    def test_malformed_store_fails_recall_and_returns_diagnostic_without_repair(self):
        root, _fixture = self.create_store("malformed-graph")
        before = self.snapshot(root)
        recall = self.cli_error(root, ["find", "cart", "--scope", "orchard"])
        diagnose = self.cli_error(root, ["diagnose"])
        after = self.snapshot(root)
        answer = json.dumps(
            {
                "node_ids": [],
                "facts": [],
                "sources": [],
                "notices": ["invalid graph"],
                "store_id": STORE_ID,
                "commit": after["main"],
            }
        )
        artifacts = {"before": before, "after": after, "actions": [recall, diagnose]}
        result = behavior_special_grader.grade_special("malformed-graph", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])
        bad = json.loads(json.dumps(artifacts))
        bad["actions"] = [recall]
        self.assertFalse(
            behavior_special_grader.grade_special("malformed-graph", answer, bad)["success"]
        )
        bad = json.loads(json.dumps(artifacts))
        bad["actions"].append({"args": ["repair"], "status": "error", "stdout": "", "stderr": ""})
        self.assertFalse(
            behavior_special_grader.grade_special("malformed-graph", answer, bad)["success"]
        )


if __name__ == "__main__":
    unittest.main()
