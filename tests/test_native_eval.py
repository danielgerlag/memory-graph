"""Offline tests: no Docker daemon, Copilot process, or model is launched."""

import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
import tarfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import native_core as core
import run_native


def options():
    return SimpleNamespace(
        run=True,
        allow_network=True,
        credential_env="COPILOT_GITHUB_TOKEN",
        image="example/image@sha256:" + "a" * 64,
        model="gpt-example",
        host_version="1.0.110-3",
        output_dir="evals/native-output",
        docker_host=None,
        session_ai_credits=30,
        run_ai_credits=60,
        session_timeout=30,
        run_timeout=120,
        cpus=1.0,
        memory_mb=512,
        max_output_mb=2,
    )


def activation_evidence(detected):
    session_id = str(uuid.UUID(int=1))
    skill_path = "/eval/workspace/.github/skills/memory-graph/SKILL.md"
    skill = b"---\nname: memory-graph\n---\n\nSynthetic skill body.\n"
    stream = [
        {
            "type": "session.skills_loaded",
            "data": {
                "skills": [
                    {
                        "name": "memory-graph",
                        "path": skill_path,
                        "source": "project",
                        "enabled": True,
                    }
                ]
            },
        },
        {"type": "assistant.idle"},
        {"type": "result", "sessionId": session_id, "exitCode": 0},
    ]
    events = []

    def add(kind, data):
        events.append(
            {
                "id": str(uuid.UUID(int=len(events) + 10)),
                "parentId": events[-1]["id"] if events else None,
                "timestamp": "2026-01-01T00:00:00Z",
                "type": kind,
                "data": data,
            }
        )

    add(
        "session.start",
        {
            "sessionId": session_id,
            "copilotVersion": "1.0.110-3",
            "producer": "copilot-agent",
            "version": 1,
        },
    )
    add("user.message", {})
    add("assistant.turn_start", {"turnId": "0"})
    if detected:
        add(
            "skill.invoked",
            {
                "name": "memory-graph",
                "path": skill_path,
                "source": "project",
                "trigger": "agent-invoked",
                "content": "Synthetic skill body.\n",
            },
        )
    add("assistant.turn_end", {"turnId": "0"})
    add("session.shutdown", {"shutdownType": "routine"})
    return stream, events, skill


def jsonl(events):
    return b"".join((json.dumps(event, separators=(",", ":")) + "\n").encode() for event in events)


def session_jsonl(events):
    events = json.loads(json.dumps(events))
    if events and events[-1]["type"] == "session.shutdown":
        events[-1]["data"]["eventsFileSizeBytes"] = len(jsonl(events[:-1]))
    return jsonl(events)


def detect(stream, events, skill, **kwargs):
    return core.activation_status(
        "1.0.110-3",
        jsonl(stream),
        session_jsonl(events),
        outcome="complete",
        exit_status=0,
        expected_skill=skill,
        **kwargs,
    )


class NativeCoreTests(unittest.TestCase):
    def test_checked_in_manifest_has_positive_and_negative(self):
        trials, dataset_hash, manifest_hash = core.load_trials(ROOT / "evals/native-smoke.json")
        self.assertEqual([item["should_activate"] for item in trials], [True, False])
        self.assertEqual(len(dataset_hash), 64)
        self.assertEqual(len(manifest_hash), 64)

    def test_manifest_duplicate_or_multi_session_is_rejected(self):
        dataset = core.read_json_bytes(core.DATASET.read_bytes())
        manifest_path = ROOT / "evals/native-smoke.json"
        manifest = core.read_json_bytes(manifest_path.read_bytes())
        manifest["trials"].append(manifest["trials"][0])
        with (
            patch.object(core, "read_json_bytes", side_effect=[dataset, manifest]),
            self.assertRaisesRegex(ValueError, "duplicate"),
        ):
            core.load_trials(manifest_path)
        manifest["trials"] = [{"case_id": "native-positive-01", "trial": 1}]
        with (
            patch.object(core, "read_json_bytes", side_effect=[dataset, manifest]),
            self.assertRaisesRegex(ValueError, "exactly one prompt"),
        ):
            core.load_trials(manifest_path)

    def test_guards_fail_before_execution(self):
        trials, _, _ = core.load_trials(ROOT / "evals/native-smoke.json")
        env = {"COPILOT_GITHUB_TOKEN": "synthetic-token-for-tests-only"}
        core.validate_options(options(), env, trials)
        local_image = options()
        local_image.image = "sha256:" + "a" * 64
        core.validate_options(local_image, env, trials)
        for field, value in [
            ("run", False),
            ("allow_network", False),
            ("model", "auto"),
            ("model", "AUTO"),
            ("host_version", "latest"),
            ("host_version", "1.0.111-1"),
            ("image", "example:latest"),
            ("session_ai_credits", 1),
            ("run_ai_credits", 30),
            ("session_timeout", 0),
            ("memory_mb", 0),
            ("cpus", float("nan")),
            ("docker_host", "tcp://example"),
        ]:
            with self.subTest(field=field):
                args = options()
                setattr(args, field, value)
                with self.assertRaises(ValueError):
                    core.validate_options(args, env, trials)
        with self.assertRaises(ValueError):
            core.validate_options(options(), {}, trials)

    def test_jsonl_requires_complete_unambiguous_json(self):
        self.assertEqual(len(core.parse_jsonl(b'{"type":"message"}\n{"type":"done"}\n')), 2)
        for raw in (
            b"",
            b"not-json\n",
            b'{"a":1}',
            b'{"a":1}\n\n',
            b'{"a":1,"a":2}\n',
            b'{"a":NaN}\n',
            b'["event"]\n',
        ):
            with self.subTest(raw=raw), self.assertRaises((ValueError, UnicodeError)):
                core.parse_jsonl(raw)

    def test_plausible_events_and_good_answers_do_not_prove_activation(self):
        for raw in (
            b'{"type":"skill.loaded","skill":"memory-graph"}\n',
            b'{"type":"assistant.message","content":"I loaded the skill"}\n',
            b'{"type":"answer","content":"correct"}\n',
        ):
            status = core.activation_status("1.0.110-3", raw)
            self.assertEqual(status["status"], "UNVERIFIED")
            self.assertNotIn("detected", status)

    def test_completed_native_sessions_detect_activation_and_nonactivation(self):
        for expected in (True, False):
            with self.subTest(detected=expected):
                stream, events, skill = activation_evidence(expected)
                result = detect(stream, events, skill)
                self.assertEqual(result["status"], "VERIFIED")
                self.assertIs(result["detected"], expected)
                self.assertTrue(result["observation_complete"])
                self.assertEqual(result["event_lines"], [4] if expected else [])
                self.assertEqual(result["source"], "session_events")

    def test_activation_fails_closed_for_incomplete_or_ambiguous_evidence(self):
        changes = (
            "disabled",
            "missing_catalog",
            "duplicate_catalog",
            "wrong_path",
            "wrong_content",
            "wrong_trigger",
            "subagent",
            "foreign_session",
            "wrong_host",
            "wrong_producer",
            "broken_parent",
            "duplicate_event",
            "bad_timestamp",
            "bad_type",
            "invalid_agent_id",
            "missing_turn_end",
            "failed_shutdown",
            "missing_idle",
            "unsuccessful_result",
        )
        for change in changes:
            with self.subTest(change=change):
                stream, events, skill = activation_evidence(True)
                if change == "disabled":
                    stream[0]["data"]["skills"][0]["enabled"] = False
                elif change == "missing_catalog":
                    stream.pop(0)
                elif change == "duplicate_catalog":
                    stream[0]["data"]["skills"].append(stream[0]["data"]["skills"][0].copy())
                elif change == "wrong_path":
                    events[3]["data"]["path"] = "/another/SKILL.md"
                elif change == "wrong_content":
                    events[3]["data"]["content"] = "different"
                elif change == "wrong_trigger":
                    events[3]["data"]["trigger"] = "user-invoked"
                elif change == "subagent":
                    events[3]["agentId"] = "other"
                elif change == "foreign_session":
                    stream[-1]["sessionId"] = str(uuid.UUID(int=2))
                elif change == "wrong_host":
                    events[0]["data"]["copilotVersion"] = "1.0.110-30"
                elif change == "wrong_producer":
                    events[0]["data"]["producer"] = "other"
                elif change == "broken_parent":
                    events[3]["parentId"] = str(uuid.UUID(int=100))
                elif change == "duplicate_event":
                    events[3]["id"] = events[2]["id"]
                elif change == "bad_timestamp":
                    events[3]["timestamp"] = 123
                elif change == "bad_type":
                    events[3]["type"] = []
                elif change == "invalid_agent_id":
                    events[3]["agentId"] = []
                elif change == "missing_turn_end":
                    events.pop(-2)
                    events[-1]["parentId"] = events[-2]["id"]
                elif change == "failed_shutdown":
                    events[-1]["data"]["shutdownType"] = "error"
                elif change == "missing_idle":
                    stream.pop(1)
                elif change == "unsuccessful_result":
                    stream[-1]["exitCode"] = 1
                result = detect(stream, events, skill)
                self.assertEqual(result["status"], "UNVERIFIED")
                self.assertNotIn("detected", result)

    def test_truncated_or_tampered_logs_never_prove_nonactivation(self):
        stream, events, skill = activation_evidence(True)
        raw = session_jsonl(events)
        with self.assertRaises(ValueError):
            core.activation_status(
                "1.0.110-3",
                jsonl(stream),
                raw[:-1],
                outcome="complete",
                exit_status=0,
                expected_skill=skill,
            )
        lines = raw.splitlines(keepends=True)
        del lines[3]
        result = core.activation_status(
            "1.0.110-3",
            jsonl(stream),
            b"".join(lines),
            outcome="complete",
            exit_status=0,
            expected_skill=skill,
        )
        self.assertEqual(result["status"], "UNVERIFIED")
        self.assertNotIn("detected", result)

    def test_failed_or_unknown_host_sessions_are_not_scored(self):
        stream, events, skill = activation_evidence(False)
        for version, outcome, exit_status in (
            ("1.0.113-1", "complete", 0),
            ("1.0.110-3", "timeout", 0),
            ("1.0.110-3", "complete", 1),
        ):
            with self.subTest(version=version, outcome=outcome, exit_status=exit_status):
                result = core.activation_status(
                    version,
                    jsonl(stream),
                    session_jsonl(events),
                    outcome=outcome,
                    exit_status=exit_status,
                    expected_skill=skill,
                )
                self.assertEqual(result["status"], "UNVERIFIED")
                self.assertNotIn("detected", result)

    def test_discovery_or_text_claims_do_not_prove_activation(self):
        stream, events, skill = activation_evidence(False)
        event = {
            "id": str(uuid.UUID(int=900)),
            "parentId": events[2]["id"],
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "assistant.message",
            "data": {"content": 'I loaded the skill: {"type":"skill.invoked"}'},
        }
        events.insert(3, event)
        events[4]["parentId"] = event["id"]
        result = detect(stream, events, skill)
        self.assertEqual(result["status"], "VERIFIED")
        self.assertFalse(result["detected"])

    def test_another_skill_does_not_count_as_the_target(self):
        stream, events, skill = activation_evidence(True)
        events[3]["data"].update(name="other-skill", path="/eval/other/SKILL.md")
        result = detect(stream, events, skill)
        self.assertEqual(result["status"], "VERIFIED")
        self.assertFalse(result["detected"])

    def test_commands_use_pinned_image_and_no_secret_or_host_mount(self):
        args = options()
        create = core.docker_create(args)
        self.assertIn("--pull=never", create)
        self.assertIn("--read-only", create)
        self.assertIn("--network=bridge", create)
        self.assertIn("--tmpfs=/eval:rw,mode=1777,size=134217728", create)
        self.assertIn("--tmpfs=/cli-tmp:rw,exec,mode=1777,size=402653184", create)
        self.assertFalse(any("--volume" in item or "--mount" in item for item in create))
        prompt = "Don't run anything; $(not-a-shell-command)"
        cli = core.copilot_command(args, prompt, 1)
        self.assertEqual(cli[-1], prompt)
        self.assertEqual(cli[:2], ["python3", "-c"])
        self.assertIn("'COPILOT_CACHE_HOME':'/cli-tmp/cache'", cli[2])
        self.assertIn("'TMPDIR':'/cli-tmp'", cli[2])
        self.assertIn("--secret-env-vars=COPILOT_GITHUB_TOKEN", cli)
        self.assertIn("--deny-tool=shell", cli)
        self.assertIn("--deny-tool=write", cli)
        self.assertIn("--disable-builtin-mcps", cli)
        self.assertIn("--disable-client-telemetry", cli)
        self.assertIn("--max-ai-credits", cli)
        self.assertNotIn("synthetic-token-for-tests-only", " ".join(cli))

    def test_native_launcher_preserves_the_working_directory_argument(self):
        import tempfile

        command = core.copilot_command(options(), "Synthetic prompt", 1)
        with tempfile.TemporaryDirectory() as temporary:
            launcher = Path(temporary) / "launcher.py"
            launcher.write_text(command[2], encoding="utf-8")
            with (
                patch.object(sys, "argv", [str(launcher), *command[3:]]),
                patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": "synthetic-token"}),
                patch.object(os, "execvpe") as launch,
            ):
                runpy.run_path(str(launcher), run_name="__main__")
        executable, argv, _ = launch.call_args.args
        self.assertEqual(executable, "copilot")
        self.assertEqual(argv[0], "copilot")
        self.assertEqual(argv[1:], command[3:])

    def test_fixture_archive_contains_only_synthetic_files_and_bundle(self):
        bundle = {
            "SKILL.md": b"synthetic",
            "scripts/memory_graph.py": b"script",
            "assets/node.schema.json": b"{}",
        }
        archive = core.fixture_archive(bundle)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            members = tar.getmembers()
            files = {item.name for item in members if item.isfile()}
            self.assertEqual(
                files,
                {
                    ".github/skills/memory-graph/SKILL.md",
                    ".github/skills/memory-graph/scripts/memory_graph.py",
                    ".github/skills/memory-graph/assets/node.schema.json",
                    "store/.gitignore",
                    "store/store.json",
                },
            )
            self.assertTrue(
                all(not item.issym() and ".." not in Path(item.name).parts for item in members)
            )
            self.assertEqual(json.load(tar.extractfile("store/store.json"))["id"], core.STORE_ID)

    def test_fixture_is_streamed_into_read_only_container(self):
        commands = []

        def fake_admin(command, env, deadline, *, data=None, limit=None):
            commands.append((command, data))
            return b""

        with patch.object(run_native, "admin", side_effect=fake_admin):
            run_native.initialize("c" * 64, b"fixture", {}, time.monotonic() + 30)
        command, data = commands[1]
        self.assertEqual(command[:4], ["docker", "exec", "-i", "c" * 64])
        self.assertIn(run_native.UNPACK_FIXTURE, command)
        self.assertEqual(data, b"fixture")

    def test_fixture_unpacker_rejects_traversal_and_links(self):
        import tempfile

        bundle = {"SKILL.md": b"synthetic"}
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-c", run_native.UNPACK_FIXTURE, temporary],
                input=core.fixture_archive(bundle),
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(
                (Path(temporary) / ".github/skills/memory-graph/SKILL.md").read_bytes(),
                b"synthetic",
            )
        for name, symlink in (("../outside", False), ("link", True)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                data = io.BytesIO()
                with tarfile.open(fileobj=data, mode="w") as archive:
                    info = tarfile.TarInfo(name)
                    if symlink:
                        info.type = tarfile.SYMTYPE
                        info.linkname = "../outside"
                    else:
                        info.size = 4
                    archive.addfile(info, None if symlink else io.BytesIO(b"data"))
                result = subprocess.run(
                    [sys.executable, "-c", run_native.UNPACK_FIXTURE, temporary],
                    input=data.getvalue(),
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_docker_environment_does_not_inherit_credentials_or_home(self):
        with patch.dict(
            os.environ, {"PATH": "/bin", "HOME": "/private", "GH_TOKEN": "other-token"}, clear=True
        ):
            env = run_native.environment(ROOT / "evals/result", None)
            self.assertNotIn("GH_TOKEN", env)
            self.assertNotEqual(env["HOME"], "/private")
            token_env = run_native.environment(ROOT / "evals/result", None, "explicit-token")
            self.assertEqual(token_env["COPILOT_GITHUB_TOKEN"], "explicit-token")
            self.assertNotIn("GH_TOKEN", token_env)
            socket_env = run_native.environment(ROOT / "evals/result", "unix:///tmp/podman.sock")
            self.assertEqual(socket_env["DOCKER_HOST"], "unix:///tmp/podman.sock")
            self.assertEqual(socket_env["CONTAINER_HOST"], "unix:///tmp/podman.sock")

    def test_version_matching_does_not_accept_other_builds(self):
        self.assertTrue(run_native.version_matches(b"Copilot CLI 1.0.110-3\n", "1.0.110-3"))
        self.assertTrue(run_native.version_matches(b"Copilot CLI v1.0.110-3\n", "1.0.110-3"))
        self.assertTrue(
            run_native.version_matches(
                b"GitHub Copilot CLI 1.0.110-3.\nRun 'copilot update' to check for updates.\n",
                "1.0.110-3",
            )
        )
        self.assertFalse(run_native.version_matches(b"Copilot CLI 1.0.110-30\n", "1.0.110-3"))
        self.assertFalse(run_native.version_matches(b"Copilot CLI 1.0.110-3.1\n", "1.0.110-3"))
        self.assertFalse(run_native.version_matches(b"Copilot CLI 1.0.110-3-extra\n", "1.0.110-3"))

    def test_activation_run_status_is_separate_from_release(self):
        for statuses, expected in (
            (["PASS", "PASS"], "PASS"),
            (["PASS", "FAIL"], "FAIL"),
            (["PASS", "BLOCKED"], "BLOCKED"),
            (["PASS", "NOT RUN"], "BLOCKED"),
            (["FAIL", "ERROR"], "ERROR"),
        ):
            with self.subTest(statuses=statuses):
                self.assertEqual(
                    run_native.activation_run_status([{"status": status} for status in statuses]),
                    expected,
                )
        with self.assertRaises(ValueError):
            run_native.activation_run_status([])

    def test_no_opt_in_starts_nothing_and_creates_no_output(self):
        argv = [
            "--manifest",
            str(ROOT / "evals/native-smoke.json"),
            "--output-dir",
            "evals/not-created",
            "--image",
            options().image,
            "--host-version",
            options().host_version,
            "--model",
            options().model,
            "--credential-env",
            "COPILOT_GITHUB_TOKEN",
            "--session-ai-credits",
            "30",
            "--run-ai-credits",
            "60",
            "--session-timeout",
            "30",
            "--run-timeout",
            "120",
            "--cpus",
            "1",
            "--memory-mb",
            "512",
            "--max-output-mb",
            "2",
        ]
        with (
            patch.dict(
                os.environ, {"COPILOT_GITHUB_TOKEN": "synthetic-token-for-tests"}, clear=True
            ),
            patch.object(run_native.subprocess, "Popen") as process,
            patch.object(Path, "mkdir") as mkdir,
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            self.assertEqual(run_native.main(argv), 2)
            process.assert_not_called()
            mkdir.assert_not_called()

    def test_passing_activation_run_does_not_certify_native_support(self):
        import tempfile

        trials = [
            {"case_id": "native-positive-02", "trial": 1, "prompts": ["positive"]},
            {"case_id": "native-negative-01", "trial": 1, "prompts": ["negative"]},
        ]
        results = [
            {"status": "PASS", "sessions": [{"activation": {"status": "VERIFIED"}}]},
            {"status": "PASS", "sessions": [{"activation": {"status": "VERIFIED"}}]},
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary) / "native"
            args = [
                "--run",
                "--allow-network",
                "--manifest",
                str(ROOT / "evals/native-smoke.json"),
                "--output-dir",
                str(output),
                "--image",
                options().image,
                "--host-version",
                options().host_version,
                "--model",
                options().model,
                "--credential-env",
                "COPILOT_GITHUB_TOKEN",
                "--session-ai-credits",
                "30",
                "--run-ai-credits",
                "60",
                "--session-timeout",
                "30",
                "--run-timeout",
                "120",
                "--cpus",
                "1",
                "--memory-mb",
                "512",
                "--max-output-mb",
                "2",
            ]
            with (
                patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": "synthetic-token-for-tests"}),
                patch.object(
                    run_native, "load_trials", return_value=(trials, "dataset", "manifest")
                ),
                patch.object(run_native, "skill_bundle", return_value={"SKILL.md": b"skill"}),
                patch.object(
                    run_native, "verify_corpus", return_value=(None, None, None, None, "corpus")
                ),
                patch.object(run_native, "collect_trial", side_effect=results),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(run_native.main(args), 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["assessment"], "native skill activation only")
            self.assertEqual(report["activation"], "VERIFIED")
            self.assertEqual(report["native_support"], "BLOCKED")
            self.assertEqual(report["release"], "BLOCKED")


class NativeCaptureTests(unittest.TestCase):
    def test_bounded_streams_preserve_bytes_without_model_or_docker(self):
        program = (
            "import sys; sys.stdout.write('{\"type\":\"done\"}\\n'); sys.stderr.write('diagnostic')"
        )
        value = run_native.captured_exec(
            [sys.executable, "-c", program],
            dict(os.environ),
            time.monotonic() + 10,
            5,
            10000,
            b"synthetic-secret-value",
        )
        self.assertEqual(value["outcome"], "complete")
        self.assertEqual(value["returncode"], 0)
        self.assertEqual(value["stdout"], b'{"type":"done"}\n')
        self.assertEqual(value["stderr"], b"diagnostic")

    def test_token_leak_and_excess_output_fail_closed(self):
        secret = b"synthetic-secret-for-test"
        leak = run_native.captured_exec(
            [sys.executable, "-c", "print('synthetic-secret-for-test')"],
            dict(os.environ),
            time.monotonic() + 10,
            5,
            10000,
            secret,
        )
        self.assertEqual(leak["outcome"], "credential-leak")
        self.assertEqual(leak["stdout"], b"")
        large = run_native.captured_exec(
            [sys.executable, "-c", "print('x'*100000)"],
            dict(os.environ),
            time.monotonic() + 10,
            5,
            1024,
            secret,
        )
        self.assertEqual(large["outcome"], "output-limit")

    def test_timeout_stops_local_capture_without_docker(self):
        value = run_native.captured_exec(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            dict(os.environ),
            time.monotonic() + 2,
            0.1,
            1000,
            b"test-secret",
        )
        self.assertEqual(value["outcome"], "timeout")

    def test_unverified_capture_is_blocked_and_cleans_its_container(self):
        args = options()
        trial = {
            "case_id": "native-positive-02",
            "trial": 1,
            "should_activate": True,
            "critical": False,
            "prompts": ["Keep this for later."],
        }
        bundle = {"SKILL.md": b"---\nname: memory-graph\n---\n\nskill\n"}
        skill_hashes = json.dumps(
            {k: hashlib.sha256(v).hexdigest() for k, v in bundle.items()}
        ).encode()
        files = {k: hashlib.sha256(v).hexdigest() for k, v in core.fixture_store().items()}
        commit = "b" * 40
        snapshot = {"files": files, "main": commit, "refs": [f"refs/heads/main {commit}"]}
        workspace = {
            ".github/skills/memory-graph/SKILL.md": hashlib.sha256(bundle["SKILL.md"]).hexdigest(),
            **{f"store/{name}": digest for name, digest in files.items()},
        }
        container = "c" * 64

        def fake_admin(command, *args, **kwargs):
            if command[1] == "create":
                return container.encode()
            if command[-1].endswith("-usage.json"):
                return b'{"tokens":10}'
            if run_native.READ_SESSION_EVENTS in command:
                return b'{"type":"session.event"}\n'
            if run_native.SKILL_HASHES in command:
                if command[-1] == "store/.git":
                    return json.dumps(workspace).encode()
                return skill_hashes
            return b""

        def fake_artifact(directory, name, data):
            return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

        captured = {
            "outcome": "complete",
            "returncode": 0,
            "stdout": b'{"type":"skill.loaded"}\n',
            "stderr": b"",
        }
        with (
            patch.object(run_native, "admin", side_effect=fake_admin),
            patch.object(run_native, "initialize"),
            patch.object(run_native, "host_version", return_value=b"1.0.110-3\n"),
            patch.object(run_native, "exec_snapshot", return_value=(snapshot, b"{}")),
            patch.object(run_native, "captured_exec", return_value=captured),
            patch.object(run_native, "artifact", side_effect=fake_artifact),
            patch.object(Path, "mkdir"),
            patch.object(
                subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ) as cleanup,
        ):
            result = run_native.collect_trial(
                args,
                trial,
                ROOT,
                {},
                b"synthetic-token-for-tests",
                b"archive",
                bundle,
                time.monotonic() + 30,
            )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["sessions"][0]["activation"]["status"], "UNVERIFIED")
        self.assertEqual(result["sessions"][0]["session_events_status"], "CAPTURED_UNVERIFIED")
        self.assertEqual(result["sessions"][0]["usage_status"], "CAPTURED_UNVERIFIED")
        cleanup.assert_called_once()
        self.assertEqual(cleanup.call_args.args[0], ["docker", "rm", "-f", container])

    def test_timeout_removes_container_without_waiting_for_more_evidence(self):
        trial = {
            "case_id": "native-negative-01",
            "trial": 1,
            "should_activate": False,
            "critical": False,
            "prompts": ["What is 1+1?"],
        }
        bundle = {"SKILL.md": b"skill"}
        skill_hashes = json.dumps(
            {k: hashlib.sha256(v).hexdigest() for k, v in bundle.items()}
        ).encode()
        files = {k: hashlib.sha256(v).hexdigest() for k, v in core.fixture_store().items()}
        commit, container = "a" * 40, "c" * 64
        snapshot = {"files": files, "main": commit, "refs": [f"refs/heads/main {commit}"]}
        workspace = {
            ".github/skills/memory-graph/SKILL.md": hashlib.sha256(b"skill").hexdigest(),
            **{f"store/{name}": digest for name, digest in files.items()},
        }

        def fake_admin(command, *args, **kwargs):
            if command[1] == "create":
                return container.encode()
            if run_native.SKILL_HASHES in command:
                if command[-1] == "store/.git":
                    return json.dumps(workspace).encode()
                return skill_hashes
            if run_native.READ_USAGE in command or run_native.READ_SESSION_EVENTS in command:
                raise AssertionError("must stop immediately after timeout")
            return b""

        captured = {"outcome": "timeout", "returncode": -9, "stdout": b"", "stderr": b""}
        with (
            patch.object(run_native, "admin", side_effect=fake_admin),
            patch.object(run_native, "initialize"),
            patch.object(run_native, "host_version", return_value=b"1.0.110-3\n"),
            patch.object(run_native, "exec_snapshot", return_value=(snapshot, b"{}")) as snapshots,
            patch.object(run_native, "captured_exec", return_value=captured),
            patch.object(run_native, "artifact", return_value={}),
            patch.object(Path, "mkdir"),
            patch.object(
                subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ) as cleanup,
        ):
            result = run_native.collect_trial(
                options(),
                trial,
                ROOT,
                {},
                b"synthetic-token-for-tests",
                b"archive",
                bundle,
                time.monotonic() + 30,
            )
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(snapshots.call_count, 1)
        cleanup.assert_called_once()

    def test_completed_native_capture_scores_only_verified_activation(self):
        for expected, detected, status in (
            (True, True, "PASS"),
            (False, False, "PASS"),
            (True, False, "FAIL"),
            (False, True, "FAIL"),
        ):
            with self.subTest(expected=expected, detected=detected):
                stream, events, skill = activation_evidence(detected)
                bundle = {"SKILL.md": skill}
                skill_hash = hashlib.sha256(skill).hexdigest()
                files = {
                    name: hashlib.sha256(value).hexdigest()
                    for name, value in core.fixture_store().items()
                }
                commit, container = "a" * 40, "c" * 64
                snapshot = {"files": files, "main": commit, "refs": [f"refs/heads/main {commit}"]}
                workspace = {
                    ".github/skills/memory-graph/SKILL.md": skill_hash,
                    **{f"store/{name}": digest for name, digest in files.items()},
                }

                def fake_admin(
                    command,
                    *args,
                    _container=container,
                    _events=events,
                    _workspace=workspace,
                    _skill_hash=skill_hash,
                    **kwargs,
                ):
                    if command[1] == "create":
                        return _container.encode()
                    if run_native.READ_SESSION_EVENTS in command:
                        return session_jsonl(_events)
                    if run_native.READ_USAGE in command:
                        return b'{"tokens":10}'
                    if run_native.SKILL_HASHES in command:
                        return (
                            json.dumps(_workspace).encode()
                            if command[-1] == "store/.git"
                            else json.dumps({"SKILL.md": _skill_hash}).encode()
                        )
                    return b""

                trial = {
                    "case_id": "native-positive-02" if expected else "native-negative-01",
                    "trial": 1,
                    "should_activate": expected,
                    "critical": False,
                    "prompts": ["Synthetic prompt"],
                }
                captured = {
                    "outcome": "complete",
                    "returncode": 0,
                    "stdout": jsonl(stream),
                    "stderr": b"",
                }
                with (
                    patch.object(run_native, "admin", side_effect=fake_admin),
                    patch.object(run_native, "initialize"),
                    patch.object(run_native, "host_version", return_value=b"1.0.110-3\n"),
                    patch.object(run_native, "exec_snapshot", return_value=(snapshot, b"{}")),
                    patch.object(run_native, "captured_exec", return_value=captured),
                    patch.object(run_native, "artifact", return_value={}),
                    patch.object(Path, "mkdir"),
                    patch.object(
                        subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
                    ),
                ):
                    result = run_native.collect_trial(
                        options(),
                        trial,
                        ROOT,
                        {},
                        b"synthetic-token-for-tests",
                        b"archive",
                        bundle,
                        time.monotonic() + 30,
                    )
                self.assertEqual(result["status"], status, result["problems"])
                self.assertEqual(result["assessment"], "native skill activation only")
                self.assertIs(result["observed_activation"], detected)
                self.assertEqual(
                    result["sessions"][0]["session_events_status"], "CAPTURED_VERIFIED"
                )

    def test_credential_in_native_events_is_withheld_and_container_removed(self):
        trial = {
            "case_id": "native-negative-01",
            "trial": 1,
            "should_activate": False,
            "critical": False,
            "prompts": ["What is 1+1?"],
        }
        bundle = {"SKILL.md": b"skill"}
        skill_hashes = json.dumps(
            {k: hashlib.sha256(v).hexdigest() for k, v in bundle.items()}
        ).encode()
        files = {k: hashlib.sha256(v).hexdigest() for k, v in core.fixture_store().items()}
        commit, container = "a" * 40, "c" * 64
        snapshot = {"files": files, "main": commit, "refs": [f"refs/heads/main {commit}"]}
        workspace = {
            ".github/skills/memory-graph/SKILL.md": hashlib.sha256(b"skill").hexdigest(),
            **{f"store/{name}": digest for name, digest in files.items()},
        }
        secret = b"synthetic-token-for-tests"

        def fake_admin(command, *args, **kwargs):
            if command[1] == "create":
                return container.encode()
            if run_native.SKILL_HASHES in command:
                return (
                    json.dumps(workspace).encode() if command[-1] == "store/.git" else skill_hashes
                )
            if run_native.READ_SESSION_EVENTS in command:
                return b'{"value":"' + secret + b'"}\n'
            if run_native.READ_USAGE in command:
                raise AssertionError("credential leakage must stop the trial")
            return b""

        def fake_artifact(directory, name, data):
            self.assertNotIn(secret, data)
            return {"path": name}

        captured = {
            "outcome": "complete",
            "returncode": 0,
            "stdout": b'{"type":"done"}\n',
            "stderr": b"",
        }
        with (
            patch.object(run_native, "admin", side_effect=fake_admin),
            patch.object(run_native, "initialize"),
            patch.object(run_native, "host_version", return_value=b"1.0.110-3\n"),
            patch.object(run_native, "exec_snapshot", return_value=(snapshot, b"{}")),
            patch.object(run_native, "captured_exec", return_value=captured),
            patch.object(run_native, "artifact", side_effect=fake_artifact),
            patch.object(Path, "mkdir"),
            patch.object(
                subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ) as cleanup,
        ):
            result = run_native.collect_trial(
                options(), trial, ROOT, {}, secret, b"archive", bundle, time.monotonic() + 30
            )
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["sessions"][0]["session_events_status"], "WITHHELD")
        self.assertNotIn("session_events", result["sessions"][0])
        cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
