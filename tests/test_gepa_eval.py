"""Offline tests for GEPA candidate isolation, budgets, and failure handling."""

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if importlib.util.find_spec("inspect_ai") is None or importlib.util.find_spec("gepa") is None:
    raise unittest.SkipTest("GEPA and Inspect AI are optional evaluation dependencies")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import gepa_reflection
import run_gepa
from inspect_task import bundle_hash


class TrialAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "candidates").mkdir()
        (self.root / "trials").mkdir()
        self.bundle = {"SKILL.md": b"---\nname: test\n---\noriginal\n", "script.py": b"fixed"}
        self.adapter = run_gepa.TrialAdapter(
            root=self.root,
            config=self.root / "config.yaml",
            prefix="---\nname: test\n---\n",
            bundle=self.bundle,
            max_trials=2,
            timeout=30,
        )

    def fake_trial(self, command, **kwargs):
        self.assertEqual(kwargs["cwd"], ROOT)
        trial_dir = Path(command[command.index("--log-dir") + 1])
        trial_dir.mkdir()
        candidate = Path(command[command.index("--skill-instructions") + 1]).read_bytes()
        report = {
            "status": "PASS",
            "case_id": command[command.index("--case-id") + 1],
            "skill_sha256": bundle_hash({**self.bundle, "SKILL.md": candidate}),
            "adapter_hashes": self.adapter.adapter_hashes,
            "corpus_source_sha256": self.adapter.corpus_hash,
            "score_detail": "correct",
        }
        (trial_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    def test_isolated_candidate_is_scored_and_budget_is_enforced(self):
        with (
            patch.object(run_gepa.subprocess, "run", side_effect=self.fake_trial) as execute,
            patch.object(run_gepa, "action_summary", return_value=[]),
        ):
            result = self.adapter.evaluate(
                ["scope-search"], {"instructions": "generic instructions\n"}, capture_traces=True
            )
            self.assertEqual(result.scores, [1.0])
            self.assertEqual(result.trajectories[0]["status"], "PASS")
            self.assertEqual(
                self.adapter.make_reflective_dataset(
                    {"instructions": ""}, result, ["instructions"]
                )["instructions"][0]["Feedback"]["checks"],
                "correct",
            )
            self.adapter.evaluate(["scope-search"], {"instructions": "generic instructions\n"})
            with self.assertRaisesRegex(run_gepa.OptimizationError, "budget"):
                self.adapter.evaluate(["scope-search"], {"instructions": "generic instructions\n"})
            self.assertEqual(execute.call_count, 2)

    def test_invalid_candidate_never_runs_model(self):
        with patch.object(run_gepa.subprocess, "run") as execute:
            result = self.adapter.evaluate(
                ["scope-search"], {"instructions": "---\nname: changed\n---\n"}, capture_traces=True
            )
            self.assertEqual(result.scores, [0.0])
            self.assertEqual(result.trajectories[0]["status"], "INVALID")
            self.adapter.make_reflective_dataset({}, result, ["instructions"])
            execute.assert_not_called()

    def test_runner_errors_are_not_scored_as_behavior_failures(self):
        def error_trial(command, **kwargs):
            trial_dir = Path(command[command.index("--log-dir") + 1])
            trial_dir.mkdir()
            (trial_dir / "report.json").write_text('{"status":"ERROR"}', encoding="utf-8")
            return SimpleNamespace(returncode=2)

        with (
            patch.object(run_gepa.subprocess, "run", side_effect=error_trial),
            self.assertRaisesRegex(run_gepa.OptimizationError, "invalid evidence"),
        ):
            self.adapter.evaluate(["scope-search"], {"instructions": "instructions"})

    def test_gepa_uses_adapter_feedback_to_select_improved_candidate(self):
        import gepa

        def fake_trial(command, **kwargs):
            trial_dir = Path(command[command.index("--log-dir") + 1])
            trial_dir.mkdir()
            candidate = Path(command[command.index("--skill-instructions") + 1]).read_bytes()
            passed = b"improved" in candidate
            report = {
                "status": "PASS" if passed else "FAIL",
                "case_id": command[command.index("--case-id") + 1],
                "skill_sha256": bundle_hash({**self.bundle, "SKILL.md": candidate}),
                "adapter_hashes": self.adapter.adapter_hashes,
                "corpus_source_sha256": self.adapter.corpus_hash,
                "score_detail": "correct" if passed else "missing evidence",
            }
            (trial_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
            return SimpleNamespace(returncode=0 if passed else 1)

        self.adapter.max_trials = 6
        with (
            patch.object(run_gepa.subprocess, "run", side_effect=fake_trial),
            patch.object(run_gepa, "action_summary", return_value=[]),
        ):
            result = gepa.optimize(
                seed_candidate={"instructions": "initial"},
                trainset=["scope-search"],
                valset=["directed-traversal"],
                adapter=self.adapter,
                reflection_lm=lambda prompt: "```text\nimproved\n```",
                reflection_minibatch_size=1,
                max_metric_calls=5,
                logger=run_gepa.QuietLogger(),
                display_progress_bar=False,
                use_wandb=False,
                use_mlflow=False,
            )
        self.assertEqual(result.best_candidate["instructions"].strip(), "improved")


class ReflectorTests(unittest.TestCase):
    def test_gepa_reflection_errors_are_recorded_without_raw_text(self):
        logger = run_gepa.QuietLogger()
        logger.log("Iteration 3: Exception building reflective dataset: sensitive details")
        self.assertEqual(logger.failures, ["GEPA could not complete a reflection or evaluation"])

    def test_reflection_limit_and_safe_instructions(self):
        reflector = run_gepa.Reflector(
            root=ROOT, config=ROOT / "unused.yaml", limit=1, timeout=30, max_tokens=100
        )
        response = {"completion": "candidate", "input_tokens": 2, "output_tokens": 3}
        with patch.object(
            run_gepa.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout=json.dumps(response)),
        ) as execute:
            self.assertEqual(reflector("prompt"), "candidate")
            self.assertIn("scenario-specific", execute.call_args.kwargs["input"])
            with self.assertRaisesRegex(run_gepa.OptimizationError, "budget"):
                reflector("prompt")
            self.assertEqual(execute.call_count, 1)

    def test_reflection_failure_stops_retries_without_exposing_stderr(self):
        reflector = run_gepa.Reflector(
            root=ROOT, config=ROOT / "unused.yaml", limit=3, timeout=30, max_tokens=100
        )
        failed = SimpleNamespace(returncode=2, stderr="private error details")
        with patch.object(run_gepa.subprocess, "run", return_value=failed) as execute:
            for _ in range(2):
                with self.assertRaisesRegex(run_gepa.OptimizationError, "unknown error"):
                    reflector("prompt")
            self.assertEqual(execute.call_count, 1)

    def test_reflection_accepts_controlled_profile_with_case_and_trial(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text(
                """
schema_version: 1
runner: controlled
options:
  case_id: scope-search
  trial: 1
  model: openai/example
  credential_env: OPENAI_API_KEY
  image: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  time_limit: 60
  wall_timeout: 90
  tool_timeout: 10
  turn_limit: 5
  tool_limit: 5
  token_limit: 5000
  max_tokens: 500
  cpus: 1
  memory_mb: 512
""",
                encoding="utf-8",
            )
            response = {"completion": "candidate", "input_tokens": 2, "output_tokens": 3}
            arguments = [
                "gepa_reflection.py",
                "--run",
                "--config",
                str(config),
                "--log-dir",
                str(root / "reflection"),
            ]
            with (
                patch.object(sys, "argv", arguments),
                patch.object(sys, "stdin", io.StringIO("synthetic prompt")),
                patch.object(sys, "stdout", new_callable=io.StringIO) as stdout,
                patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
                patch.object(gepa_reflection.run_inspect, "isolate_environment"),
                patch.object(gepa_reflection, "generate", return_value=response),
            ):
                self.assertEqual(gepa_reflection.main(), 0)
                self.assertEqual(json.loads(stdout.getvalue()), response)


class GepaAlgorithmTests(unittest.TestCase):
    def test_installed_gepa_can_optimize_with_local_adapter(self):
        import gepa
        from gepa.core.adapter import EvaluationBatch

        class Adapter:
            propose_new_texts = None

            def evaluate(self, batch, candidate, capture_traces=False):
                score = float("improved" in candidate["instructions"])
                traces = [{"feedback": "Add the missing generic instruction."}] * len(batch)
                return EvaluationBatch(
                    outputs=traces,
                    scores=[score] * len(batch),
                    trajectories=traces if capture_traces else None,
                )

            def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
                return {name: eval_batch.trajectories for name in components_to_update}

        result = gepa.optimize(
            seed_candidate={"instructions": "initial"},
            trainset=["training"],
            valset=["validation"],
            adapter=Adapter(),
            reflection_lm=lambda prompt: "```text\nimproved\n```",
            reflection_minibatch_size=1,
            max_metric_calls=5,
            logger=run_gepa.QuietLogger(),
            display_progress_bar=False,
            use_wandb=False,
            use_mlflow=False,
        )
        self.assertEqual(result.best_candidate["instructions"].strip(), "improved")
