"""Verify that fresh comparison cases remain separate and internally consistent."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if importlib.util.find_spec("inspect_ai") is None:
    raise unittest.SkipTest("Inspect AI is an optional evaluation dependency")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import gepa_holdout
import run_gepa
from inspect_task import bundle_hash, skill_files


class HoldoutTests(unittest.TestCase):
    def test_holdout_summary_preserves_case_warnings(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            suite = root / "suite"
            suite.mkdir()
            candidate = "candidate instructions"
            config = root / "config.yaml"
            config.write_text("test", encoding="utf-8")
            bundle = bundle_hash({**skill_files(), "SKILL.md": candidate.encode()})

            def fake_trial(command, **kwargs):
                trial_dir = Path(command[command.index("--log-dir") + 1])
                trial_dir.mkdir(parents=True)
                report = {
                    "case_id": command[command.index("--case-id") + 1],
                    "status": "PASS",
                    "skill_sha256": bundle,
                    "score_detail": "passed",
                    "observations": ["schema warning"],
                }
                (trial_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
                return SimpleNamespace(returncode=0)

            with (
                patch.object(gepa_holdout, "prepare_suite", return_value=suite),
                patch.object(gepa_holdout.subprocess, "run", side_effect=fake_trial),
            ):
                result = gepa_holdout.run_holdout(root, config, candidate, candidate, 30)
            self.assertEqual(len(result["results"]["baseline"]), len(gepa_holdout.CASE_IDS))
            self.assertEqual(result["results"]["baseline"][0]["observations"], ["schema warning"])

    def test_generated_corpus_is_distinct_valid_and_keeps_aliases(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            suite = gepa_holdout.prepare_suite(Path(directory).relative_to(ROOT))
            self.assertTrue(suite.is_absolute())
            original = json.loads((ROOT / "evals/scenarios.json").read_text())
            fresh = json.loads((suite / "evals/scenarios.json").read_text())
            original_cases = {case["id"]: case for case in original["cases"]}
            fresh_cases = {case["id"]: case for case in fresh["cases"]}
            for case_id in gepa_holdout.CASE_IDS:
                with self.subTest(case_id=case_id):
                    self.assertNotEqual(
                        original_cases[case_id]["prompts"], fresh_cases[case_id]["prompts"]
                    )
                    self.assertNotEqual(
                        original_cases[case_id]["scope"], fresh_cases[case_id]["scope"]
                    )
            self.assertEqual(original_cases["novel-vocabulary"], fresh_cases["novel-vocabulary"])
            self.assertFalse(
                set(gepa_holdout.CASE_IDS)
                & (set(run_gepa.TRAIN_CASES) | set(run_gepa.VALIDATION_CASES))
            )
            program = """
import sys, runpy
sys.path.insert(0, "evals")
import behavior_cases, run_behavior, graders.behavior_grader as grader
namespace = runpy.run_path(sys.argv[1])
for case_id in sys.argv[2:]:
    answer, artifacts = namespace["evidence"](case_id)
    result = grader.grade(case_id, answer, artifacts)
    assert result["success"], (case_id, result["failures"])
"""
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    program,
                    str(ROOT / "tests/test_behavior_eval.py"),
                    *gepa_holdout.CASE_IDS,
                ],
                cwd=suite,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
