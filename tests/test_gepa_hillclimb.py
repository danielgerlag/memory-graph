"""Offline tests for the multi-model GEPA hill-climbing adapter."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if importlib.util.find_spec("inspect_ai") is None or importlib.util.find_spec("gepa") is None:
    raise unittest.SkipTest("GEPA and Inspect AI are optional evaluation dependencies")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import gepa_holdout
import run_gepa_hillclimb as hill
from inspect_task import bundle_hash


class HillClimbAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("candidates", "trials"):
            (self.root / name).mkdir()
        self.bundle = {"SKILL.md": b"---\nname: test\n---\noriginal\n", "script.py": b"fixed"}
        self.adapter = hill.MultiModelAdapter(
            root=self.root,
            config=self.root / "config.yaml",
            prefix="---\nname: test\n---\n",
            bundle=self.bundle,
            models=["model-a", "model-b"],
            workers=4,
            max_trials=50,
            timeout=30,
            env=None,
        )
        self.verdicts = {}
        self.calls = []
        self.lock = threading.Lock()

    def fake_trial(self, command, **kwargs):
        model = command[command.index("--model") + 1].rsplit("/", 1)[1]
        case = command[command.index("--case-id") + 1]
        with self.lock:
            self.calls.append((model, case))
            attempt = sum(1 for call in self.calls if call == (model, case))
        verdict = self.verdicts.get((model, case), "PASS")
        trial_dir = Path(command[command.index("--log-dir") + 1])
        trial_dir.mkdir()
        candidate = Path(command[command.index("--skill-instructions") + 1]).read_bytes()
        report = {
            "case_id": case,
            "skill_sha256": bundle_hash({**self.bundle, "SKILL.md": candidate}),
            "adapter_hashes": self.adapter.adapter_hashes,
            "corpus_source_sha256": self.adapter.corpus_hash,
            "score_detail": "checks text",
            "inspect_log": "unused",
        }
        if verdict == "INFRA_ONCE" and attempt == 1:
            report.update(status="ERROR", problems=["Inspect run did not complete successfully"])
        elif verdict == "CRASH_ONCE" and attempt == 1:
            report = {"status": "ERROR", "reason": "RuntimeError"}  # no identity fields at all
        elif verdict == "API_ERROR_ONCE" and attempt == 1:
            report.update(
                status="ERROR", problems=["sample error or limit exhaustion"], limit_kind="none"
            )
        elif verdict == "BUDGET":
            report.update(
                status="ERROR", problems=["sample error or limit exhaustion"], limit_kind="budget"
            )
        else:
            report["status"] = "FAIL" if verdict == "FAIL" else "PASS"
        (trial_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    def evaluate(self, cases, **kwargs):
        facts = ([{"command": "find x", "result": "ok"}], "answer", "token")

        def classify(report):
            return ("budget", "token") if report.get("limit_kind") == "budget" else ("infra", None)

        with (
            patch.object(hill.subprocess, "run", side_effect=self.fake_trial),
            patch.object(hill, "trial_facts", return_value=facts),
            patch.object(hill, "classify_error", side_effect=classify),
        ):
            return self.adapter.evaluate(cases, {"instructions": "generic\n"}, **kwargs)

    def test_case_score_is_the_mean_over_models_and_budget_counts_as_failure(self):
        self.verdicts = {("model-a", "scope-search"): "FAIL", ("model-b", "read-only"): "BUDGET"}
        result = self.evaluate(["scope-search", "read-only", "remember-fact"])
        self.assertEqual(result.scores, [0.5, 0.5, 1.0])
        self.assertEqual(
            result.outputs[1]["models"]["model-b"]["checks"],
            "ran out of its token budget before finishing",
        )

    def test_infrastructure_errors_are_retried_not_scored_as_failures(self):
        self.verdicts = {("model-a", "scope-search"): "INFRA_ONCE"}
        result = self.evaluate(["scope-search"])
        self.assertEqual(result.scores, [1.0])
        self.assertEqual(self.calls.count(("model-a", "scope-search")), 2)

    def test_a_crash_report_or_an_api_error_is_retried_not_fatal_and_not_a_budget_failure(self):
        for verdict in ("CRASH_ONCE", "API_ERROR_ONCE"):
            with self.subTest(verdict=verdict):
                self.calls.clear()
                self.verdicts = {("model-a", "scope-search"): verdict}
                result = self.evaluate(["scope-search"])
                self.assertEqual(result.scores, [1.0])
                self.assertEqual(self.calls.count(("model-a", "scope-search")), 2)
                self.assertEqual(result.outputs[0]["models"]["model-a"]["outcome"], "PASS")

    def test_reflection_feedback_shows_commands_and_verdicts_per_model(self):
        self.verdicts = {("model-b", "scope-search"): "FAIL"}
        result = self.evaluate(["scope-search"], capture_traces=True)
        record = self.adapter.make_reflective_dataset({}, result, ["instructions"])["instructions"][
            0
        ]
        self.assertIn("model-b: FAILED. checks text", record["Feedback"])
        self.assertIn("model-a: passed", record["Feedback"])
        self.assertEqual(set(record["Generated Outputs"]), {"model-a", "model-b"})
        self.assertTrue(record["Inputs"]["user_request"])

    def test_every_attempt_is_logged_for_the_breakdown(self):
        self.evaluate(["scope-search"])
        rows = [
            json.loads(line) for line in (self.root / "evaluations.jsonl").read_text().splitlines()
        ]
        self.assertEqual({row["model"] for row in rows}, {"model-a", "model-b"})
        self.assertTrue(all(row["outcome"] == "PASS" for row in rows))

    def test_trial_budget_is_a_hard_cap(self):
        self.adapter.max_trials = 1
        with self.assertRaisesRegex(hill.OptimizationError, "budget"):
            self.evaluate(["scope-search"])


class ReflectionPromptTests(unittest.TestCase):
    def test_targeted_edit_guidance_is_appended_once_and_survives_retries(self):
        seen = []

        class Inner:
            failure = None

            def __call__(self, prompt):
                seen.append(prompt)
                if len(seen) == 1:
                    raise hill.OptimizationError("transient")
                return "new instructions"

        with patch.object(hill.time, "sleep"):
            reflector = hill.RetryingReflector(Inner(), suffix=hill.TARGETED_EDITS)
            self.assertEqual(reflector("base prompt"), "new instructions")
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(p.count(hill.TARGETED_EDITS) == 1 for p in seen))
        self.assertTrue(seen[0].startswith("base prompt"))
        self.assertEqual(hill.RetryingReflector(Inner())("x"), "new instructions")


class RunManagementTests(unittest.TestCase):
    def args(self, **changes):
        base = {
            "models": ["m1", "m2"],
            "reflection_model": "r",
            "targeted_edits": True,
            "minibatch": 3,
            "resume": False,
        }
        return SimpleNamespace(**{**base, **changes})

    def test_resume_validates_before_spending_and_keeps_budgets_cumulative(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory) / "run"
            self.assertEqual(hill.prepare_run(root, self.args(), "seed"), (0, 0))
            with self.assertRaisesRegex(ValueError, "pass --resume"):
                hill.prepare_run(root, self.args(), "seed")
            (root / "evaluations.jsonl").write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
            (root / "reflections" / "one").mkdir()
            resumed = self.args(resume=True)
            self.assertEqual(hill.prepare_run(root, resumed, "seed"), (2, 1))
            for changed in (
                self.args(resume=True, models=["m1"]),
                self.args(resume=True, minibatch=2),
            ):
                with self.assertRaisesRegex(ValueError, "differ"):
                    hill.prepare_run(root, changed, "seed")
            with self.assertRaisesRegex(ValueError, "differ"):
                hill.prepare_run(root, resumed, "another seed")
            (root / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already finished"):
                hill.prepare_run(root, resumed, "seed")

    def test_holdout_candidates_with_one_name_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            first, second = Path(directory) / "a" / "best.md", Path(directory) / "b" / "best.md"
            for path in (first, second):
                path.parent.mkdir()
                path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "used twice"):
                hill.candidate_texts([first, second])
            self.assertEqual(hill.candidate_texts([first]), {"best": "x"})


class HoldoutMatrixTests(unittest.TestCase):
    def run_matrix(self, reports):
        """Run a one-case matrix whose trials write the given report text, in order."""
        calls = []
        real_run = subprocess.run

        def fake(command, **kwargs):
            if "--log-dir" not in command:  # building the variant corpus runs for real
                return real_run(command, **kwargs)
            trial_dir = Path(command[command.index("--log-dir") + 1])
            trial_dir.mkdir(parents=True)
            text = reports[min(len(calls), len(reports) - 1)]
            calls.append(command)
            candidate = Path(command[command.index("--skill-instructions") + 1]).read_bytes()
            bundle = bundle_hash({**hill.skill_files(), "SKILL.md": candidate})
            case = command[command.index("--case-id") + 1]
            text = text.replace("BUNDLE", bundle).replace("CASE", case)
            (trial_dir / "report.json").write_text(text, encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with (
            tempfile.TemporaryDirectory(dir=ROOT) as directory,
            patch.object(gepa_holdout.subprocess, "run", side_effect=fake),
        ):
            result = gepa_holdout.run_holdout_matrix(
                Path(directory),
                ROOT / "evals/configs/foundry.example.yaml",
                {"c": "---\nname: memory-graph\n---\nbody\n"},
                ["m"],
                ["hostile-node"],
                1,
                1,
                30,
            )
        return result["rows"], len(calls)

    def test_a_malformed_report_is_retried_and_never_aborts_the_matrix(self):
        ok = '{"status":"PASS","case_id":"CASE","skill_sha256":"BUNDLE","score_detail":"ok"}'
        rows, calls = self.run_matrix(["not json", ok])
        self.assertEqual((rows[0]["status"], calls), ("PASS", 2))

    def test_a_report_for_another_case_or_skill_is_an_error_not_a_result(self):
        wrong = '{"status":"PASS","case_id":"other","skill_sha256":"BUNDLE"}'
        rows, _ = self.run_matrix([wrong])
        self.assertEqual(rows[0]["status"], "ERROR")
        stale = '{"status":"PASS","case_id":"CASE","skill_sha256":"stale"}'
        self.assertEqual(self.run_matrix([stale])[0][0]["status"], "ERROR")

    def test_a_trial_that_never_produces_a_report_ends_as_error_after_retries(self):
        rows, calls = self.run_matrix(['{"status":"ERROR","reason":"RuntimeError"}'])
        self.assertEqual((rows[0]["status"], calls), ("ERROR", 3))


class CaseSetTests(unittest.TestCase):
    def test_training_validation_and_holdout_cases_are_disjoint_and_supported(self):
        sets = [set(hill.TRAIN_CASES), set(hill.VALIDATION_CASES), set(hill.HOLDOUT_CASES)]
        for index, first in enumerate(sets):
            for second in sets[index + 1 :]:
                self.assertFalse(first & second)
        self.assertTrue(set().union(*sets) <= set(hill.SPECS))
        self.assertFalse(set().union(*sets) & set(hill.UNSUPPORTED))

    def test_extended_holdout_variants_compile_with_fresh_vocabulary(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            suite = gepa_holdout.prepare_suite(Path(directory), gepa_holdout.EXTENDED_CASE_IDS)
            original = (ROOT / "evals/cases/behavior/novel-vocabulary.yaml").read_text()
            variant = (suite / "evals/cases/behavior/novel-vocabulary.yaml").read_text()
            self.assertIn("glimmer", original)
            self.assertNotIn("glimmer", variant)
            second = (suite / "evals/cases/behavior/remember-second-domain.yaml").read_text()
            self.assertNotIn("ceramic", second)


if __name__ == "__main__":
    unittest.main()
