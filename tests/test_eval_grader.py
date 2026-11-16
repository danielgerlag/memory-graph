"""Tests of the evaluator itself; these synthetic values are not live evidence."""

import hashlib
import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from graders import imported as grader


def trace(**overrides):
    value = {
        "complete": True,
        "integrity_violations": [],
        "mutations": [],
        "mutation_records": [],
        "changed_paths": [],
        "actions": [],
    }
    value.update(overrides)
    return value


def simple_case():
    return {
        "id": "test",
        "lane": "behavior",
        "category": "recall",
        "fixture": {},
        "allowed_mutations": [],
        "forbidden_actions": [],
        "checks": [{"path": "/answer/facts", "op": "equals", "value": ["correct"]}],
    }


class PredicateTests(unittest.TestCase):
    def test_bad_answer_and_missing_outcome_fail(self):
        case = simple_case()
        for answer in ({"facts": ["incorrect"]}, {}):
            results = grader.grade_observation(case, {"trace": trace(), "answer": answer})
            self.assertFalse(all(item["pass"] for item in results))

    def test_scope_and_sources_are_not_ignored_by_contains(self):
        check = {
            "path": "/facts",
            "op": "contains",
            "value": {"label": "item", "scope": "a", "sources": ["source:1"]},
        }
        self.assertFalse(
            grader.grade_check(
                {"facts": [{"label": "item", "scope": "b", "sources": ["source:1"]}]}, check
            )["pass"]
        )
        self.assertFalse(
            grader.grade_check({"facts": [{"label": "item", "scope": "a", "sources": []}]}, check)[
                "pass"
            ]
        )
        self.assertTrue(
            grader.grade_check(
                {"facts": [{"label": "item", "scope": "a", "sources": ["source:1"], "id": "x"}]},
                check,
            )["pass"]
        )

    def test_generated_id_relationship_is_verified(self):
        check = {
            "path": "/facts",
            "op": "linked",
            "value": {
                "source": {"id": "old"},
                "target": {"label": "new"},
                "field": "superseded_by",
            },
        }
        facts = [
            {"id": "old", "superseded_by": "generated-123"},
            {"id": "generated-123", "label": "new"},
        ]
        self.assertTrue(grader.grade_check({"facts": facts}, check)["pass"])
        facts[0]["superseded_by"] = "elsewhere"
        self.assertFalse(grader.grade_check({"facts": facts}, check)["pass"])

    def test_missing_trace_and_forbidden_actions_are_failures(self):
        case = simple_case()
        case["forbidden_actions"] = ["repair"]
        for observed in (
            {},
            trace(actions=["repair"]),
            trace(actions=["force_push"]),
            trace(complete=False),
            trace(integrity_violations=["scope leak"]),
            trace(
                mutations=["node:update"],
                changed_paths=["node.md"],
                mutation_records=[{"path": "node.md", "kind": "node:update"}],
            ),
            trace(changed_paths=["../escape"]),
        ):
            checks = grader.grade_observation(
                case, {"trace": observed, "answer": {"facts": ["correct"]}}
            )
            self.assertFalse(all(item["pass"] for item in checks))

    def test_type_confusion_and_inconsistent_mutations_fail(self):
        self.assertFalse(
            grader.grade_check({"x": True}, {"path": "/x", "op": "equals", "value": 1})["pass"]
        )
        value = trace(mutations=["node:update"], changed_paths=["node.md"], mutation_records=[])
        checks = grader.grade_observation(
            simple_case(), {"trace": value, "answer": {"facts": ["correct"]}}
        )
        self.assertFalse(next(x for x in checks if x["path"] == "/trace/changed_paths")["pass"])


class GateTests(unittest.TestCase):
    def full_results(self):
        cases = []
        results = []
        for lane, count in (("behavior", 24), ("positive", 20), ("negative", 20)):
            for i in range(count):
                native = lane != "behavior"
                case = {
                    "id": f"{lane}-{i}",
                    "lane": "native" if native else "behavior",
                    "category": lane,
                    "critical": lane == "positive" and i == 0,
                }
                if native:
                    case["should_activate"] = lane == "positive"
                cases.append(case)
                for trial in range(1, 4):
                    result = {
                        "case_id": case["id"],
                        "lane": case["lane"],
                        "category": lane,
                        "trial": trial,
                        "status": "SCORED",
                        "success": True,
                        "critical": case["critical"],
                        "checks": [],
                    }
                    if native:
                        result.update(
                            expected_activation=case["should_activate"],
                            observed_activation=case["should_activate"],
                        )
                    results.append(result)
        return cases, results

    def test_complete_thresholds_are_not_a_release_or_host_claim(self):
        cases, results = self.full_results()
        report = {"results": results, "gates": {}}
        grader._gates(report, cases)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["gates"]["activation"], "PASS")
        self.assertEqual(report["gates"]["native_host_support"], "BLOCKED")
        self.assertEqual(report["gates"]["release"], "BLOCKED")

    def test_incomplete_coverage_and_undefined_activation_block(self):
        cases, results = self.full_results()
        results[0].update(status="MISSING", success=False)
        for item in results:
            if item.get("lane") == "native":
                item["observed_activation"] = False
        report = {"results": results, "gates": {}}
        grader._gates(report, cases)
        self.assertEqual(report["gates"]["coverage"], "BLOCKED")
        self.assertEqual(report["gates"]["behavior"], "BLOCKED")
        self.assertEqual(report["gates"]["activation"], "BLOCKED")
        self.assertIsNone(report["activation"]["precision"])

    def test_below_threshold_and_critical_failure(self):
        cases, results = self.full_results()
        for item in results:
            if item["case_id"].startswith("positive-") and int(item["case_id"].split("-")[1]) < 3:
                item.update(observed_activation=False, success=False)
        report = {"results": results, "gates": {}}
        grader._gates(report, cases)
        self.assertEqual(report["gates"]["activation"], "FAIL")
        self.assertEqual(report["gates"]["critical_native"], "FAIL")

    def test_category_failure_and_any_integrity_failure_block(self):
        cases, results = self.full_results()
        for case in cases:
            if case["id"] == "behavior-0":
                case["category"] = "separate"
        for item in results:
            if item["case_id"] == "behavior-0":
                item.update(category="separate", success=False)
                item["checks"] = [{"pass": False, "path": "/trace/actions"}]
        report = {"results": results, "gates": {}}
        grader._gates(report, cases)
        self.assertGreater(report["behavior"]["macro_success"], 0.9)
        self.assertEqual(report["gates"]["behavior"], "FAIL")
        self.assertEqual(report["gates"]["integrity"], "FAIL")

    def test_missing_negative_scenarios_cannot_pass(self):
        cases, results = self.full_results()
        cases = [c for c in cases if c["category"] != "negative"]
        results = [r for r in results if r["category"] != "negative"]
        report = {"results": results, "gates": {}}
        grader._gates(report, cases)
        self.assertEqual(report["gates"]["activation"], "BLOCKED")


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.prefix = ROOT / "tests" / (".eval-scratch-" + uuid.uuid4().hex)
        self.paths = []

    def tearDown(self):
        for path in self.paths:
            path.unlink(missing_ok=True)

    def write(self, name, value):
        path = self.prefix.with_name(self.prefix.name + "-" + name)
        path.write_text(value, encoding="utf-8")
        self.paths.append(path)
        return path

    def descriptor(self, path, role=None):
        value = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if role:
            value["role"] = role
        return value

    def fixture(self):
        dataset = {
            "schema_version": 1,
            "dataset_version": "test",
            "trial_count": 1,
            "cases": [simple_case()],
        }
        dataset_path = self.write("dataset.json", json.dumps(dataset))
        obs = self.write(
            "observation.json", json.dumps({"trace": trace(), "answer": {"facts": ["correct"]}})
        )
        transcript = self.write("transcript.jsonl", '{"type":"answer"}\n')
        before = self.write("before.json", '{"files":{},"refs":{}}')
        after = self.write("after.json", '{"files":{},"refs":{}}')
        actions = self.write("actions.json", "[]")
        artifacts = [
            self.descriptor(p, role)
            for p, role in (
                (transcript, "transcript"),
                (before, "graph_before"),
                (after, "graph_after"),
                (actions, "action_trace"),
            )
        ]
        entry = {
            "case_id": "test",
            "trial": 1,
            "status": "completed",
            "observation": self.descriptor(obs),
            "artifacts": artifacts,
        }
        manifest = {
            "schema_version": 1,
            "dataset_version": "test",
            "dataset_sha256": grader.digest(dataset_path),
            "condition": "candidate",
            "run_id": "synthetic-test",
            "adapter": "unit-test",
            "source_revision": "test",
            "skill_sha256": "a" * 64,
            "cli_sha256": "b" * 64,
            "schema_version_under_test": "test",
            "sandbox_image": "test@sha256:" + "c" * 64,
            "dependency_lock_sha256": "none",
            "python_version": "test",
            "git_version": "test",
            "isolation_profile": "test",
            "clock_policy": "fixed",
            "host": {"name": "test", "version": "1"},
            "model": {"id": "none", "configuration": {"mode": "synthetic"}},
            "limits": {
                "concurrency": 1,
                "timeout_seconds": 1,
                "budget": 1,
                "budget_currency": "USD",
            },
            "results": [entry],
        }
        return dataset_path, manifest, after

    def test_import_grades_and_mismatched_hash_is_error(self):
        dataset, manifest, after = self.fixture()
        evidence = self.write("manifest.json", json.dumps(manifest))
        report = grader.run(dataset, evidence)
        self.assertEqual(report["results"][0]["status"], "SCORED")
        self.assertTrue(report["results"][0]["success"])
        self.assertEqual(report["gates"]["behavior"], "BLOCKED")  # undersized dataset
        after.write_text('{"files":{"node.md":"' + "a" * 64 + '"},"refs":{}}', encoding="utf-8")
        report = grader.run(dataset, evidence)
        self.assertEqual(report["results"][0]["status"], "ERROR")
        self.assertIn("hash mismatch", report["results"][0]["error"])

    def test_unchanged_snapshot_is_checked_independently_of_trace(self):
        dataset, manifest, after = self.fixture()
        after.write_text('{"files":{"node.md":"' + "a" * 64 + '"},"refs":{}}', encoding="utf-8")
        manifest["results"][0]["artifacts"][2] = self.descriptor(after, "graph_after")
        evidence = self.write("manifest.json", json.dumps(manifest))
        report = grader.run(dataset, evidence)
        self.assertFalse(report["results"][0]["success"])
        self.assertEqual(report["gates"]["integrity"], "FAIL")

    def test_missing_outcome_is_scoring_error_not_wrong_answer(self):
        dataset, manifest, _ = self.fixture()
        entry = manifest["results"][0]
        observation = ROOT / "tests" / entry["observation"]["path"]
        observation.write_text(json.dumps({"trace": trace(), "answer": {}}), encoding="utf-8")
        entry["observation"] = self.descriptor(observation)
        evidence = self.write("manifest.json", json.dumps(manifest))
        report = grader.run(dataset, evidence)
        self.assertEqual(report["results"][0]["status"], "ERROR")
        self.assertIn("/answer/facts", report["results"][0]["error"])
        self.assertEqual(report["gates"]["coverage"], "BLOCKED")

    def test_missing_and_duplicate_trials_do_not_disappear(self):
        dataset, manifest, _ = self.fixture()
        manifest["results"] = []
        evidence = self.write("manifest.json", json.dumps(manifest))
        report = grader.run(dataset, evidence)
        self.assertEqual(report["counts"]["missing"], 1)
        self.assertEqual(report["gates"]["coverage"], "BLOCKED")
        _, manifest, _ = self.fixture()
        manifest["results"].append(manifest["results"][0])
        evidence.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(grader.EvidenceError, "duplicate case/trial"):
            grader.run(dataset, evidence)

    def test_escape_and_missing_artifact_rejected(self):
        dataset, manifest, _ = self.fixture()
        manifest["results"][0]["artifacts"][0]["path"] = "../outside"
        evidence = self.write("manifest.json", json.dumps(manifest))
        report = grader.run(dataset, evidence)
        self.assertEqual(report["results"][0]["status"], "ERROR")
        self.assertIn("escapes", report["results"][0]["error"])

    def test_native_activation_needs_complete_observation_and_real_line_reference(self):
        _, manifest, _ = self.fixture()
        entry = manifest["results"][0]
        observation_path = ROOT / "tests" / entry["observation"]["path"]
        observation = {
            "trace": trace(),
            "activation": {"detected": True, "observation_complete": True, "event_lines": [1]},
        }
        observation_path.write_text(json.dumps(observation), encoding="utf-8")
        entry["observation"] = self.descriptor(observation_path)
        case = {"lane": "native", "critical": False}
        _, activation, _ = grader._read_trial(ROOT / "tests", entry, case)
        self.assertTrue(activation["detected"])
        observation["activation"]["event_lines"] = [2]
        observation_path.write_text(json.dumps(observation), encoding="utf-8")
        entry["observation"] = self.descriptor(observation_path)
        with self.assertRaisesRegex(grader.EvidenceError, "line references"):
            grader._read_trial(ROOT / "tests", entry, case)
        observation["activation"]["event_lines"] = []
        observation_path.write_text(json.dumps(observation), encoding="utf-8")
        entry["observation"] = self.descriptor(observation_path)
        with self.assertRaisesRegex(grader.EvidenceError, "matching event line"):
            grader._read_trial(ROOT / "tests", entry, case)

    def test_no_evidence_is_not_run(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "evals" / "run.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "NOT RUN")
        self.assertEqual(report["gates"]["release"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
