"""Regression tests for declarative cases, real fixtures, and corpus integrity."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import corpus
from graders import imported


class CorpusTests(unittest.TestCase):
    def test_declarative_sources_reproduce_the_evidence_dataset(self):
        data, supported, blocked, fixtures, _ = corpus.verify_corpus()
        self.assertEqual(corpus.dataset_bytes(data), corpus.DATASET.read_bytes())
        imported.validate_dataset(data)
        self.assertEqual(len(data["cases"]), 64)
        self.assertEqual(len(supported), 21)
        self.assertEqual(
            set(blocked), {"proposal-not-main", "accepted-after-fetch", "concurrent-changes"}
        )
        self.assertEqual(len(fixtures), 24)
        scope_case = next(case for case in data["cases"] if case["id"] == "scope-search")
        self.assertEqual(scope_case["fixture"]["main"]["facts"][0]["label"], "ladder is in shed")

    def test_mutated_fixture_or_case_cannot_bypass_the_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(corpus.ROOT / "cases", root / "cases")
            shutil.copytree(corpus.ROOT / "fixtures" / "graphs", root / "fixtures" / "graphs")
            for name in ("corpus.yaml", "corpus.lock.json", "scenarios.json"):
                shutil.copyfile(corpus.ROOT / name, root / name)
            with patch.multiple(
                corpus,
                ROOT=root,
                DATASET=root / "scenarios.json",
                LOCK=root / "corpus.lock.json",
            ):
                corpus.verify_corpus()
                node = next((root / "fixtures/graphs/scope-search").glob("*.md"))
                original = node.read_bytes()
                node.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "corpus source or snapshot changed"):
                    corpus.verify_corpus()
                node.write_bytes(original)
                case = root / "cases/behavior/scope-search.yaml"
                case.write_text(case.read_text() + "\n# changed\n")
                with self.assertRaisesRegex(ValueError, "corpus source or snapshot changed"):
                    corpus.verify_corpus()

    def test_yaml_rejects_duplicate_keys_and_aliases(self):
        for text in (b"key: a\nkey: b\n", b"a: &value [1]\nb: *value\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                corpus.load_yaml(text, "fixture")

    def test_unknown_controlled_grader_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(corpus.ROOT / "cases", root / "cases")
            shutil.copytree(corpus.ROOT / "fixtures" / "graphs", root / "fixtures" / "graphs")
            shutil.copyfile(corpus.ROOT / "corpus.yaml", root / "corpus.yaml")
            case = root / "cases/behavior/scope-search.yaml"
            original = case.read_text()
            case.write_text(original.replace("kind: find", "kind: arbitrary-code"))
            with (
                patch.object(corpus, "ROOT", root),
                self.assertRaisesRegex(ValueError, "invalid controlled specification"),
            ):
                corpus.compile_corpus()

    def test_checked_in_smoke_manifest_still_targets_same_dataset(self):
        manifest = json.loads((corpus.ROOT / "native-smoke.json").read_text())
        lock = json.loads(corpus.LOCK.read_text())
        self.assertEqual(manifest["dataset_sha256"], lock["dataset_sha256"])

    def test_retry_seed_must_match_fixture(self):
        _, specs, _, fixtures, _ = corpus.compile_corpus()
        spec = specs["lost-ack-retry"]
        spec["seed_request"]["expires_at"] = "2027-01-01T00:00:00Z"
        fixture = fixtures["lost-ack-retry"]
        with self.assertRaisesRegex(ValueError, "retry seed differs"):
            corpus._validate_spec("lost-ack-retry", spec, fixture["ids"], fixture["nodes"])


if __name__ == "__main__":
    unittest.main()
