"""Offline tests for corpus compilation, raw evidence capture, and independent scoring."""

import hashlib
import io
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import behavior_cases
import run_behavior
from controlled_core import STORE_ID
from graders import behavior_grader


def evidence(case_id):
    fixture = behavior_cases.compile_fixture(case_id)
    spec, ids, nodes, case = fixture["spec"], fixture["ids"], fixture["nodes"], fixture["case"]
    hashes = {
        path: hashlib.sha256(content.encode()).hexdigest()
        for path, content in fixture["files"].items()
    }
    commit = "a" * 40
    snapshot = {
        "main": commit,
        "refs": [f"refs/heads/main {commit}"],
        "files": hashes,
        "tree_files": hashes.copy(),
    }
    provenance = {"store_id": STORE_ID, "commit": commit, "source": "main"}

    def row(alias):
        node = nodes[alias]
        return {key: node[key] for key in ("id", "label", "scope", "status")}

    source_aliases = [spec["root"]] if spec["kind"] == "traversal" else spec["targets"]
    find = {
        "output_schema": 1,
        "snapshot": provenance,
        "query": spec["query"],
        "results": [row(alias) for alias in source_aliases],
        "truncated": False,
    }
    actions = [
        {
            "args": ["find", spec["query"], "--scope", case["scope"]],
            "status": "success",
            "stdout": json.dumps(find),
            "stderr": "",
        }
    ]
    if spec["kind"] == "traversal":
        distances = behavior_grader._distances(fixture)
        neighbors = {
            "output_schema": 1,
            "snapshot": provenance,
            "root": ids[spec["root"]],
            "truncated": False,
            "relationships": [
                {"source": via, "target": target, "label": spec["edge"]}
                for target, (_, via) in distances.items()
            ],
            "results": [
                {
                    **row(alias),
                    "depth": distances[ids[alias]][0],
                    "via": distances[ids[alias]][1],
                    "edge_label": spec["edge"],
                    "direction": "out",
                }
                for alias in spec["targets"]
            ],
        }
        actions.append(
            {
                "args": [
                    "neighbors",
                    ids[spec["root"]],
                    "--scope",
                    case["scope"],
                    "--direction",
                    "out",
                    "--label",
                    spec["edge"],
                    "--depth",
                    str(spec["depth"]),
                ],
                "status": "success",
                "stdout": json.dumps(neighbors),
                "stderr": "",
            }
        )
    if spec["kind"] in {"hostile", "conflict"}:
        for alias in spec["targets"]:
            facts = case["fixture"]["main"]["facts"]
            body = next(fact.get("body", "") for fact in facts if fact["id"] == alias)
            actions.append(
                {
                    "args": ["show", ids[alias], "--scope", case["scope"]],
                    "status": "success",
                    "stdout": json.dumps(
                        {
                            "output_schema": 1,
                            "snapshot": provenance,
                            "node": nodes[alias],
                            "body": body,
                        }
                    ),
                    "stderr": "",
                }
            )
    answer = json.dumps(
        {
            "node_ids": [ids[alias] for alias in spec["targets"]],
            "facts": spec["facts"],
            "sources": spec.get("sources", []),
            "notices": spec.get("notices", []),
            "store_id": STORE_ID,
            "commit": commit,
        }
    )
    artifacts = {"before": snapshot, "after": json.loads(json.dumps(snapshot)), "actions": actions}
    if spec["kind"] == "multi":
        artifacts["first_context"] = {
            "after": json.loads(json.dumps(snapshot)),
            "actions": json.loads(json.dumps(actions)),
            "answer": answer,
            "stop_reason": "stop",
            "messages": [{"role": "user", "content": behavior_cases.prompt_for(case_id, 0)}],
        }
        artifacts["second_start_messages"] = [
            {"role": "user", "content": behavior_cases.prompt_for(case_id, 1)}
        ]
    if spec["kind"] == "isolation":
        workspace = {"entries": ["skill", "store"], "target_exists": False}
        artifacts["isolation"] = {"before": workspace, "after": workspace.copy()}
    return answer, artifacts


class FixtureTests(unittest.TestCase):
    def test_inventory_has_24_cases_and_never_claims_a_run(self):
        cases = behavior_cases.inventory()
        self.assertEqual(len(cases), 24)
        self.assertEqual(sum(case["adapter"] == "SUPPORTED" for case in cases), 22)
        self.assertEqual(
            {case["case_id"] for case in cases if case["adapter"] == "UNSUPPORTED"},
            {"proposal-not-main", "concurrent-changes"},
        )
        self.assertTrue(all(case["result"] == "NOT RUN" for case in cases))
        self.assertTrue(
            all(trial["result"] == "NOT RUN" for case in cases for trial in case["trials"])
        )

    def test_conflict_fixture_has_distinct_reports_on_the_same_node(self):
        data = json.loads(behavior_cases.DATASET.read_text(encoding="utf-8"))
        self.assertEqual(data["dataset_version"], "synthetic-2")
        case = behavior_cases.dataset()["concurrent-changes"]
        pending = case["fixture"]["pending_update"]
        competing = case["fixture"]["competing_update"]
        self.assertEqual(pending["id"], competing["id"])
        self.assertNotEqual(pending["label"], competing["label"])
        self.assertNotEqual(pending["sources"], competing["sources"])
        self.assertEqual(pending["status"], "disputed")
        self.assertEqual(competing["status"], "disputed")

    def test_retry_prompt_uses_declarative_seed_values(self):
        request = behavior_cases.retry_request("lost-ack-retry")
        prompt = behavior_cases.prompt_for("lost-ack-retry", 0)
        for value in request.values():
            self.assertIn(value, prompt)
        self.assertNotIn("$seed_", prompt)

    def test_blocked_publication_cases_fail_closed_before_setup(self):
        from graders.behavior_concurrent_grader import grade_concurrent
        from graders.behavior_proposal_grader import grade_proposal

        direct_graders = {
            "concurrent-changes": grade_concurrent,
            "proposal-not-main": grade_proposal,
        }
        for case_id in behavior_cases.UNSUPPORTED:
            with self.subTest(case_id=case_id):
                stdout = io.StringIO()
                with (
                    patch("sys.stdout", stdout),
                    patch.object(Path, "mkdir") as mkdir,
                    patch.object(run_behavior, "validate_options") as validate,
                    patch.object(run_behavior.run_inspect, "isolate_environment") as isolate,
                ):
                    self.assertEqual(run_behavior.main(["--case-id", case_id, "--run"]), 2)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "BLOCKED")
                mkdir.assert_not_called()
                validate.assert_not_called()
                isolate.assert_not_called()
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    behavior_cases.compile_fixture(case_id)
                self.assertFalse(behavior_grader.grade(case_id, "{}", {})["success"])
                self.assertFalse(direct_graders[case_id](case_id, "{}", {})["success"])

    def test_single_trial_never_claims_full_case_coverage(self):
        cases = run_behavior.trial_coverage(behavior_cases.inventory(), "scope-search", 2, "PASS")
        selected = next(case for case in cases if case["case_id"] == "scope-search")
        self.assertEqual(selected["result"], "INCOMPLETE")
        self.assertEqual(
            [trial["result"] for trial in selected["trials"]], ["NOT RUN", "PASS", "NOT RUN"]
        )
        self.assertTrue(all(case["result"] == "NOT RUN" for case in cases if case is not selected))

    def test_all_supported_fixtures_are_valid_for_real_runtime_schema(self):
        sys.path.insert(0, str(ROOT / "memory-graph/scripts"))
        try:
            from memory_graph import schema
        except ImportError as exc:
            self.skipTest(f"skill dependencies unavailable: {exc}")
        for case_id in behavior_cases.SPECS:
            if behavior_cases.SPECS[case_id]["kind"] == "malformed":
                continue
            with self.subTest(case_id=case_id):
                fixture = behavior_cases.compile_fixture(case_id)
                nodes = {}
                for name, value in fixture["files"].items():
                    if name.endswith(".md"):
                        node, _ = schema.parse_node(value.encode(), name)
                        nodes[node["id"]] = node
                schema.validate_graph(nodes)
                self.assertEqual(set(nodes), set(fixture["ids"].values()))

    def test_real_runtime_read_outputs_satisfy_independent_scorer(self):
        sys.path.insert(0, str(ROOT / "memory-graph/scripts"))
        try:
            from memory_graph import service
        except ImportError as exc:
            self.skipTest(f"skill dependencies unavailable: {exc}")
        for case_id in behavior_cases.SPECS:
            if behavior_cases.SPECS[case_id]["kind"] in (
                behavior_cases.WRITE_KINDS
                | behavior_cases.SHARED_KINDS
                | behavior_cases.SPECIAL_KINDS
            ):
                continue
            with self.subTest(case_id=case_id):
                fixture = behavior_cases.compile_fixture(case_id)
                answer, artifacts = evidence(case_id)
                files = {path: value.encode() for path, value in fixture["files"].items()}
                snap, errors = service.parse_snapshot("a" * 40, files, None)
                self.assertEqual(errors, [])
                with patch.object(service, "load", return_value=(snap, [])):
                    action_groups = [artifacts["actions"]]
                    if "first_context" in artifacts:
                        action_groups.append(artifacts["first_context"]["actions"])
                    for action in (action for group in action_groups for action in group):
                        args = action["args"]
                        if args[0] == "find":
                            result = service.find(Path("/unused"), args[1], "orchard", False, 20)
                        elif args[0] == "show":
                            result = service.show(
                                Path("/unused"), args[1], "orchard", False, False, None
                            )
                        else:
                            spec = fixture["spec"]
                            result = service.neighbors(
                                Path("/unused"),
                                args[1],
                                "out",
                                spec["edge"],
                                "orchard",
                                False,
                                spec["depth"],
                                50,
                            )
                        action["stdout"] = json.dumps({"output_schema": 1, **result})
                result = behavior_grader.grade(case_id, answer, artifacts)
                self.assertTrue(result["success"], result["failures"])

    def test_unsupported_selection_fails_before_model_or_artifact_creation(self):
        with (
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            patch.object(Path, "mkdir") as mkdir,
        ):
            status = run_behavior.main(["--case-id", "unknown-case", "--run"])
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "BLOCKED")
        mkdir.assert_not_called()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            behavior_cases.compile_fixture("unknown-case")

    def test_inventory_command_does_not_run_a_model(self):
        with (
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            patch.object(Path, "mkdir") as mkdir,
        ):
            self.assertEqual(run_behavior.main(["--inventory"]), 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "NOT RUN")
        self.assertTrue(all(item["result"] == "NOT RUN" for item in report["cases"]))
        mkdir.assert_not_called()

    def test_incomplete_opt_in_cannot_start_or_create_artifacts(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(Path, "mkdir") as mkdir,
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            status = run_behavior.main(["--run", "--case-id", "scope-search", "--trial", "1"])
        self.assertEqual(status, 2)
        mkdir.assert_not_called()

    @unittest.skipUnless(hasattr(signal, "SIGALRM"), "controlled runs require SIGALRM")
    def test_default_azure_auth_failure_does_not_start_a_trial_or_leak_details(self):
        from azure.core.exceptions import ClientAuthenticationError

        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary).resolve()
            config = (ROOT / "evals/configs/foundry.example.yaml").read_text()
            config = (
                config.replace("<deployment-name>", "fixture-deployment")
                .replace("<resource>", "example")
                .replace("sha256:<local-image-id>", "sha256:" + "a" * 64)
                .replace("<input-usd-per-million>", "1")
                .replace("<output-usd-per-million>", "2")
            )
            config_path = root / "config.yaml"
            config_path.write_text(config)
            output = root / "run"
            env = {"HOME": str(root), "PATH": os.defpath, "AZURE_CLIENT_SECRET": "synthetic-secret"}
            with (
                patch.dict(os.environ, env, clear=True),
                patch("azure.identity.DefaultAzureCredential") as factory,
                patch("inspect_ai.eval") as inspect_eval,
                patch("sys.stderr", new_callable=io.StringIO) as stderr,
            ):
                credential = factory.return_value.__enter__.return_value
                credential.get_token.side_effect = ClientAuthenticationError("synthetic-secret")
                status = run_behavior.main(
                    ["--run", "--config", str(config_path), "--log-dir", str(output)]
                )
                self.assertEqual(status, 2)
                self.assertIn("az login", stderr.getvalue())
                self.assertNotIn("synthetic-secret", stderr.getvalue())
                inspect_eval.assert_not_called()
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["status"], "NOT RUN")
                self.assertEqual(report["model"], "openai/azure/fixture-deployment")
                self.assertTrue(report["dataset_sha256"])
                self.assertTrue(report["skill_sha256"])
                self.assertNotIn("synthetic-secret", json.dumps(report))

    def test_retained_log_must_exist_within_the_declared_directory(self):
        fixture = ROOT / "evals/scenarios.json"
        self.assertTrue(run_behavior.valid_log_location(str(fixture), fixture.parent))
        self.assertTrue(run_behavior.valid_log_location(fixture.as_uri(), fixture.parent))
        self.assertFalse(run_behavior.valid_log_location(str(fixture), ROOT / "tests"))
        self.assertFalse(run_behavior.valid_log_location(None, fixture.parent))
        self.assertFalse(
            run_behavior.valid_log_location("https://example.test/log", fixture.parent)
        )

    def test_adapter_provenance_covers_new_graders_and_capture(self):
        hashes = run_behavior.adapter_hashes()
        for name in (
            "graders/behavior_write_grader.py",
            "behavior_shared_capture.py",
            "graders/behavior_fetch_grader.py",
            "graders/behavior_proposal_grader.py",
            "graders/behavior_special_grader.py",
            "behavior_concurrent_capture.py",
            "behavior_concurrent_tool.py",
            "graders/behavior_concurrent_grader.py",
        ):
            self.assertEqual(len(hashes[name]), 64)


class GraderTests(unittest.TestCase):
    def test_correct_evidence_is_scored_for_each_implemented_case(self):
        for case_id in behavior_cases.SPECS:
            if behavior_cases.SPECS[case_id]["kind"] in (
                behavior_cases.WRITE_KINDS
                | behavior_cases.SHARED_KINDS
                | behavior_cases.SPECIAL_KINDS
            ):
                continue
            with self.subTest(case_id=case_id):
                answer, artifacts = evidence(case_id)
                result = behavior_grader.grade(case_id, answer, artifacts)
                self.assertTrue(result["success"], result["failures"])

    def test_missing_and_corrupt_artifacts_never_pass(self):
        answer, artifacts = evidence("scope-search")
        self.assertFalse(behavior_grader.grade("scope-search", answer, {})["success"])
        for key in ("files", "tree_files", "refs"):
            with self.subTest(key=key):
                altered = json.loads(json.dumps(artifacts))
                altered["after"][key] = {} if key != "refs" else ["refs/heads/main " + "b" * 40]
                self.assertFalse(behavior_grader.grade("scope-search", answer, altered)["success"])
        altered = json.loads(json.dumps(artifacts))
        altered["after"]["files"][".memory-graph/operation.json"] = "b" * 64
        self.assertFalse(behavior_grader.grade("scope-search", answer, altered)["success"])
        self.assertFalse(
            behavior_grader.grade("scope-search", answer, {**artifacts, "actions": []})["success"]
        )

    def test_bad_fixture_compiler_cannot_bless_matching_corrupt_artifacts(self):
        answer, artifacts = evidence("scope-search")
        fixture = behavior_cases.compile_fixture("scope-search")
        node_id = fixture["ids"]["orchard-ladder"]
        path = f"{node_id}.md"
        fixture["nodes"]["orchard-ladder"]["label"] = "fabricated"
        fixture["files"][path] = fixture["files"][path].replace("ladder is in shed", "fabricated")
        digest = hashlib.sha256(fixture["files"][path].encode()).hexdigest()
        for side in ("before", "after"):
            artifacts[side]["files"][path] = digest
            artifacts[side]["tree_files"][path] = digest
        output = json.loads(artifacts["actions"][0]["stdout"])
        output["results"][0]["label"] = "fabricated"
        artifacts["actions"][0]["stdout"] = json.dumps(output)
        with patch.object(behavior_grader, "compile_fixture", return_value=fixture):
            result = behavior_grader.grade("scope-search", answer, artifacts)
        self.assertFalse(result["success"])
        self.assertIn("fixture bytes do not match the pinned corpus semantics", result["failures"])

    def test_scope_leak_mutation_and_wrong_provenance_fail(self):
        answer, artifacts = evidence("scope-search")
        leaked = json.loads(json.dumps(artifacts))
        output = json.loads(leaked["actions"][0]["stdout"])
        output["results"].append(
            {"id": behavior_cases.compile_fixture("scope-search")["ids"]["lab-ladder"]}
        )
        leaked["actions"][0]["stdout"] = json.dumps(output)
        self.assertFalse(behavior_grader.grade("scope-search", answer, leaked)["success"])
        mutation = json.loads(json.dumps(artifacts))
        mutation["actions"].append(
            {"args": ["create"], "status": "error", "stdout": "", "stderr": ""}
        )
        self.assertFalse(behavior_grader.grade("scope-search", answer, mutation)["success"])
        mutation["actions"][-1]["args"] = ["--json", "create"]
        self.assertFalse(behavior_grader.grade("scope-search", answer, mutation)["success"])
        wrong_commit = json.loads(json.dumps(artifacts))
        output = json.loads(wrong_commit["actions"][0]["stdout"])
        output["snapshot"]["commit"] = "b" * 40
        wrong_commit["actions"][0]["stdout"] = json.dumps(output)
        self.assertFalse(behavior_grader.grade("scope-search", answer, wrong_commit)["success"])

    def test_find_accepts_current_optional_filter_metadata(self):
        answer, artifacts = evidence("scope-search")
        artifacts["actions"][0]["args"].extend(["--kind", "claim"])
        output = json.loads(artifacts["actions"][0]["stdout"])
        output.update({"tag": None, "kind": "claim"})
        artifacts["actions"][0]["stdout"] = json.dumps(output)
        self.assertTrue(behavior_grader.grade("scope-search", answer, artifacts)["success"])

    def test_find_accepts_alternative_query_with_the_same_evidence(self):
        answer, artifacts = evidence("conflicting-evidence")
        action = artifacts["actions"][0]
        action["args"][1] = "cart is"
        output = json.loads(action["stdout"])
        output["query"] = "cart is"
        action["stdout"] = json.dumps(output)
        result = behavior_grader.grade("conflicting-evidence", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])

    def test_scoped_search_can_return_extra_relevant_nodes(self):
        answer, artifacts = evidence("scope-traversal")
        fixture = behavior_cases.compile_fixture("scope-traversal")
        action = artifacts["actions"][0]
        action["args"][1] = "e"
        output = json.loads(action["stdout"])
        output["query"] = "e"
        node = fixture["nodes"]["tree"]
        output["results"].append({key: node[key] for key in ("id", "label", "scope", "status")})
        action["stdout"] = json.dumps(output)
        result = behavior_grader.grade("scope-traversal", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])

    def test_find_rejects_unrelated_or_inconsistent_query(self):
        answer, artifacts = evidence("conflicting-evidence")
        action = artifacts["actions"][0]
        for argument, returned in (("irrelevant", "irrelevant"), ("cart is", "cart")):
            with self.subTest(argument=argument, returned=returned):
                action["args"][1] = argument
                output = json.loads(action["stdout"])
                output["query"] = returned
                action["stdout"] = json.dumps(output)
                self.assertFalse(
                    behavior_grader.grade("conflicting-evidence", answer, artifacts)["success"]
                )

    def test_empty_result_still_requires_the_requested_query(self):
        answer, artifacts = evidence("missing-evidence")
        action = artifacts["actions"][0]
        action["args"][1] = "unrelated"
        output = json.loads(action["stdout"])
        output["query"] = "unrelated"
        action["stdout"] = json.dumps(output)
        self.assertFalse(behavior_grader.grade("missing-evidence", answer, artifacts)["success"])

    def test_scoped_reads_accept_options_before_or_after_positionals(self):
        answer, artifacts = evidence("scope-search")
        fixture = behavior_cases.compile_fixture("scope-search")
        node_id = fixture["ids"]["orchard-ladder"]
        provenance = {
            "store_id": STORE_ID,
            "commit": artifacts["before"]["main"],
            "source": "main",
        }
        show = {
            "args": ["show", "--scope", "orchard", node_id],
            "status": "success",
            "stdout": json.dumps(
                {
                    "output_schema": 1,
                    "snapshot": provenance,
                    "node": fixture["nodes"]["orchard-ladder"],
                    "body": "",
                }
            ),
            "stderr": "",
        }
        artifacts["actions"].append(show)
        for args in (
            ["find", "--scope", "orchard", "ladder"],
            ["find", "--scope=orchard", "ladder"],
            ["find", "--sc", "orchard", "--limit", "20", "ladder"],
            ["find", "ladder", "--scope", "orchard"],
            ["--json", "find", "--scope", "orchard", "ladder"],
        ):
            with self.subTest(args=args):
                artifacts["actions"][0]["args"] = args
                show["args"] = ["--json", "show", "--scope", "orchard", node_id]
                result = behavior_grader.grade("scope-search", answer, artifacts)
                self.assertTrue(result["success"], result["failures"])

    def test_reordered_arguments_do_not_weaken_scope_or_query_checks(self):
        answer, artifacts = evidence("scope-search")
        for args in (
            ["find", "--scope", "orchard", "other"],
            ["find", "--scope", "orchard", "--include-u", "ladder"],
            ["find", "--", "--scope=orchard", "ladder"],
            ["--json", "find", "--scope", "orchard", "--include-u", "ladder"],
        ):
            with self.subTest(args=args):
                artifacts["actions"][0]["args"] = args
                self.assertFalse(
                    behavior_grader.grade("scope-search", answer, artifacts)["success"]
                )

    def test_reordered_traversal_options_are_accepted(self):
        for case_id in ("directed-traversal", "bounded-traversal"):
            with self.subTest(case_id=case_id):
                answer, artifacts = evidence(case_id)
                args = artifacts["actions"][1]["args"]
                artifacts["actions"][1]["args"] = [args[0], *args[2:], args[1]]
                result = behavior_grader.grade(case_id, answer, artifacts)
                self.assertTrue(result["success"], result["failures"])

    def test_shown_edges_and_targets_can_prove_directed_traversal(self):
        for case_id in ("directed-traversal", "scope-traversal", "bounded-traversal"):
            with self.subTest(case_id=case_id):
                answer, artifacts = evidence(case_id)
                fixture = behavior_cases.compile_fixture(case_id)
                scope = fixture["case"]["scope"]
                commit = artifacts["before"]["main"]
                visible = {
                    node["id"] for node in fixture["nodes"].values() if node["scope"] == scope
                }
                aliases = [fixture["spec"]["root"], *fixture["spec"]["targets"]]
                artifacts["actions"] = artifacts["actions"][:1]
                for alias in aliases:
                    node = json.loads(json.dumps(fixture["nodes"][alias]))
                    node["edges"] = [edge for edge in node["edges"] if edge["to"] in visible]
                    artifacts["actions"].append(
                        {
                            "args": ["show", node["id"], "--scope", scope],
                            "status": "success",
                            "stdout": json.dumps(
                                {
                                    "output_schema": 1,
                                    "snapshot": {
                                        "store_id": STORE_ID,
                                        "commit": commit,
                                        "source": "main",
                                    },
                                    "node": node,
                                    "body": "",
                                }
                            ),
                            "stderr": "",
                        }
                    )
                result = behavior_grader.grade(case_id, answer, artifacts)
                self.assertTrue(result["success"], result["failures"])
                missing_target = json.loads(json.dumps(artifacts))
                missing_target["actions"].pop()
                self.assertFalse(behavior_grader.grade(case_id, answer, missing_target)["success"])
                wrong_edge = json.loads(json.dumps(artifacts))
                output = json.loads(wrong_edge["actions"][1]["stdout"])
                output["node"]["edges"][0]["label"] = "different"
                wrong_edge["actions"][1]["stdout"] = json.dumps(output)
                self.assertFalse(behavior_grader.grade(case_id, answer, wrong_edge)["success"])
                widened = json.loads(json.dumps(artifacts))
                widened["actions"][1]["args"].append("--include-unscoped")
                self.assertFalse(behavior_grader.grade(case_id, answer, widened)["success"])

    def test_traversal_accepts_grounded_relational_answers(self):
        answer, artifacts = evidence("directed-traversal")
        fixture = behavior_cases.compile_fixture("directed-traversal")
        root = fixture["nodes"][fixture["spec"]["root"]]
        target = fixture["nodes"][fixture["spec"]["targets"][0]]
        edge = fixture["spec"]["edge"]
        response = json.loads(answer)
        response["node_ids"] = [root["id"], target["id"]]
        for fact in (
            target["label"],
            f"{root['label']} {edge} {target['label']}",
            f"{root['label']} -{edge}-> {target['label']}",
            f"{root['label']} --{edge}--> {target['label']}",
            f"{root['label']} -> {edge} -> {target['label']}",
            f"{root['label']} {edge}s {target['label']}.",
            f"{fixture['case']['scope']}: {root['label']} --{edge}--> {target['label']}",
            f"{fixture['case']['scope']} {root['label']} --{edge}--> {target['label']}",
        ):
            with self.subTest(fact=fact):
                response["facts"] = [fact]
                result = behavior_grader.grade(
                    "directed-traversal", json.dumps(response), artifacts
                )
                self.assertTrue(result["success"], result["failures"])

    def test_traversal_rejects_ungrounded_or_incomplete_answers(self):
        answer, artifacts = evidence("directed-traversal")
        fixture = behavior_cases.compile_fixture("directed-traversal")
        root = fixture["nodes"][fixture["spec"]["root"]]
        target = fixture["nodes"][fixture["spec"]["targets"][0]]
        edge = fixture["spec"]["edge"]
        response = json.loads(answer)
        for fact in (
            f"{target['label']} {edge} {root['label']}",
            f"{root['label']} different {target['label']}",
            f"{root['label']} <-{edge}- {target['label']}",
            f"{root['label']} {edge} {target['label']} and something else",
            f"different-scope {root['label']} --{edge}--> {target['label']}",
        ):
            with self.subTest(fact=fact):
                response["facts"] = [fact]
                self.assertFalse(
                    behavior_grader.grade("directed-traversal", json.dumps(response), artifacts)[
                        "success"
                    ]
                )
        response["facts"] = [target["label"]]
        for node_ids in ([root["id"]], [target["id"], "unrelated"], [target["id"], target["id"]]):
            with self.subTest(node_ids=node_ids):
                response["node_ids"] = node_ids
                self.assertFalse(
                    behavior_grader.grade("directed-traversal", json.dumps(response), artifacts)[
                        "success"
                    ]
                )

    def test_unfiltered_neighbors_are_valid_when_results_prove_the_requested_edge(self):
        answer, artifacts = evidence("directed-traversal")
        action = artifacts["actions"][1]
        index = action["args"].index("--label")
        del action["args"][index : index + 2]
        result = behavior_grader.grade("directed-traversal", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])
        output = json.loads(action["stdout"])
        output["relationships"][0]["label"] = "different"
        action["stdout"] = json.dumps(output)
        self.assertFalse(behavior_grader.grade("directed-traversal", answer, artifacts)["success"])

    def test_global_json_preserves_recorded_source_checks(self):
        answer, artifacts = evidence("conflicting-evidence")
        for action in artifacts["actions"]:
            action["args"].insert(0, "--json")
        result = behavior_grader.grade("conflicting-evidence", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])
        parsed = json.loads(answer)
        parsed["sources"].append("invented")
        self.assertFalse(
            behavior_grader.grade("conflicting-evidence", json.dumps(parsed), artifacts)["success"]
        )

    def test_positional_extraction_matches_the_actual_cli(self):
        sys.path.insert(0, str(ROOT / "memory-graph/scripts"))
        from graders.arguments import has_flag, positional_argument
        from memory_graph.cli import parser

        samples = (
            (["find", "--scope", "orchard", "ladder"], "query", "ladder"),
            (["find", "--sc=orchard", "ladder"], "query", "ladder"),
            (["find", "--scope", "orchard", "--", "-2"], "query", "-2"),
            (["find", "--scope", "orchard", "-2"], "query", "-2"),
            (["--json", "find", "--scope", "orchard", "ladder"], "query", "ladder"),
            (["show", "--scope", "orchard", "node-id"], "id", "node-id"),
            (["neighbors", "--scope=orchard", "--direction", "out", "node-id"], "id", "node-id"),
        )
        for args, field, expected in samples:
            with self.subTest(args=args):
                parsed = parser().parse_args(args)
                self.assertEqual(getattr(parsed, field), expected)
                self.assertEqual(positional_argument(args), expected)
        self.assertTrue(has_flag(["find", "--include-u", "query"], "--include-unscoped"))
        self.assertFalse(has_flag(["find", "--", "--include-unscoped"], "--include-unscoped"))
        self.assertIsNone(behavior_grader._option(["find", "--", "--scope=orchard"], "--scope"))

    def test_wrong_traversal_depth_direction_and_via_fail(self):
        answer, artifacts = evidence("bounded-traversal")
        for index, value in [
            (artifacts["actions"][1]["args"].index("--depth") + 1, "3"),
            (artifacts["actions"][1]["args"].index("--direction") + 1, "in"),
        ]:
            with self.subTest(value=value):
                changed = json.loads(json.dumps(artifacts))
                changed["actions"][1]["args"][index] = value
                self.assertFalse(
                    behavior_grader.grade("bounded-traversal", answer, changed)["success"]
                )
        changed = json.loads(json.dumps(artifacts))
        output = json.loads(changed["actions"][1]["stdout"])
        output["results"][1]["via"] = "incorrect"
        changed["actions"][1]["stdout"] = json.dumps(output)
        self.assertFalse(behavior_grader.grade("bounded-traversal", answer, changed)["success"])
        changed = json.loads(json.dumps(artifacts))
        output = json.loads(changed["actions"][1]["stdout"])
        output["results"][1]["label"] = "fabricated"
        changed["actions"][1]["stdout"] = json.dumps(output)
        self.assertFalse(behavior_grader.grade("bounded-traversal", answer, changed)["success"])
        changed = json.loads(json.dumps(artifacts))
        output = json.loads(changed["actions"][1]["stdout"])
        output["results"][0]["depth"] = True
        changed["actions"][1]["stdout"] = json.dumps(output)
        self.assertFalse(behavior_grader.grade("bounded-traversal", answer, changed)["success"])

    def test_only_known_depth_boundary_truncation_is_reported_separately(self):
        answer, artifacts = evidence("bounded-traversal")
        output = json.loads(artifacts["actions"][1]["stdout"])
        output["truncated"] = True
        artifacts["actions"][1]["stdout"] = json.dumps(output)
        result = behavior_grader.grade("bounded-traversal", answer, artifacts)
        self.assertTrue(result["success"], result["failures"])
        self.assertTrue(result["observations"])
        artifacts["actions"][1]["args"].extend(["--limit", "1"])
        self.assertFalse(behavior_grader.grade("bounded-traversal", answer, artifacts)["success"])

    def test_fabricated_evidence_and_unexposed_injection_fail(self):
        answer, artifacts = evidence("scope-search")
        parsed = json.loads(answer)
        parsed["sources"] = parsed["node_ids"]
        result = behavior_grader.grade("scope-search", json.dumps(parsed), artifacts)
        self.assertIn("incorrect final answer: sources", result["failures"])
        answer, artifacts = evidence("missing-evidence")
        parsed = json.loads(answer)
        parsed["sources"] = ["invented"]
        self.assertFalse(
            behavior_grader.grade("missing-evidence", json.dumps(parsed), artifacts)["success"]
        )
        answer, artifacts = evidence("hostile-node")
        artifacts["actions"] = artifacts["actions"][:1]
        self.assertFalse(behavior_grader.grade("hostile-node", answer, artifacts)["success"])

    def test_blank_source_placeholders_do_not_count_as_evidence(self):
        answer, artifacts = evidence("scope-search")
        response = json.loads(answer)
        response["sources"] = ["", "  ", []]
        result = behavior_grader.grade("scope-search", json.dumps(response), artifacts)
        self.assertTrue(result["success"])
        self.assertIn(
            "empty source placeholder removed; answer schema differs", result["observations"]
        )
        response["sources"].append("invented")
        self.assertFalse(
            behavior_grader.grade("scope-search", json.dumps(response), artifacts)["success"]
        )
        response["sources"] = [["invented"]]
        self.assertFalse(
            behavior_grader.grade("scope-search", json.dumps(response), artifacts)["success"]
        )

        answer, artifacts = evidence("conflicting-evidence")
        response = json.loads(answer)
        response["sources"].append("")
        self.assertTrue(
            behavior_grader.grade("conflicting-evidence", json.dumps(response), artifacts)[
                "success"
            ]
        )
        response["sources"] = [""]
        self.assertFalse(
            behavior_grader.grade("conflicting-evidence", json.dumps(response), artifacts)[
                "success"
            ]
        )

    def test_unknown_notice_on_an_unsourced_fact_is_a_visible_warning(self):
        answer, artifacts = evidence("scope-search")
        response = json.loads(answer)
        response["notices"] = ["unknown"]
        result = behavior_grader.grade("scope-search", json.dumps(response), artifacts)
        self.assertTrue(result["success"], result["failures"])
        self.assertIn(
            "ambiguous unknown notice on an unsourced retrieved claim", result["observations"]
        )
        response["facts"] = []
        self.assertFalse(
            behavior_grader.grade("scope-search", json.dumps(response), artifacts)["success"]
        )
        answer, artifacts = evidence("conflicting-evidence")
        response = json.loads(answer)
        response["notices"] = ["unknown"]
        self.assertFalse(
            behavior_grader.grade("conflicting-evidence", json.dumps(response), artifacts)[
                "success"
            ]
        )

    def test_conflict_requires_actual_disputed_sources(self):
        answer, artifacts = evidence("conflicting-evidence")
        show = json.loads(artifacts["actions"][1]["stdout"])
        show["node"]["sources"] = []
        artifacts["actions"][1]["stdout"] = json.dumps(show)
        self.assertFalse(
            behavior_grader.grade("conflicting-evidence", answer, artifacts)["success"]
        )
        answer, artifacts = evidence("conflicting-evidence")
        show = json.loads(artifacts["actions"][1]["stdout"])
        show["node"]["sources"].append({"kind": "interaction", "ref": "invented"})
        artifacts["actions"][1]["stdout"] = json.dumps(show)
        self.assertFalse(
            behavior_grader.grade("conflicting-evidence", answer, artifacts)["success"]
        )

    def test_multi_session_requires_fresh_context_and_first_handoff(self):
        answer, artifacts = evidence("fresh-session-recall")
        for change in (
            "missing_first",
            "changed_graph",
            "carried_transcript",
            "first_mutation",
            "unexpected_fetch",
        ):
            with self.subTest(change=change):
                altered = json.loads(json.dumps(artifacts))
                if change == "missing_first":
                    altered.pop("first_context")
                elif change == "changed_graph":
                    altered["first_context"]["after"]["refs"] = ["refs/heads/main " + "b" * 40]
                elif change == "carried_transcript":
                    altered["second_start_messages"].append(
                        {"role": "assistant", "content": "prior"}
                    )
                elif change == "unexpected_fetch":
                    altered["first_context"]["actions"].append(
                        {"args": ["fetch"], "status": "invalid"}
                    )
                else:
                    altered["first_context"]["actions"].append(
                        {"args": ["create"], "status": "error", "stdout": "", "stderr": ""}
                    )
                self.assertFalse(
                    behavior_grader.grade("fresh-session-recall", answer, altered)["success"]
                )

    def test_isolation_requires_absent_target_and_explicit_approval(self):
        answer, artifacts = evidence("bound-store")
        exposed = json.loads(json.dumps(artifacts))
        exposed["isolation"]["after"]["target_exists"] = True
        self.assertFalse(behavior_grader.grade("bound-store", answer, exposed)["success"])
        leak = json.loads(answer)
        leak["notices"] = ["no fresh context or separate approval required"]
        self.assertFalse(
            behavior_grader.grade("bound-store", json.dumps(leak), artifacts)["success"]
        )
        attempted = json.loads(json.dumps(artifacts))
        attempted["actions"].append(
            {"args": ["create"], "status": "error", "stdout": "", "stderr": ""}
        )
        self.assertFalse(behavior_grader.grade("bound-store", answer, attempted)["success"])

    def test_distractor_body_and_truncated_results_fail(self):
        answer, artifacts = evidence("aliases-distractors")
        fixture = behavior_cases.compile_fixture("aliases-distractors")
        wrong_id = fixture["ids"]["d0"]
        artifacts["actions"].append(
            {
                "args": ["show", wrong_id, "--scope", "orchard"],
                "status": "success",
                "stdout": json.dumps(
                    {
                        "output_schema": 1,
                        "snapshot": {"store_id": STORE_ID, "commit": "a" * 40, "source": "main"},
                        "node": {"id": wrong_id},
                    }
                ),
                "stderr": "",
            }
        )
        self.assertFalse(behavior_grader.grade("aliases-distractors", answer, artifacts)["success"])
        answer, artifacts = evidence("aliases-distractors")
        output = json.loads(artifacts["actions"][0]["stdout"])
        output["truncated"] = True
        artifacts["actions"][0]["stdout"] = json.dumps(output)
        self.assertFalse(behavior_grader.grade("aliases-distractors", answer, artifacts)["success"])

    def test_runner_requires_real_captured_score_and_usage_evidence(self):
        answer, artifacts = evidence("scope-search")
        score = SimpleNamespace(value="C", explanation="passed", metadata={"artifacts": artifacts})
        sample = SimpleNamespace(
            id="scope-search",
            epoch=1,
            completed_at="observed",
            error=None,
            limit=None,
            output=SimpleNamespace(choices=[object()], stop_reason="stop", completion=answer),
            scores={"behavior": score},
            model_usage={"model": SimpleNamespace(total_tokens=100, total_cost=0.1)},
        )
        logs = [SimpleNamespace(status="success", samples=[sample])]
        self.assertEqual(run_behavior.result_problems(logs, "scope-search", 1.0)[0], [])
        score.metadata = {}
        self.assertIn(
            "missing captured grading artifacts",
            run_behavior.result_problems(logs, "scope-search", 1.0)[0],
        )
        score.metadata = {"artifacts": {**artifacts, "actions": []}}
        self.assertIn(
            "inconsistent captured grading artifacts",
            run_behavior.result_problems(logs, "scope-search", 1.0)[0],
        )
        score.metadata = {"artifacts": artifacts}
        sample.model_usage["model"].total_cost = None
        problems, usage, _, _ = run_behavior.result_problems(
            logs, "scope-search", None, pricing=False
        )
        self.assertEqual(problems, [])
        self.assertEqual(usage["model"], {"tokens": 100, "cost_usd": None})
        self.assertIn(
            "missing model cost evidence",
            run_behavior.result_problems(logs, "scope-search", 1.0, pricing=True)[0],
        )
        sample.model_usage = {}
        self.assertIn(
            "missing model token usage", run_behavior.result_problems(logs, "scope-search", 1.0)[0]
        )

    def test_runner_reports_provider_timeout_with_empty_model_choices(self):
        sample = SimpleNamespace(
            id="scope-search",
            epoch=1,
            completed_at="observed",
            error=SimpleNamespace(message="RetryError: APITimeoutError"),
            limit=None,
            output=SimpleNamespace(choices=[]),
            scores={},
            model_usage={},
        )
        logs = [SimpleNamespace(status="error", samples=[sample])]
        problems, _, _, _ = run_behavior.result_problems(logs, "scope-search", None, pricing=False)
        self.assertIn("model API request timed out", problems)
        self.assertIn("missing or truncated model completion", problems)


try:
    import behavior_capture
    import behavior_concurrent_tool
    import behavior_inspect
except ImportError:
    behavior_capture = None
    behavior_inspect = None


@unittest.skipIf(behavior_inspect is None, "optional eval dependencies not installed")
class InspectAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_conflict_tool_requires_fetched_main_and_records_real_check(self):
        from inspect_ai.tool import ToolError
        from inspect_ai.util import ExecResult

        class FakeStore:
            def __init__(self):
                self.data = {
                    "memory_calls": [],
                    "conflict_setup": {
                        "winner_commit": "b" * 40,
                        "candidate": "c" * 40,
                    },
                }

            def get(self, key, default=None):
                return self.data.get(key, default)

            def set(self, key, value):
                self.data[key] = value

        class FakeSandbox:
            def __init__(self, tracking):
                self.tracking = tracking
                self.commands = []

            async def exec(self, command, **kwargs):
                self.commands.append(command)
                if command[0] == "git":
                    return ExecResult(success=True, returncode=0, stdout=self.tracking, stderr="")
                return ExecResult(
                    success=False, returncode=1, stdout="", stderr='{"error":{"message":"stale"}}'
                )

        store = FakeStore()
        sandbox = FakeSandbox("a" * 40)
        with (
            patch.object(behavior_concurrent_tool, "store", return_value=store),
            patch.object(behavior_concurrent_tool, "sandbox", return_value=sandbox),
            self.assertRaises(ToolError),
        ):
            await behavior_concurrent_tool.conflict_check(5, 10)()
        self.assertEqual(len(sandbox.commands), 1)
        self.assertEqual(store.data["memory_calls"][0]["status"], "invalid")

        store = FakeStore()
        sandbox = FakeSandbox("b" * 40)
        with (
            patch.object(behavior_concurrent_tool, "store", return_value=store),
            patch.object(behavior_concurrent_tool, "sandbox", return_value=sandbox),
        ):
            output = await behavior_concurrent_tool.conflict_check(5, 10)()
        self.assertIn("stale", output)
        self.assertEqual(store.data["memory_calls"][0]["status"], "error")
        self.assertEqual(
            store.data["memory_calls"][0]["args"],
            ["check-merge", "--base", "b" * 40, "--candidate", "c" * 40],
        )
        self.assertEqual(sandbox.commands[1][0], "python3")

    async def test_preparation_writes_only_fixture_and_skill_via_sandbox(self):
        class FakeStore:
            def __init__(self):
                self.data = {}

            def set(self, key, value):
                self.data[key] = value

        class FakeSandbox:
            def __init__(self):
                self.writes = []

            async def write_file(self, path, value):
                self.writes.append((path, value))

        bundle = {"SKILL.md": b"synthetic"}
        fake = FakeSandbox()
        state = SimpleNamespace(store=FakeStore())
        _, artifacts = evidence("scope-search")
        commands = []

        async def fake_exec(command, *, timeout):
            commands.append(command)
            return ""

        async def fake_snapshot(timeout):
            return artifacts["before"]

        with (
            patch.object(behavior_inspect, "sandbox", return_value=fake),
            patch.object(behavior_inspect, "skill_files", return_value=bundle),
            patch.object(behavior_inspect, "checked_exec", side_effect=fake_exec),
            patch.object(behavior_inspect, "capture_snapshot", side_effect=fake_snapshot),
        ):
            await behavior_inspect.prepare_case(
                "scope-search", behavior_inspect.bundle_hash(bundle), 10
            )(state, None)
        expected = behavior_cases.compile_fixture("scope-search")["files"]
        self.assertEqual(len(fake.writes), len(expected) + len(bundle))
        self.assertTrue(all(path.startswith("/workspace/") for path, _ in fake.writes))
        self.assertTrue(any("init" in command for command in commands))
        self.assertTrue(any("commit" in command for command in commands))
        self.assertEqual(state.store.data["behavior_before"], artifacts["before"])
        self.assertEqual(state.store.data["memory_calls"], [])

    def test_trace_capture_is_independent_and_requires_evidence(self):
        calls = [{"args": ["find"], "stdout": "actual"}]
        state = SimpleNamespace(store=SimpleNamespace(get=lambda key: calls))
        captured = behavior_capture.capture_trace(state)
        calls[0]["stdout"] = "changed"
        self.assertEqual(captured[0]["stdout"], "actual")
        with self.assertRaisesRegex(TypeError, "missing action trace"):
            behavior_capture.capture_trace(
                SimpleNamespace(store=SimpleNamespace(get=lambda key: None))
            )

    async def test_fresh_agent_context_drops_prior_messages_and_hands_off_only_graph(self):
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        class FakeStore:
            def __init__(self):
                self.data = {"memory_calls": [{"args": ["find"], "status": "success"}]}

            def get(self, key, default=None):
                return self.data.get(key, default)

            def set(self, key, value):
                self.data[key] = value

        first_prompt = behavior_cases.prompt_for("fresh-session-recall", 0)
        store = FakeStore()
        state = SimpleNamespace(
            store=store,
            messages=[
                ChatMessageUser(content=first_prompt),
                ChatMessageAssistant(content="prior private text"),
            ],
            output=SimpleNamespace(completion="old answer", stop_reason="stop"),
        )

        async def fake_snapshot(timeout):
            return {"main": "a" * 40}

        with patch.object(behavior_inspect, "capture_snapshot", side_effect=fake_snapshot):
            await behavior_inspect.fresh_context("fresh-session-recall", 10)(state, None)
        self.assertEqual(len(state.messages), 1)
        self.assertEqual(
            state.messages[0].content, behavior_cases.prompt_for("fresh-session-recall", 1)
        )
        self.assertEqual(state.output.completion, "")
        self.assertEqual(store.data["first_action_count"], 1)
        self.assertEqual(
            store.data["first_context"]["messages"][-1]["content"], "prior private text"
        )
        self.assertNotIn("prior private text", json.dumps(store.data["second_start_messages"]))

    async def test_capture_reads_sandbox_git_artifacts_separately(self):
        base = {
            "main": "a" * 40,
            "refs": ["refs/heads/main " + "a" * 40],
            "files": {"node": "b" * 64},
        }

        async def fake_snapshot(timeout):
            return base

        calls = []

        async def fake_exec(command, *, timeout):
            calls.append((command, timeout))
            return json.dumps({"node": "b" * 64})

        with (
            patch.object(behavior_capture, "snapshot", side_effect=fake_snapshot),
            patch.object(behavior_capture, "checked_exec", side_effect=fake_exec),
        ):
            result = await behavior_capture.capture_snapshot(10)
        self.assertEqual(result["tree_files"], {"node": "b" * 64})
        self.assertEqual(calls[0][0][:2], ["python3", "-c"])
        self.assertEqual(calls[0][0][-1], base["main"])

    def test_actual_inspect_task_is_docker_scoped_and_contains_no_targets(self):
        args = SimpleNamespace(
            case_id="directed-traversal",
            trial=1,
            image="example/image@sha256:" + "a" * 64,
            memory_mb=512,
            cpus=1.0,
            tool_timeout=10,
            tool_limit=5,
            time_limit=60,
            turn_limit=5,
            token_limit=4000,
            budget_usd=1.0,
        )
        task = behavior_inspect.build_task(args)
        self.assertEqual(task.sandbox.type, "docker")
        self.assertEqual(task.sandbox.config.services["default"].network_mode, "none")
        self.assertEqual(
            task.sandbox.config.services["default"].environment, {"HOME": "/workspace"}
        )
        self.assertEqual(task.metadata["trial"], 1)
        self.assertEqual(task.dataset[0].target, "")
        self.assertNotIn(
            behavior_cases.compile_fixture(args.case_id)["ids"]["tools"], task.dataset[0].input
        )

    def test_all_behavior_harnesses_disable_budget_without_prices(self):
        args = SimpleNamespace(
            case_id="scope-search",
            trial=1,
            image="example/image@sha256:" + "a" * 64,
            tool_timeout=10,
            tool_limit=5,
            memory_mb=512,
            cpus=1,
            time_limit=60,
            turn_limit=5,
            token_limit=4000,
            budget_usd=1.0,
            input_price=None,
            output_price=None,
        )
        for case_id in ("scope-search", "remember-fact", "shared-read"):
            with self.subTest(case_id=case_id):
                args.case_id = case_id
                self.assertIsNone(behavior_inspect.build_task(args).cost_limit)
                args.input_price = 1.0
                args.output_price = 2.0
                self.assertEqual(behavior_inspect.build_task(args).cost_limit, 1.0)
                args.input_price = None
                args.output_price = None

    def test_unsupported_shared_tasks_cannot_be_built(self):
        args = SimpleNamespace(
            case_id="concurrent-changes",
            trial=1,
            image="example/image@sha256:" + "a" * 64,
            memory_mb=512,
            cpus=1.0,
            tool_timeout=10,
            tool_limit=10,
            time_limit=60,
            turn_limit=5,
            token_limit=4000,
            budget_usd=1.0,
        )
        for case_id in behavior_cases.UNSUPPORTED:
            with self.subTest(case_id=case_id):
                args.case_id = case_id
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    behavior_inspect.build_task(args)


class EvaluationToleranceTests(unittest.TestCase):
    def test_help_for_known_commands_is_safe_but_other_arguments_are_not(self):
        from controlled_core import is_safe_help

        for args in (
            ["--help"],
            ["create", "--help"],
            ["find", "-h"],
            ["--json", "apply", "--help"],
        ):
            self.assertTrue(is_safe_help(args), args)
        for args in (["create", "--input", "-"], ["stores", "--help"], ["create", "x", "--help"]):
            self.assertFalse(is_safe_help(args), args)

    def test_any_query_naming_the_target_is_accepted(self):
        fixture = behavior_cases.compile_fixture("read-only")
        for query in ("seed tin", "Seed", "tin", "dry"):
            self.assertTrue(behavior_grader.query_names_targets(query, fixture), query)
        for query in ("orchard", "", "  ", None):
            self.assertFalse(behavior_grader.query_names_targets(query, fixture), query)

    def test_natural_phrasing_of_a_created_fact_is_the_same_fact(self):
        from graders.behavior_write_grader import same_fact

        self.assertTrue(same_fact("Amber kite is stored in the loft", "amber kite stored in loft"))
        self.assertTrue(same_fact("ceramic wheel uses a blue belt", "ceramic wheel uses blue belt"))
        self.assertFalse(same_fact("amber kite stored in barn", "amber kite stored in loft"))
        self.assertFalse(
            same_fact("amber kite stored in the loft too", "amber kite stored in loft")
        )
        self.assertFalse(same_fact(None, "amber kite"))

    def test_extra_unknown_notice_is_ignored_only_beside_a_retrieved_unsourced_fact(self):
        from graders.answers import normalize_notices

        self.assertEqual(
            normalize_notices(["review due: x", "unknown"], ["review due: x"], [], ["f"]),
            (["review due: x"], True),
        )
        self.assertEqual(normalize_notices(["unknown"], ["unknown"], [], []), (["unknown"], False))
        self.assertEqual(normalize_notices(["unknown"], [], ["src"], ["f"]), (["unknown"], False))

    def test_relationship_statements_may_include_to(self):
        fixture = behavior_cases.compile_fixture("bounded-traversal")
        self.assertTrue(
            behavior_grader._traversal_answer(
                "facts", ["A connects to B", "B connects to C"], fixture
            )
        )
        self.assertFalse(
            behavior_grader._traversal_answer(
                "facts", ["A connects to C", "B connects to C"], fixture
            )
        )

    def test_placeholder_sources_in_a_write_answer_are_not_a_failure(self):
        from graders.behavior_write_grader import _answer

        ids, commit = ["n1"], "a" * 40
        answer = json.dumps(
            {
                "node_ids": ids,
                "facts": ["f"],
                "sources": [""],
                "notices": [],
                "store_id": STORE_ID,
                "commit": commit,
            }
        )
        self.assertTrue(_answer(answer, ids, ["f"], [], [], commit))
        self.assertFalse(_answer(answer.replace('""', '"real"'), ids, ["f"], [], [], commit))

    def test_help_calls_do_not_fail_a_read_trial(self):
        answer, artifacts = evidence("scope-search")
        artifacts["actions"].insert(
            0,
            {
                "args": ["find", "--help"],
                "status": "success",
                "stdout": "usage: memory-graph find [-h]\n",
                "stderr": "",
            },
        )
        self.assertTrue(behavior_grader.grade("scope-search", answer, artifacts)["success"])


if __name__ == "__main__":
    unittest.main()
