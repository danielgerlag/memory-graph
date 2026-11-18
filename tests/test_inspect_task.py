"""Offline tests only; no model, Docker container, or provider call is launched."""

import asyncio
import hashlib
import io
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
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


def foundry_options():
    opts = options()
    opts.model = "openai/azure/example-deployment"
    opts.credential_env = "AZUREAI_OPENAI_API_KEY"
    opts.azure_endpoint = "https://example.openai.azure.com"
    opts.azure_api_version = "2025-03-01-preview"
    return opts


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
    def test_candidate_instructions_replace_only_skill_markdown(self):
        import inspect_task

        original = inspect_task.skill_files()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            candidate = Path(directory) / "candidate.md"
            candidate.write_text("candidate instructions\n", encoding="utf-8")
            with patch.object(inspect_task, "SKILL_INSTRUCTIONS", candidate):
                overridden = inspect_task.skill_files()
        self.assertEqual(overridden["SKILL.md"], b"candidate instructions\n")
        self.assertNotEqual(
            inspect_task.bundle_hash(original), inspect_task.bundle_hash(overridden)
        )
        self.assertEqual(
            {key: value for key, value in original.items() if key != "SKILL.md"},
            {key: value for key, value in overridden.items() if key != "SKILL.md"},
        )

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
        opts.image = "sha256:" + "a" * 64
        core.validate_options(opts, env)
        opts.image = "example/image@sha256:" + "a" * 64
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

    def test_pricing_and_budget_are_optional_but_prices_must_be_paired(self):
        opts = options()
        env = {"OPENAI_API_KEY": "synthetic-test-value"}
        opts.input_price = None
        opts.output_price = None
        self.assertIsNone(core.effective_budget(opts))
        self.assertEqual(run_inspect.cost_control(opts)["budget"], "DISABLED")
        with self.assertRaisesRegex(ValueError, "requires both prices"):
            core.validate_options(opts, env)
        opts.budget_usd = None
        core.validate_options(opts, env)
        opts.input_price = 1.0
        with self.assertRaisesRegex(ValueError, "supplied together"):
            core.validate_options(opts, env)
        opts.output_price = 2.0
        core.validate_options(opts, env)
        self.assertIsNone(core.effective_budget(opts))
        opts.budget_usd = 1.0
        core.validate_options(opts, env)
        self.assertEqual(core.effective_budget(opts), 1.0)
        self.assertEqual(run_inspect.cost_control(opts)["budget"], "ENABLED")

    def test_unpriced_usage_is_unknown_and_still_requires_tokens(self):
        sample = SimpleNamespace(
            model_usage={"model": SimpleNamespace(total_tokens=100, total_cost=None)}
        )
        usage, problems = run_inspect.usage_summary(sample, pricing=False, budget=None)
        self.assertEqual(problems, [])
        self.assertEqual(usage["model"], {"tokens": 100, "cost_usd": None})
        sample.model_usage["model"].total_cost = 0.1
        usage, problems = run_inspect.usage_summary(sample, pricing=False, budget=None)
        self.assertEqual(problems, [])
        self.assertIsNone(usage["model"]["cost_usd"])
        sample.model_usage["model"].total_cost = None
        self.assertIn(
            "missing model cost evidence",
            run_inspect.usage_summary(sample, pricing=True, budget=1.0)[1],
        )
        sample.model_usage["model"].total_tokens = None
        self.assertIn(
            "missing model token usage",
            run_inspect.usage_summary(sample, pricing=False, budget=None)[1],
        )
        sample.model_usage["model"].total_tokens = 100
        sample.model_usage["model"].total_cost = 2.0
        self.assertIn(
            "reported cost exceeds the budget",
            run_inspect.usage_summary(sample, pricing=True, budget=1.0)[1],
        )
        self.assertEqual(run_inspect.usage_summary(sample, pricing=True, budget=None)[1], [])

    def test_unpriced_run_warns_that_dollar_budget_is_disabled(self):
        opts = options()
        opts.input_price = None
        opts.output_price = None
        opts.budget_usd = None
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            run_inspect.warn_if_unbudgeted(opts)
        self.assertIn("no dollar budget is enforced", stderr.getvalue())
        self.assertIn("can still be billed", stderr.getvalue())

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

    def test_foundry_requires_explicit_safe_endpoint_and_key(self):
        opts = foundry_options()
        env = {"AZUREAI_OPENAI_API_KEY": "synthetic-key"}
        core.validate_options(opts, env)
        for endpoint in (
            "https://example.cognitiveservices.azure.com",
            "https://example.services.ai.azure.com",
        ):
            accepted = foundry_options()
            accepted.azure_endpoint = endpoint
            core.validate_options(accepted, env)
        alternate = foundry_options()
        alternate.credential_env = "AZURE_OPENAI_API_KEY"
        core.validate_options(alternate, {"AZURE_OPENAI_API_KEY": "synthetic-key"})
        for endpoint in (
            "http://example.openai.azure.com",
            "https://example.openai.azure.com.evil.test",
            "https://example.openai.azure.com@evil.test",
            "https://example.openai.azure.com/path",
            "https://example.openai.azure.com?secret=x",
            "https://example.openai.azure.com?",
            "https://example.openai.azure.com#fragment",
            "https://example.openai.azure.com#",
            "https://example.openai.azure.com:8080",
            "https://localhost",
        ):
            with self.subTest(endpoint=endpoint):
                bad = foundry_options()
                bad.azure_endpoint = endpoint
                with self.assertRaises(ValueError):
                    core.validate_options(bad, env)
        for field, value in (
            ("credential_env", "OPENAI_API_KEY"),
            ("azure_endpoint", None),
            ("azure_api_version", "invalid"),
            ("model_family", "not a model"),
            ("model", "openai/azure/"),
        ):
            with self.subTest(field=field):
                bad = foundry_options()
                setattr(bad, field, value)
                with self.assertRaises(ValueError):
                    core.validate_options(bad, env)
        with self.assertRaises(ValueError):
            core.validate_options(opts, {})

    def test_default_azure_auth_is_explicit_and_does_not_require_an_api_key(self):
        opts = foundry_options()
        opts.azure_auth = "default"
        core.validate_options(opts, {})
        opts.credential_env = None
        core.validate_options(opts, {})
        opts.azure_auth = "invalid"
        with self.assertRaises(ValueError):
            core.validate_options(opts, {})
        direct = options()
        direct.azure_auth = "default"
        with self.assertRaises(ValueError):
            core.validate_options(direct, {"OPENAI_API_KEY": "synthetic"})

    def test_default_azure_environment_preserves_only_identity_context_on_host(self):
        opts = foundry_options()
        opts.azure_auth = "default"
        host_home = ROOT / "synthetic-home"
        azure_dir = ROOT / "synthetic-azure"
        azd_dir = ROOT / "synthetic-azd"
        original = {
            "PATH": "/bin",
            "HOME": str(host_home),
            "AZURE_CONFIG_DIR": str(azure_dir),
            "AZD_CONFIG_DIR": str(azd_dir),
            "AZURE_CLIENT_ID": "client",
            "AZURE_CLIENT_SECRET": "synthetic-secret",
            "AZURE_TOKEN_CREDENTIALS": "dev",
            "AZUREAI_OPENAI_API_KEY": "unselected-key",
            "OPENAI_API_KEY": "other-key",
            "HTTPS_PROXY": "http://proxy.invalid",
        }
        with patch.dict(os.environ, original, clear=True), patch.object(Path, "mkdir"):
            run_inspect.isolate_environment(opts, ROOT / "evals" / "explicit-run")
            self.assertEqual(os.environ["HOME"], str(ROOT / "evals/explicit-run/home"))
            self.assertEqual(os.environ["AZURE_CONFIG_DIR"], str(azure_dir.resolve()))
            self.assertEqual(os.environ["AZD_CONFIG_DIR"], str(azd_dir.resolve()))
            self.assertEqual(os.environ["AZURE_CLIENT_ID"], "client")
            self.assertEqual(os.environ["AZURE_CLIENT_SECRET"], "synthetic-secret")
            self.assertEqual(os.environ["AZURE_TOKEN_CREDENTIALS"], "dev")
            self.assertNotIn("AZUREAI_OPENAI_API_KEY", os.environ)
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            self.assertNotIn("HTTPS_PROXY", os.environ)
        with (
            patch.dict(os.environ, {"HOME": str(host_home), "PATH": "/bin"}, clear=True),
            patch.object(Path, "mkdir"),
        ):
            run_inspect.isolate_environment(opts, ROOT / "evals" / "explicit-run")
            self.assertEqual(os.environ["AZURE_CONFIG_DIR"], str((host_home / ".azure").resolve()))
            self.assertEqual(os.environ["AZD_CONFIG_DIR"], str((host_home / ".azd").resolve()))

    def test_default_azure_preflight_reports_safe_errors(self):
        from azure.core.exceptions import ClientAuthenticationError

        opts = foundry_options()
        opts.azure_auth = "default"
        with patch("azure.identity.DefaultAzureCredential") as factory:
            credential = factory.return_value.__enter__.return_value
            run_inspect.preflight_default_credentials(opts)
            credential.get_token.assert_called_once_with(run_inspect.AZURE_SCOPE)
            factory.return_value.__exit__.assert_called_once()
            credential.get_token.side_effect = ClientAuthenticationError("synthetic-secret")
            with self.assertRaisesRegex(ValueError, "az login") as caught:
                run_inspect.preflight_default_credentials(opts)
            self.assertNotIn("synthetic-secret", str(caught.exception))
        opts.azure_auth = "api-key"
        with patch("azure.identity.DefaultAzureCredential") as factory:
            run_inspect.preflight_default_credentials(opts)
            factory.assert_not_called()

    def test_default_azure_sdk_warning_output_is_suppressed_and_restored(self):
        opts = foundry_options()
        opts.azure_auth = "default"
        logger = logging.getLogger("azure.identity")
        previous = logger.level
        with (
            self.assertNoLogs("azure.identity", level="WARNING"),
            run_inspect.azure_credential_log_guard(opts),
        ):
            logging.getLogger("azure.identity.synthetic").warning("synthetic-secret")
        self.assertEqual(logger.level, previous)

    @unittest.skipIf(os.name == "nt", "developer CLI fixture requires a Unix shell")
    def test_default_azure_uses_developer_cli_with_original_config_paths(self):
        opts = foundry_options()
        opts.azure_auth = "default"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            host_home = root / "host-home"
            host_home.mkdir()
            binary_dir = root / "bin"
            binary_dir.mkdir()
            cases = (
                (
                    "az",
                    "AzureCliCredential",
                    host_home / ".azure",
                    host_home / ".azd",
                    '{"accessToken":"synthetic-token","expires_on":4102444800}',
                    {},
                ),
                (
                    "azd",
                    "AzureDeveloperCliCredential",
                    root / "custom-azure",
                    root / "custom-azd",
                    '{"token":"synthetic-token","expiresOn":"2100-01-01T00:00:00Z"}',
                    {
                        "AZURE_CONFIG_DIR": str(root / "custom-azure"),
                        "AZD_CONFIG_DIR": str(root / "custom-azd"),
                    },
                ),
            )
            for index, (name, selection, azure_dir, azd_dir, response, overrides) in enumerate(
                cases
            ):
                with self.subTest(credential=name):
                    cli = binary_dir / name
                    cli.write_text(
                        "#!/bin/sh\n"
                        f'[ "$AZURE_CONFIG_DIR" = {shlex.quote(str(azure_dir))} ] || exit 10\n'
                        f'[ "$AZD_CONFIG_DIR" = {shlex.quote(str(azd_dir))} ] || exit 11\n'
                        f"printf '%s\\n' {shlex.quote(response)}\n",
                        encoding="utf-8",
                    )
                    cli.chmod(0o700)
                    run_root = root / f"run-{index}"
                    run_root.mkdir()
                    env = {
                        "HOME": str(host_home),
                        "PATH": f"{binary_dir}:/usr/bin:/bin",
                        "AZURE_TOKEN_CREDENTIALS": selection,
                        **overrides,
                    }
                    with patch.dict(os.environ, env, clear=True):
                        run_inspect.isolate_environment(opts, run_root)
                        run_inspect.preflight_default_credentials(opts)

    def test_default_azure_provider_refreshes_tokens_without_api_key(self):
        from azure.core.credentials import AccessToken
        from httpx2 import MockTransport, Response
        from inspect_ai.model import get_model
        from openai import DefaultAsyncHttpxClient

        opts = foundry_options()
        opts.azure_auth = "default"
        tokens = []
        requests = []
        host_home = ROOT / "synthetic-home"

        class FakeCredential:
            def get_token(self, *scopes, **_kwargs):
                assert scopes == (run_inspect.AZURE_SCOPE,)
                assert os.environ["AZURE_CONFIG_DIR"] == str((host_home / ".azure").resolve())
                assert os.environ["AZD_CONFIG_DIR"] == str((host_home / ".azd").resolve())
                value = f"synthetic-token-{len(tokens) + 1}"
                tokens.append(value)
                return AccessToken(value, int(time.time()) + 30)

        async def handler(request):
            requests.append(request)
            return Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "example-deployment",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        original = {
            "HOME": str(host_home),
            "PATH": "/bin",
            "AZUREAI_OPENAI_API_KEY": "must-not-be-used",
        }
        with patch.dict(os.environ, original, clear=True), patch.object(Path, "mkdir"):
            run_inspect.isolate_environment(opts, ROOT / "evals" / "explicit-run")
            with (
                patch("azure.identity.DefaultAzureCredential", return_value=FakeCredential()),
                patch(
                    "openai.DefaultAsyncHttpxClient",
                    side_effect=lambda **kwargs: DefaultAsyncHttpxClient(
                        transport=MockTransport(handler), **kwargs
                    ),
                ),
            ):
                model = get_model(opts.model, memoize=False, **run_inspect.model_args(opts))

            async def invoke():
                try:
                    for _ in range(2):
                        await model.api.client.chat.completions.create(
                            model="example-deployment",
                            messages=[{"role": "user", "content": "hello"}],
                        )
                finally:
                    await model.api.aclose()

            asyncio.run(invoke())
        self.assertEqual(len(tokens), 2)
        self.assertEqual(
            [request.headers["authorization"] for request in requests],
            [f"Bearer {token}" for token in tokens],
        )
        self.assertTrue(all("api-key" not in request.headers for request in requests))

    def test_foundry_environment_and_sdk_use_only_configured_provider(self):
        from httpx2 import MockTransport, Response
        from inspect_ai.model import get_model
        from openai import DefaultAsyncHttpxClient

        opts = foundry_options()
        requests = []

        async def handler(request):
            requests.append(request)
            return Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "example-deployment",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        original = {
            "PATH": "/bin",
            "HOME": "/synthetic-home",
            "AZUREAI_OPENAI_API_KEY": "synthetic-key",
            "OPENAI_API_KEY": "unrelated-key",
            "AZUREAI_OPENAI_BASE_URL": "https://unrelated.openai.azure.com",
            "HTTPS_PROXY": "http://proxy.invalid",
        }
        with patch.dict(os.environ, original, clear=True), patch.object(Path, "mkdir"):
            run_inspect.isolate_environment(opts, ROOT / "evals" / "explicit-run")
            self.assertEqual(os.environ["AZUREAI_OPENAI_BASE_URL"], opts.azure_endpoint)
            self.assertEqual(os.environ["AZUREAI_OPENAI_API_VERSION"], opts.azure_api_version)
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            self.assertNotIn("HTTPS_PROXY", os.environ)
            with patch(
                "openai.DefaultAsyncHttpxClient",
                side_effect=lambda **kwargs: DefaultAsyncHttpxClient(
                    transport=MockTransport(handler), **kwargs
                ),
            ):
                model_args = run_inspect.model_args(opts)
            client = model_args["http_client"]
            self.assertFalse(client.follow_redirects)
            self.assertFalse(client.trust_env)
            model = get_model(opts.model, memoize=False, **model_args)
            self.assertEqual(model.api.client.max_retries, 0)

            async def invoke():
                try:
                    return await model.api.client.chat.completions.create(
                        model="example-deployment",
                        messages=[{"role": "user", "content": "hello"}],
                    )
                finally:
                    await model.api.aclose()

            response = asyncio.run(invoke())
            self.assertEqual(response.choices[0].message.content, "ok")
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].url.host, "example.openai.azure.com")
            self.assertEqual(requests[0].headers["api-key"], "synthetic-key")
            self.assertIn("api-version=2025-03-01-preview", str(requests[0].url))

    def test_alternate_foundry_key_does_not_inherit_other_credentials(self):
        opts = foundry_options()
        opts.credential_env = "AZURE_OPENAI_API_KEY"
        original = {
            "PATH": "/bin",
            "HOME": "/synthetic-home",
            "AZURE_OPENAI_API_KEY": "selected",
            "AZUREAI_OPENAI_API_KEY": "unrelated",
        }
        with patch.dict(os.environ, original, clear=True), patch.object(Path, "mkdir"):
            run_inspect.isolate_environment(opts, ROOT / "evals" / "explicit-run")
            self.assertEqual(os.environ["AZURE_OPENAI_API_KEY"], "selected")
            self.assertNotIn("AZUREAI_OPENAI_API_KEY", os.environ)

    def test_foundry_http_client_refuses_redirects(self):
        from httpx2 import MockTransport, Response
        from inspect_ai.model import get_model
        from openai import APIStatusError, DefaultAsyncHttpxClient

        requests = []

        async def handler(request):
            requests.append(request)
            return Response(307, headers={"location": "https://other.invalid"})

        with patch(
            "openai.DefaultAsyncHttpxClient",
            side_effect=lambda **kwargs: DefaultAsyncHttpxClient(
                transport=MockTransport(handler), **kwargs
            ),
        ):
            client = run_inspect.model_args(foundry_options())["http_client"]

        env = {
            "AZUREAI_OPENAI_API_KEY": "synthetic-key",
            "AZUREAI_OPENAI_BASE_URL": "https://example.openai.azure.com",
            "AZUREAI_OPENAI_API_VERSION": "2025-03-01-preview",
        }
        with patch.dict(os.environ, env, clear=True):
            model = get_model(foundry_options().model, memoize=False, http_client=client)

            async def invoke():
                try:
                    return await model.api.client.chat.completions.create(
                        model="example-deployment",
                        messages=[{"role": "user", "content": "hello"}],
                    )
                finally:
                    await model.api.aclose()

            with self.assertRaises(APIStatusError):
                asyncio.run(invoke())
        self.assertEqual(len(requests), 1)

    def test_isolated_home_links_only_the_compose_executable(self):
        opts = options()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original_home = base / "original-home"
            plugins = original_home / ".docker" / "cli-plugins"
            plugins.mkdir(parents=True)
            compose = plugins / "docker-compose"
            compose.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            compose.chmod(0o700)
            (original_home / ".docker" / "config.json").write_text('{"auths":"synthetic"}')
            root = base / "run"
            root.mkdir()
            env = {"HOME": str(original_home), "PATH": "/bin", "OPENAI_API_KEY": "test-key"}
            with patch.dict(os.environ, env, clear=True):
                run_inspect.isolate_environment(opts, root)
                clean_docker = root / "home" / ".docker"
                linked = clean_docker / "cli-plugins" / "docker-compose"
                self.assertTrue(linked.is_symlink())
                self.assertEqual(linked.resolve(), compose.resolve())
                self.assertFalse((clean_docker / "config.json").exists())

    def test_unregistered_foundry_deployment_has_explicit_costs(self):
        from inspect_ai.model import get_model, get_model_info
        from inspect_ai.model._model_info import clear_model_info_cache

        opts = foundry_options()
        opts.model_family = "gpt-4o"
        try:
            prices = run_inspect.pricing_config(opts)
            info = get_model_info(opts.model)
            self.assertIsNotNone(info)
            self.assertEqual(info.family, "gpt-4o")
            self.assertEqual(prices[opts.model].input, opts.input_price)
            self.assertEqual(prices[opts.model].output, opts.output_price)
            self.assertEqual(prices[opts.model].input_cache_read, opts.output_price)
            env = {
                "AZUREAI_OPENAI_API_KEY": "synthetic-key",
                "AZUREAI_OPENAI_BASE_URL": opts.azure_endpoint,
            }
            with patch.dict(os.environ, env, clear=True):
                model = get_model(opts.model, memoize=False, **run_inspect.model_args(opts))
                try:
                    self.assertEqual(model.api.model_family(), "gpt-4o")
                    self.assertEqual(model.api.service_model_name(), "example-deployment")
                finally:
                    asyncio.run(model.api.aclose())
        finally:
            clear_model_info_cache()

    def test_unpriced_foundry_deployment_keeps_its_model_family(self):
        from inspect_ai.model import get_model_info
        from inspect_ai.model._model_info import clear_model_info_cache

        opts = foundry_options()
        opts.input_price = None
        opts.output_price = None
        opts.model_family = "gpt-4o"
        try:
            self.assertIsNone(run_inspect.pricing_config(opts))
            info = get_model_info(opts.model)
            self.assertIsNotNone(info)
            self.assertEqual(info.family, "gpt-4o")
            self.assertIsNone(info.cost)
        finally:
            clear_model_info_cache()

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
        self.assertEqual(service.environment, {"HOME": "/workspace"})
        self.assertTrue(service.read_only)
        self.assertEqual(service.pull_policy, "never")
        self.assertEqual(service.tmpfs, ["/workspace:rw,mode=1777,size=67108864"])
        self.assertEqual(len(task.dataset), 1)

    def test_smoke_task_disables_dollar_limit_without_pricing(self):
        opts = options()
        opts.input_price = None
        opts.output_price = None
        self.assertIsNone(inspect_task.build_task(opts).cost_limit)
        opts.input_price = 1.0
        opts.output_price = 2.0
        self.assertEqual(inspect_task.build_task(opts).cost_limit, 1.0)

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
            self.assertEqual(await tool(["--json", "find", "ladder", "--scope", "orchard"]), "{}")
        self.assertEqual(
            fake.command[:4], ["python3", inspect_task.SCRIPT, "--root", inspect_task.STORE]
        )
        self.assertEqual(fake.command[4:], ["--json", "find", "ladder", "--scope", "orchard"])
        self.assertFalse(fake.kwargs["timeout_retry"])
        self.assertEqual(state.data["memory_calls"][0]["status"], "success")
        self.assertEqual(
            state.data["memory_calls"][1]["args"],
            ["--json", "find", "ladder", "--scope", "orchard"],
        )

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
                await tool(["--json", "find", "ladder", "--root", "/other"])
            with self.assertRaises(ToolError):
                await tool(["find", "ladder"])
            sandbox.assert_not_called()
        self.assertEqual([x["status"] for x in state.data["memory_calls"]], ["invalid", "invalid"])

    async def test_graph_tool_accepts_global_json_without_weakening_mutation_guards(self):
        import behavior_write_tool
        from inspect_ai.tool import ToolError

        class StateStore:
            def __init__(self):
                self.data = {"memory_calls": []}

            def get(self, key, default=None):
                return self.data.get(key, default)

            def set(self, key, value):
                self.data[key] = value

        class FakeSandbox:
            def __init__(self):
                self.commands = []

            async def exec(self, command, **_kwargs):
                self.commands.append(command)
                return ExecResult(success=True, returncode=0, stdout="{}", stderr="")

        state, fake = StateStore(), FakeSandbox()
        with (
            patch.object(behavior_write_tool, "sandbox", return_value=fake),
            patch.object(behavior_write_tool, "store", return_value=state),
        ):
            tool = behavior_write_tool.graph_cli(6, 10)
            valid = ["--json", "create", "--operation", "test", "--input", "-"]
            self.assertEqual(await tool(valid, "{}"), "{}")
            self.assertEqual(await tool(["--help"]), "{}")
            self.assertEqual(await tool(["fetch", "--help"]), "{}")
            with self.assertRaises(ToolError):
                await tool(["--json", "create", "--operation", "test", "--input", "/private"])
            with self.assertRaises(ToolError):
                await tool(["--json", "find", "query", "--root", "/other"])
            with self.assertRaises(ToolError):
                await tool(["fetch", "--help", "--root", "/other"])
        self.assertEqual(len(fake.commands), 3)
        self.assertEqual(fake.commands[0][4:], valid)
        self.assertEqual(fake.commands[1][4:], ["--help"])
        self.assertEqual(fake.commands[2][4:], ["fetch", "--help"])
        self.assertEqual(state.data["memory_calls"][0]["args"], valid)
        self.assertEqual(
            [call["status"] for call in state.data["memory_calls"]],
            ["success", "success", "success", "invalid", "invalid", "invalid"],
        )


if __name__ == "__main__":
    unittest.main()
