"""Offline profile tests: no runner, credential, network, or container is used."""

import argparse
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
from controlled_core import validate_options as validate_controlled
from native_core import validate_options as validate_native
from profiles import apply_profiles


@pytest.fixture
def profile_files():
    paths = []

    def create(text=None, *, target=None, directory=False):
        path = ROOT / f".profile-test-{uuid.uuid4().hex}.yaml"
        paths.append(path)
        if target is not None:
            path.symlink_to(target, target_is_directory=directory)
        else:
            path.write_text(text, encoding="utf-8")
        return path

    yield create
    for path in reversed(paths):
        path.unlink(missing_ok=True)


def profile(runner, options, *, name="Example"):
    return f"schema_version: 1\nrunner: {runner}\nname: {name}\noptions:\n{options}\n"


@pytest.mark.parametrize("runner", ["controlled", "native"])
def test_precedence_and_missing_namespace_attributes(profile_files, runner):
    agent = profile_files(profile(runner, "  model: yaml-model\n  credential_env: MODEL_KEY"))
    harness = profile_files(profile(runner, "  image: yaml-image\n  cpus: 2"))
    args = argparse.Namespace(
        agent=agent, harness=harness, model="cli-model", image=None, cpus=0, run=False
    )

    assert apply_profiles(args, runner=runner) is args
    assert (args.model, args.credential_env, args.image, args.cpus) == (
        "cli-model",
        "MODEL_KEY",
        "yaml-image",
        0,
    )
    assert args.run is False
    assert not hasattr(args, "allow_network")


@pytest.mark.parametrize(
    ("runner", "options", "expected"),
    [
        (
            "controlled",
            """  image: image
  docker_host: unix:///socket
  budget_usd: 1
  input_price: 2.5
  output_price: 3
  time_limit: 4
  wall_timeout: 5
  tool_timeout: 6
  turn_limit: 7
  tool_limit: 8
  token_limit: 9
  max_tokens: 10
  cpus: 1.5
  memory_mb: 11""",
            {
                "image": "image",
                "docker_host": "unix:///socket",
                "budget_usd": 1,
                "input_price": 2.5,
                "output_price": 3,
                "time_limit": 4,
                "wall_timeout": 5,
                "tool_timeout": 6,
                "turn_limit": 7,
                "tool_limit": 8,
                "token_limit": 9,
                "max_tokens": 10,
                "cpus": 1.5,
                "memory_mb": 11,
            },
        ),
        (
            "native",
            """  image: image
  host_version: "1.0"
  docker_host: unix:///socket
  session_ai_credits: 1
  run_ai_credits: 2
  session_timeout: 3
  run_timeout: 4
  cpus: 0.5
  memory_mb: 5
  max_output_mb: 6""",
            {
                "image": "image",
                "host_version": "1.0",
                "docker_host": "unix:///socket",
                "session_ai_credits": 1,
                "run_ai_credits": 2,
                "session_timeout": 3,
                "run_timeout": 4,
                "cpus": 0.5,
                "memory_mb": 5,
                "max_output_mb": 6,
            },
        ),
    ],
)
def test_harness_option_destinations(profile_files, runner, options, expected):
    args = argparse.Namespace(harness=profile_files(profile(runner, options)))
    apply_profiles(args, runner=runner)
    assert {key: getattr(args, key) for key in expected} == expected


def test_absent_profile_attributes_and_empty_options(profile_files):
    args = argparse.Namespace(run=False)
    assert apply_profiles(args, runner="controlled") is args
    assert vars(args) == {"run": False}
    args.agent = profile_files("schema_version: 1\nrunner: controlled\noptions: {}\n")
    apply_profiles(args, runner="controlled")
    assert not hasattr(args, "model")


@pytest.mark.parametrize("runner", ["controlled", "native"])
def test_runner_reads_profiles_but_still_requires_explicit_opt_in(
    profile_files, runner, monkeypatch, capsys
):
    agent = profile_files(profile(runner, "  model: yaml-model\n  credential_env: MODEL_KEY"))
    harness = profile_files(profile(runner, "  image: yaml-image"))
    monkeypatch.setattr(Path, "mkdir", lambda *_args, **_kwargs: pytest.fail("created output"))
    if runner == "controlled":
        from run_inspect import main, parser

        argv = ["--agent", str(agent), "--harness", str(harness)]
    else:
        from run_native import main, parser

        argv = [
            "--agent",
            str(agent),
            "--harness",
            str(harness),
            "--manifest",
            str(ROOT / "evals/native-smoke.json"),
            "--output-dir",
            "unused",
        ]
    args = parser().parse_args(argv)
    apply_profiles(args, runner=runner)
    assert args.model == "yaml-model"
    assert args.image == "yaml-image"
    assert main(argv) == 2
    assert "requires" in capsys.readouterr().err


def test_combined_config_selects_case_and_respects_cli_precedence(profile_files):
    from run_behavior import parser

    config = profile_files(
        profile(
            "controlled",
            "  model: openai/azure/deployment\n"
            "  credential_env: AZUREAI_OPENAI_API_KEY\n"
            "  azure_endpoint: https://example.openai.azure.com\n"
            "  image: example/image@sha256:" + "a" * 64 + "\n"
            "  case_id: scope-search\n"
            "  trial: 1",
        )
    )
    args = parser().parse_args(["--config", str(config), "--trial", "2"])
    apply_profiles(args, runner="controlled")
    assert args.model == "openai/azure/deployment"
    assert args.azure_endpoint == "https://example.openai.azure.com"
    assert args.case_id == "scope-search"
    assert args.trial == 2
    assert not args.run


def test_default_azure_auth_overrides_a_key_profile_without_using_the_key(profile_files):
    from run_behavior import parser

    config = profile_files(
        profile(
            "controlled",
            "  model: openai/azure/deployment\n"
            "  credential_env: AZUREAI_OPENAI_API_KEY\n"
            "  azure_auth: api-key\n"
            "  azure_endpoint: https://example.openai.azure.com\n"
            "  image: example/image@sha256:" + "a" * 64 + "\n"
            "  case_id: scope-search\n"
            "  trial: 1\n"
            "  budget_usd: 1\n"
            "  input_price: 1\n"
            "  output_price: 2\n"
            "  cpus: 1\n"
            "  time_limit: 60\n"
            "  wall_timeout: 90\n"
            "  tool_timeout: 10\n"
            "  turn_limit: 5\n"
            "  tool_limit: 5\n"
            "  token_limit: 4000\n"
            "  max_tokens: 500\n"
            "  memory_mb: 512",
        )
    )
    args = parser().parse_args(["--config", str(config), "--run", "--azure-auth", "default"])
    apply_profiles(args, runner="controlled")
    args.log_dir = "eval-runs/test"
    assert args.azure_auth == "default"
    assert args.credential_env == "AZUREAI_OPENAI_API_KEY"
    validate_controlled(args, {})


def test_combined_config_can_omit_prices_and_budget(profile_files):
    from run_behavior import parser
    from run_inspect import cost_control, default_log_dir

    config = profile_files(
        profile(
            "controlled",
            "  model: openai/azure/deployment\n"
            "  azure_auth: default\n"
            "  azure_endpoint: https://example.openai.azure.com\n"
            "  image: example/image@sha256:" + "a" * 64 + "\n"
            "  case_id: scope-search\n"
            "  trial: 1\n"
            "  cpus: 1\n"
            "  time_limit: 60\n"
            "  wall_timeout: 90\n"
            "  tool_timeout: 10\n"
            "  turn_limit: 5\n"
            "  tool_limit: 5\n"
            "  token_limit: 4000\n"
            "  max_tokens: 500\n"
            "  memory_mb: 512",
        )
    )
    args = parser().parse_args(["--config", str(config), "--run"])
    apply_profiles(args, runner="controlled")
    default_log_dir(args, "behavior")
    validate_controlled(args, {})
    assert args.input_price is None and args.output_price is None and args.budget_usd is None
    assert cost_control(args)["pricing"] == "UNKNOWN"
    assert cost_control(args)["budget"] == "DISABLED"


def test_separate_profiles_override_combined_config(profile_files):
    from run_behavior import parser

    config = profile_files(profile("controlled", "  model: openai/azure/from-config"))
    agent = profile_files(profile("controlled", "  model: openai/azure/from-agent"))
    args = parser().parse_args(["--config", str(config), "--agent", str(agent)])
    apply_profiles(args, runner="controlled")
    assert args.model == "openai/azure/from-agent"


@pytest.mark.parametrize("forbidden", ["run", "allow_network", "api_key", "log_dir"])
def test_combined_config_cannot_authorize_or_contain_credentials(profile_files, forbidden):
    path = profile_files(profile("controlled", f"  {forbidden}: value"))
    with pytest.raises(ValueError, match="unknown"):
        apply_profiles(argparse.Namespace(config=path), runner="controlled")


def test_smoke_runner_rejects_case_selection_from_combined_config(profile_files):
    from run_inspect import parser

    path = profile_files(profile("controlled", "  case_id: scope-search"))
    args = parser().parse_args(["--config", str(path)])
    with pytest.raises(ValueError, match="not supported"):
        apply_profiles(args, runner="controlled")


def test_default_log_dir_is_unique_and_remains_uncreated():
    from run_inspect import default_log_dir

    first = argparse.Namespace(run=True, log_dir=None)
    second = argparse.Namespace(run=True, log_dir=None)
    default_log_dir(first, "behavior")
    default_log_dir(second, "behavior")
    assert first.log_dir != second.log_dir
    assert Path(first.log_dir).parts[0] == "eval-runs"
    assert not Path(first.log_dir).exists()
    existing = argparse.Namespace(run=True, log_dir="my-explicit-dir")
    default_log_dir(existing, "behavior")
    assert existing.log_dir == "my-explicit-dir"


@pytest.mark.parametrize(
    "content",
    [
        "",
        "[]",
        "schema_version: 1\nrunner: controlled\n",
        "schema_version: 1\nrunner: controlled\noptions:\n",
        "schema_version: 1\nrunner: controlled\noptions: []",
        "schema_version: 1\nrunner: controlled\noptions: {}\nextra: bad",
        "schema_version: true\nrunner: controlled\noptions: {}",
        "schema_version: '1'\nrunner: controlled\noptions: {}",
        "schema_version: 2\nrunner: controlled\noptions: {}",
        "schema_version: 1\nrunner: controlled\nname: null\noptions: {}",
        "schema_version: 1\nrunner: controlled\noptions: {unknown: value}",
        "schema_version: 1\nrunner: controlled\noptions: {1: value}",
        "schema_version: 1\nschema_version: 1\nrunner: controlled\noptions: {}",
        "schema_version: 1\nrunner: controlled\noptions: {model: x, model: y}",
        "schema_version: 1\nrunner: controlled\noptions: {model: [}",
        "schema_version: 1\nrunner: controlled\noptions: {model: !!str x}",
        "schema_version: 1\nrunner: controlled\noptions: {model: !custom x}",
        "schema_version: 1\nrunner: controlled\noptions: {model: &anchor x}",
        "schema_version: 1\nrunner: controlled\noptions: {model: &anchor x, y: *anchor}",
        "schema_version: 1\nrunner: controlled\noptions: {<<: {model: x}}",
        "schema_version: 1\nrunner: controlled\noptions: {}\n---\noptions: {}",
    ],
)
def test_malformed_unknown_and_unsafe_yaml_is_rejected(profile_files, content):
    args = argparse.Namespace(agent=profile_files(content), model="cli-wins")
    with pytest.raises(ValueError):
        apply_profiles(args, runner="controlled")
    assert args.model == "cli-wins"


@pytest.mark.parametrize("kind", ["agent", "harness"])
@pytest.mark.parametrize(
    "key",
    [
        "credential",
        "api_key",
        "token",
        "run",
        "allow_network",
        "output_dir",
        "log_dir",
        "manifest",
        "case",
        "case_id",
        "trial",
    ],
)
def test_credential_authorization_and_run_selection_options_are_rejected(profile_files, kind, key):
    path = profile_files(profile("native", f"  {key}: value"))
    with pytest.raises(ValueError, match="unknown"):
        apply_profiles(argparse.Namespace(**{kind: path}), runner="native")


@pytest.mark.parametrize(
    ("kind", "runner", "options"),
    [
        ("agent", "controlled", "  model: 4"),
        ("agent", "controlled", "  credential_env: secret-with-punctuation"),
        ("agent", "controlled", "  credential_env: null"),
        ("harness", "controlled", "  image: 42"),
        ("harness", "controlled", "  time_limit: true"),
        ("harness", "controlled", "  time_limit: 1.0"),
        ("harness", "controlled", '  time_limit: "1"'),
        ("harness", "controlled", "  cpus: true"),
        ("harness", "controlled", "  cpus: .inf"),
        ("harness", "controlled", "  budget_usd: .nan"),
        ("harness", "controlled", f"  budget_usd: {2**1024}"),
        ("harness", "native", "  session_ai_credits: false"),
        ("harness", "native", "  host_version: 1.0"),
    ],
)
def test_strict_option_types(profile_files, kind, runner, options):
    path = profile_files(profile(runner, options))
    with pytest.raises(ValueError):
        apply_profiles(argparse.Namespace(**{kind: path}), runner=runner)


def test_wrong_runner_invalid_runner_and_atomic_failure(profile_files):
    agent = profile_files(profile("controlled", "  model: yaml-model"))
    harness = profile_files(profile("native", "  image: yaml-image"))
    args = argparse.Namespace(agent=agent, harness=harness, model=None)
    with pytest.raises(ValueError, match="runner"):
        apply_profiles(args, runner="controlled")
    assert args.model is None
    assert not hasattr(args, "image")
    with pytest.raises(ValueError, match="runner"):
        apply_profiles(args, runner="other")
    with pytest.raises(ValueError, match="runner"):
        apply_profiles(args, runner=[])


def test_wrong_option_for_runner_or_profile_kind(profile_files):
    for kind, runner, option in [
        ("agent", "controlled", "image"),
        ("harness", "controlled", "model"),
        ("harness", "native", "budget_usd"),
        ("harness", "controlled", "host_version"),
    ]:
        path = profile_files(profile(runner, f"  {option}: value"))
        with pytest.raises(ValueError, match="unknown"):
            apply_profiles(argparse.Namespace(**{kind: path}), runner=runner)


def test_rejects_symlink_file_and_parent(profile_files):
    path = profile_files(profile("controlled", "  model: example"))
    alias = profile_files(target=path)
    parent_alias = profile_files(target=ROOT, directory=True)
    for link in (alias, parent_alias / path.name):
        with pytest.raises(ValueError):
            apply_profiles(argparse.Namespace(agent=link), runner="controlled")


def test_examples_are_loadable_placeholders():
    for runner in ("controlled", "native"):
        args = argparse.Namespace(
            agent=ROOT / f"evals/agents/{runner}.example.yaml",
            harness=ROOT / f"evals/harnesses/{runner}.example.yaml",
        )
        apply_profiles(args, runner=runner)
        assert args.model == "<EDIT_MODEL>"
        assert args.image == "<EDIT_IMAGE>"
        assert args.credential_env.endswith("_KEY") or args.credential_env.endswith("_TOKEN")
        assert not hasattr(args, "run")
        assert not hasattr(args, "allow_network")

        args.run = True
        if runner == "controlled":
            with pytest.raises(ValueError, match="model"):
                validate_controlled(args, {"OPENAI_API_KEY": "test-key"})
            args.model = "openai/example"
            with pytest.raises(ValueError, match="image"):
                validate_controlled(args, {"OPENAI_API_KEY": "test-key"})
        else:
            args.allow_network = True
            env = {"COPILOT_GITHUB_TOKEN": "test-key-for-testing"}
            with pytest.raises(ValueError, match="image"):
                validate_native(args, env, [])
            args.image = "example/image@sha256:" + "a" * 64
            with pytest.raises(ValueError, match="model"):
                validate_native(args, env, [])


def test_existing_validators_remain_authoritative(profile_files):
    controlled = argparse.Namespace(
        agent=profile_files(profile("controlled", "  model: openai/example")),
        harness=profile_files(profile("controlled", "  time_limit: -1")),
        run=True,
        credential_env="OPENAI_API_KEY",
        image="example/image@sha256:" + "a" * 64,
        budget_usd=1,
        input_price=1,
        output_price=1,
        cpus=1,
        wall_timeout=10,
        tool_timeout=1,
        turn_limit=1,
        tool_limit=1,
        token_limit=1,
        max_tokens=1,
        memory_mb=128,
        log_dir="logs",
        docker_host=None,
    )
    apply_profiles(controlled, runner="controlled")
    with pytest.raises(ValueError, match="time-limit"):
        validate_controlled(controlled, {"OPENAI_API_KEY": "test-key"})

    native = argparse.Namespace(
        harness=profile_files(profile("native", "  session_ai_credits: 1")),
        run=True,
        allow_network=True,
        credential_env="COPILOT_GITHUB_TOKEN",
        image="example/image@sha256:" + "a" * 64,
        model="example",
        host_version="1.0.110-3",
        output_dir="output",
        docker_host=None,
        run_ai_credits=30,
        session_timeout=30,
        run_timeout=30,
        cpus=1,
        memory_mb=128,
        max_output_mb=1,
    )
    apply_profiles(native, runner="native")
    with pytest.raises(ValueError, match="session-ai-credits"):
        validate_native(native, {"COPILOT_GITHUB_TOKEN": "test-key-for-testing"}, [])
