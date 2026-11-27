import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from memory_graph import cli, service
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "memory-graph"


def test_complete_skill_has_portable_metadata_and_dependencies():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, _ = text.split("---", 2)
    metadata = YAML(typ="safe").load(frontmatter)
    assert metadata["name"] == "memory-graph"
    assert metadata["description"]
    for path in [
        "scripts/memory_graph.py",
        "scripts/memory_graph/cli.py",
        "scripts/memory_graph/service.py",
        "scripts/memory_graph/git_store.py",
        "scripts/memory_graph/schema.py",
        "references/recipes.md",
        "references/format.md",
        "assets/node.schema.json",
        "assets/store.schema.json",
        "assets/operation.schema.json",
        "pyproject.toml",
        "uv.lock",
    ]:
        assert (SKILL / path).is_file(), path


def test_bundled_schemas_are_valid_draft_2020_12():
    for name in ("node", "store", "operation"):
        schema = json.loads((SKILL / "assets" / f"{name}.schema.json").read_text())
        Draft202012Validator.check_schema(schema)


def test_export_is_whole_bundle_and_does_not_overwrite(tmp_path):
    dest = tmp_path / "installed"
    script = SKILL / "scripts/memory_graph.py"
    command = [
        sys.executable,
        str(script),
        "export-skill",
        "--dest",
        str(dest),
    ]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert (dest / "SKILL.md").read_bytes() == (SKILL / "SKILL.md").read_bytes()
    assert (dest / "scripts/memory_graph.py").is_file()
    assert (dest / "uv.lock").is_file()
    assert not (dest / ".venv").exists()
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_store_commands_never_guess_a_store(tmp_path):
    env = {
        **os.environ,
        "MEMORY_GRAPH_CONFIG_DIR": str(tmp_path / "config"),
        "MEMORY_GRAPH_DATA_DIR": str(tmp_path / "data"),
    }
    response = subprocess.run(
        [sys.executable, str(SKILL / "scripts/memory_graph.py"), "find", "example"],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=env,
    )
    assert response.returncode != 0
    assert "no store selected" in json.loads(response.stderr)["error"]["message"]


def test_export_rejects_source_symlink_and_parent_escape(tmp_path, monkeypatch):
    bundle = tmp_path / "source"
    module = bundle / "scripts" / "memory_graph" / "cli.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    outside = tmp_path / "outside"
    outside.write_text("synthetic private data")
    try:
        (bundle / "scripts" / "link").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    monkeypatch.setattr(cli, "__file__", str(module))
    args = argparse.Namespace(
        root=tmp_path / "store", command="export-skill", dest=tmp_path / "out"
    )
    with pytest.raises(service.GraphError, match="symlink"):
        cli.execute(args)
    assert not args.dest.exists()
    (bundle / "scripts" / "link").unlink()
    (bundle / ".env").write_text("synthetic secret")
    with pytest.raises(service.GraphError, match="unexpected"):
        cli.execute(args)
    assert not args.dest.exists()
    (bundle / ".env").unlink()
    alias = tmp_path / "alias"
    alias.symlink_to(bundle, target_is_directory=True)
    args.dest = alias / "nested"
    with pytest.raises(service.GraphError, match="inside the skill"):
        cli.execute(args)
    assert not (bundle / "nested").exists()
