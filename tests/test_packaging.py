import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tools import package

ROOT = Path(__file__).resolve().parents[1]


def test_plugin_copy_matches_skill():
    assert package.plugin_drift() == [], "run: python3 tools/package.py plugin"


def test_bundle_matches_skill_export(tmp_path):
    exported = tmp_path / "exported"
    script = ROOT / "memory-graph/scripts/memory_graph.py"
    subprocess.run(
        [sys.executable, str(script), "export-skill", "--dest", str(exported)],
        check=True,
        capture_output=True,
    )
    assert package.bundle_files(exported) == package.bundle_files()


def test_install_replaces_only_memory_graph_skill(tmp_path):
    destination = package.install("claude", tmp_path, force=False, sync=False)
    assert destination == tmp_path.resolve() / ".claude/skills/memory-graph"
    assert (destination / "SKILL.md").is_file()
    with pytest.raises(package.PackageError, match="already exists"):
        package.install("claude", tmp_path, force=False, sync=False)
    package.install("claude", tmp_path, force=True, sync=False)
    (destination / "SKILL.md").write_text("---\nname: other\n---\n", encoding="utf-8")
    with pytest.raises(package.PackageError, match="not a memory-graph skill"):
        package.install("claude", tmp_path, force=True, sync=False)


def test_archive_has_one_skill_folder(tmp_path):
    output = package.archive(tmp_path / "skill.zip")
    with zipfile.ZipFile(output) as bundle:
        names = bundle.namelist()
    assert "memory-graph/SKILL.md" in names
    assert all(name.startswith("memory-graph/") for name in names)
    assert not any(".venv" in name or "__pycache__" in name for name in names)
    with pytest.raises(package.PackageError, match="already exists"):
        package.archive(output)


def test_lockfiles_use_only_public_pypi():
    for lock in (ROOT / "uv.lock", ROOT / "memory-graph/uv.lock"):
        text = lock.read_text(encoding="utf-8")
        assert 'registry = "https://pypi.org/simple"' in text
        assert "visualstudio" not in text and "microsoft.io" not in text, lock
        assert all(
            url.startswith("https://files.pythonhosted.org/")
            for url in re.findall(r'url = "(https://[^"]+)"', text)
        ), lock
