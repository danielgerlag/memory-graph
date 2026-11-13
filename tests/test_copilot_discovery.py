import subprocess
import sys
from pathlib import Path

import pytest

from host_checks import copilot_discovery


def test_discovery_requires_explicit_opt_in():
    script = Path(copilot_discovery.__file__)
    response = subprocess.run(
        [sys.executable, str(script), "--copilot", "/does/not/exist"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert response.returncode != 0
    assert "pass --run" in response.stderr


def test_discovery_does_not_infer_success_from_an_error_mention():
    assert copilot_discovery._found("  memory-graph - synthetic skill\n")
    assert not copilot_discovery._found("failed to load memory-graph\n")


def test_copilot_not_started_if_network_isolation_fails(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox-exec"
    sandbox.touch()
    copilot = tmp_path / "copilot"
    copilot.touch()
    monkeypatch.setattr(copilot_discovery.sys, "platform", "darwin")
    monkeypatch.setattr(copilot_discovery, "SANDBOX", str(sandbox))
    monkeypatch.setattr(copilot_discovery.os, "access", lambda *_: True)
    calls = []

    def failed_probe(args, cwd, env):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(copilot_discovery, "_run", failed_probe)
    with pytest.raises(RuntimeError, match="Copilot was not launched"):
        copilot_discovery.check(copilot)
    assert len(calls) == 1
    assert calls[0][-2:] == ["-c", copilot_discovery.NETWORK_PROBE]
