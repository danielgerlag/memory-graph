"""Offline tests for the explicit, network-disabled native host probe."""

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import validate_native_activation as probe


def valid_results():
    return {
        "results": [
            {
                "expected": False,
                "native_launcher": True,
                "usage_captured": True,
                "read_tools_denied": False,
                "skill_offered": True,
                "model_requests": 1,
                "activation": {
                    "status": "VERIFIED",
                    "detected": False,
                    "observation_complete": True,
                    "event_lines": [],
                    "source": "session_events",
                },
                "tampered": None,
            },
            {
                "expected": True,
                "native_launcher": True,
                "usage_captured": True,
                "read_tools_denied": False,
                "skill_offered": True,
                "model_requests": 2,
                "activation": {
                    "status": "VERIFIED",
                    "detected": True,
                    "observation_complete": True,
                    "event_lines": [4],
                    "source": "session_events",
                },
                "tampered": {"status": "UNVERIFIED"},
            },
        ]
    }


def test_probe_container_has_no_network_or_host_mounts():
    command = probe.probe_container("sha256:" + "a" * 64)
    assert "--network=none" in command
    assert "--pull=never" in command and "--read-only" in command
    assert "--user=1000:1000" in command
    assert not any("--volume" in item or "--mount" in item for item in command)


def test_probe_requires_both_observations_and_rejects_tampering():
    assert len(probe.validate_results(valid_results())) == 2
    deny_read = valid_results()
    for result in deny_read["results"]:
        result["read_tools_denied"] = True
    assert len(probe.validate_results(deny_read, deny_read=True)) == 2
    for update in (
        lambda value: value["results"].pop(),
        lambda value: value["results"][0].update(skill_offered=False),
        lambda value: value["results"][0].update(native_launcher=False),
        lambda value: value["results"][1].update(usage_captured=False),
        lambda value: value["results"][1].update(read_tools_denied=True),
        lambda value: value["results"][0]["activation"].update(detected=True),
        lambda value: value["results"][1]["activation"].update(event_lines=[]),
        lambda value: value["results"][1]["activation"].update(event_lines=[4, 4]),
        lambda value: value["results"][1].update(tampered={"status": "VERIFIED"}),
    ):
        value = valid_results()
        update(value)
        with pytest.raises(ValueError):
            probe.validate_results(value)


def test_probe_requires_opt_in_and_an_immutable_image():
    for argv in (
        ["--image", "sha256:" + "a" * 64],
        ["--run", "--image", "example:latest"],
    ):
        with (
            patch.object(probe.run_native, "admin") as admin,
            patch.object(Path, "mkdir") as mkdir,
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            assert probe.main(argv) == 2
            admin.assert_not_called()
            mkdir.assert_not_called()
