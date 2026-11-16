"""Strict, opt-in YAML profiles for the existing eval runner arguments.

Call ``apply_profiles(args, runner="controlled" | "native")`` after parsing
arguments, before the runner's existing validation. Missing ``agent``,
``harness``, and ``config`` attributes mean no profile. All files are
validated before any arguments are changed; non-None argument values win.

Each file requires ``schema_version: 1``, a matching ``runner``, and an
``options`` mapping (which may be empty); ``name`` is optional. Agent options
include ``model`` and ``credential_env``. Harness options are the destinations
in ``_HARNESS_OPTIONS`` below. A controlled config combines both and may select
a case and trial. Values must have their CLI types, with integer
values also accepted for float options; numeric range and format checks remain
the runner's responsibility. Invalid files raise ``ValueError``.

Profiles do not select defaults, authorize execution or network access, or
read credentials. ``credential_env`` is only an environment-variable name.
The existing runner validators remain required.
"""

from __future__ import annotations

import math
import os
import re
import stat
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.events import AliasEvent, ScalarEvent

_AGENT_OPTIONS = {"model": str, "credential_env": str}
_CONTROLLED_AGENT_OPTIONS = {
    **_AGENT_OPTIONS,
    "azure_endpoint": str,
    "azure_api_version": str,
    "azure_auth": str,
    "model_family": str,
}
_HARNESS_OPTIONS = {
    "controlled": {
        "image": str,
        "docker_host": str,
        "budget_usd": float,
        "input_price": float,
        "output_price": float,
        "time_limit": int,
        "wall_timeout": int,
        "tool_timeout": int,
        "turn_limit": int,
        "tool_limit": int,
        "token_limit": int,
        "max_tokens": int,
        "cpus": float,
        "memory_mb": int,
    },
    "native": {
        "image": str,
        "host_version": str,
        "docker_host": str,
        "session_ai_credits": int,
        "run_ai_credits": int,
        "session_timeout": int,
        "run_timeout": int,
        "cpus": float,
        "memory_mb": int,
        "max_output_mb": int,
    },
}
_MAX_PROFILE_BYTES = 1_048_576


def _read_profile(path: str | os.PathLike[str]) -> str:
    """Read a regular file, rejecting symlinks in every component when supported."""
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    with ExitStack() as stack:
        if (
            hasattr(os, "O_NOFOLLOW")
            and hasattr(os, "O_DIRECTORY")
            and os.open in os.supports_dir_fd
        ):
            parts = path.parts
            absolute = path.is_absolute()
            parent = os.open(path.anchor if absolute else ".", os.O_RDONLY)
            stack.callback(os.close, parent)
            components = parts[1:] if absolute else parts
            if not components:
                raise ValueError("profile must be a regular file")
            for component in components[:-1]:
                parent = os.open(
                    component, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY, dir_fd=parent
                )
                stack.callback(os.close, parent)
            fd = os.open(components[-1], flags | os.O_NOFOLLOW, dir_fd=parent)
        else:
            if any(part.is_symlink() for part in (path, *path.parents)):
                raise ValueError("profile paths must not contain symlinks")
            fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0))
        stack.callback(os.close, fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("profile must be a regular file")
        with os.fdopen(fd, "rb", closefd=False) as file:
            data = file.read(_MAX_PROFILE_BYTES + 1)
        if len(data) > _MAX_PROFILE_BYTES:
            raise ValueError("profile exceeds size limit")
        return data.decode("utf-8")


def _load_profile(path: str | os.PathLike[str], runner: str, kind: str) -> dict:
    try:
        text = _read_profile(path)
        yaml = YAML(typ="safe", pure=True)
        yaml.allow_duplicate_keys = False
        for event in yaml.parse(text):
            if (
                isinstance(event, AliasEvent)
                or getattr(event, "anchor", None) is not None
                or getattr(event, "tag", None) is not None
                or (isinstance(event, ScalarEvent) and event.style is None and event.value == "<<")
            ):
                raise ValueError("YAML aliases, anchors, tags and merges are not allowed")
        profile = yaml.load(text)
    except (OSError, UnicodeError, YAMLError, TypeError, RecursionError) as exc:
        raise ValueError(f"cannot read or parse {kind} profile") from exc

    if not isinstance(profile, dict) or set(profile) - {
        "schema_version",
        "runner",
        "options",
        "name",
    }:
        raise ValueError(f"invalid {kind} profile envelope")
    if type(profile.get("schema_version")) is not int or profile["schema_version"] != 1:
        raise ValueError(f"invalid {kind} profile schema_version")
    if type(profile.get("runner")) is not str or profile["runner"] != runner:
        raise ValueError(f"{kind} profile runner does not match {runner}")
    if "name" in profile and (type(profile["name"]) is not str or not profile["name"].strip()):
        raise ValueError(f"invalid {kind} profile name")

    options = profile.get("options")
    if kind == "agent":
        allowed = _CONTROLLED_AGENT_OPTIONS if runner == "controlled" else _AGENT_OPTIONS
    elif kind == "harness":
        allowed = _HARNESS_OPTIONS[runner]
    else:
        if runner != "controlled":
            raise ValueError("combined config requires the controlled runner")
        allowed = {
            **_CONTROLLED_AGENT_OPTIONS,
            **_HARNESS_OPTIONS[runner],
            "case_id": str,
            "trial": int,
        }
    if not isinstance(options, dict) or not all(type(key) is str for key in options):
        raise ValueError(f"{kind} profile options must be a mapping with string keys")
    if options.keys() - allowed.keys():
        raise ValueError(f"unknown {kind} profile option")
    for key, value in options.items():
        expected = allowed[key]
        if expected is float:
            try:
                valid = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                valid = False
        else:
            valid = type(value) is expected
        if not valid:
            raise ValueError(f"invalid type or value for {kind} option {key}")
        if key == "credential_env" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(
                "credential_env must be an environment variable name, not a credential"
            )
    return options


def apply_profiles(args: Namespace, *, runner: str) -> Namespace:
    """Fill unset arguments from agent and harness YAML profiles and return args.

    The envelope requires ``schema_version: 1``, ``runner``, and an ``options``
    mapping; ``name`` is optional. Unknown keys and options are errors, even
    when an explicit argument would override them. Runner validation is still
    responsible for all semantic constraints and execution authorization.
    """
    if type(runner) is not str or runner not in _HARNESS_OPTIONS:
        raise ValueError("runner must be controlled or native")
    loaded = []
    for kind in ("agent", "harness", "config"):
        path = getattr(args, kind, None)
        if path is not None:
            loaded.append(_load_profile(path, runner, kind))
    for options in loaded:
        for key in ("case_id", "trial"):
            if key in options and not hasattr(args, key):
                raise ValueError(f"{key} is not supported by this runner")
    for options in loaded:
        for key, value in options.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    return args
