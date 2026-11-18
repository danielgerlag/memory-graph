"""Offline validation and command construction for the native-host trial runner."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tarfile
import uuid
from datetime import datetime
from pathlib import Path

from corpus import verify_corpus

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "scenarios.json"
SKILL = ROOT / "memory-graph"
CONTAINER_WORKSPACE = "/eval/workspace"
CONTAINER_STORE = f"{CONTAINER_WORKSPACE}/store"
STORE_ID = "52bd8904-9a68-4391-9c0c-03af22bf2211"
AUTH_NAMES = {"COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"}
DOCUMENTED_BUILD = "1.0.110-3"

VERIFIED_ACTIVATION_ADAPTERS = {"1.0.110-3": "session-skill-invoked-v1"}


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json_bytes(raw):
    return json.loads(
        raw,
        object_pairs_hook=_unique_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def load_trials(manifest_path):
    verify_corpus()
    dataset_raw = DATASET.read_bytes()
    dataset = read_json_bytes(dataset_raw)
    manifest_raw = Path(manifest_path).read_bytes()
    manifest = read_json_bytes(manifest_raw)
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "dataset_sha256",
        "trials",
    }:
        raise ValueError("manifest must contain schema_version, dataset_sha256 and trials")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported native manifest schema")
    dataset_hash = hashlib.sha256(dataset_raw).hexdigest()
    if manifest["dataset_sha256"] != dataset_hash:
        raise ValueError("native manifest dataset hash mismatch")
    if not isinstance(manifest["trials"], list) or not 1 <= len(manifest["trials"]) <= 10:
        raise ValueError("manifest must select 1 to 10 trials")
    cases = {case["id"]: case for case in dataset["cases"] if case.get("lane") == "native"}
    seen = set()
    trials = []
    for entry in manifest["trials"]:
        if not isinstance(entry, dict) or set(entry) != {"case_id", "trial"}:
            raise ValueError("native trial requires only case_id and trial")
        case_id, trial = entry["case_id"], entry["trial"]
        if (
            not isinstance(case_id, str)
            or not re.fullmatch(r"native-(positive|negative)-\d{2}", case_id)
            or case_id not in cases
            or type(trial) is not int
            or not 1 <= trial <= dataset["trial_count"]
            or (case_id, trial) in seen
        ):
            raise ValueError("unknown or duplicate native case/trial")
        seen.add((case_id, trial))
        case = cases[case_id]
        if (
            type(case.get("should_activate")) is not bool
            or case.get("fixture") != {"main": {"facts": [], "edges": []}}
            or not isinstance(case.get("prompts"), list)
            or len(case["prompts"]) != 1
            or any(
                not isinstance(p, str) or not p.strip() or len(p) > 4000 for p in case["prompts"]
            )
        ):
            raise ValueError(
                "native case must have an empty synthetic fixture and exactly one prompt"
            )
        trials.append(
            {
                "case_id": case_id,
                "trial": trial,
                "prompts": case["prompts"],
                "should_activate": case["should_activate"],
                "critical": case.get("critical", False),
            }
        )
    return trials, dataset_hash, hashlib.sha256(manifest_raw).hexdigest()


def validate_options(args, env, trials):
    if not args.run or not args.allow_network:
        raise ValueError("native execution requires --run and --allow-network")
    if args.credential_env not in AUTH_NAMES or not env.get(args.credential_env):
        raise ValueError("explicit nonempty GitHub token environment variable is required")
    if len(env[args.credential_env]) < 16:
        raise ValueError("token is too short for safe leak detection")
    if not isinstance(args.image, str) or not re.fullmatch(
        r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}", args.image
    ):
        raise ValueError("image must be an immutable image ID or name@sha256 digest")
    if (
        not isinstance(args.model, str)
        or not re.fullmatch(r"[A-Za-z0-9._:-]+", args.model)
        or args.model.lower() == "auto"
    ):
        raise ValueError("an explicit non-auto Copilot model is required")
    if args.host_version != DOCUMENTED_BUILD:
        raise ValueError(f"only the documented CLI build {DOCUMENTED_BUILD} is selectable")
    if not args.output_dir:
        raise ValueError("--output-dir is required")
    if args.docker_host and not args.docker_host.startswith("unix:///"):
        raise ValueError("--docker-host must be an explicit unix:/// socket")
    caps = {
        "session_timeout": (1, 1800),
        "run_timeout": (1, 7200),
        "memory_mb": (128, 8192),
        "session_ai_credits": (30, 1_000_000),
        "run_ai_credits": (30, 10_000_000),
        "max_output_mb": (1, 64),
    }
    for field, (low, high) in caps.items():
        value = getattr(args, field)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{field.replace('_', '-')} must be between {low} and {high}")
    if (
        type(args.cpus) not in (int, float)
        or not math.isfinite(args.cpus)
        or not 0 < args.cpus <= 8
        or args.run_timeout < args.session_timeout
    ):
        raise ValueError("invalid CPU or timeout limits")
    sessions = sum(len(trial["prompts"]) for trial in trials)
    if sessions * args.session_ai_credits > args.run_ai_credits:
        raise ValueError("selected sessions exceed the declared run AI-credit budget")


def skill_bundle():
    if SKILL.is_symlink() or not SKILL.is_dir():
        raise ValueError("skill root must be a real directory")
    files = [SKILL / "SKILL.md", SKILL / "pyproject.toml", SKILL / "uv.lock"]
    for folder, pattern in (("scripts", "*.py"), ("references", "*.md"), ("assets", "*.json")):
        files.extend((SKILL / folder).rglob(pattern))
    result = {}
    for path in sorted(files):
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(SKILL.resolve())
        ):
            raise ValueError("invalid or escaping skill bundle file")
        if "__pycache__" in path.parts:
            raise ValueError("unexpected bytecode in skill bundle")
        data = path.read_bytes()
        if len(data) > 1_048_576:
            raise ValueError("skill bundle file exceeds size limit")
        result[path.relative_to(SKILL).as_posix()] = data
    if not {"SKILL.md", "scripts/memory_graph.py", "assets/node.schema.json"}.issubset(result):
        raise ValueError("incomplete skill bundle")
    return result


def skill_digest(bundle):
    digest = hashlib.sha256()
    for name, value in sorted(bundle.items()):
        digest.update(name.encode() + b"\0" + hashlib.sha256(value).digest())
    return digest.hexdigest()


def fixture_archive(bundle):
    """Create an in-memory archive with only the bundle and an empty synthetic store."""
    files = {f".github/skills/memory-graph/{name}": value for name, value in bundle.items()}
    files.update({f"store/{name}": value for name, value in fixture_store().items()})
    directories = set()
    for name in files:
        path = Path(name).parent
        while str(path) != ".":
            directories.add(path.as_posix())
            path = path.parent
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name in sorted(directories, key=lambda x: (x.count("/"), x)):
            entry = tarfile.TarInfo(name)
            entry.type, entry.mode, entry.uid, entry.gid, entry.mtime = (
                tarfile.DIRTYPE,
                0o755,
                1000,
                1000,
                0,
            )
            archive.addfile(entry)
        for name, value in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.mode, entry.uid, entry.gid, entry.mtime, entry.size = (
                0o644,
                1000,
                1000,
                0,
                len(value),
            )
            archive.addfile(entry, io.BytesIO(value))
    return output.getvalue()


def fixture_store():
    return {
        "store.json": json.dumps({"schema": 1, "id": STORE_ID}, sort_keys=True).encode() + b"\n",
        ".gitignore": b".memory-graph/\n",
    }


def docker_create(args):
    return [
        "docker",
        "create",
        "--rm",
        "--pull=never",
        "--network=bridge",
        "--read-only",
        "--user=1000:1000",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--cpus={args.cpus}",
        f"--memory={args.memory_mb}m",
        "--pids-limit=128",
        "--tmpfs=/eval:rw,mode=1777,size=134217728",
        "--tmpfs=/cli-tmp:rw,exec,mode=1777,size=402653184",
        "--workdir=/eval",
        "--entrypoint=/bin/sh",
        args.image,
        "-c",
        f"exec sleep {args.run_timeout + 15}",
    ]


def copilot_command(args, prompt, session):
    """Keep the secret in the process environment, never in the command arguments."""
    wrapper = (
        "import os,sys; "
        "env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/eval/home',"
        "'COPILOT_HOME':'/eval/home/.copilot','COPILOT_CACHE_HOME':'/cli-tmp/cache',"
        "'TMPDIR':'/cli-tmp',"
        "'COPILOT_PLUGIN_DIR_ONLY':'true','COPILOT_DISABLE_LOGIN_SHELL_ENV':'true',"
        "'COPILOT_AUTO_UPDATE':'false','COPILOT_TASK_WAIT_TIMEOUT_SECONDS':'5',"
        "'COPILOT_SYSTEM_PROXY':'0','COPILOT_GITHUB_TOKEN':os.environ['COPILOT_GITHUB_TOKEN']}; "
        "os.execvpe('copilot',['copilot',*sys.argv[1:]],env)"
    )
    return [
        "python3",
        "-c",
        wrapper,
        "-C",
        CONTAINER_WORKSPACE,
        "--trust-folder",
        "--no-auto-update",
        "--no-custom-instructions",
        "--no-ask-user",
        "--disable-builtin-mcps",
        "--disable-client-telemetry",
        "--secret-env-vars=COPILOT_GITHUB_TOKEN",
        "--deny-tool=shell",
        "--deny-tool=write",
        "--deny-tool=task",
        "--deny-tool=agent",
        "--deny-tool=read(/proc/**)",
        "--deny-tool=read(/sys/**)",
        "--allow-tool=read(/eval/workspace/**)",
        "--model",
        args.model,
        "--max-ai-credits",
        str(args.session_ai_credits),
        "--output-format",
        "json",
        "--usage-output-file",
        f"/eval/session-{session}-usage.json",
        "--prompt",
        prompt,
    ]


def parse_jsonl(raw):
    if not raw:
        raise ValueError("empty JSONL transcript")
    events = []
    for line in raw.splitlines():
        if not line.strip():
            raise ValueError("blank JSONL line")
        event = read_json_bytes(line)
        if not isinstance(event, dict):
            raise ValueError("JSONL events must be objects")  # noqa: TRY004 - malformed serialized data
        events.append(event)
    if not events or not raw.endswith(b"\n"):
        raise ValueError("truncated JSONL transcript")
    return events


def _unverified(reason):
    return {"status": "UNVERIFIED", "reason": reason}


def _uuid(value):
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _skill_body(value):
    if not isinstance(value, bytes):
        raise TypeError("expected skill bytes are required")
    text = value.decode("utf-8")
    if not text.startswith("---\n"):
        raise ValueError("expected skill has no frontmatter")
    _, delimiter, body = text[4:].partition("\n---\n")
    body = body.lstrip("\r\n")
    if not delimiter or not body:
        raise ValueError("expected skill body is missing")
    return body


def activation_status(
    host_version,
    raw,
    session_events=None,
    *,
    outcome=None,
    exit_status=None,
    expected_skill=None,
    skill_name="memory-graph",
    skill_path=f"{CONTAINER_WORKSPACE}/.github/skills/memory-graph/SKILL.md",
):
    """Detect an agent-invoked skill only in a verified, complete host session."""
    events = parse_jsonl(raw)
    if host_version not in VERIFIED_ACTIVATION_ADAPTERS:
        return _unverified("no validated skill-load event mapping")
    if session_events is None or expected_skill is None:
        return _unverified("complete session events and expected skill are required")
    if outcome != "complete" or type(exit_status) is not int or exit_status != 0:
        return _unverified("native process did not complete successfully")
    persisted = parse_jsonl(session_events)
    body = _skill_body(expected_skill)
    if not isinstance(skill_name, str) or not skill_name or not isinstance(skill_path, str):
        raise ValueError("expected skill name and path are required")
    expected_path = Path(skill_path)
    if (
        not expected_path.is_absolute()
        or ".." in expected_path.parts
        or str(expected_path) != skill_path
    ):
        raise ValueError("expected skill path must be canonical and absolute")
    if not all(isinstance(event.get("type"), str) and event["type"] for event in events):
        return _unverified("native output event type is invalid")
    results = [event for event in events if event.get("type") == "result"]
    if (
        len(results) != 1
        or events[-1] is not results[0]
        or type(results[0].get("exitCode")) is not int
        or results[0]["exitCode"] != 0
        or not _uuid(results[0].get("sessionId"))
        or not any(event.get("type") == "assistant.idle" for event in events[:-1])
        or any(event.get("type") in {"session.error", "abort"} for event in events)
    ):
        return _unverified("native output lacks a complete successful session")
    catalogs = [event for event in events if event.get("type") == "session.skills_loaded"]
    if not catalogs or not isinstance(catalogs[-1].get("data"), dict):
        return _unverified("skill discovery evidence is missing")
    skills = catalogs[-1]["data"].get("skills")
    if not isinstance(skills, list) or not all(isinstance(item, dict) for item in skills):
        return _unverified("skill discovery evidence is malformed")
    matching = [
        item for item in skills if item.get("name") == skill_name or item.get("path") == skill_path
    ]
    if (
        len(matching) != 1
        or matching[0].get("name") != skill_name
        or matching[0].get("path") != skill_path
        or matching[0].get("source") != "project"
        or matching[0].get("enabled") is not True
    ):
        return _unverified("expected project skill is missing, disabled, or ambiguous")
    if (
        persisted[0].get("type") != "session.start"
        or persisted[-1].get("type") != "session.shutdown"
        or sum(event.get("type") == "session.start" for event in persisted) != 1
        or sum(event.get("type") == "session.shutdown" for event in persisted) != 1
    ):
        return _unverified("session event log is incomplete")
    start = persisted[0].get("data")
    shutdown = persisted[-1].get("data")
    if (
        not isinstance(start, dict)
        or start.get("sessionId") != results[0]["sessionId"]
        or start.get("copilotVersion") != host_version
        or start.get("producer") != "copilot-agent"
        or type(start.get("version")) is not int
        or start["version"] != 1
        or not isinstance(shutdown, dict)
        or shutdown.get("shutdownType") != "routine"
        or type(shutdown.get("eventsFileSizeBytes")) is not int
        or shutdown["eventsFileSizeBytes"]
        != len(session_events) - len(session_events.splitlines(keepends=True)[-1])
    ):
        return _unverified("session identity, host version, or log completeness is invalid")

    seen = set()
    current_turn = None
    completed_turns = 0
    matches = []
    for index, event in enumerate(persisted, 1):
        event_id, parent = event.get("id"), event.get("parentId")
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            return _unverified("session event type is invalid")
        try:
            timestamp = datetime.fromisoformat(event.get("timestamp", ""))
        except (ValueError, TypeError, AttributeError):
            return _unverified("session event timestamp is invalid")
        if (
            not _uuid(event_id)
            or event_id in seen
            or (index == 1 and parent is not None)
            or (index > 1 and (not _uuid(parent) or parent not in seen))
            or timestamp.utcoffset() is None
            or (event.get("ephemeral") is not None and event.get("ephemeral") is not False)
            or event_type in {"session.error", "abort"}
        ):
            return _unverified("session event chain is invalid")
        seen.add(event_id)
        data = event.get("data")
        agent_id = event.get("agentId")
        if agent_id is not None and (not isinstance(agent_id, str) or not agent_id):
            return _unverified("session agent identity is invalid")
        if event_type in {"assistant.turn_start", "assistant.turn_end"} and agent_id is None:
            if not isinstance(data, dict) or not isinstance(data.get("turnId"), str):
                return _unverified("assistant turn evidence is malformed")
            if event_type == "assistant.turn_start":
                if current_turn is not None:
                    return _unverified("assistant turns overlap")
                current_turn = data["turnId"]
            else:
                if current_turn != data["turnId"]:
                    return _unverified("assistant turn evidence is incomplete")
                current_turn = None
                completed_turns += 1
        if event_type == "skill.invoked":
            if not isinstance(data, dict):
                return _unverified("skill invocation evidence is malformed")
            if data.get("name") != skill_name and data.get("path") != skill_path:
                continue
            if (
                data.get("name") != skill_name
                or data.get("path") != skill_path
                or data.get("source") != "project"
                or data.get("trigger") != "agent-invoked"
                or data.get("content") != body
                or agent_id is not None
                or current_turn is None
            ):
                return _unverified("skill invocation identity or content is ambiguous")
            matches.append(index)
    if current_turn is not None or not completed_turns:
        return _unverified("assistant turn evidence is incomplete")
    return {
        "status": "VERIFIED",
        "detected": bool(matches),
        "observation_complete": True,
        "event_lines": matches,
        "source": "session_events",
        "events": len(persisted),
    }
