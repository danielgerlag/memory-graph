"""Offline validation and command construction for the native-host trial runner."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "scenarios.json"
SKILL = ROOT / "memory-graph"
CONTAINER_WORKSPACE = "/eval/workspace"
CONTAINER_STORE = f"{CONTAINER_WORKSPACE}/store"
STORE_ID = "52bd8904-9a68-4391-9c0c-03af22bf2211"
AUTH_NAMES = {"COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"}
DOCUMENTED_BUILD = "1.0.110-3"

# No shape has been verified against a pinned Copilot build.
VERIFIED_ACTIVATION_ADAPTERS = {}


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
        r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", args.image
    ):
        raise ValueError("image must be an immutable name@sha256 digest")
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
        "'COPILOT_HOME':'/eval/home/.copilot','COPILOT_CACHE_HOME':'/eval/home/cache',"
        "'COPILOT_PLUGIN_DIR_ONLY':'true','COPILOT_DISABLE_LOGIN_SHELL_ENV':'true',"
        "'COPILOT_AUTO_UPDATE':'false','COPILOT_TASK_WAIT_TIMEOUT_SECONDS':'5',"
        "'COPILOT_SYSTEM_PROXY':'0','COPILOT_GITHUB_TOKEN':os.environ['COPILOT_GITHUB_TOKEN']}; "
        "os.execvpe('copilot',sys.argv[1:],env)"
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


def activation_status(host_version, raw):
    """Unknown host event shapes can never produce a positive or negative observation."""
    events = parse_jsonl(raw)
    adapter = VERIFIED_ACTIVATION_ADAPTERS.get(host_version)
    if adapter is None:
        return {
            "status": "UNVERIFIED",
            "reason": "no validated skill-load event mapping",
            "events": len(events),
        }
    raise RuntimeError("no verified activation adapter is implemented")
