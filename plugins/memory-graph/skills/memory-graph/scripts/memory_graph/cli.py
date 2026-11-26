"""Command-line entry point for the bundled runtime."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import NoReturn

from . import cache, git_store, remotes, service


def _input(path: str) -> dict:
    if path == "-":
        data = sys.stdin.buffer.read(2_000_001)
    else:
        with Path(path).open("rb") as stream:
            data = stream.read(2_000_001)
    if len(data) > 2_000_000:
        raise service.GraphError("input exceeds 2 MB")
    return service._json_file(data, "input")


def _validate_skill_bundle(src: Path) -> None:
    allowed_root = {"SKILL.md", "pyproject.toml", "uv.lock", "scripts", "references", "assets"}

    def fail_on_error(error: OSError) -> None:
        raise error

    for parent, directories, files in os.walk(src, followlinks=False, onerror=fail_on_error):
        directories[:] = [name for name in directories if name not in {".venv", "__pycache__"}]
        if Path(parent) == src:
            unexpected = (set(directories) | set(files)) - allowed_root
            unexpected = {name for name in unexpected if not name.endswith(".pyc")}
            if unexpected:
                raise service.GraphError(
                    f"unexpected skill bundle entries: {', '.join(sorted(unexpected))}"
                )
        for name in [*directories, *files]:
            if name.endswith(".pyc"):
                continue
            if name.startswith("."):
                raise service.GraphError(f"hidden skill bundle entry is not allowed: {name}")
            path = Path(parent) / name
            if path.is_symlink():
                raise service.GraphError(f"skill bundle contains a symlink: {path}")
            if not (path.is_file() or path.is_dir()):
                raise service.GraphError(f"skill bundle contains a non-regular entry: {path}")


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise service.GraphError(f"invalid arguments: {message}")


def parser() -> argparse.ArgumentParser:
    p = ArgumentParser(prog="memory-graph", description="Git-backed, committed-main graph memory")
    p.add_argument("--root", type=Path, help="dedicated Git memory-store root")
    p.add_argument("--json", action="store_true", help="JSON output (the default)")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="initialize a new standalone store")
    for action in ("create", "update", "apply", "supersede", "link", "unlink", "repair", "revert"):
        sub = commands.add_parser(action, help=f"perform an idempotent {action}")
        sub.add_argument(
            "--operation", required=True, help="caller-generated UUIDv4; reuse on retries"
        )
        sub.add_argument("--input", required=True, help="JSON request file, or - for stdin")
    show = commands.add_parser("show")
    show.add_argument("id")
    show.add_argument("--scope")
    show.add_argument("--include-unscoped", action="store_true")
    show.add_argument("--include-inactive", action="store_true")
    show.add_argument("--commit", help="explicit full commit ID from selected main history")
    find = commands.add_parser("find")
    find.add_argument("query", nargs="?", default="")
    find.add_argument("--tag")
    find.add_argument("--kind")
    find.add_argument("--scope")
    find.add_argument("--include-unscoped", action="store_true")
    find.add_argument("--limit", type=int, default=20)
    neighbors = commands.add_parser("neighbors")
    neighbors.add_argument("id")
    neighbors.add_argument("--direction", choices=("in", "out", "both"), default="both")
    neighbors.add_argument("--label")
    neighbors.add_argument("--scope")
    neighbors.add_argument("--include-unscoped", action="store_true")
    neighbors.add_argument("--depth", type=int, default=2)
    neighbors.add_argument("--limit", type=int, default=50)
    review = commands.add_parser("review")
    review.add_argument("--limit", type=int, default=20)
    review.add_argument("--scope")
    review.add_argument("--include-unscoped", action="store_true")
    commands.add_parser("check")
    commands.add_parser("diagnose")
    merge = commands.add_parser("check-merge")
    merge.add_argument("--base", required=True, help="full object ID of actual protected main")
    merge.add_argument(
        "--candidate", required=True, help="full object ID of actual integration commit"
    )
    history = commands.add_parser("history")
    history.add_argument("--limit", type=int, default=20)
    operation = commands.add_parser("operation")
    operation_commands = operation.add_subparsers(dest="operation_command", required=True)
    op_status = operation_commands.add_parser("status")
    op_status.add_argument("id")
    attach = commands.add_parser("attach")
    attach.add_argument("--remote", required=True, help="existing configured Git remote name")
    attach.add_argument("--mode", choices=("read", "propose", "write"), default="propose")
    commands.add_parser("fetch")
    commands.add_parser("status")
    publish = commands.add_parser("publish")
    publish.add_argument("--operation", required=True)
    view = commands.add_parser("export-view")
    view.add_argument("--dest", type=Path, required=True)
    export = commands.add_parser("export-skill")
    export.add_argument("--dest", type=Path, required=True)
    return p


def execute(args: argparse.Namespace) -> dict:
    if args.command == "export-skill":
        src = Path(__file__).resolve().parents[2]
        requested = args.dest.expanduser().absolute()
        if requested.exists() or requested.is_symlink():
            raise service.GraphError("destination already exists")
        dest = requested.resolve()
        if dest.is_relative_to(src):
            raise service.GraphError("destination cannot be inside the skill")
        _validate_skill_bundle(src)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc"))
        return {"destination": str(dest)}
    if args.root is None:
        raise service.GraphError("--root is required for store operations")
    root = args.root.expanduser().absolute()
    if args.command == "init":
        commit = git_store.init_store(root)
        snap, _ = service.load(root)
        return {"state": "initialized", "commit": commit, "store_id": snap.store_id}
    if args.command in (
        "create",
        "update",
        "apply",
        "supersede",
        "link",
        "unlink",
        "repair",
        "revert",
    ):
        return service.mutate(root, args.operation, args.command, _input(args.input))
    if args.command == "show":
        return service.show(
            root, args.id, args.scope, args.include_unscoped, args.include_inactive, args.commit
        )
    if args.command == "find":
        return service.find(
            root, args.query, args.scope, args.include_unscoped, args.limit, args.tag, args.kind
        )
    if args.command == "neighbors":
        return service.neighbors(
            root,
            args.id,
            args.direction,
            args.label,
            args.scope,
            args.include_unscoped,
            args.depth,
            args.limit,
        )
    if args.command == "review":
        return service.review(root, args.limit, args.scope, args.include_unscoped)
    if args.command in ("check", "diagnose"):
        snap, errors = service.load(root, diagnostic=True)
        return {"snapshot": service.provenance(snap), "valid": not errors, "errors": errors}
    if args.command == "history":
        return service.history(root, args.limit)
    if args.command == "check-merge":
        return service.check_merge(root, args.base, args.candidate)
    if args.command == "operation":
        return service.operation_status(root, args.id)
    if args.command == "attach":
        return remotes.attach(root, args.remote, args.mode)
    if args.command == "fetch":
        return remotes.fetch(root)
    if args.command == "status":
        return remotes.status(root)
    if args.command == "publish":
        service.valid_uuid(args.operation, "operation id")
        return remotes.publish(root, args.operation)
    if args.command == "export-view":
        return service.export_view(root, args.dest)
    raise service.GraphError("unsupported command")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        output = execute(args)
        sys.stdout.buffer.write(service.canonical({"output_schema": 1, **output}) + b"\n")
        return 1 if args.command in ("check", "diagnose") and not output["valid"] else 0
    except (
        ValueError,
        KeyError,
        OSError,
        RecursionError,
        sqlite3.Error,
        cache.CacheError,
        git_store.GitStoreError,
        BrokenProcessPool,
        subprocess.SubprocessError,
    ) as exc:
        error = {"output_schema": 1, "error": {"type": type(exc).__name__, "message": str(exc)}}
        sys.stderr.buffer.write(json.dumps(error, ensure_ascii=True).encode("ascii") + b"\n")
        return 1
