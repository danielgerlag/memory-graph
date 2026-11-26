#!/usr/bin/env python3
"""Install, archive, or refresh the plugin copy of the memory-graph skill bundle.

Uses only the standard library, so it runs before the skill's dependencies exist.

  install --host copilot|claude|codex [--project DIR]   copy into a host skills directory
  archive --output FILE.zip                             portable zip with a memory-graph/ folder
  plugin [--check]                                      refresh or verify plugins/memory-graph
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "memory-graph"
PLUGIN_SKILL = ROOT / "plugins" / "memory-graph" / "skills" / "memory-graph"
ALLOWED_TOP_LEVEL = {"SKILL.md", "pyproject.toml", "uv.lock", "scripts", "references", "assets"}
EXCLUDED_DIRECTORIES = {".venv", "__pycache__"}
HOST_DIRECTORIES = {
    "copilot": (Path(".copilot/skills"), Path(".github/skills")),
    "claude": (Path(".claude/skills"), Path(".claude/skills")),
    "codex": (Path(".agents/skills"), Path(".agents/skills")),
}


class PackageError(Exception):
    pass


def bundle_files(source: Path = SKILL) -> dict[str, bytes]:
    """Return the distributable bundle, mirroring the skill's own export-skill rules."""
    if source.is_symlink() or not (source / "SKILL.md").is_file():
        raise PackageError(f"not a skill directory: {source}")
    files: dict[str, bytes] = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        parts = relative.parts
        if any(part in EXCLUDED_DIRECTORIES for part in parts) or path.suffix == ".pyc":
            continue
        if parts[0] not in ALLOWED_TOP_LEVEL:
            raise PackageError(f"unexpected skill bundle entry: {parts[0]}")
        if path.is_symlink():
            raise PackageError(f"skill bundle contains a symlink: {relative}")
        if path.is_file():
            files[relative.as_posix()] = path.read_bytes()
        elif not path.is_dir():
            raise PackageError(f"skill bundle contains a non-regular entry: {relative}")
    return files


def write_tree(files: dict[str, bytes], destination: Path) -> None:
    for name, data in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _is_memory_graph_skill(path: Path) -> bool:
    skill_file = path / "SKILL.md"
    if path.is_symlink() or not skill_file.is_file():
        return False
    text = skill_file.read_text(encoding="utf-8")
    return re.search(r"^name: memory-graph$", text, re.MULTILINE) is not None


def install(host: str, project: Path | None, *, force: bool, sync: bool) -> Path:
    personal, in_project = HOST_DIRECTORIES[host]
    if project is None:
        destination = Path.home() / personal / "memory-graph"
    else:
        project = project.expanduser().resolve()
        if not project.is_dir():
            raise PackageError(f"project directory does not exist: {project}")
        destination = project / in_project / "memory-graph"
    files = bundle_files()
    if destination.exists() or destination.is_symlink():
        if not force:
            raise PackageError(f"{destination} already exists; use --force to replace it")
        if not _is_memory_graph_skill(destination):
            raise PackageError(f"refusing to replace {destination}: not a memory-graph skill")
        shutil.rmtree(destination)
    write_tree(files, destination)
    if sync:
        uv = shutil.which("uv")
        if uv is None:
            raise PackageError(
                f"installed to {destination}, but uv is not on PATH; install uv, then run: "
                f"uv sync --locked --project {destination}"
            )
        subprocess.run([uv, "sync", "--locked", "--project", str(destination)], check=True)
    return destination


def archive(output: Path) -> Path:
    output = output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise PackageError(f"{output} already exists")
    if not output.parent.is_dir():
        raise PackageError(f"output directory does not exist: {output.parent}")
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as bundle:
        for name, data in sorted(bundle_files().items()):
            entry = zipfile.ZipInfo(f"memory-graph/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, data)
    return output


def plugin_drift() -> list[str]:
    """Names that differ between the source bundle and the committed plugin copy."""
    expected = bundle_files()
    actual = bundle_files(PLUGIN_SKILL) if PLUGIN_SKILL.is_dir() else {}
    names = expected.keys() | actual.keys()
    return sorted(name for name in names if expected.get(name) != actual.get(name))


def refresh_plugin() -> None:
    if PLUGIN_SKILL.exists():
        shutil.rmtree(PLUGIN_SKILL)
    write_tree(bundle_files(), PLUGIN_SKILL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    install_parser = commands.add_parser("install")
    install_parser.add_argument("--host", choices=sorted(HOST_DIRECTORIES), required=True)
    install_parser.add_argument(
        "--project", type=Path, help="install into this project, not your home"
    )
    install_parser.add_argument("--force", action="store_true", help="replace an existing install")
    install_parser.add_argument("--no-sync", action="store_true", help="skip 'uv sync --locked'")
    archive_parser = commands.add_parser("archive")
    archive_parser.add_argument("--output", type=Path, required=True)
    plugin_parser = commands.add_parser("plugin")
    plugin_parser.add_argument("--check", action="store_true", help="fail if the copy is stale")
    args = parser.parse_args(argv)
    try:
        if args.command == "install":
            destination = install(args.host, args.project, force=args.force, sync=not args.no_sync)
            print(f"Installed {destination}")
        elif args.command == "archive":
            print(f"Wrote {archive(args.output)}")
        elif args.check:
            drift = plugin_drift()
            if drift:
                print(
                    "plugins/memory-graph is stale; run: python3 tools/package.py plugin",
                    file=sys.stderr,
                )
                print("\n".join(drift), file=sys.stderr)
                return 1
            print("plugin copy matches memory-graph/")
        else:
            refresh_plugin()
            print(f"Refreshed {PLUGIN_SKILL.relative_to(ROOT)}")
    except (PackageError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
