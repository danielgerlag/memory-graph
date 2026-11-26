"""Run the runtime bundled with this skill."""

import sys

try:
    from memory_graph.cli import main
except ModuleNotFoundError as exc:
    if exc.name in {"filelock", "jsonschema", "ruamel"}:
        print(
            "Missing runtime dependencies. Run `uv sync --locked --project <skill-directory>` first.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    raise

if __name__ == "__main__":
    raise SystemExit(main())
