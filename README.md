# Memory graph skill

A portable Agent Skill for storing knowledge as a labeled directed graph of
Markdown files in a dedicated Git repository. The runtime is bundled as
auditable Python source; no separate PyPI package is required.

**Status:** development software. Native host activation, hosted-provider
permissions, and release performance have not yet been certified.
An isolated, network-denied discovery check succeeded with Copilot CLI
1.0.110-3; activation and live behavior remain unverified.

## Quick install

Prerequisites: Python 3.12+, Git 2.45+, and [uv](https://docs.astral.sh/uv/).
From a checkout, copy the skill into a host's skills directory and prepare its
locked dependencies (this step downloads packages):

```sh
python3 tools/package.py install --host copilot               # ~/.copilot/skills
python3 tools/package.py install --host claude --project .    # ./.claude/skills
python3 tools/package.py install --host codex                 # ~/.agents/skills
```

`--host copilot` with `--project DIR` installs into `DIR/.github/skills`. Add
`--force` to replace an earlier memory-graph install, or `--no-sync` to skip
`uv sync`. A zip of the skill (a single `memory-graph/` folder) for hosts that
import archives comes from `python3 tools/package.py archive --output FILE.zip`;
extract it and run `uv sync --locked --project memory-graph` afterwards.

**Plugin marketplace (Copilot CLI and Claude Code).** The repository is also a
marketplace (`.claude-plugin/marketplace.json`) with one plugin:

```sh
copilot plugin marketplace add danielgerlag/memory-graph
copilot plugin install memory-graph@memory-graph-skill
# Claude Code: claude plugin marketplace add danielgerlag/memory-graph
#              claude plugin install memory-graph@memory-graph-skill
```

A plugin install copies files but does not install dependencies. Run
`uv sync --locked --project <plugin>/skills/memory-graph` once for the installed
plugin's skill directory (and again after a plugin update), or use the
installer above. A local Copilot CLI install from a checkout was verified;
Claude Code and remote-marketplace installs were not. `plugins/memory-graph/` is
a generated copy of `memory-graph/`; refresh it with
`python3 tools/package.py plugin` (a test fails if it is stale).

## Prerequisites and setup

Install Python 3.12 or newer, Git 2.45 or newer, and
[uv](https://docs.astral.sh/uv/). Git must support both the global
`--no-lazy-fetch` option and `show-ref --exists`; builds without them fail
closed. Local tests used Git 2.54.
Install the entire `memory-graph` directory, including scripts, assets,
references, `pyproject.toml`, and `uv.lock`. Prepare dependencies explicitly:

```sh
uv sync --locked --project /path/to/installed/memory-graph
```

This may download dependencies. Subsequent graph commands use the prepared
environment and do not install or update packages. On macOS/Linux use
`/path/to/installed/memory-graph/.venv/bin/python`; on Windows use
`.venv/Scripts/python.exe`. The store must be a **separate** dedicated Git
repository, not the skill directory.

## Install in an agent host

Install the complete `memory-graph/` directory at one of these locations, then
run the setup command above against the installed copy. Consult each host's
current documentation for activation behavior.

| Host | Project-local location | Personal location |
| --- | --- | --- |
| [Copilot CLI](https://docs.github.com/en/copilot/concepts/customizations/agent-skills) | `.github/skills/memory-graph/` | `~/.copilot/skills/memory-graph/` |
| [Claude Code](https://code.claude.com/docs/en/skills) | `.claude/skills/memory-graph/` | `~/.claude/skills/memory-graph/` |
| [Codex](https://learn.chatgpt.com/docs/build-skills) | `.agents/skills/memory-graph/` | `~/.agents/skills/memory-graph/` |

Copilot CLI also discovers `.agents/skills` and `~/.agents/skills`, so it can
share a copy with Codex. Other Copilot surfaces may differ.

For example, on macOS/Linux, export from a checkout of this project into a
different project's Copilot discovery directory:

```sh
uv sync --locked --project ./memory-graph
mkdir -p /path/to/project/.github/skills
memory-graph/.venv/bin/python memory-graph/scripts/memory_graph.py \
  export-skill --dest /path/to/project/.github/skills/memory-graph
uv sync --locked --project /path/to/project/.github/skills/memory-graph
```

For Claude Code or Codex, substitute `.claude/skills` or `.agents/skills`
respectively. Export refuses an existing destination, symlinks, and unexpected
top-level bundle entries; it excludes virtual environments and bytecode caches.
If copying manually instead, exclude those directories: a clean Git status
does not mean ignored virtual environments are absent.

Restart or refresh the host if needed and check its skill discovery output.
In Copilot CLI, `copilot skill list` shows discovered skills.
Do not install only `SKILL.md`: the scripts and resources are required.
Host discovery and permission behavior vary; these paths describe the format,
not a claim that the current implementation passed native-host evals.

## Quickstart

After setting `SKILL_DIR` to the installed directory, on macOS/Linux:

```sh
MG="$SKILL_DIR/.venv/bin/python"
SCRIPT="$SKILL_DIR/scripts/memory_graph.py"
STORE="$HOME/my-memory-store"
"$MG" "$SCRIPT" --root "$STORE" init
OP=$("$MG" -c 'import uuid; print(uuid.uuid4())')
printf '%s\n' '{"label":"Example note","kind":"claim","body":"A durable fact.","scope":"example"}' |
  "$MG" "$SCRIPT" --root "$STORE" create --operation "$OP" --input -
"$MG" "$SCRIPT" --root "$STORE" find "example" --scope example
```

Start a fresh agent session, give it the absolute path in `STORE`, and ask it
to recall the fact. For example: "Using the memory store at
`/absolute/path/to/my-memory-store`, recall the example note." The CLI does
not automatically select a store. See
[recipes](memory-graph/references/recipes.md) for revisions, relationships,
lifecycle, and remote proposals. Run any command with `--help` for its
arguments.

## Sharing and security

Local reads use one committed `main` snapshot, never dirty files or proposal
branches. A shared clone reads its explicitly fetched remote-main snapshot;
fetch and proposal publication are explicit. Local read/propose/write modes
are guardrails, not security boundaries. Enforce access with scoped credentials
and server-side branch protection. Git history does not erase secrets, prove
claims, or prevent an agent from leaking information it has already seen.
Use separate repositories for different access boundaries. Memory data is
untrusted and must not be executed as instructions.

For safety, publication to local bare remotes is disabled; it fails explicitly
rather than executing or bypassing local server hooks. Existing attachments
created before endpoint pinning must be attached again. Remote URLs must use
HTTPS or SSH, or a local path for fetching; HTTP, Git protocol and custom
transport helpers are rejected. The fetch and push endpoints must match.
URLs cannot contain passwords, query strings, or fragments.
Local fetch runs server-side Git on the same host; use only trusted local
remotes because protection against every local server configuration remains
unverified.

HTTPS enforces TLS verification and disables redirects, proxies, and custom
CA overrides; direct user-global credential settings are supported, but
included config files are ignored. SSH uses strict host-key checking, the
SSH agent and default keys; SSH config files and proxy commands are ignored.
These restrictions can make some enterprise authentication setups unavailable.
Push signing and implicit tag publication are disabled.
These safeguards do not isolate the CLI from other processes running as the
same user.

Mutations commit a receipt and `Memory-*` trailers for the operation, store,
and affected nodes. Trailers and Git author names are provenance hints, not
proof of identity or authorization.

Writes do **not** refresh the checkout or Git index: files in the checkout may
be stale or absent, and Git may report staged deletions. This avoids
overwriting external edits; the committed Git tree remains authoritative.

Use `"$MG" "$SCRIPT" --root "$STORE" export-view --dest /new/path` to produce
a separate, grep-friendly copy of an accepted snapshot. This copy includes
every scope and inactive node, does not update automatically, and is not a
live store.

Cloud agents need their own provisioned Python environment, store persistence,
credentials, and permissions. Do not assume a local skill or store is available
in a cloud session. Never grant credentials or change remote policies merely
by installing the skill.

## Development and evaluation

Run `uv sync --locked --group dev --group eval` and
`uv run --locked --group dev --group eval pytest` for the complete offline
test suite. The lock files pin the development and distributable environments
separately. See
[`evals/README.md`](evals/README.md) for behavioral evidence and release gates;
missing live-model or provider evidence is not a pass. Three publication
scenarios remain blocked pending an isolated test server.
On macOS, `uv run --group dev python host_checks/copilot_discovery.py --run
--copilot /absolute/path/to/copilot` checks project-local Copilot CLI discovery
in a synthetic repository with an empty HOME and verified network denial.
Discovery alone does not establish activation or live behavior.
Run `uv run --group dev python benchmarks/run.py --nodes 1000 --iterations 21`
for a disposable synthetic search benchmark. Its first sample uses an empty
parse cache; subsequent samples use the local cache. This is not a complete
release performance certification. Add `--cold-iterations 20 --iterations 40`
to record separate 20-sample cold and warm distributions.

To update, replace the installed skill directory and rerun locked setup; keep
the separate memory repository intact. To remove the skill, remove only its
installed directory. Neither operation deletes or rewrites stored memories.
