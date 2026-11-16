# Evaluation runner and evidence reference

This is evaluation infrastructure. The native runner is an opt-in evidence
capture spike, **not a verified native-host support adapter**. The offline
report and evidence importer validate the declarative corpus before use. The
generated `scenarios.json` contains synthetic **development/regression** cases:
24 behavior cases and 20 positive plus 20 negative activation prompts, with
three trials each. These are not held-out generalization evidence. The
fixtures, scopes, clocks, allowed mutations, and expected outcomes are data,
not product logic. See [the evaluation overview](../README.md) for the source
layout and snapshot regeneration.

## Run

From the repository root:

```sh
mkdir -p eval-runs
uv run --locked python evals/run.py
uv run --locked python evals/run.py --evidence ./eval-runs/import/manifest.json --lane all --output ./eval-runs/offline-report.json
uv run --locked python evals/run.py --evidence ./eval-runs/import/manifest.json --lane behavior
uv run --locked python evals/run.py --evidence ./eval-runs/import/manifest.json --lane native
uv run --locked python -m unittest discover -s tests -p test_eval_grader.py -v
```

Without a manifest the report says **NOT RUN**, usage/cost are unknown and
release is blocked. No model or CLI is launched, no credentials are used and
no network access is needed. The process exits 0 only if the selected
**imported-evidence** gates pass, 1 for NOT RUN/BLOCKED/FAIL, and 2 for an
invalid manifest or dataset. Even a 0 is *not* a release or host-support
claim: those gates remain BLOCKED. `--output` writes the JSON report; otherwise
it is printed. The reports for individual lanes cannot substitute for the
other lane.

The root `.gitignore` excludes `/eval-runs/`, but ignoring files is **not an
access-control boundary**. Restrict filesystem and artifact permissions, and
never rely on Git's ignore rule to protect sensitive logs or credentials.

## Opt-in controlled Inspect smoke

The optional `eval` dependency group adds pinned Inspect AI. The controlled
runner executes **one** `scope-search` trial with an explicitly loaded copy of
the skill and the bundled CLI, not a native host or activation test. It runs
only when explicitly invoked; offline reports and ordinary tests never launch
it. This is a development smoke scenario, not the 24-case release corpus.

First install the optional group with `uv sync --locked --group eval`. Build
the supplied `evals/Dockerfile` or prepare another trusted local image. It
needs Python 3.12+, Git 2.45+ with `--no-lazy-fetch` and `show-ref --exists`,
`filelock`, `jsonschema`, `ruamel.yaml`, `/bin/sh`, `tail`, `tar`,
`/usr/bin/timeout`, `tee`, `base64`, and a `/workspace` directory. Git 2.54
was used for the broader local suite; Git 2.47.3 was exercised in the
synthetic sandbox run. Other Git 2.45–2.53 versions remain unverified.
Earlier versions are unsupported, including the unverified fsmonitor
configuration behavior in Git 2.35.2 and earlier.
The installed versions should match the skill's locked dependencies; record
the image digest. The runner does not build, pull or install anything inside
the container. It uses the digest-pinned image with a read-only root,
64 MiB writable `/workspace` tmpfs, no network, UID 1000, dropped
capabilities, and explicit CPU, memory, process, tool, token and time limits.
There are no host bind mounts or developer-home copies. The image itself
must be trusted; Docker/container isolation is not a substitute for host
security updates or provider-side spending controls.

With an explicitly configured provider API key and a locally available image,
an example invocation (replace all angle-bracketed values) is:

```sh
uv run --locked --group eval python evals/run_inspect.py --run \
  --model 'openai/<model-id>' --credential-env OPENAI_API_KEY \
  --image '<image-name>@sha256:<64-hex-digest>' \
  --log-dir ./eval-runs/controlled-001 \
  --budget-usd 1 --input-price 1 --output-price 2 \
  --time-limit 120 --wall-timeout 180 --tool-timeout 15 --turn-limit 8 --tool-limit 8 \
  --token-limit 6000 --max-tokens 500 --cpus 1 --memory-mb 512
```

For Anthropic use `--model anthropic/<model-id> --credential-env
ANTHROPIC_API_KEY`. For an Azure OpenAI API deployment in Foundry, use
`--model openai/azure/<deployment>` with `--azure-endpoint` set to the HTTPS
resource endpoint. Set `--azure-auth default` to use DefaultAzureCredential,
including an existing `az login` or `azd auth login`. For an API key, use
`--azure-auth api-key --credential-env AZUREAI_OPENAI_API_KEY` (or
`AZURE_OPENAI_API_KEY`). Optionally set `--azure-api-version`; if omitted,
Inspect uses its default. Set `--model-family` to the underlying model name
when the deployment uses a different alias, so Inspect can choose the proper
request shape. Standard public Azure resource hostnames are
accepted; custom domains and other Foundry API protocols are not supported.
For key authentication the key must already be in the named environment
variable; neither keys nor Azure credential files are sent to the sandbox.
The runner clears other inherited environment variables, creates an isolated
HOME, and passes the selected key or Azure identity settings, PATH, and explicit
Foundry endpoint/version to the host model client. DefaultAzureCredential runs
on the host, with the original Azure CLI and Azure Developer CLI configuration
directories available for token refresh. Host identity environment settings can
select another credential earlier in the SDK's default chain; set
`AZURE_TOKEN_CREDENTIALS=dev` to restrict the chain to developer credentials.
In default mode a configured `credential_env` is ignored and API keys are not
passed through. The runner obtains a token before starting the trial and fails
with a sign-in hint if default authentication is unavailable.
The sanitized environment does not preserve proxy settings; VS Code and
PowerShell credential sources that require the original HOME are not
available. It disables Foundry HTTP redirects and proxy environment settings.
If a standard per-user Docker Compose plugin exists,
the clean HOME links only that executable; Docker configuration and
credentials are not copied. For a nondefault Docker socket, explicitly add
`--docker-host unix:///path/to/socket`.
The log directory must be a **new** path inside the current working directory;
when omitted, a unique directory under `eval-runs/` is used.
Logs may contain prompts, transcripts, and artifacts; restrict access and do
not commit them.

Token, time, tool and resource limits, the model, image, and selected
authentication are mandatory; prices and the dollar budget are optional. The
image may be a digest-qualified reference or a local `sha256:<image-id>`.
The outer wall timeout uses Unix SIGALRM; the runner refuses to
start on platforms without it. Inspect cleanup is requested, but check for
orphaned containers if a run is forcibly interrupted. Supply both input and
output prices in USD per million tokens to estimate costs. Without them, cost
is reported as unknown and no dollar budget is enforced; the runner prints a
warning and records this in its report. A requested dollar budget requires
both prices and fails validation otherwise. With prices, the optional dollar
budget is an Inspect soft limit checked between
requests, **not a hard billing ceiling**. A request can overshoot. Configure a
separate provider-side spending cap. Inspect logs may contain catalog-based
cost estimates; the runner does not treat them as your billing rate when prices
are omitted.
Missing token usage, score, snapshots,
successful retrieval evidence, errors, or limit exhaustion cannot pass; missing
cost evidence also fails when prices are configured. The report and Inspect
logs are written under the log directory.
An exit code of 0 means only that this smoke trial's deterministic checks
passed; release remains BLOCKED. This smoke does not test native discovery
or activation. The ordinary `evals/run.py` report does not automatically
import or treat this smoke result as release evidence.

The setup copies the exact skill bundle and hash-pinned synthetic store through
Inspect's `sandbox().write_file()` and invokes every sandbox command through
`sandbox().exec()`. The controlled tool fixes the store root, does not expose
a shell, and records calls. The scorer outside the sandbox checks the actual
initial/final file hashes and Git refs, tool outputs, scope isolation and the
provenanced final answer. Only `.memory-graph/parse-cache.sqlite3` and its
`-journal`, `-wal`, and `-shm` sidecars are excluded from file hashes;
symlinks and other local files remain checked. The agent can attempt CLI
mutations, which fail the read-only rubric even if the command fails.
Expectations are not mounted into
the agent sandbox. This does not prove security against a hostile image or
compromised Docker daemon; nor does it test native skill discovery.
The supplied image was exercised with a synthetic Foundry HTTP response and
real network-disabled Docker sandbox. Real provider compatibility and model
outcomes remain unverified until the operator explicitly runs them.

## Opt-in controlled corpus trials

The separate `run_behavior.py` adapter selects cases from the pinned 24-case
behavior corpus. It uses the actual bundled skill and CLI with the same
isolated, prebuilt Docker image, opt-in credentials, and limits described
above. List the complete implementation inventory without a model call:

```sh
uv run --locked python evals/run_behavior.py --inventory
```

`SUPPORTED` in that inventory means the adapter and deterministic scorer are
implemented; every case still says **NOT RUN** until actually executed.
Implementations cover 21 cases, including standalone creates, reuse, lost-ack
reconciliation, atomic correction, pinned-time reads, malformed-store
diagnosis, shared read, stale recall, and a tool-bound single-store isolation
scenario. `--inventory` lists every ID.
The inventory is implementation coverage, **not evidence that any model
trial passed**. Provider authorization and arbitrary-host information-flow
isolation are not claimed.

For the shorter Foundry setup, see [Run a Foundry trial](../README.md#run-a-foundry-trial).
For one explicitly approved trial with flags (replace the placeholders):

```sh
uv run --locked --group eval python evals/run_behavior.py --run \
  --case-id scope-search --trial 1 \
  --model 'openai/<model-id>' --credential-env OPENAI_API_KEY \
  --image '<image-name>@sha256:<64-hex-digest>' \
  --log-dir ./eval-runs/behavior-scope-1 \
  --budget-usd 1 --input-price 1 --output-price 2 \
  --time-limit 120 --wall-timeout 180 --tool-timeout 15 --turn-limit 8 --tool-limit 8 \
  --token-limit 6000 --max-tokens 500 --cpus 1 --memory-mb 512
```

`--config` loads the combined YAML configuration; separate `--agent` and
`--harness` profiles remain available. Command-line options take precedence.
An unsupported case is rejected **before** creating an output directory or
calling a model. Each invocation is one fresh case/trial, with no automatic
behavioral retries. Set `--trial` to 1, 2, or 3 and use a separate new log
directory for every attempt. The report marks the selected case INCOMPLETE,
the other two trials and all other cases NOT RUN, and keeps release,
comparison and native discovery blocked/not run. It is not a
24-case aggregate or an evidence-import manifest.

The declarative corpus and real Markdown fixtures are pinned by the source
lock; changes require review and repinning. Synthetic aliases for pre-existing
nodes and operation IDs map to stable UUIDs. Capture modules independently record working
files, all Git refs, committed-tree contents/hashes, commit ancestry and raw
CLI action inputs/outputs through the sandbox API. Separate graders check
the corpus semantics, allowed changes, operation journals, request hashes,
receipts, commit trailers, provenance, scope, retrieval and source references.
Read-only cases reject mutation attempts. Write cases permit only their
designated operation and independently verify its single committed effect;
reuse permits no update or an update that changes no factual metadata. The
lost-ack case performs a real seed operation, withholds its acknowledgment
from the agent, and checks reconciliation without a second effect or renewed
expiration. For synthetic evidence references, the prompt explicitly states
that these are user reports rather than direct artifact inspections.

The bounded-traversal fixture can report truncation at its requested two-hop
boundary; all in-bound results are independently checked and that observation
is reported separately. Other truncation fails. The injection case must
actually expose the hostile node body. Its forbidden-action check is limited
to the provided memory CLI tool surface, not an unrestricted-shell security
test. The answer format uses exact node labels, source refs and documented
notice values; strict formatting and explicit scoped queries can produce
conservative failures.

The supported shared-read and offline-stale cases use a hook-free synthetic
local bare remote seeded by cloning the fixture, not by publication. The stale
case moves the remote away before retrieval. The grader verifies the pinned
endpoint digest as well as local and remote refs and data. Legacy attachments
without an endpoint digest must be explicitly reattached. The container has no
network and the model-facing tool is restricted to the `file` protocol.

The following cases are **UNSUPPORTED/BLOCKED**, not passed: `proposal-not-main`,
`accepted-after-fetch`, and `concurrent-changes`. The hardened runtime refuses
local publication because server policy cannot be safely enforced. The present
sandbox has no verified isolated SSH/HTTPS transport with a separately
authorized acceptance actor or the required review/current-base server policy.
The old raw local pushes, ref updates and hook fixture are not an acceptable
substitute. A loopback SSH authentication probe on the developer host does not
establish that such a transport works in the constrained evaluation image or
enforces those policies. No provider authorization or branch protection is
claimed. The isolation case checks one tool-bound store and an absent target
workspace, not unrestricted host or cross-provider information flow.

Restoring these cases requires a trusted, locally prebuilt digest-pinned image
with Git 2.54, Python 3.12 and the skill dependencies, plus an SSH client,
`sshd`, and `ssh-keygen`. The current single-service adapter also has no
separately protected harness-owned server and hooks. A future adapter must
isolate that server from the agent, bind SSH only to loopback with no external
network or developer credentials, pin the ephemeral host key, and verify that
the trusted server hooks reject unauthorized and stale updates. The available
devcontainer, manylinux, and Node image candidates were probed in disposable
networkless containers; none exposed `sshd` on PATH or at standard locations,
and none met all runtime prerequisites. No image was pulled. The cases remain
blocked until both the image and the server/hook isolation are implemented and
verified end to end.

The corpus is version `synthetic-2`; `synthetic-1` evidence is not silently
accepted against the revised dataset hash.

`fresh-session-recall` uses two distinct Inspect agent contexts, resets the
message history and model output between them, and shares only the synthetic
disk graph. Both contexts' actions, first-context transcript, reset input and
before/intermediate/after snapshots are recorded and scored. Its corpus
fixture **already contains the fact**: it tests read-only reuse and fresh
recall, not creation from an empty store. The two time-boundary cases invoke
the actual CLI module with only its service clock injected; this does not
assert that the shipped CLI has a clock-selection flag. No live corpus
trial has been run; the supplied image was tested with synthetic HTTP only.
Actual provider and model compatibility remain unverified. Missing artifacts, usage, scoring, or limit
evidence never pass. Raw evidence is retained in the Inspect log, separate
from the deterministic grader; the operator must control access to that log.

## Opt-in native Copilot evidence capture

`evals/run_native.py` starts an actual noninteractive Copilot CLI process in a
fresh container for each selected case. It does not run unless both `--run`
and `--allow-network` are supplied. It uses no Inspect dependency. The
checked-in `evals/native-smoke.json` selects one positive and one negative
prompt from the hash-pinned scenario dataset. A custom manifest has this
shape:

```json
{
  "schema_version": 1,
  "dataset_sha256": "<exact SHA-256 of evals/scenarios.json>",
  "trials": [
    {"case_id": "native-positive-02", "trial": 1},
    {"case_id": "native-negative-01", "trial": 1}
  ]
}
```

Only existing native case/trial keys with the empty synthetic fixture and a
single prompt are accepted. Duplicate trials are rejected; the multi-session
critical case is **not supported**. Prompts are read from the dataset rather
than supplied as host commands. Each trial receives its own empty Git store,
the bundled skill in `.github/skills/memory-graph`, a fresh HOME, and a fresh
Copilot process. The runner verifies the copied skill and fixture hashes and
the requested CLI version before starting the model. Only build `1.0.110-3`,
the build whose flags were inspected, is selectable. A separate local,
network-denied skill-list check confirmed discovery on that build, but this
container runner's authentication, activation, and behavior are unverified.
The host image must be
trusted, contain no credentials, and be prebuilt locally, referenced by
immutable digest. It must contain `copilot` and its runtime dependencies,
`/bin/sh`, `/usr/bin/env`, `sleep`, `mkdir`, Git 2.54 (the tested version) and
Python 3, plus an existing `/eval` directory for the writable tmpfs. The runtime
requires Git 2.45 or newer with `--no-lazy-fetch` and `show-ref --exists`;
Git 2.45–2.53 is unverified. Earlier versions are unsupported, including the
unverified fsmonitor configuration behavior in Git 2.35.2 and earlier.
No image is built or pulled. There are no host bind mounts or developer HOME
copies. The container has a read-only root, a 128 MiB writable `/eval` tmpfs,
UID 1000, dropped capabilities, and explicit memory, CPU, PID, output and
wall-time limits. It self-terminates after the run deadline; cleanup is
best-effort and should be checked after interruption.

The following is an invocation template; replace the placeholders and supply
an explicitly approved, dedicated token in the named environment variable:

```sh
uv run --locked python evals/run_native.py --run --allow-network \
  --manifest evals/native-smoke.json --output-dir ./eval-runs/native-001 \
  --image '<local-image-name>@sha256:<64-hex-digest>' \
  --host-version '1.0.110-3' --model '<copilot-model-id>' \
  --credential-env COPILOT_GITHUB_TOKEN \
  --session-ai-credits 30 --run-ai-credits 60 \
  --session-timeout 120 --run-timeout 360 \
  --cpus 1 --memory-mb 512 --max-output-mb 8
```

The output directory must be a **new** path inside the current working
directory. Accepted credential names are `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`,
and `GITHUB_TOKEN`; the selected value is passed explicitly to the Copilot
process, not as a command argument. The host Docker client receives a
separate clean HOME and no inherited credentials or Docker config; use
`--docker-host unix:///path/to/socket` if needed. Noninteractive token
authentication and the image's CLI version have **not** been validated.
Use a dedicated least-privilege test account, never a developer token.

The container **does have network access** for the Copilot service, without
an enforced egress allowlist. The runner disables built-in MCPs and updates,
denies shell, write, and delegation tools, restricts read permissions, and
uses the CLI's `--secret-env-vars`; these are defense in depth, not a proven
credential or information-flow boundary. The Docker daemon, the image, the
Copilot process, and same-UID processes can potentially access the token.
Restrict egress externally and use disposable credentials. Literal token
occurrences in captured streams or usage/session files are withheld, but
encoded or transformed secrets cannot be reliably detected. Treat all outputs
as sensitive, keep them access-controlled, and do not commit them.

The CLI's `--max-ai-credits` is a **post-paid soft cap**, with a minimum of
30 credits per session in the inspected build. The runner requires an
explicit per-session cap and rejects a manifest whose planned session caps
exceed the declared run budget. A final response can overshoot; usage format
and actual account billing are not independently validated. Use a separate
provider/account spending cap. The runner captures raw stdout JSONL, stderr,
exit status, usage JSON, optional native `events.jsonl`, before/after graph
and ref snapshots, workspace file hashes, and SHA-256 artifact metadata in
`report.json`. Missing,
oversized, malformed, timed-out, or credential-containing evidence is
reported rather than counted as success.

**Activation always remains UNVERIFIED/BLOCKED today.** The installed
documentation does not specify a reliable skill-load event schema or prove
that absence of an event means non-activation. No answer text, model
self-report, event name guessed from a transcript, or absence of JSONL events
is used as activation evidence. A future adapter must validate both positive
and negative capture against the pinned host version before implementing a
mapping. Shell/write denial also means this spike cannot establish the
critical remember-then-recall workflow; behavioral and native support release
gates remain BLOCKED. The raw output is not automatically converted to the
separate importer format below. No native Docker or model run has been
performed; the runner and parser have only offline tests.

## Producing an import

A future independent host adapter for the **separate imported-evidence
release report** must execute each selected case and trial
in a fresh, isolated environment, using the supplied fixture, clock, prompts,
scope and store policy. Multi-session trials must use separate contexts and
explicitly hand off only the graph. It must extract observations from the
actual artifacts rather than a model's self-report, record all attempts, and
preserve the raw files. No such adapter or host event mapping is implemented
here. The controlled Inspect smoke above is not that adapter and does not
produce this import format. Raw transcripts and artifacts can be sensitive;
keep them in access-controlled storage and out of version control. Do not
mount personal stores, home directories, credentials, or unrelated data.

The manifest is JSON, schema version `1`. All artifact `path` values are
relative to the manifest directory and must remain inside it, including after
resolving symlinks. `sha256` values are lowercase hashes of the exact file
bytes. The dataset hash is the hash of `evals/scenarios.json` itself; it ties
an imported run to its normalized-observation checks. The report also records
the grader SHA-256. For the default corpus it records the source hash separately,
covering the YAML cases, controlled expectations, and Markdown fixtures.
Every manifest represents **one** condition,
host build, model/configuration and isolated execution profile. Required
top-level fields:

| Field | Format |
| --- | --- |
| `schema_version`, `dataset_version`, `dataset_sha256` | `1`, exact dataset version, exact dataset SHA-256 |
| `condition` | `candidate`, `previous`, or `no-skill` |
| `run_id`, `adapter`, `source_revision` | Nonempty identifiers; record the actual tested revision |
| `skill_sha256`, `cli_sha256` | Hashes of the actual tested artifacts, 64 lowercase hex characters |
| `schema_version_under_test`, `clock_policy`, `isolation_profile` | Nonempty strings; describe what was actually used |
| `sandbox_image` | Immutable image reference ending in `@sha256:` and 64 hex characters |
| `dependency_lock_sha256` | Lock hash, or literal `none` if no lock was used |
| `python_version`, `git_version` | Nonempty actual version strings |
| `host` | `{"name": "...", "version": "..."}` |
| `model` | `{"id": "...", "configuration": {"...": "..."}}`; configuration must be nonempty |
| `limits` | `{"concurrency": 1, "timeout_seconds": 120, "budget": 1.0, "budget_currency": "USD"}` with positive values |
| `results` | Array of result entries described below |

These are field shapes, not evidence that any such run happened. An entry
contains `case_id`, integer `trial` (1 through 3), `status`, and, for
`status: "completed"`, an `observation` descriptor
`{"path":"observation.json","sha256":"..."}` and `artifacts` descriptors:

```json
[
  {"role":"transcript", "path":"transcript.jsonl", "sha256":"<64 lowercase hex>"},
  {"role":"graph_before", "path":"before.json", "sha256":"<64 lowercase hex>"},
  {"role":"graph_after", "path":"after.json", "sha256":"<64 lowercase hex>"},
  {"role":"action_trace", "path":"actions.json", "sha256":"<64 lowercase hex>"}
]
```

Those strings illustrate the schema and are not valid hashes. Each role is
required exactly once. Missing entries, non-completed statuses, bad hashes,
missing outcomes, malformed snapshots and invalid artifacts are visible as
missing/scoring errors; they cannot pass or disappear from denominators.
Unknown cases, out-of-range trial numbers and duplicate case/trial entries
reject the manifest. Do not replace a failed attempt with a retry. The same
case/trial key can occur only once. Separate manifests are needed for
different conditions or model/host configurations.
Grade the no-skill control with `--lane behavior`; activation expectations
apply to a condition that actually installs a skill.

`graph_before` and `graph_after` are canonical snapshot manifests:
`{"files":{"relative/path":"<file SHA-256>"},"refs":{"heads/main":"<commit>"}}`.
Include all store files and relevant refs, not only expected changes.
The grader independently diffs these maps, reports their changed keys prefixed
with `files/` or `refs/`, and compares them with the trace. File keys must be
safe relative paths; file hashes are SHA-256. No-mutation cases require
identical snapshot contents. `action_trace` is the complete, raw action log;
the transcript is the full raw host transcript. For native cases it must be
JSONL, one JSON event per line. These raw files are hash-checked and referenced
in the report; apart from the snapshot manifests and native JSONL line
references, the grader **does not parse or verify their semantics**.

The observation JSON is a normalized, externally produced view of actual
evidence. Each case's `checks` in `cases/` (compiled into `scenarios.json`)
specifies JSON pointer paths and expected values. In addition every observation
must include:

```json
{
  "trace": {
    "complete": true,
    "integrity_violations": [],
    "mutations": [],
    "mutation_records": [],
    "changed_paths": [],
    "actions": []
  }
}
```

This is the shape for a complete trace with **no changes**, not a passing
result by itself. `mutations` is the list of observed semantic mutation kinds
from the case's `allowed_mutations`; `changed_paths` lists all changed
snapshot keys. `mutation_records` has one `{"path":"files/...","kind":"..."}`
per changed key; its path and kind sets must match the two lists. Actions are
complete action-kind strings; forbidden kinds are graded independently.
Fixtures use synthetic aliases for pre-existing node IDs. The adapter must
consistently resolve those aliases in observations; IDs generated by the
system must remain actual stable IDs, not guessed constants. Checks support
exact equality, list containment with partial-object matching, allowed subsets,
unique context IDs, and a cross-node identity link. A `/length` path segment
following a list means its length. Synthetic fixture records, including source
shorthand such as `witness:a`, describe semantic inputs rather than ready-made
product-format files; a future adapter must map them to the actual storage
schema and preserve the source identity when normalizing outcomes. Full observed and expected
values and raw artifact references appear in the report. An adapter must
provide *all* required case-specific fields, including empty arrays when
observed; absence never means empty or success.

For native cases add
`"activation":{"observation_complete":true,"detected":false,"event_lines":[]}`.
For a detected activation, set `detected` true and list the 1-based JSONL
transcript line(s) containing actual host skill-load events. The critical
two-context case requires distinct lines for both contexts, distinct
`sessions.context_ids`, an accepted main commit containing the fact, and
fresh-context recall. Negative cases may not mutate the graph. **The importer
can validate that referenced JSONL events exist, but cannot recognize a
host's undocumented load-event semantics or establish capture completeness.**
The mapping and completeness are external attestations, not proven activation.
Good answers, a model's claim that it loaded a skill, and unverified event
interpretations are not native discovery evidence.

## Gates and limitations

The imported behavior gate requires at least 24 cases, three trials each,
complete scoring, >=90% macro case success, and >=90% in every category.
The imported activation gate requires at least 20 positive and 20 negative
cases, three trials each, complete scoring, and >=90% precision and recall;
undefined metrics block it. Every critical native trial must pass. Any
observed forbidden action, integrity violation, out-of-policy mutation,
inconsistent trace, or unexplained snapshot change fails integrity regardless
of aggregate scores. Reports include individual case/trial checks and category
scores. Missing data blocks coverage. Do not interpret initial sample sizes as
population reliability bounds; uncertainty intervals are NOT RUN.

Normalized observations, snapshots, action classifications, event mappings,
and provenance are supplied by the importer and may be incorrect or
fabricated. Hashes establish file integrity, not origin or truth; semantic
facts and host event meanings need an independently validated extractor.
Therefore imported threshold gates can pass while native support and release
remain BLOCKED. Paired candidate/baseline comparisons, clustered confidence
intervals, performance, provider permissions, model-judged qualities, usage
and cost collection for the **imported release report** are explicitly NOT
RUN. Verified native activation is also NOT RUN: imported activation
attestations can be scored, but do not prove host activation. A separate local,
network-denied Copilot skill-list check confirmed discovery only; it did not
test activation. The separate opt-in smoke can record its own reported usage
and cost. No live run is claimed until actually executed.
