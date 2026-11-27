# Graph-memory evaluations

The evals check the skill's behavior, native activation, and imported evidence
separately. The cases are synthetic **development and regression fixtures**,
not held-out evidence of generalization.

## Start here

1. Use the [case guide](docs/cases.md) to understand each scenario, its starting
   state, and what counts as success. The [grader guide](graders/README.md)
   explains which evidence is actually checked and the limits of each lane.
2. Read [scope-search.yaml](cases/behavior/scope-search.yaml): its prompt,
   expected results, allowed changes, and controlled grader are in one place.
3. Open its [Markdown graph](fixtures/graphs/scope-search/). The harness copies
   these real node files into a fresh Git store and commits the initial snapshot.
4. Run the offline checks from the repository root:

   ```sh
   uv sync --locked --group dev --group eval
   uv run --locked python evals/build_dataset.py
   uv run --locked python evals/run_behavior.py --inventory
   uv run --locked --group dev --group eval pytest -q tests/test_eval_corpus.py tests/test_behavior_eval.py tests/test_behavior_write_eval.py tests/test_eval_grader.py tests/test_native_eval.py tests/test_inspect_task.py tests/test_eval_profiles.py
   ```

Inventory says whether a case has an adapter, **not** whether a model passed.
These commands do not call a model or require Docker.

## Run a Foundry trial

The controlled runner supports Microsoft Foundry deployments exposed through
the Azure OpenAI API, using DefaultAzureCredential or an API key. Other Foundry
protocols are not yet supported.

One-time setup, from the repository root:

1. Run `uv sync --locked --group eval`.
2. Build the sandbox: `docker build -f evals/Dockerfile -t memory-graph-eval:local .`.
   Get its immutable image ID with
   `docker image inspect memory-graph-eval:local --format '{{.Id}}'`.
3. Copy `evals/configs/foundry.example.yaml` to
   `evals/configs/foundry.local.yaml`. Edit the deployment, HTTPS resource
   endpoint, and image ID. Review the token and time limits. If the deployment
   name differs from the underlying model name, set `model_family` too. This
   local file is gitignored.
4. Sign in with `az login` or `azd auth login`, and make sure that identity
   can invoke the deployment. The example uses `azure_auth: default`.
   To use an API key instead, set `azure_auth: api-key` and `credential_env` to
   `AZUREAI_OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY`; set that variable
   securely in your environment, never in the YAML. Use only a trusted config
   and image: the configured endpoint receives the credential.

Then run the selected case:

```sh
uv run --locked --group eval python evals/run_behavior.py --run --config evals/configs/foundry.local.yaml
```

For an existing API-key config, add `--azure-auth default` to use your Azure
login without changing the file. In default mode any configured
`credential_env` is ignored, and API keys are not forwarded.

The config selects a case and trial; edit it or override them with `--case-id`
and `--trial`. A unique directory under `eval-runs/` is created automatically,
and its path appears in the report. Each invocation runs one trial. Prices are
optional; without them, cost is unknown and **no dollar budget is enforced**.
Token and time limits still apply. A configured `budget_usd` requires both
USD-per-million prices and is a soft limit that can overshoot. Azure may bill
either way; configure an Azure-side spending cap.

Selected controlled trials have run against a Foundry model. They do not
establish release or native-host support. See the
[runner reference](docs/reference.md#opt-in-controlled-corpus-trials) for
requirements and limitations.

## Compare saved runs

For a point-in-time snapshot of the latest saved result per case and trial
for the current skill:

```sh
uv run --locked python evals/snapshot.py
```

The command prints the path of a new, owner-only CSV under `eval-runs/`; it
never overwrites an earlier snapshot or runs a model. Each corpus case has
one row per planned trial, including `NOT RUN` and `BLOCKED` cases. The
`description` column contains the case prompt. Native host support is still
unverified; a passing behavior trial does not establish release readiness.
Native rows include expected and observed activation separately.

Only reports matching the current skill, dataset, and corpus hashes are
eligible. Fresh holdout variants and reports with missing identity metadata
are not mixed with the pinned corpus. The latest report is selected by file
modification time, which is not necessarily the execution time. Imported
evidence is shown separately from controlled and native trials.

The `status` column includes a matching saved offline regrade when available;
`recorded_status` preserves the original trial result and `result_basis`
identifies regrading. Existing regrade files identify their source by path and
version, not a content hash; `regrade_provenance` makes that limitation
explicit. `evaluator_match` shows whether saved evidence used the current
evaluation adapter. Blank costs mean unknown, not zero.

For a table of **all** saved historical results instead:

After running evaluations, create a spreadsheet-friendly CSV:

```sh
uv run --locked python evals/results.py
```

The command prints the new file's path under `eval-runs/`. Rows include saved
case/trial results and explicitly mark missing or unrun trials, with status,
model, usage, failure details, and version hashes. Run it again after new
trials to create a fresh comparison; existing files are never overwritten.

The exporter finds `report.json` and `*-report.json` under `eval-runs/`. Add
`--report PATH` for a report with another name, `--root DIRECTORY` for another
report directory, or `--output FILE` for a new, explicit output path. Symbolic
links are not followed. It reads saved reports only; it does not run models or
collect pytest results.

Blank usage or cost means unknown, not zero; a reported cost is an estimate,
not an Azure bill. `report_status` and
`release_status` are separate from an individual trial's `status`: an imported
trial passing does not certify the release. Full native host support remains
unverified. `report_modified_utc` is the report file's modification time, not
necessarily the run's start time. Compare the dataset, corpus, and grader
hashes before treating rows as equivalent; different fixtures or graders can
make results incomparable. `adapter_sha256` fingerprints the complete
recorded adapter-hash map. Some older or failed reports lack these fields.

The CSV is created with owner-only permissions where supported. It can contain
failure details and paths; the gitignored output directory is not an
access-control boundary. Text that could be interpreted as a spreadsheet
formula is prefixed with an apostrophe.

## Optimize skill instructions with GEPA

GEPA proposes instruction changes using feedback from the controlled behavior
runner. This is opt-in and makes billable model calls:

```sh
uv sync --locked --group eval --group optimize
uv run --locked --group eval --group optimize python evals/run_gepa.py --run \
  --config evals/configs/foundry.local.yaml
```

By default, the search uses at most 18 controlled trials and 3 reflection
requests. It then runs up to 6 additional trials for the fresh holdout
comparison. Each trial can make multiple model requests and uses the
configured trial token and time limits. Reflection requests are limited to
4,096 output tokens each. Prices are optional, so these limits are **not a
dollar cap**. Use `--max-trials`, `--max-reflections`,
`--reflection-max-tokens`, and `--max-seconds` to adjust the search limits.

The optimizer evaluates isolated candidate copies; it never edits the live
skill or the graders. Runner errors stop the search rather than counting as
behavior failures. The generated `best.md`, trial reports, and `summary.json`
are saved under a new `eval-runs/gepa-*/` directory. Review a candidate and
its safety requirements before applying it.

Training, selection, and holdout cases are disjoint. The holdout is generated
after selection with fresh randomized vocabulary and is evaluated against both
the original and selected skill. Its three cases are variants of existing
development scenarios, not proof of broad generalization. Each runs once per
skill, so results are noisy. Neither this search nor its holdout establishes
native activation or unblocks the two publication scenarios.

## Hill-climb the skill across models

`run_gepa_hillclimb.py` is the multi-model optimizer (the older `run_gepa.py`
above is the small single-model search). Each candidate `SKILL.md` is scored by
controlled trials on several models; a case's score is the fraction of models that
pass it. Reflection uses a separate, stronger model and receives, for each failing
trial, the user request, the commands run with their errors, the final answer, and a
plain-language reading of the grader's failure message.

```sh
uv sync --locked --group eval --group optimize
uv run --frozen --group eval --group optimize python evals/run_gepa_hillclimb.py optimize \
  --config evals/configs/foundry.gepa.local.yaml --name gepa-1 \
  --models gpt-5.4-mini gpt-4.1 --reflection-model gpt-6.1-terra \
  --seed eval-runs/cand/seed.md --max-metric-calls 220 --max-reflections 20
uv run --frozen --group eval --group optimize python evals/run_gepa_hillclimb.py holdout \
  --config evals/configs/foundry.gepa.local.yaml --name holdout-1 \
  --models gpt-5.4-mini gpt-4.1 --candidates eval-runs/cand/seed.md eval-runs/gepa-1/best.md
```

Training, validation, and holdout cases are disjoint (see `TRAIN_CASES`,
`VALIDATION_CASES`, and `HOLDOUT_CASES`). Holdouts are fresh randomized-vocabulary
variants of withheld cases (`gepa_holdout.yaml`). The run writes every trial to
`evaluations.jsonl`, GEPA's reasoning to `gepa.log`, and each candidate to
`candidates/`. Infrastructure errors are retried and budget exhaustion counts as a
failed trial. The seed must live inside the repository.

Requirements: a real Docker Engine 24.0.6 or newer with Compose v2 (Inspect AI rejects
Podman), and a controlled config whose `docker_host` points at it; pass `--docker-bin`
if another `docker` shadows it on `PATH`. A relaxed-budget config (for example 16 turns,
40,000 tokens, 4,000-token generations, 240 s) avoids scoring budget exhaustion as a
skill failure. Keep the machine awake: sleeping pauses a long run. Optimization on the
development corpus overfits it by construction, so judge a candidate on the holdout
variants and on cases that were never optimized.

## Layout

```text
evals/
  cases/behavior/        prompts, fixtures, and expectations
  cases/native/          positive and negative activation prompts
  fixtures/graphs/       actual store manifests and Markdown nodes
  agents/                example model/credential-variable profiles
  harnesses/             example isolation/runtime profiles
  configs/               combined runner configuration templates
  graders/               independent artifact and imported-evidence graders
  results.py              export saved trial reports as CSV
  run_gepa_hillclimb.py   multi-model GEPA optimizer and holdout comparison
  corpus.yaml            dataset metadata and trial count
  corpus.lock.json       pinned source and generated-dataset hashes
  scenarios.json         generated compatibility snapshot, not authoring source
  docs/reference.md      runner commands, evidence format, and release gates
```

Each behavior case starts with its prompt and a readable preview of the
starting graph. Node names in that preview are aliases; arrows are written as
`edges` with a label and target alias. Omitted node scopes use the case scope.
The actual UUID-named Markdown nodes remain in `fixtures/graphs/<case-id>/`.
The corpus loader checks the preview against those files, so they cannot
silently disagree. The deliberately malformed fixture is an exception.

The `controlled` section selects an allowlisted grader and summarizes expected
facts, sources, and targets; it is not the grader's complete rubric. Controlled
graders also inspect captured files, Git refs, and tool calls independently.
The main read grader accepts alternate search terms and additional in-scope
results, and accepts independently verified `show` calls as an alternative to
`neighbors` for traversal. Traversal answers can name the starting node and
state a verified relationship instead of listing only its destination.
Blank strings and empty nested arrays in the source list are treated as no
source and reported as a schema warning; nonempty sources must match recorded
evidence. An `unknown` notice on a correctly retrieved, unsourced claim is
reported as ambiguous rather than treated as a wrong fact; missing or
incorrect facts still fail. The fetch case grades a single JSON object even
when surrounded by prose; it reports that schema deviation and does not
grade the surrounding prose.
Empty-result cases and some specialized graders still require the specified
query. The later `checks` section applies only to
**imported observations**, not controlled trials. Mutation and forbidden-action
lists describe the case contract.

Expectation files and grader code stay outside the agent's sandbox. YAML does
not execute commands, and its named graders are not arbitrary plugins.

## Execution lanes

| Lane | What it establishes |
| --- | --- |
| Controlled Inspect | Opt-in model trial with an explicitly loaded skill and a restricted tool; not native activation. |
| Native Copilot | Opt-in activation-only trials for CLI 1.0.110-3; overall native support remains unverified. |
| Imported evidence | Offline grading of supplied observations and hash-checked artifacts; it cannot establish that attestations are truthful. |

The controlled corpus has 24 behavior cases: 22 have adapters, while
`proposal-not-main` and `concurrent-changes` remain blocked pending a verified
publication and policy harness. `accepted-after-fetch` uses a separate
fixture actor to seed an accepted snapshot; it tests fetch and recall, not
review, merge, or provider permissions. Selected live trials are not a
complete release evaluation. See the [runner reference](docs/reference.md)
for requirements, explicit opt-in commands, limits, and evidence formats.

Optional `--agent <file>` and `--harness <file>` flags on the controlled and
native runners load the matching YAML profiles. Controlled runners also accept
`--config <file>` to combine them. The examples contain
placeholders: edit them and supply all missing limits and credentials before
an opt-in run. Command-line values override separate profiles, which override
the combined config. Profiles must not contain secrets and cannot authorize
`--run` or network access.

## Changing a case

Edit the readable YAML preview and its matching Markdown fixture together.
The `order` values are unique and contiguous. Aliases map to stable UUIDv4
names derived from the case ID and alias. For example, to obtain a filename
for a new node:

```sh
uv run --locked python -c "from evals.corpus import semantic_id; print(semantic_id('scope-search', 'new-alias') + '.md')"
```

The controlled runner currently expects three trials. Its supported grader
kinds and required fields are validated by `corpus.py`; a new kind requires
an implemented grader, not an arbitrary command in YAML.
Review the changes, then explicitly regenerate the compatibility snapshot and
lock:

```sh
uv run --locked python evals/build_dataset.py --write
uv run --locked python evals/build_dataset.py
```

Changing the generated dataset invalidates manifests pinned to its old hash,
including `native-smoke.json`; update and review them explicitly. Do not edit
`scenarios.json` or the lock by hand. Keep cases and expectations synthetic,
and keep new grader behavior independent of product code. Unknown grader kinds,
stale locks, missing fixtures, and duplicate YAML keys fail closed.
