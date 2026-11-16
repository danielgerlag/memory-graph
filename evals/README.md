# Graph-memory evaluations

The evals check the skill's behavior, native activation, and imported evidence
separately. The cases are synthetic **development and regression fixtures**,
not held-out evidence of generalization.

## Start here

1. Read [scope-search.yaml](cases/behavior/scope-search.yaml): its prompt,
   expected results, allowed changes, and controlled grader are in one place.
2. Open its [Markdown graph](fixtures/graphs/scope-search/). The harness copies
   these real node files into a fresh Git store and commits the initial snapshot.
3. Run the offline checks from the repository root:

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

The image recipe was exercised with a synthetic Foundry HTTP endpoint and a
network-disabled sandbox; **no real Foundry model has been tested**. See the
[runner reference](docs/reference.md#opt-in-controlled-corpus-trials) for
requirements and limitations.

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
  corpus.yaml            dataset metadata and trial count
  corpus.lock.json       pinned source and generated-dataset hashes
  scenarios.json         generated compatibility snapshot, not authoring source
  docs/reference.md      runner commands, evidence format, and release gates
```

In a behavior case, `fixture.main.nodes` maps readable aliases to actual
UUID-named files. `fields` identifies which node fields appear in the legacy
normalized observation format; it does not change the Markdown node. `checks`
are predicates for **imported observations**. `controlled` selects an
allowlisted built-in grader and gives it the expected facts, targets, and
case-specific instructions. Controlled graders inspect captured files, Git
refs, and tool calls independently; they do not blindly trust the agent's
answer. The deliberately malformed fixture is an exception to valid-node
format.

Expectation files and grader code stay outside the agent's sandbox. YAML does
not execute commands, and its named graders are not arbitrary plugins.

## Execution lanes

| Lane | What it establishes |
| --- | --- |
| Controlled Inspect | Opt-in model trial with an explicitly loaded skill and a restricted tool; not native activation. |
| Native Copilot | Opt-in evidence-capture spike; activation-event interpretation is not yet verified. |
| Imported evidence | Offline grading of supplied observations and hash-checked artifacts; it cannot establish that attestations are truthful. |

The controlled corpus has 24 behavior cases: 21 have adapters, while
`proposal-not-main`, `accepted-after-fetch`, and `concurrent-changes` remain
blocked pending a verified remote and acceptance harness. **No live model or
provider evaluation is claimed.** See the [runner reference](docs/reference.md)
for requirements, explicit opt-in commands, limits, and evidence formats.

Optional `--agent <file>` and `--harness <file>` flags on the controlled and
native runners load the matching YAML profiles. Controlled runners also accept
`--config <file>` to combine them. The examples contain
placeholders: edit them and supply all missing limits and credentials before
an opt-in run. Command-line values override separate profiles, which override
the combined config. Profiles must not contain secrets and cannot authorize
`--run` or network access.

## Changing a case

Edit its YAML and, if needed, its Markdown fixture. The `order` values are
unique and contiguous. Aliases map to stable UUIDv4 names derived from the
case ID and alias. For example, to obtain a filename for a new node:

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
