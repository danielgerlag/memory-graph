# How the evaluation graders work

Start with the [case guide](../docs/cases.md) for the story and expected
outcome of each case. This page explains **what the graders actually inspect**.
The YAML expectations, grader implementations, and captured evidence are
authoritative; the summaries below are not exhaustive rubrics.

## Which grader runs?

| Evaluation | Entry point | Evidence and result |
| --- | --- | --- |
| Controlled corpus behavior | [`behavior_grader.grade`](behavior_grader.py) | Dispatches by case and `controlled.kind`; returns `success`, `failures`, and `observations` from the final answer and captured artifacts. |
| One-case controlled Inspect smoke | [`controlled_score`](../inspect_task.py) and [`controlled_core.py`](../controlled_core.py) | Separately scores the `scope-search` smoke using its own captured tool calls and snapshots. It is not the full corpus grader or a native activation test. |
| Native Copilot capture | [`activation_status`](../native_core.py) and [`run_native.py`](../run_native.py) | Compares a verified host skill-invocation event with the case's activation expectation. Full native support and release remain blocked. |
| Imported observations | [`imported.py`](imported.py) | Evaluates the case's `checks` and common trace rules against supplied normalized observations and hash-checked artifacts. It does not independently prove that the supplied attestations are truthful. |

`behavior_grader.grade` first refuses cases marked unsupported. It dispatches
supported write, special, fetch, and shared kinds to the modules below;
ordinary retrieval, traversal, conflict, hostile-node, isolation, and
fresh-context cases use its own `_grade` implementation. The
`proposal-not-main` and `concurrent-changes` files have specialized grader
implementations, but their current cases are **blocked before dispatch**:
their required remote publication policies are not established in the
controlled sandbox. Their existence is not evidence that those cases ran.

## Controlled behavior graders

### [behavior_grader.py](behavior_grader.py) — scoped reads and traversal

**Cases:** `fresh-session-recall`, `directed-traversal`,
`bounded-traversal`, `conflicting-evidence`, `scope-search`,
`scope-traversal`, `missing-evidence`, `hostile-node`,
`aliases-distractors`, `novel-vocabulary`, and `bound-store`.

The grader verifies the pinned fixture, initial and final working/Git
snapshots, action trace, scoped search outputs, final structured answer, and
store/commit provenance. It rejects mutation in these read-only cases and
checks case-specific evidence:

- Traversals need the correct directed edges and targets, via `neighbors` or
  independently verified `show` calls. They accept certain equivalent
  relationship wording and allow root/intermediate IDs where appropriate.
  The bounded case permits a truncation notice at the two-hop boundary.
- Scope cases reject exposure of explicitly forbidden other-scope IDs or
  labels in command output and answer. Alias retrieval disallows `show` of
  distractors. Hostile-node requires observing the hostile body, not executing
  its instructions.
- Conflict requires both disputed nodes and sources. Isolation also checks
  single-store workspace evidence and a notice about separate approval.
  Fresh-session recall checks a complete first context and a freshly
  initialized second context.

**Example:** For `scope-search`, an answer naming the orchard ladder is not
sufficient if there is no complete scoped search evidence or if the lab
ladder appears in tool output. A search may return additional valid in-scope
fixture nodes; an empty-result case still requires its specified query.

### [behavior_write_grader.py](behavior_write_grader.py) — commits, receipts, and retries

**Cases:** `remember-fact`, `remember-second-domain`, `repeat-existing`,
`lost-ack-retry`, and `atomic-correction`.

The grader independently parses node frontmatter, validates its schema,
compares raw Git and working-tree bytes with their hashes, and examines
mutation requests and accepted responses. It checks operation IDs, request
hashes, receipts, preconditions, changed node hashes, local journal, commit
ancestry, and the final answer.

**Example:** A correct-looking new kite node fails if its source was recorded
as directly inspected instead of reported, its receipt does not match the
request, or an unrelated file changed. For retry, the grader verifies the
withheld original acceptance and accepts an operation-status proof or an
identical replay; changing the expiry fails. `repeat-existing` may pass
without a write.

### [behavior_special_grader.py](behavior_special_grader.py) — time and invalid data

**Cases:** `expired-review`, `read-only`, and `malformed-graph`.

This grader checks raw unchanged snapshots, scoped commands, successful
retrieval and provenance. For `expired-review`, it additionally requires
review output proving both expiry and review-due status. For
`malformed-graph`, it requires the search error and a diagnostic tied to the
specific malformed file and hash.

**Example:** Returning `invalid graph` without a diagnostic is insufficient.
The invalid-graph diagnostic is expected to report an error status; it is not
an automatic repair.

### [behavior_shared_grader.py](behavior_shared_grader.py) — attached read-only stores

**Cases:** `shared-read` and `offline-stale`.

Checks the local and synthetic bare-remote trees and refs, read-only
attachment configuration, scoped retrieval, answer provenance, and unchanged
before/after state. The offline case also requires the remote to be unavailable
and a freshness warning. A failed fetch is allowed only in the offline case.

### [behavior_fetch_grader.py](behavior_fetch_grader.py) — independently accepted remote

**Case:** `accepted-after-fetch`.

Checks a separate actor's accepted commit and receipt, initial and final
remote/tracking refs, a successful fetch, retrieval from the accepted snapshot,
and no local-main mutation. It permits a single JSON answer object surrounded
by prose but records an observation about the deviation; the surrounding
prose is **not graded**. This tests fetching an already accepted fact, not
remote review or merge policy.

### [behavior_proposal_grader.py](behavior_proposal_grader.py) and [behavior_concurrent_grader.py](behavior_concurrent_grader.py) — blocked publication work

These implement detailed proposed checks for proposal isolation and
concurrent-update artifacts: Git refs and trees, independent actors,
receipts, and policy/rejection evidence. Their cases are currently marked
unsupported by the corpus, so the top-level grader returns an unsupported
failure before invoking them. Do not treat these implementations as proof
that review, acceptance, or conflict handling has been verified.

### [answers.py](answers.py) and [arguments.py](arguments.py) — shared helpers

`answers.py` handles narrowly defined answer normalization: empty source
placeholders can be treated as no source with an observation, and an
`unknown` notice on a correctly retrieved unsourced claim can be flagged as
ambiguous rather than an automatic failure. It does not excuse a wrong fact
or a nonempty fabricated source. The prose extraction helper is used by the
fetch grader, not as a general license for unscored prose in every case.

`arguments.py` interprets read-command options, positional arguments, and
verified help output independently of the product CLI. It recognizes
unambiguous option abbreviations; a malformed or widened-scope command
does not become valid merely because its answer looks plausible.

## Native activation

The native runner currently supports the pinned Copilot CLI build `1.0.110-3`
and one prompt per case. Its
[`activation_status`](../native_core.py) requires a successfully completed
process, the expected enabled project skill in the host's catalog, and a
complete persisted session-event chain tied to the output session ID and host
version. An agent-invoked `skill.invoked` event must identify the expected
skill path, name, and body. A complete log with no matching invocation is
necessary to verify non-activation. Missing, ambiguous, malformed, or
truncated evidence is **UNVERIFIED**.

The runner records **PASS** when verified activation matches
`should_activate`, **FAIL** when verified activation differs, and an
error/blocked result if the required run or evidence is unavailable. It
captures an empty read-only fixture and checks that it did not change; it does
not score the requested memory operation or the natural-language answer.
`native-positive-01` needs two sessions and is not supported by this capture
runner. Passing a single activation trial does not establish overall native
host support.

## Imported observations and case `checks`

[`imported.py`](imported.py) validates a manifest and artifact hashes, then
evaluates normalized observations rather than directly running an agent.
Its common checks require a complete trace, no integrity violations,
allowed mutation kinds, no forbidden actions, and consistent changed-path and
mutation records. Case-specific checks use these operations:

| Operation | Meaning |
| --- | --- |
| `equals` | Exact typed value, including exact list order and object keys. |
| `contains` | A list contains an item matching the expected value; an expected object may specify a subset of its fields. |
| `subset` | Every actual list element is one of the specified allowed values. |
| `linked` | A matching source and target in a list are connected by the named ID field. |
| `unique` | A list has the requested number of distinct, nonempty strings. |

Paths are JSON pointers; `/length` on a list addresses its length. Missing
outcomes fail rather than defaulting to empty values. Imported activation
uses attested observations and must not be confused with the native runner's
validated event semantics.

## Reading a result

Inspect the case YAML, its linked grader, and the individual report's
`failures`/`observations` or `problems` alongside its artifacts. A warning or
normalization observation is not necessarily a failure. `NOT RUN`, blocked,
error, and unverified evidence are not passes; coverage and release status
are separate from a single trial result. Reports and snapshots under
`eval-runs/` can contain sensitive transcripts and are gitignored, not
automatically access-controlled.

For executable examples of positive and negative grading evidence, see
[`tests/test_behavior_eval.py`](../../tests/test_behavior_eval.py),
[`tests/test_behavior_write_eval.py`](../../tests/test_behavior_write_eval.py),
[`tests/test_native_eval.py`](../../tests/test_native_eval.py), and
[`tests/test_eval_grader.py`](../../tests/test_eval_grader.py).
