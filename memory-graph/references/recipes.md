# Usage and request recipes

Set `SKILL_DIR` to the installed skill directory and `STORE` to a separate,
dedicated store. After installation, on macOS/Linux:

```sh
MG="$SKILL_DIR/.venv/bin/python"
SCRIPT="$SKILL_DIR/scripts/memory_graph.py"
"$MG" "$SCRIPT" --root "$STORE" init
```

On Windows use `"$SKILL_DIR/.venv/Scripts/python.exe"`. All results are JSON
on stdout; errors are JSON on stderr with nonzero exit status. `check` and
`diagnose` also exit nonzero when the snapshot is invalid. No command
automatically installs dependencies, fetches, or pushes.

Each mutation takes `--operation UUID` and `--input FILE`; use `-` for stdin.
Generate the UUID before the first attempt, retain the input, and reuse both
for retries. For example:

```sh
OP=$("$MG" -c 'import uuid; print(uuid.uuid4())')
printf '%s\n' '{"label":"An example fact","kind":"claim","body":"A qualified statement.","scope":"example"}' |
  "$MG" "$SCRIPT" --root "$STORE" create --operation "$OP" --input -
"$MG" "$SCRIPT" --root "$STORE" find "example" --scope example
"$MG" "$SCRIPT" --root "$STORE" operation status "$OP"
```

`show ID` returns `node`, `body`, and the SHA-256 `revision`. An update input
is `{"id":"UUID","expected_revision":"SHA256","patch":{"label":"New label"}}`.
Omitted fields and body remain unchanged; supplied arrays and objects replace
the entire field; nullable fields may be cleared with `null`. `ttl_seconds`
is an optional nonnegative expiry duration, mutually exclusive with
`expires_at`.

`link` and `unlink` inputs contain `id`, `expected_revision`, and `edge`
(`{"label":"...","to":"UUID"}`). `apply` takes
`{"changes":[{"op":"create","node":{"id":"UUID","label":"...","kind":"..."}},{"op":"update","id":"UUID","expected_revision":"SHA256","patch":{...}}]}`;
batch-created IDs must be chosen in advance. `supersede` takes `id`,
`expected_revision`, and either `replacement_id` or `replacement` (a create
object). The entire candidate graph must validate before publication.

`find QUERY --scope SCOPE` matches that scope exactly; `--include-unscoped`
also includes nodes without scope. Omitting scope searches all scopes. `--tag`
and `--kind` are exact, case-sensitive filters and may be used without a query.
`neighbors ID` has `--direction in|out|both`, `--label`, `--depth`, and
`--limit`. All hops enforce the same visibility and scope filters. `show
--include-inactive` is an explicit inspection of expired/superseded nodes.
The `show` result filters outgoing edges to the same visible scope and
lifecycle; it is not necessarily the complete stored edge list. `review`
lists overdue, expired, disputed, and superseded nodes and accepts the same
scope options. Traversal's `relationships` list preserves distinct labels,
directions, edge properties, and cycles among the returned nodes; a response
marks truncation when its node, relationship, or byte budget is exhausted.

If `check` or ordinary recall reports an invalid main snapshot, `diagnose`
reports affected paths and raw SHA-256 hashes. With explicit authorization,
`repair` accepts `{"changes":[{"path":"UUID.md","expected_sha256":"HASH",
"action":"quarantine"}]}` or an `action` of `replace` with full `content`.
Include all necessary dependent changes in the same request; the entire
result must validate. Quarantine preserves the original bytes in Git history
and in a tracked quarantine directory. Damaged store identity, damaged
receipts, and unsupported future schemas require operator recovery rather
than guessing.

`history` lists commits reachable from selected main. `show ID --commit
FULL_COMMIT` explicitly inspects an ancestor and labels the result historical.
`revert` takes `{"target_operation":"UUID"}` and creates a new validated
commit, preserving the original receipt. It refuses to overwrite later
changes or produce an invalid graph; repair and revert receipts are not
automatically reversible.

For an existing clone of a shared store, configure its Git remote separately,
then `attach --remote origin --mode read|propose|write` and explicitly
`fetch`. Reads use only the last fetched remote main. Mutations in propose
and write modes prepare a local operation branch; `publish --operation UUID`
pushes that proposal, not main. Review and merge are external. Fetch after
merge to make it visible to normal recall. Git credentials and server rules,
not the local mode, enforce authorization. `status` reports attachment state.
Publication to local bare remotes is deliberately unavailable: the client
will not execute or bypass their server hooks. Earlier attachments without
endpoint pins must be explicitly reattached.
Local fetch starts server-side Git on the same host. Use only trusted local
remotes: protection against every local server configuration is unverified.
Only one directly configured endpoint is supported; push and fetch URLs must
match, and changes require reattachment. Supported transports are HTTPS, SSH,
and local paths for fetching. HTTP, `git://`, custom helpers, URL passwords,
queries, and fragments are rejected. HTTPS requires verified TLS and disables
redirects, proxies, and custom CA overrides. SSH enforces host-key checking
and uses the agent/default keys without SSH config files or proxy commands.
Direct user-global HTTPS credential settings are supported, but included
config files are not. Push signing and automatic tag following are disabled.
For protected-branch integration, a trusted required check must run
`check-merge --base FULL_MAIN_COMMIT --candidate FULL_MERGE_COMMIT` on the
actual merge result. It checks graph validity, existing receipt preservation,
affected-node preconditions, and declared changes. Configure the check and
reviews on the server independently; a local check is not authorization.

Checkout files may not reflect committed main after a mutation: the runtime
updates Git refs atomically without overwriting the checkout or staging area.
Do not treat uncommitted edits as memories.

For a grep-friendly, non-authoritative view of every node in one accepted
snapshot, use `export-view --dest /new/empty-parent/snapshot`. The destination
must not exist or be inside the store. The exported directory includes
all scopes and inactive nodes, `store.json`, and snapshot provenance, so it
may contain sensitive information. It is not a live Git store and will not
update as main changes. A failed export can leave an incomplete destination;
the provenance file is written only after all node files.

The ignored `.memory-graph/parse-cache.sqlite3` is a disposable local parse
cache, keyed by committed node content and parser version. If the CLI reports
cache corruption, remove that specific file and retry; it is not graph data.
