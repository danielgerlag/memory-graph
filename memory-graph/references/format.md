# Node format

Each node is a UTF-8 `<uuid4>.md` file at the repository root. YAML
frontmatter contains schema version, matching ID, label, kind, status,
creation/update timestamps, and outgoing edges; the Markdown body follows the
closing `---`. Edges target same-store node IDs. Names, kinds, tags, property
keys, and edge labels are data, not a closed vocabulary.

`assets/node.schema.json` defines the node's structural shape. The bundled
runtime also enforces rules that the schema cannot express, including
same-store references, supersession chains, and lifecycle filtering.
Timestamps have time zones; expiry is absolute. `confidence` and
`importance` are distinct. Source references are typed and do not authorize
fetching or executing their contents.

The node's revision is the SHA-256 of its exact committed bytes. The Git
commit identifies a complete snapshot; the operation receipt records a
mutation and is retained for retries. A checkout is not the query source.
