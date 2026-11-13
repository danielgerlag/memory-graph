# Local scaling measurements

These are engineering observations, **not release certification**. They cover
one search workload, not writes, traversal, provider access, or agent behavior.
The runner was not pinned or isolated, and operating-system cache state was
not controlled. There is no previous-release baseline.

On 2026-11-12, using Python 3.12.13 and Git 2.54.0 on macOS/Apple M5, each
workload ran 20 empty-parse-cache samples and 20 warm samples:

| Nodes | Receipts | Cold p95 | Warm p95 | Peak process RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0 | 2.263 s | 0.237 s | 53.5 MiB |
| 10,000 | 0 | 7.553 s | 1.841 s | 159.6 MiB |
| 1,000 | 10,000 | 2.713 s | 0.424 s | 135.7 MiB |

Peak RSS includes fixture construction and excludes child processes. Each
query used 7 Git processes for the small corpus and 9 for the other corpora.
Large cold parses also used four Python workers. The fixture corpus hashes
were, respectively:

```text
a7d13209ad7e30ea4013dffe9ed025be42bea89766052f155e2f918404953420
f3e8b1554b6fc9fa90352e4bdedf3b6ca0c06a4fdef4bfe55566af37adb540fe
b15e1d5f726a1fff9e9acaccc9e72cc205784c92deb104b864b33cf229f8e082
```

The tested source was uncommitted. The benchmark script SHA-256 was
`683227cf26b559be080e62bc06d8c88d3977d2cba673db0680f54e5f378a741e`;
the development lock SHA-256 was
`81ab3e1cebe519d0248b2114b398b487c1b537d2ce4ccb3b74983ecd9cb4de69`.
The 16-file skill bundle hash was
`918f8d4c027665e856496b95e4edc32dbb000b21f40be8cbcce4df514a03a894`,
computed from sorted relative paths and file content hashes, excluding virtual
environments and bytecode caches. Runtime hardening continued after these
measurements, so they are not a benchmark of the current bundle.

After transport and lazy-fetch hardening, the 10,000-node search workload was
remeasured on the same machine with 20 cold and 20 warm samples. Cold p95 was
9.201 s, warm p95 was 1.842 s, and peak process RSS was 149.9 MiB. This was
still a shared, unpinned runner and is not release certification. The tested
skill bundle hash was
`019d2421ff477c2ea748353cc0f3834badc47e26ab8c9dc4181178d8965828ce`;
the benchmark script hash was
`703f3dccec2d419b4e929d226151ee27d4c5899bd00061182dc1b8efdeed2da6`.

A later 1,000-node bounded-traversal run used the same sample counts and
reported 2.203 s cold p95 and 0.242 s warm p95. Its skill bundle hash was
`7c6c5e050d18fdd7425696c688474d6f026bdcf25b05f92c38cbd1e2860aec5d`;
the benchmark script hash was unchanged. These remain exploratory results.
With that same bundle, a 10,000-node traversal run reported 11.217 s cold p95
and 1.798 s warm p95. The cold result exceeded the provisional 10-second
target on this shared, unpinned runner. A repeat measured 9.054 s cold p95 and
1.623 s warm p95. This variability is why the release runner must be pinned
before making a performance claim.

Reproduce a workload with:

```sh
uv run --locked --group dev python benchmarks/run.py \
  --nodes 1000 --iterations 40 --cold-iterations 20
```

The runtime requires Git with `--no-lazy-fetch` and `show-ref --exists`
(upstream Git 2.45+).
Use `--nodes 10000` for the larger graph and `--receipts 10000` for the
long-history case. Add `--workload neighbors` to measure bounded traversal
or `--workload tag` to measure exact tag filtering instead of lexical search.
The generator and store are disposable synthetic data.
Release comparisons still require a pinned runner, representative additional
workloads, and a previous release measured under the same conditions.
