# LastSeek

Memory-mapped semantic vector search for LastDB. Canonical remote
`lastdb:///lastseek` (LastGit).

LastSeek runs **outside Mini**, so a default `lastdbd` ships no model weights
and no inference runtime. The index is **local-only and regenerable** from
LastDB — it is not CloudSync product data, and losing it costs a rebuild, not
data.

## Why it exists

The predecessor (the TypeScript Search app) held the entire index in the heap
and re-serialised all of it on every flush. Two costs followed, and both scaled
with corpus size instead of with work done:

- **Residency** — 108 MB of payload cost 373 MB of heap, and one query required
  loading every byte first.
- **Write amplification** — a flush rewrote the whole snapshot. On 2026-08-07
  that wrote 549.80 GB in 20 minutes and took the machine down.

LastSeek separates the two things those costs came from.

## Design

| | |
|---|---|
| `vectors.f32` | one contiguous `f32` slab, memory-mapped, never parsed or copied |
| `meta.jsonl` | identity + text, touched **only for rows that make the result set** |
| `pending.jsonl` | append-only log; a flush costs O(changed), not O(corpus) |
| `manifest.json` | dims, embedder id, row count |

Keeping text out of the scan is the load-bearing split: text is roughly 4x the
bytes of a vector, so a query reads ~1.5 KB per candidate rather than ~7 KB.
Because vectors are mapped rather than loaded, the OS page cache decides
residency and resident memory tracks the working set instead of the corpus.

Scoring is a linear pass. That is the right default while a scan is fast; an
approximate index adds recall risk and build cost, so it should arrive when the
corpus earns it, not before.

## Schema scope is structural

When a caller names schemas, only those schemas' rows are scored. Scope is
never a post-filter over a global top-k — that silently returns fewer than `k`
results, or none, whenever the scoped schema is not already in the global head.
This is a correctness property, not an optimisation.

## Schema scope resolves through the Schema Service

`--schema` takes what a caller would naturally write:

```bash
lastseek query "durable compaction" --schema Card        # readable name
lastseek query "durable compaction" --schema bc941dbc…   # identity hash
lastseek query "durable compaction" --schema 183416…     # registry name
```

The index keys rows on a canonical **schema identity hash**, and the readable
name → identity table comes from the **Schema Service**, directly — SS holds the
definitive names. Refresh it with `lastseek schema-table`; it caches to
`~/.lastseek/schema-table.json`.

A readable name is deliberately **set-valued**: `Card` is six distinct
identities on the live registry, and resolving to one of them would be a quiet
version of the bug below.

**A schema name that resolves to nothing is an error, never an empty result.**

```
$ lastseek query "anything" --schema Crad
lastseek: unknown schema "Crad"; known names include: Card
```

That distinction is the whole point. The predecessor returned `Ok([])` for both
"you spelled the schema differently than ingest wrote it" and "that schema has
no rows yet", live, on brain's and kanban's own corpora — `--schema Card`
returned zero hits while the same query scoped by the interned hash returned
correctly ranked results.

Ingest performs the same translation from the same table, so ingest and query
key off one authority and agree by construction. `Record.schema_identity` holds
that key, and never a readable name.

## Coverage: the owning app declares the denominator

`lastseek status` reports, per schema, vectors held against the **live** records
that exist — and it only rates schemas whose owning app has said how many that
is:

```bash
lastseek coverage-declare --schema Reference \
  --live-records "$(brain list --type reference --count --json | jq -r .count)" \
  --source 'brain list --type reference --count'
```

Two reasons it works this way, both measured.

**The node catalog's `record_count` is not a denominator.** It counts rows in the
partition, which includes every superseded version of every row. Dividing by it
reported `Reference` at 63% when brain's own census is 1,788 and LastSeek holds
1,788 — 100%. On `Milestone` the catalog says 415 rows for 67 milestones. The
ratio was `live / (live + history)`, which cannot reach the floor and falls as
the corpus ages. It is still reported, as `partition_rows`, because it is
informative — just not a denominator.

The rule: **a coverage metric must divide by a count the owning app would
recognise as its own.** The app is the only thing that knows, because it
dual-writes the keyed secondary that makes its corpus enumerable without a scan.

**The declared count is the app's claim, not a census LastSeek vouches for** —
which is why `--source` is required rather than decorative. The example above
uses `brain list --count`, and that instrument has measured defects in both
directions: the list projection under-delivers (993 of 999 reference rows), while
`--count` over-counts because deleted records stay countable. The residual error
inflates the denominator, so it can only make coverage look *worse* than reality
— a false red, never a false green.

Recording the method keeps the fix where it belongs: when an app gets a better
access pattern, its declaration changes and LastSeek does not. And a reader who
distrusts a ratio can read the method that produced its denominator.

**Declaring is opting in.** Without that, the floor applied to all 1,137 schemas
on the node and produced 139 `degraded` entries dominated by machine schemas
nobody wants searchable — `LastgitRepoPolicyEventV2` (12,633 rows),
`LastDB Request Ops Rollup` (9,892), `RoutineRunSummary` (5,465). A signal that
is always red is not a signal, and the thing this report exists to make visible —
a stopped drain — goes straight back to being invisible.

So `state` is one of:

| | |
|---|---|
| `healthy` | every tracked schema is at or above the floor |
| `degraded` | a tracked schema is below it |
| `unknown` | nothing has declared a count, or the catalog was unreachable |

`unknown` is deliberately not `healthy`: "no app declared a denominator" and
"every schema is covered" are different facts.

## Ingest: a push feed and an app-driven backfill — no scan

LastDB is Dynamo-shaped: point get, range under one hash, **no scan**. So
"page every schema to build the first index" is not a path LastSeek has.

```bash
lastseek drain                  # apply the host's IndexChangeBatch feed
lastseek add < records.jsonl    # an app feeding its own corpus
```

`drain` reads the batch files the host's outbox sink already writes to
`$LASTDB_HOME/apps/search/inbox/`. It is **bounded** (`--max-files`, default
2000) and reports `remaining`, because an unbounded drain over a large backlog
is what took the machine down on 2026-08-07 — and a capped run that reports
nothing reads as "inbox empty", which is how that backlog reached 99k files
unnoticed. It advances a watermark rather than deleting batch files, so the
incumbent Search app can keep reading the same inbox until cutover; `--consume`
deletes them.

Schema owners can opt an entire schema out of semantic indexing by declaring
an explicit empty `searchable_fields` list in the host change feed. LastSeek
treats absence of that field as the legacy "unspecified" case, but treats an
empty list as "index nothing": upserts are suppressed and converted to
tombstones so replaying owner records removes copies indexed under an older
policy. `lastseek status` exposes the cumulative
`non_searchable_records_skipped` count from the durable drain watermark.

Backfill is the owning app feeding its own records: brain and kanban can
enumerate theirs without a scan because they dual-write the keyed secondaries
that make enumeration possible. A central backfill would have to guess, for
every app forever, which of its fields are prose.

## Coverage status reads LastDB, by design

`lastseek status` compares vectors held with source records available for each
schema and reports `state: healthy|degraded`. The default healthy floor is 95%;
override it with `--coverage-floor` only when an operator deliberately wants a
different policy. Source counts come from
`GET /api/schemas?include_counts=true` over the LastDB Unix socket. This is the
status half of LastSeek's existing LastDB relationship (records and changes),
not a third outside dependency; Schema Service remains the independent source
of definitive names and canonical identities.

Coverage deliberately joins `catalog.name -> Schema Service identity ->
LastSeek interned id`. It never keys on the node catalog's `identity_hash`,
which is recomputed by the running node and disagrees with Schema Service on
the live brain/kanban schemas. Multiple catalog aliases that resolve to one
canonical identity are collapsed using the maximum record count rather than
summed, so the six `Card` identities cannot multiply one logical denominator.
If the catalog cannot be read, status fails closed as `degraded` with a
`coverage.available: false` diagnostic instead of claiming health from
internal consistency alone.

## Embedders

`LASTSEEK_EMBEDDER=deterministic` selects a test fixture with no semantics, so
CI can prove storage, scope and ranking offline without downloading weights.
`LASTSEEK_EMBEDDER=bge` selects real neural embedding — `bge-small-en-v1.5`
via the vendored FastEmbed crate (`vendor/fastembed-4.9.0-no-image`, the same
fork `fold` uses) — and requires a build with `--features neural`.

The fixture reports `is_neural() == false`, and that is deliberate: the
predecessor let 2,021 deterministic hash vectors reach its production index,
where they shared a space with neural ones and were scored against them as if
comparable. The boundary must be refusable at write time rather than
discoverable in a later coverage audit.

BGE is **asymmetric**: `lastseek query` prepends the model's query
instruction, `lastseek add` does not. Embedding the same text through both
paths produces different vectors on purpose.

### Fetching the bge-small-en-v1.5 weights

Neither the weights nor a fetch step ship in this repo — `neural`-feature
tests skip themselves when the model files are absent, and CI never fetches
them. To use `LASTSEEK_EMBEDDER=bge` locally, download the five files FastEmbed
itself would pull from `Xenova/bge-small-en-v1.5` into
`~/.lastseek/models/bge-small-en-v1.5` (or `$LASTSEEK_MODEL_DIR`):

```bash
dir="${LASTSEEK_MODEL_DIR:-$HOME/.lastseek/models/bge-small-en-v1.5}"
mkdir -p "$dir"
base="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main"
curl -L "$base/onnx/model.onnx" -o "$dir/model.onnx"
for f in tokenizer.json config.json special_tokens_map.json tokenizer_config.json; do
  curl -L "$base/$f" -o "$dir/$f"
done
```

## Usage

```bash
lastseek schema-table                                   # refresh the SS name table
lastseek status                                          # rows, embedder, table, SS gaps
lastseek query "write amplification" --k 5 --schema Concept
lastseek drain                                           # host feed -> index
lastseek add < records.jsonl                             # an app's own corpus
lastseek compact                                         # fold the log into the slab
```

A cold `lastseek query` — fork, load the model, map the index, embed, score,
print — is **~100 ms**. That is why there is no resident daemon: the reason to
keep a search process warm was the predecessor's multi-second Node +
transformers.js startup, and `ort` does not have it.

## Development

```bash
cargo test
bash .lastgit/ci.sh
```

The required gate builds an **explicit** feature set, not `--all-features`.
`--all-features` turns on `neural`, which downloads ONNX Runtime at build time
(contradicting the gate's own weightless contract) and takes ~11.5 minutes in
release — past LastGit's 600 s CI timeout. Run the neural tests locally, on a
machine that has the weights:

```bash
cargo test --release --features neural,fault-injection
```

### Artifact release context

The installable binary is produced by the separate LastGit
`artifact-release` context. It builds with `--features neural`, stages only
`dist/lastseek`, and publishes an immutable artifact whose manifest records the
exact source commit. The required `ci-required` merge gate never invokes this
slow build.

On the forge host, run the context against a cursor seeded immediately before
the target `main` update so the watcher builds that exact new tip without
replaying historical commits:

```bash
lastgit ci watch --repo lastseek \
  --context artifact-release \
  --ref refs/heads/main \
  --state-file /path/to/pre-update.cursor \
  --timeout-ms 900000 \
  --once
```
