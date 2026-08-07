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

## Embedders

Neural embedding is **not wired up yet**; `LASTSEEK_EMBEDDER=deterministic`
selects a test fixture with no semantics, so CI can prove storage, scope and
ranking offline without downloading weights.

The fixture reports `is_neural() == false`, and that is deliberate: the
predecessor let 2,021 deterministic hash vectors reach its production index,
where they shared a space with neural ones and were scored against them as if
comparable. The boundary must be refusable at write time rather than
discoverable in a later coverage audit.

## Usage

```bash
lastseek status
lastseek query "write amplification" --k 5 --schema concept
lastseek add < records.jsonl
lastseek compact
```

## Development

```bash
cargo test
bash .lastgit/ci.sh
```
