//! Store and search behaviour.
//!
//! Several of these exist because the predecessor got them wrong in production:
//! whole-corpus rewrites per flush, a log that was unreadable until a snapshot
//! existed, and scope implemented as a post-filter.

use lastseek::embed::{DeterministicEmbedder, Embedder};
use lastseek::index::{QueryOpts, Searcher};
use lastseek::store::{PendingOp, Record, VectorStore};
use lastseek::LastSeekError;

const DIMS: usize = 64;

/// Named stand-in for tests that need an arbitrary embedder identity without
/// exercising `embed()`.
struct NamedEmbedder {
    id: String,
    dims: usize,
    is_neural: bool,
}

impl Embedder for NamedEmbedder {
    fn id(&self) -> &str {
        &self.id
    }
    fn dimensions(&self) -> usize {
        self.dims
    }
    fn is_neural(&self) -> bool {
        self.is_neural
    }
    fn embed(&mut self, _texts: &[String]) -> Result<Vec<Vec<f32>>, LastSeekError> {
        unimplemented!("NamedEmbedder is identity-only")
    }
}

fn rec(schema: &str, key: &str, text: &str) -> Record {
    Record {
        schema_identity: schema.to_string(),
        key_hash: Some(key.to_string()),
        key_range: None,
        fragment_key: "body".to_string(),
        text: text.to_string(),
        mutation_id: None,
        deleted: false,
    }
}

fn seed(dir: &std::path::Path, docs: &[(&str, &str, &str)]) -> VectorStore {
    let mut emb = DeterministicEmbedder::new(DIMS);
    let mut store = VectorStore::open(dir).unwrap();
    store.bind_embedder(&emb).unwrap();
    let texts: Vec<String> = docs.iter().map(|d| d.2.to_string()).collect();
    let vecs = emb.embed(&texts).unwrap();
    let ops: Vec<PendingOp> = docs
        .iter()
        .zip(vecs)
        .map(|(d, v)| PendingOp {
            record: rec(d.0, d.1, d.2),
            vector: Some(v),
        })
        .collect();
    store.append(&ops).unwrap();
    store
}

#[test]
fn reads_back_from_the_log_before_any_compaction() {
    // A young store has a log and no slab. The predecessor ignored its log
    // unless a snapshot already existed, which was silent total loss on
    // restart.
    let dir = tempfile::tempdir().unwrap();
    {
        seed(
            dir.path(),
            &[("concept", "a", "alpha beta"), ("concept", "b", "gamma")],
        );
    }
    let store = VectorStore::open(dir.path()).unwrap();
    assert_eq!(store.committed_rows(), 0, "nothing compacted yet");
    assert_eq!(store.pending_ops(), 2, "but the log must still be readable");

    let mut emb = DeterministicEmbedder::new(DIMS);
    let q = emb.embed(&["alpha beta".to_string()]).unwrap().remove(0);
    let hits = Searcher::new(&store, emb.id())
        .unwrap()
        .search(&q, &QueryOpts::default())
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].record.key_hash.as_deref(), Some("a"));
}

#[test]
fn compaction_folds_the_log_and_preserves_vectors_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = seed(
        dir.path(),
        &[
            ("concept", "a", "alpha beta gamma"),
            ("concept", "b", "delta epsilon"),
            ("design", "c", "alpha zeta"),
        ],
    );
    let before: Vec<Vec<f32>> = store
        .pending()
        .iter()
        .map(|o| o.vector.clone().unwrap())
        .collect();

    store.compact().unwrap();
    assert_eq!(store.committed_rows(), 3);
    assert_eq!(store.pending_ops(), 0);
    assert_eq!(store.log_bytes(), 0, "log dropped after compaction");

    // f32 in, f32 out — the slab is the same bytes, so this is exact.
    for (i, want) in before.iter().enumerate() {
        assert_eq!(store.vector_at(i).unwrap(), want.as_slice());
    }

    // And it survives a reopen.
    let reopened = VectorStore::open(dir.path()).unwrap();
    assert_eq!(reopened.committed_rows(), 3);
    assert_eq!(reopened.vector_at(0).unwrap(), before[0].as_slice());
}

#[test]
fn appending_is_proportional_to_changes_not_to_corpus_size() {
    // The defect that took the machine down: a flush cost O(corpus). Here a
    // flush over a large corpus must cost about one record.
    let dir = tempfile::tempdir().unwrap();
    let docs: Vec<(String, String, String)> = (0..2000)
        .map(|i| {
            (
                "concept".to_string(),
                format!("k{i}"),
                format!("record {i} {}", "lorem ipsum dolor sit amet ".repeat(20)),
            )
        })
        .collect();
    let refs: Vec<(&str, &str, &str)> = docs
        .iter()
        .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
        .collect();
    let mut store = seed(dir.path(), &refs);
    store.compact().unwrap();

    let slab = store.slab_bytes();
    assert!(slab > 0);

    let mut emb = DeterministicEmbedder::new(DIMS);
    let v = emb
        .embed(&["one more record".to_string()])
        .unwrap()
        .remove(0);
    store
        .append(&[PendingOp {
            record: rec("concept", "extra", "one more record"),
            vector: Some(v),
        }])
        .unwrap();

    let log = store.log_bytes() as usize;
    assert!(
        log < slab / 100,
        "one append wrote {log} bytes against a {slab}-byte corpus; \
         a flush must not scale with corpus size"
    );
}

#[test]
fn schema_scope_is_structural_not_a_post_filter() {
    // The distinction that matters: scoping must SEARCH the named schema, not
    // filter a global top-k. Here the scoped schema's only row is a poor match,
    // so a post-filter over the global head would return nothing.
    let dir = tempfile::tempdir().unwrap();
    let mut docs: Vec<(String, String, String)> = (0..50)
        .map(|i| {
            (
                "concept".to_string(),
                format!("c{i}"),
                "alpha beta gamma delta".to_string(),
            )
        })
        .collect();
    docs.push((
        "sop".to_string(),
        "s1".to_string(),
        "completely unrelated wording".to_string(),
    ));
    let refs: Vec<(&str, &str, &str)> = docs
        .iter()
        .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
        .collect();
    let mut store = seed(dir.path(), &refs);
    store.compact().unwrap();

    let mut emb = DeterministicEmbedder::new(DIMS);
    let q = emb
        .embed(&["alpha beta gamma delta".to_string()])
        .unwrap()
        .remove(0);

    let scoped = Searcher::new(&store, emb.id())
        .unwrap()
        .search(
            &q,
            &QueryOpts {
                schemas: vec!["sop".to_string()],
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(scoped.len(), 1, "scoped search must find the sop row");
    assert_eq!(scoped[0].record.schema_identity, "sop");
}

#[test]
fn pending_ops_shadow_committed_rows() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = seed(dir.path(), &[("concept", "a", "original text")]);
    store.compact().unwrap();

    let mut emb = DeterministicEmbedder::new(DIMS);
    let v = emb.embed(&["replaced text".to_string()]).unwrap().remove(0);
    store
        .append(&[PendingOp {
            record: rec("concept", "a", "replaced text"),
            vector: Some(v.clone()),
        }])
        .unwrap();

    let hits = Searcher::new(&store, emb.id())
        .unwrap()
        .search(&v, &QueryOpts::default())
        .unwrap();
    assert_eq!(hits.len(), 1, "the same identity must not appear twice");
    assert_eq!(hits[0].record.text, "replaced text");

    store.compact().unwrap();
    assert_eq!(store.committed_rows(), 1);
    assert_eq!(store.record_at(0).unwrap().text, "replaced text");
}

#[test]
fn deletes_survive_compaction_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = seed(
        dir.path(),
        &[("concept", "a", "keep me"), ("concept", "b", "remove me")],
    );
    store.compact().unwrap();

    let mut gone = rec("concept", "b", "remove me");
    gone.deleted = true;
    store
        .append(&[PendingOp {
            record: gone,
            vector: None,
        }])
        .unwrap();
    store.compact().unwrap();

    let reopened = VectorStore::open(dir.path()).unwrap();
    assert_eq!(reopened.committed_rows(), 1);
    assert_eq!(
        reopened.record_at(0).unwrap().key_hash.as_deref(),
        Some("a")
    );
}

#[test]
fn a_torn_log_tail_costs_one_op_not_the_log() {
    let dir = tempfile::tempdir().unwrap();
    seed(
        dir.path(),
        &[("concept", "a", "first"), ("concept", "b", "second")],
    );
    let log = dir.path().join("pending.jsonl");
    let mut bytes = std::fs::read(&log).unwrap();
    bytes.truncate(bytes.len() - 10); // interrupted mid-append
    std::fs::write(&log, bytes).unwrap();

    let store = VectorStore::open(dir.path()).unwrap();
    assert_eq!(store.pending_ops(), 1, "the intact op must still load");
}

#[test]
fn fixture_embedder_is_marked_non_neural() {
    // Mixing fixture vectors into a neural space is what produced 2,021
    // meaningless vectors in the predecessor. The boundary must be checkable.
    let emb = DeterministicEmbedder::new(DIMS);
    assert!(!emb.is_neural());
    assert_ne!(emb.id(), "all-MiniLM-L6-v2");
}

#[test]
fn embedder_identity_survives_a_reopen_before_any_compaction() {
    // A separate `lastseek status` process must see the embedder that a
    // prior `lastseek add` recorded, even though nothing has compacted yet
    // and there is no generation directory to carry it.
    let dir = tempfile::tempdir().unwrap();
    {
        let mut store = VectorStore::open(dir.path()).unwrap();
        store
            .bind_embedder(&NamedEmbedder {
                id: "some-embedder".to_string(),
                dims: DIMS,
                is_neural: true,
            })
            .unwrap();
    }
    let reopened = VectorStore::open(dir.path()).unwrap();
    assert_eq!(reopened.embedder_id(), "some-embedder");
    assert_eq!(reopened.dimensions(), DIMS);
    assert!(reopened.embedder_is_neural());
}

#[test]
fn search_touches_the_text_segment_at_most_k_times() {
    // The residency property, made observable: scoring and schema-scope
    // filtering must come entirely from rows.bin, so a k=5 query over a
    // 10k-row corpus parses at most 5 meta.jsonl lines — not 10,000.
    let dir = tempfile::tempdir().unwrap();
    let docs: Vec<(String, String, String)> = (0..10_000)
        .map(|i| {
            (
                "concept".to_string(),
                format!("k{i}"),
                format!("record number {i} filler text goes here"),
            )
        })
        .collect();
    let refs: Vec<(&str, &str, &str)> = docs
        .iter()
        .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
        .collect();
    let mut store = seed(dir.path(), &refs);
    store.compact().unwrap();
    drop(store);

    let reopened = VectorStore::open(dir.path()).unwrap();
    let mut emb = DeterministicEmbedder::new(DIMS);
    let q = emb
        .embed(&["record number 42 filler text goes here".to_string()])
        .unwrap()
        .remove(0);
    let hits = Searcher::new(&reopened, emb.id())
        .unwrap()
        .search(
            &q,
            &QueryOpts {
                k: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 5);
    assert!(
        reopened.meta_reads() <= 5,
        "open()+search(k=5) over 10k rows touched the text segment {} times; \
         residency requires the scan to filter via rows.bin only",
        reopened.meta_reads()
    );
}

#[test]
fn residency_rss_growth_is_bounded_best_effort() {
    // Loose secondary signal only (the deterministic access-counter test
    // above is the real gate): if `ps` is unavailable or behaves oddly on
    // this platform, skip rather than flake.
    fn rss_kb() -> Option<u64> {
        let pid = std::process::id().to_string();
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8(out.stdout).ok()?.trim().parse().ok()
    }

    let Some(before) = rss_kb() else {
        return;
    };

    let dir = tempfile::tempdir().unwrap();
    let docs: Vec<(String, String, String)> = (0..5_000)
        .map(|i| {
            (
                "concept".to_string(),
                format!("k{i}"),
                format!(
                    "record number {i} {}",
                    "lorem ipsum dolor sit amet ".repeat(10)
                ),
            )
        })
        .collect();
    let refs: Vec<(&str, &str, &str)> = docs
        .iter()
        .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
        .collect();
    let mut store = seed(dir.path(), &refs);
    store.compact().unwrap();
    drop(store);

    let reopened = VectorStore::open(dir.path()).unwrap();
    let meta_bytes = reopened.meta_bytes();
    let mut emb = DeterministicEmbedder::new(DIMS);
    let q = emb
        .embed(&["record number 1 lorem ipsum dolor".to_string()])
        .unwrap()
        .remove(0);
    let _ = Searcher::new(&reopened, emb.id())
        .unwrap()
        .search(
            &q,
            &QueryOpts {
                k: Some(5),
                ..Default::default()
            },
        )
        .unwrap();

    let Some(after) = rss_kb() else {
        return;
    };
    let growth_kb = after.saturating_sub(before);
    let bound_kb = (meta_bytes / 1024) / 4; // 25% of meta bytes, loose signal
                                            // Diagnostic only, never fatal: whole-process RSS also counts whatever
                                            // sibling tests are concurrently allocating in this same test binary, so
                                            // a bound here would flake on test-harness parallelism rather than catch
                                            // a residency regression. search_touches_the_text_segment_at_most_k_times
                                            // is the real (deterministic) gate for this property.
    if growth_kb > bound_kb.max(4096) {
        eprintln!(
            "note: opening + querying a {meta_bytes}-byte text segment grew \
             RSS by {growth_kb}KB (loose bound {bound_kb}KB) — likely test-binary \
             noise from concurrently running tests, not necessarily a regression"
        );
    }
}

#[test]
fn append_rejects_a_vector_with_the_wrong_dimensions() {
    // The exact reproduction that used to wedge the store permanently: append
    // a wrong-dims vector, then discover the damage only at compact() or
    // query time. It must now be refused at the write itself, and the store
    // must remain fully usable afterward — never wedged.
    let dir = tempfile::tempdir().unwrap();
    let mut store = seed(dir.path(), &[("concept", "a", "alpha beta")]);

    let wrong_dims = vec![0.0f32; DIMS * 2];
    let err = store
        .append(&[PendingOp {
            record: rec("concept", "b", "wrong dims"),
            vector: Some(wrong_dims),
        }])
        .unwrap_err();
    assert!(matches!(err, LastSeekError::DimensionMismatch { .. }));

    // Not wedged: a correctly-dimensioned append and compact still succeed.
    let mut emb = DeterministicEmbedder::new(DIMS);
    let v = emb.embed(&["still fine".to_string()]).unwrap().remove(0);
    store
        .append(&[PendingOp {
            record: rec("concept", "c", "still fine"),
            vector: Some(v),
        }])
        .unwrap();
    store.compact().unwrap();
    assert_eq!(store.committed_rows(), 2);
}

#[test]
fn rebind_a_different_embedder_on_a_non_empty_store_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = seed(dir.path(), &[("concept", "a", "alpha beta")]);

    let err = store
        .bind_embedder(&NamedEmbedder {
            id: "other-embedder".to_string(),
            dims: DIMS,
            is_neural: false,
        })
        .unwrap_err();
    assert!(matches!(err, LastSeekError::EmbedderBoundary(_)));

    let err = store
        .bind_embedder(&NamedEmbedder {
            id: "deterministic-test-fixture".to_string(),
            dims: DIMS * 2,
            is_neural: false,
        })
        .unwrap_err();
    assert!(matches!(err, LastSeekError::EmbedderBoundary(_)));
}

#[test]
fn fixture_vector_into_a_neural_manifest_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = VectorStore::open(dir.path()).unwrap();
    let neural = NamedEmbedder {
        id: "all-MiniLM-L6-v2".to_string(),
        dims: DIMS,
        is_neural: true,
    };
    store.bind_embedder(&neural).unwrap();
    store
        .append(&[PendingOp {
            record: rec("concept", "a", "real"),
            vector: Some(vec![0.1; DIMS]),
        }])
        .unwrap();

    let fixture = DeterministicEmbedder::new(DIMS);
    let err = store.bind_embedder(&fixture).unwrap_err();
    assert!(matches!(err, LastSeekError::EmbedderBoundary(_)));
}

#[test]
fn neural_vector_into_a_fixture_manifest_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = seed(dir.path(), &[("concept", "a", "alpha beta")]);

    let neural = NamedEmbedder {
        id: "all-MiniLM-L6-v2".to_string(),
        dims: DIMS,
        is_neural: true,
    };
    let err = store.bind_embedder(&neural).unwrap_err();
    assert!(matches!(err, LastSeekError::EmbedderBoundary(_)));
}

#[test]
fn query_with_a_mismatched_embedder_id_errors_instead_of_returning_empty() {
    // A store queried with the wrong embedder must not return a confident
    // empty hit list — that is indistinguishable from "no matches" and is
    // exactly the silent-success pattern this boundary exists to prevent.
    let dir = tempfile::tempdir().unwrap();
    let mut store = seed(dir.path(), &[("concept", "a", "alpha beta")]);
    store.compact().unwrap();

    match Searcher::new(&store, "some-other-embedder") {
        Err(LastSeekError::EmbedderBoundary(_)) => {}
        Err(e) => panic!("expected EmbedderBoundary, got {e:?}"),
        Ok(_) => panic!("expected Err, got Ok"),
    }
}

/// Regression gate for the bounded top-k rewrite: `search()` must stay
/// within 3x of a raw dot-product pass over the same corpus. Release-only —
/// a debug build's cost is dominated by unoptimized scalar math and
/// `debug_assert!`s that say nothing about the algorithmic fix (sorting the
/// whole corpus vs. a size-k heap), and `.lastgit/ci.sh` already runs
/// `cargo test --release`.
#[cfg(not(debug_assertions))]
#[test]
fn search_stays_within_3x_of_the_raw_scan() {
    use std::time::Instant;

    const ROWS: usize = 50_000;
    const BENCH_DIMS: usize = 384;
    const BATCH: usize = 5_000;

    let dir = tempfile::tempdir().unwrap();
    let mut emb = DeterministicEmbedder::new(BENCH_DIMS);
    let mut store = VectorStore::open(dir.path()).unwrap();
    store.bind_embedder(&emb).unwrap();

    let mut i = 0;
    while i < ROWS {
        let end = (i + BATCH).min(ROWS);
        let texts: Vec<String> = (i..end)
            .map(|n| format!("record {n} filler benchmark corpus text goes here {n}"))
            .collect();
        let vecs = emb.embed(&texts).unwrap();
        let ops: Vec<PendingOp> = (i..end)
            .zip(&texts)
            .zip(vecs)
            .map(|((n, text), v)| PendingOp {
                record: rec("bench", &format!("k{n}"), text),
                vector: Some(v),
            })
            .collect();
        store.append(&ops).unwrap();
        i = end;
    }
    store.compact().unwrap();
    drop(store);
    let store = VectorStore::open(dir.path()).unwrap();

    let q = emb
        .embed(&["record 42 filler benchmark corpus text goes here 42".to_string()])
        .unwrap()
        .remove(0);

    // The floor cost this query could ever beat: the raw dot-product pass
    // with no ranking machinery at all. Best-of-3 to damp scheduler noise.
    let raw = (0..3)
        .map(|_| {
            let start = Instant::now();
            let mut acc = 0.0f32;
            for i in 0..store.committed_rows() {
                acc += lastseek::index::dot(&q, store.vector_at(i).unwrap());
            }
            std::hint::black_box(acc);
            start.elapsed()
        })
        .min()
        .unwrap();

    let searcher = Searcher::new(&store, emb.id()).unwrap();
    let searched = (0..3)
        .map(|_| {
            let start = Instant::now();
            let hits = searcher
                .search(
                    &q,
                    &QueryOpts {
                        k: Some(20),
                        ..Default::default()
                    },
                )
                .unwrap();
            std::hint::black_box(&hits);
            start.elapsed()
        })
        .min()
        .unwrap();

    assert!(
        searched <= raw * 3,
        "search() took {searched:?} against a {raw:?} raw scan over {ROWS} rows \
         ({:.1}x) — bounded top-k must stay within 3x of the raw dot-product pass",
        searched.as_secs_f64() / raw.as_secs_f64().max(1e-9)
    );
}

#[cfg(feature = "fault-injection")]
mod fault_injection {
    use lastseek::embed::{DeterministicEmbedder, Embedder};
    use lastseek::index::{QueryOpts, Searcher};
    use lastseek::store::fault::{self, FaultPoint};
    use lastseek::store::{PendingOp, Record, VectorStore};

    const DIMS: usize = 32;

    fn rec(schema: &str, key: &str, text: &str) -> Record {
        Record {
            schema_identity: schema.to_string(),
            key_hash: Some(key.to_string()),
            key_range: None,
            fragment_key: "body".to_string(),
            text: text.to_string(),
            mutation_id: None,
            deleted: false,
        }
    }

    /// Every interior point of `compact()` must leave the store either wholly
    /// on the previous generation or wholly on the new one, with every
    /// identity's latest value still findable — never a slab/meta desync,
    /// never a lost op. This is the direct regression test for the bug this
    /// card exists to fix: a crash between the old code's two renames left a
    /// new slab paired with old meta, so `vector_at(i)` and `record_at(i)`
    /// described different documents.
    #[test]
    fn compaction_survives_a_crash_at_every_interior_point() {
        assert_eq!(
            FaultPoint::ALL.len(),
            8,
            "compact() grew a new interior step without a matching fault case"
        );

        let docs = [
            ("concept", "a", "alpha beta"),
            ("concept", "b", "gamma delta"),
            ("design", "c", "epsilon zeta"),
        ];

        for &point in FaultPoint::ALL {
            let dir = tempfile::tempdir().unwrap();
            let mut emb = DeterministicEmbedder::new(DIMS);
            let mut store = VectorStore::open(dir.path()).unwrap();
            store.bind_embedder(&emb).unwrap();
            let texts: Vec<String> = docs.iter().map(|d| d.2.to_string()).collect();
            let vecs = emb.embed(&texts).unwrap();
            let ops: Vec<PendingOp> = docs
                .iter()
                .zip(vecs)
                .map(|(d, v)| PendingOp {
                    record: rec(d.0, d.1, d.2),
                    vector: Some(v),
                })
                .collect();
            store.append(&ops).unwrap();

            fault::arm(point);
            let crashed =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| store.compact())).is_err();
            fault::disarm();
            drop(store);

            let reopened = VectorStore::open(dir.path()).unwrap();
            let mut checker = DeterministicEmbedder::new(DIMS);
            for d in &docs {
                let q = checker.embed(&[d.2.to_string()]).unwrap().remove(0);
                let hits = Searcher::new(&reopened, checker.id())
                    .unwrap()
                    .search(
                        &q,
                        &QueryOpts {
                            exact: Some(d.2.to_string()),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(
                    hits.len(),
                    1,
                    "fault at {point:?} (crashed={crashed}): identity {:?} must \
                     appear exactly once after reopen, found {}",
                    d,
                    hits.len()
                );
                assert_eq!(hits[0].record.text, d.2);
                assert_eq!(hits[0].record.schema_identity, d.0);
            }
        }
    }

    /// Direct regression test for the gap in the crash matrix above: a panic
    /// mid-`compact()` proves nothing about durability, because the OS page
    /// cache survives a process crash whether or not the preceding `fsync`
    /// ran. Deleting any `sync_all()` call from the compaction path left that
    /// test fully green. This test instead counts every durability sync that
    /// actually fires and pins the count, so dropping one fails loudly here
    /// instead of silently passing the crash matrix.
    ///
    /// One full non-empty `compact()` performs exactly 8 syncs:
    /// `build_generation` syncs `vectors.f32`, `rows.bin`, `meta.jsonl`, and
    /// `manifest.json` (4), then `compact()` itself syncs the staging
    /// directory, the directory entry after the generation rename, the
    /// `CURRENT.tmp` file, and the directory entry after the `CURRENT` rename
    /// (4 more).
    #[test]
    fn compaction_performs_every_expected_durability_sync() {
        let dir = tempfile::tempdir().unwrap();
        let mut emb = DeterministicEmbedder::new(DIMS);
        let mut store = VectorStore::open(dir.path()).unwrap();
        store.bind_embedder(&emb).unwrap();
        let docs = [("concept", "a", "alpha"), ("concept", "b", "beta")];
        let texts: Vec<String> = docs.iter().map(|d| d.2.to_string()).collect();
        let vecs = emb.embed(&texts).unwrap();
        let ops: Vec<PendingOp> = docs
            .iter()
            .zip(vecs)
            .map(|(d, v)| PendingOp {
                record: rec(d.0, d.1, d.2),
                vector: Some(v),
            })
            .collect();
        store.append(&ops).unwrap();

        fault::reset_sync_count();
        store.compact().unwrap();
        assert_eq!(
            fault::sync_count(),
            8,
            "compact() performed {} durability syncs, expected exactly 8 — a \
             sync was added or removed from the generation-swap path without \
             updating this pinned count",
            fault::sync_count()
        );
    }
}

// ---------------------------------------------------------------------------
// Schema scope resolution.
//
// These are the assertions `papercut-search-schema-scope-by-name-returns-
// confident-empty` names as missing. The predecessor returned `Ok(vec![])` for
// a mistyped schema and `Ok(vec![])` for an empty one, live, on brain's and
// kanban's own corpora. Nothing in the suite could tell the two apart, so
// nothing failed when they became the same answer.
// ---------------------------------------------------------------------------

/// `/v1/schemas/available` shaped like the live registry: one readable name
/// wearing two identities (as `Card` wears six), a registry `name` that differs
/// from the identity it maps to, and a schema that is real but holds no rows.
const AVAILABLE: &str = r#"{"schemas":[
  {"descriptive_name":"Card","name":"card-registry-1","identity_hash":"card-ident-1"},
  {"descriptive_name":"Card","name":"card-registry-2","identity_hash":"card-ident-2"},
  {"descriptive_name":"Reference","name":"ref-registry","identity_hash":"ref-ident"},
  {"descriptive_name":"Milestone","name":"ms-registry","identity_hash":"ms-ident"}
]}"#;

fn scope_table() -> lastseek::schema::SchemaTable {
    lastseek::schema::SchemaTable::from_available_json(AVAILABLE.as_bytes(), "test").unwrap()
}

/// A corpus keyed the way ingest keys it: on identity hashes, never on readable
/// names. `Card` rows are split across both `Card` identities on purpose.
fn scope_store(dir: &std::path::Path) -> VectorStore {
    let mut store = seed(
        dir,
        &[
            ("card-ident-1", "a", "durable compaction vector store"),
            ("card-ident-2", "b", "durable compaction generation swap"),
            ("ref-ident", "c", "durable compaction reference notes"),
        ],
    );
    store.compact().unwrap();
    store
}

fn scope_query(
    store: &VectorStore,
    terms: &[&str],
) -> Result<Vec<lastseek::index::Hit>, LastSeekError> {
    let mut emb = DeterministicEmbedder::new(DIMS);
    let q = emb
        .embed(&["durable compaction".to_string()])
        .unwrap()
        .remove(0);
    Searcher::new(store, emb.id())
        .unwrap()
        .with_schema_table(scope_table())
        .search(
            &q,
            &QueryOpts {
                schemas: terms.iter().map(|s| s.to_string()).collect(),
                ..Default::default()
            },
        )
}

#[test]
fn scoping_by_a_readable_name_finds_every_identity_wearing_it() {
    // The live measurement this replaces: `--schema Card` returned 0 hits
    // while the same query scoped by the interned hash returned correctly
    // ranked results.
    let dir = tempfile::tempdir().unwrap();
    let store = scope_store(dir.path());
    let hits = scope_query(&store, &["Card"]).unwrap();
    assert_eq!(hits.len(), 2, "both Card identities must answer");
    let mut got: Vec<&str> = hits
        .iter()
        .map(|h| h.record.schema_identity.as_str())
        .collect();
    got.sort();
    assert_eq!(got, vec!["card-ident-1", "card-ident-2"]);
}

#[test]
fn a_wrong_schema_name_and_an_empty_schema_are_different_observables() {
    let dir = tempfile::tempdir().unwrap();
    let store = scope_store(dir.path());

    // Real schema, no rows in this corpus: an honest empty answer.
    let empty = scope_query(&store, &["Milestone"]).unwrap();
    assert!(empty.is_empty(), "Milestone holds no rows here");

    // Not a schema at all: an error naming what is known. This is the whole
    // defect — before this, both of these were `Ok(vec![])`.
    match scope_query(&store, &["Crad"]) {
        Err(LastSeekError::UnknownSchema { name, known }) => {
            assert_eq!(name, "Crad");
            assert!(!known.is_empty(), "the error must say what IS known");
        }
        other => panic!("expected UnknownSchema, got {other:?}"),
    }
}

#[test]
fn scoping_by_a_registry_name_resolves_through_to_the_identity() {
    // A caller holding the registry `name` (what the host's IndexChangeBatch
    // files carry) must reach the rows, which are keyed on the identity hash.
    let dir = tempfile::tempdir().unwrap();
    let store = scope_store(dir.path());
    let hits = scope_query(&store, &["card-registry-2"]).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.schema_identity, "card-ident-2");
}

#[test]
fn scoping_by_an_identity_hash_still_works_without_any_table() {
    // No table at all: hash-scoped and unscoped queries must not need one, or
    // a missing table would take down the whole query surface.
    let dir = tempfile::tempdir().unwrap();
    let store = scope_store(dir.path());
    let mut emb = DeterministicEmbedder::new(DIMS);
    let q = emb
        .embed(&["durable compaction".to_string()])
        .unwrap()
        .remove(0);
    let hits = Searcher::new(&store, emb.id())
        .unwrap()
        .search(
            &q,
            &QueryOpts {
                schemas: vec!["card-ident-1".to_string()],
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.schema_identity, "card-ident-1");
}

#[test]
fn a_readable_name_without_a_table_errors_rather_than_matching_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let store = scope_store(dir.path());
    let mut emb = DeterministicEmbedder::new(DIMS);
    let q = emb
        .embed(&["durable compaction".to_string()])
        .unwrap()
        .remove(0);
    let got = Searcher::new(&store, emb.id()).unwrap().search(
        &q,
        &QueryOpts {
            schemas: vec!["Card".to_string()],
            ..Default::default()
        },
    );
    // An unconfigured table is a config error, and it says so. Silently
    // returning no hits is what made the predecessor's missing table invisible
    // for as long as it was missing.
    assert!(matches!(got, Err(LastSeekError::UnknownSchema { .. })));
}

#[test]
fn an_uncompacted_row_is_reachable_by_readable_name() {
    // Pending ops are scoped separately from committed rows. Comparing the
    // pending record's identity key against the caller's raw term would drop
    // every uncompacted row from a name-scoped query — a window that lasts
    // exactly as long as the pending log does.
    let dir = tempfile::tempdir().unwrap();
    let mut store = scope_store(dir.path());
    let mut emb = DeterministicEmbedder::new(DIMS);
    let v = emb
        .embed(&["durable compaction fresh pending row".to_string()])
        .unwrap()
        .remove(0);
    store
        .append(&[PendingOp {
            record: rec(
                "ms-ident",
                "pending",
                "durable compaction fresh pending row",
            ),
            vector: Some(v),
        }])
        .unwrap();

    let hits = scope_query(&store, &["Milestone"]).unwrap();
    assert_eq!(hits.len(), 1, "the pending Milestone row must be reachable");
    assert_eq!(hits[0].record.schema_identity, "ms-ident");
}
