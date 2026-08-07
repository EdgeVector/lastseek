//! Store and search behaviour.
//!
//! Several of these exist because the predecessor got them wrong in production:
//! whole-corpus rewrites per flush, a log that was unreadable until a snapshot
//! existed, and scope implemented as a post-filter.

use lastseek::embed::{DeterministicEmbedder, Embedder};
use lastseek::index::{QueryOpts, Searcher};
use lastseek::store::{PendingOp, Record, VectorStore};

const DIMS: usize = 64;

fn rec(schema: &str, key: &str, text: &str) -> Record {
    Record {
        schema_name: schema.to_string(),
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
    store.set_embedder(emb.id(), emb.dimensions()).unwrap();
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
    let hits = Searcher::new(&store).search(&q, &QueryOpts::default());
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

    let scoped = Searcher::new(&store).search(
        &q,
        &QueryOpts {
            schemas: vec!["sop".to_string()],
            ..Default::default()
        },
    );
    assert_eq!(scoped.len(), 1, "scoped search must find the sop row");
    assert_eq!(scoped[0].record.schema_name, "sop");
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

    let hits = Searcher::new(&store).search(&v, &QueryOpts::default());
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
