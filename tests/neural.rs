//! BGE neural-embedder tests.
//!
//! Gated on the `neural` feature at the file level, so a default `cargo test`
//! (no `--features neural`) compiles this into an empty test binary that
//! trivially passes — CI stays offline and links no inference runtime.
//! Individually, each test also skips (rather than fails) if model weights
//! are not present at `BgeSmallEmbedder::default_model_dir()`, since CI must
//! not require a network fetch of weights to stay green.
#![cfg(feature = "neural")]

use lastseek::embed::{BgeSmallEmbedder, DeterministicEmbedder, Embedder};
use lastseek::store::{PendingOp, Record, VectorStore};

fn load_or_skip() -> Option<BgeSmallEmbedder> {
    let dir = BgeSmallEmbedder::default_model_dir();
    let needed = ["model.onnx", "tokenizer.json", "config.json"];
    if needed.iter().any(|f| !dir.join(f).is_file()) {
        eprintln!(
            "skipping: no bge-small-en-v1.5 weights at {} (set LASTSEEK_MODEL_DIR or fetch them first)",
            dir.display()
        );
        return None;
    }
    Some(BgeSmallEmbedder::load(&dir).expect("weights present but failed to load"))
}

#[test]
fn embedding_the_same_text_twice_is_bit_identical() {
    let Some(mut emb) = load_or_skip() else {
        return;
    };
    let a = emb.embed(&["the quick brown fox".to_string()]).unwrap();
    let b = emb.embed(&["the quick brown fox".to_string()]).unwrap();
    assert_eq!(a, b, "inference must be deterministic for identical input");
}

#[test]
fn vectors_are_the_declared_dimensionality_and_normalised() {
    let Some(mut emb) = load_or_skip() else {
        return;
    };
    let v = emb
        .embed(&["LastDB is a Dynamo-style key-value store".to_string()])
        .unwrap()
        .remove(0);
    assert_eq!(v.len(), BgeSmallEmbedder::DIMENSIONS);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "expected a unit vector, got norm {norm}"
    );
}

#[test]
fn query_and_document_paths_produce_different_vectors_for_the_same_text() {
    let Some(mut emb) = load_or_skip() else {
        return;
    };
    let text = "how do I ship a new build to production".to_string();
    let doc = emb.embed(std::slice::from_ref(&text)).unwrap().remove(0);
    let query = emb.embed_query(&[text]).unwrap().remove(0);
    assert_ne!(
        doc, query,
        "BGE is asymmetric: the query instruction prefix must change the vector"
    );
}

#[test]
fn round_trip_through_the_store_preserves_neural_vectors_exactly() {
    let Some(mut emb) = load_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let mut store = VectorStore::open(dir.path()).unwrap();
    store.bind_embedder(&emb).unwrap();

    let v = emb.embed(&["exact round trip check".to_string()]).unwrap();
    store
        .append(&[PendingOp {
            record: Record {
                schema_identity: "concept".into(),
                key_hash: Some("a".into()),
                key_range: None,
                fragment_key: "body".into(),
                text: "exact round trip check".into(),
                mutation_id: None,
                deleted: false,
            },
            vector: Some(v[0].clone()),
        }])
        .unwrap();
    store.compact().unwrap();

    assert_eq!(store.vector_at(0).unwrap(), v[0].as_slice());
}

#[test]
fn a_neural_store_refuses_a_fixture_vector() {
    let Some(mut emb) = load_or_skip() else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let mut store = VectorStore::open(dir.path()).unwrap();
    store.bind_embedder(&emb).unwrap();
    let v = emb.embed(&["warm the session".to_string()]).unwrap();
    store
        .append(&[PendingOp {
            record: Record {
                schema_identity: "concept".into(),
                key_hash: Some("a".into()),
                key_range: None,
                fragment_key: "body".into(),
                text: "warm the session".into(),
                mutation_id: None,
                deleted: false,
            },
            vector: Some(v[0].clone()),
        }])
        .unwrap();

    let fixture = DeterministicEmbedder::new(emb.dimensions());
    let err = store.bind_embedder(&fixture).unwrap_err();
    assert!(err.to_string().contains("refus"));
}
