//! Embedders.
//!
//! The neural embedder is deliberately behind a trait and an optional feature.
//! CI must be able to prove the store and the search semantics without
//! downloading model weights or linking an inference runtime — the predecessor
//! made the same call and it is the reason its tests stayed fast and offline.
//!
//! A deterministic embedder is a **test fixture only**. The predecessor let
//! deterministic vectors reach the production index, where 2,021 of them ended
//! up sharing a space with neural ones and being scored against them as if they
//! were comparable. `Embedder::is_neural` exists so that mixing is refusable at
//! the boundary rather than discovered later in a coverage audit.

use crate::LastSeekError;

pub trait Embedder {
    /// Identity recorded in the manifest. Vectors from different ids are not
    /// comparable and must never share a store.
    fn id(&self) -> &str;
    fn dimensions(&self) -> usize;
    /// False for fixtures. A production index refuses these.
    fn is_neural(&self) -> bool;
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, LastSeekError>;
}

/// Hash-based stand-in with the right shape and no semantics.
///
/// Exists so tests can exercise storage, scope and ranking offline. It is not
/// an approximation of a language model and must never be written to a
/// production store.
pub struct DeterministicEmbedder {
    dims: usize,
}

impl DeterministicEmbedder {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Embedder for DeterministicEmbedder {
    fn id(&self) -> &str {
        "deterministic-test-fixture"
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn is_neural(&self) -> bool {
        false
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, LastSeekError> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; self.dims];
                // Token-level hashing, so shared words produce genuine overlap
                // and ranking tests are meaningful rather than arbitrary.
                for tok in t.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
                    if tok.is_empty() {
                        continue;
                    }
                    let mut h: u64 = 1469598103934665603;
                    for b in tok.as_bytes() {
                        h ^= *b as u64;
                        h = h.wrapping_mul(1099511628211);
                    }
                    v[(h % self.dims as u64) as usize] += 1.0;
                }
                crate::index::normalize(&mut v);
                v
            })
            .collect())
    }
}
