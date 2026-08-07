//! Schema-scoped k-nearest-neighbour search over the mapped slab.
//!
//! Scope is **structural**: when a caller names schemas, only those schemas'
//! rows are scored. It is never a post-filter over a global top-k, because that
//! silently returns fewer than `k` results — or none — whenever the scoped
//! schema is not already in the global head. That distinction is a correctness
//! property, not an optimisation, and it is inherited deliberately from the
//! predecessor's design.
//!
//! Scoring is a linear pass today. That is the right default: at corpus sizes
//! where a scan is fast, an approximate index only adds recall risk and build
//! cost. `Scorer` exists so an ANN backend can replace the pass when the corpus
//! earns it, without moving the scope semantics.

use crate::store::{PendingOp, Record, VectorStore};

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub score: f32,
    pub record: Record,
}

#[derive(Debug, Clone, Default)]
pub struct QueryOpts {
    pub k: Option<usize>,
    /// Structural allow-list. Empty means every schema.
    pub schemas: Vec<String>,
    pub min_score: Option<f32>,
    /// Keep only hits whose text contains the query, case-insensitively.
    pub exact: Option<String>,
}

/// Cosine similarity for L2-normalised vectors, which is their dot product.
///
/// Written as a plain indexed loop so LLVM can autovectorise it; the bounds
/// check is hoisted by the length assertion.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// L2-normalise in place. Vectors are stored normalised so query time is a dot
/// product rather than a division per candidate.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub struct Searcher<'a> {
    store: &'a VectorStore,
}

impl<'a> Searcher<'a> {
    pub fn new(store: &'a VectorStore) -> Self {
        Self { store }
    }

    /// Score `query` against the corpus and return the best `k`.
    ///
    /// Reads vectors straight out of the mapping and touches a record's text
    /// only for rows that survive into the result set.
    pub fn search(&self, query: &[f32], opts: &QueryOpts) -> Vec<Hit> {
        let k = opts.k.unwrap_or(20).max(1);
        let min = opts.min_score.unwrap_or(f32::NEG_INFINITY);
        let scoped = !opts.schemas.is_empty();

        // Pending ops shadow committed rows, so collect their identities first
        // and let the pending value win.
        let mut shadowed: std::collections::HashSet<String> = std::collections::HashSet::new();
        for op in self.store.pending() {
            shadowed.insert(op.record.id());
        }

        let mut scored: Vec<(f32, ScoredRef)> = Vec::new();

        for i in 0..self.store.committed_rows() {
            let Some(rec) = self.store.record_at(i) else {
                continue;
            };
            if rec.deleted || shadowed.contains(&rec.id()) {
                continue;
            }
            if scoped && !opts.schemas.iter().any(|s| s == &rec.schema_name) {
                continue;
            }
            let Some(v) = self.store.vector_at(i) else {
                continue;
            };
            if v.len() != query.len() {
                continue;
            }
            let s = dot(query, v);
            if s < min {
                continue;
            }
            scored.push((s, ScoredRef::Committed(i)));
        }

        for (j, op) in self.store.pending().iter().enumerate() {
            if op.record.deleted {
                continue;
            }
            let Some(v) = op.vector.as_ref() else {
                continue;
            };
            if v.len() != query.len() {
                continue;
            }
            if scoped && !opts.schemas.iter().any(|s| s == &op.record.schema_name) {
                continue;
            }
            let s = dot(query, v);
            if s < min {
                continue;
            }
            scored.push((s, ScoredRef::Pending(j)));
        }

        // Deterministic order: score desc, then identity, so equal scores do
        // not reorder between runs.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.id_of(&a.1).cmp(&self.id_of(&b.1)))
        });

        scored
            .into_iter()
            .filter_map(|(score, r)| {
                let record = self.record_of(&r)?.clone();
                Some(Hit { score, record })
            })
            .filter(|h| match &opts.exact {
                Some(q) => h.record.text.to_lowercase().contains(&q.to_lowercase()),
                None => true,
            })
            .take(k)
            .collect()
    }

    fn record_of(&self, r: &ScoredRef) -> Option<&Record> {
        match r {
            ScoredRef::Committed(i) => self.store.record_at(*i),
            ScoredRef::Pending(j) => self
                .store
                .pending()
                .get(*j)
                .map(|op: &PendingOp| &op.record),
        }
    }

    fn id_of(&self, r: &ScoredRef) -> String {
        self.record_of(r).map(|x| x.id()).unwrap_or_default()
    }
}

enum ScoredRef {
    Committed(usize),
    Pending(usize),
}
