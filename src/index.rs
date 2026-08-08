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
//!
//! The scan itself never touches `meta.jsonl`: candidate scoring, the
//! tombstone check and the schema-scope check all read out of `rows.bin`
//! (`RowMeta`), which is fixed-width and mmap'd. A record's text is parsed
//! only for rows that survive sorting into the final `k`, which is what makes
//! the store's residency property observable from `Searcher` rather than just
//! from `VectorStore` in isolation.

use crate::store::{id_hash, PendingOp, Record, VectorStore};
use crate::LastSeekError;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

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
    debug_assert_eq!(a.len(), b.len());
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
    /// Bind a searcher to `store`, refusing a querying embedder that does not
    /// match the one the store was built with. A query embedded with the
    /// wrong embedder is not a corpus with zero matches — it is a config
    /// error, and returning an empty hit list for it is exactly the
    /// confident-but-wrong answer this boundary exists to prevent.
    pub fn new(store: &'a VectorStore, embedder_id: &str) -> Result<Self, LastSeekError> {
        if !store.embedder_id().is_empty() && store.embedder_id() != embedder_id {
            return Err(LastSeekError::EmbedderBoundary(format!(
                "store is bound to embedder {:?}; refusing to query it with embedder {:?}",
                store.embedder_id(),
                embedder_id
            )));
        }
        Ok(Self { store })
    }

    /// Score `query` against the corpus and return the best `k`.
    ///
    /// Reads vectors and row metadata straight out of their mappings and
    /// touches a record's text only for rows that survive into the result
    /// set.
    pub fn search(&self, query: &[f32], opts: &QueryOpts) -> Result<Vec<Hit>, LastSeekError> {
        let dims = self.store.dimensions();
        if dims != 0 && query.len() != dims {
            return Err(LastSeekError::DimensionMismatch {
                expected: dims,
                got: query.len(),
            });
        }
        let k = opts.k.unwrap_or(20).max(1);
        let min = opts.min_score.unwrap_or(f32::NEG_INFINITY);
        let scoped = !opts.schemas.is_empty();
        // Committed rows whose schema isn't interned in this generation at
        // all can never match a scoped query, so an empty set (rather than
        // None) correctly excludes every committed row for that case.
        let scope_ids: Option<HashSet<u32>> = scoped.then(|| {
            opts.schemas
                .iter()
                .filter_map(|s| self.store.schema_id(s))
                .collect()
        });

        // Pending ops shadow committed rows, so collect their identity
        // hashes first and let the pending value win.
        let mut shadowed: HashSet<u64> = HashSet::new();
        for op in self.store.pending() {
            shadowed.insert(id_hash(&op.record.id()));
        }

        // `--exact` stays structural: a candidate must pass it to occupy a
        // heap slot at all, so a low-scoring exact match can still displace
        // a higher-scoring non-match. The text read that requires is paid
        // only for candidates that clear the current heap's score bar (see
        // `consider`), not for every row in the corpus.
        let exact = opts.exact.as_ref().map(|q| q.to_lowercase());

        // Bounded top-k: a min-heap capped at size k, so ranking the corpus
        // costs O(n log k) instead of sorting all n scored candidates.
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);

        for i in 0..self.store.committed_rows() {
            let Some(row) = self.store.row_at(i) else {
                continue;
            };
            if row.deleted() || shadowed.contains(&row.id_hash) {
                continue;
            }
            if let Some(ids) = &scope_ids {
                if !ids.contains(&row.schema_id) {
                    continue;
                }
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
            self.consider(
                &mut heap,
                k,
                s,
                row.id_hash,
                ScoredRef::Committed(i),
                &exact,
            );
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
            let idh = id_hash(&op.record.id());
            self.consider(&mut heap, k, s, idh, ScoredRef::Pending(j), &exact);
        }

        // Deterministic order: score desc, then identity hash, so equal
        // scores do not reorder between runs. Compares hashes rather than
        // strings so the sort itself never touches meta.jsonl either.
        let mut winners: Vec<Candidate> = heap.into_vec();
        winners.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id_hash.cmp(&b.id_hash))
        });

        Ok(winners
            .into_iter()
            .filter_map(|c| {
                let record = c.record.or_else(|| self.record_of(&c.r))?;
                Some(Hit {
                    score: c.score,
                    record,
                })
            })
            .collect())
    }

    /// Admit `r` into the bounded top-k heap if it beats the current worst
    /// kept candidate (or the heap isn't full yet). When `--exact` is set,
    /// the text read and match check happen here — only for candidates that
    /// clear the score bar — so a non-matching high scorer never pays for a
    /// text read it doesn't need, and a matching low scorer still gets a
    /// fair look for as long as a heap slot is contested.
    fn consider(
        &self,
        heap: &mut BinaryHeap<Candidate>,
        k: usize,
        score: f32,
        id_hash: u64,
        r: ScoredRef,
        exact: &Option<String>,
    ) {
        let in_contention = heap.len() < k
            || heap
                .peek()
                .map(|worst| {
                    score > worst.score || (score == worst.score && id_hash < worst.id_hash)
                })
                .unwrap_or(true);
        if !in_contention {
            return;
        }

        let record = match exact {
            Some(q) => {
                let Some(record) = self.record_of(&r) else {
                    return;
                };
                if !record.text.to_lowercase().contains(q) {
                    return;
                }
                Some(record)
            }
            None => None,
        };

        if heap.len() >= k {
            heap.pop();
        }
        heap.push(Candidate {
            score,
            id_hash,
            r,
            record,
        });
    }

    fn record_of(&self, r: &ScoredRef) -> Option<Record> {
        match r {
            ScoredRef::Committed(i) => self.store.record_at(*i),
            ScoredRef::Pending(j) => self
                .store
                .pending()
                .get(*j)
                .map(|op: &PendingOp| op.record.clone()),
        }
    }
}

enum ScoredRef {
    Committed(usize),
    Pending(usize),
}

/// One kept top-k slot. `record` caches the text read from an `--exact`
/// check so the final hydration pass never reads a row's text twice.
struct Candidate {
    score: f32,
    id_hash: u64,
    r: ScoredRef,
    record: Option<Record>,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.id_hash == other.id_hash
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    /// Orders by "worseness" so `BinaryHeap::peek()` returns the current
    /// worst kept candidate in O(1): a lower score is worse (`Greater`), and
    /// for equal scores a larger `id_hash` is worse — the same tie-break the
    /// final sort applies, so the heap only ever evicts the slot the full
    /// sort would have dropped too.
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .reverse()
            .then_with(|| self.id_hash.cmp(&other.id_hash))
    }
}
