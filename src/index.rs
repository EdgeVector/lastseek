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
//! (`RowMeta`), which is fixed-width and mmap'd. A record's identity line is
//! parsed only for rows that survive sorting into the final `k`, which is
//! what makes the store's residency property observable from `Searcher`
//! rather than just from `VectorStore` in isolation.
//!
//! A hit carries identity and score, never text: the index does not retain
//! record bodies (see `store` module doc). There is consequently no exact
//! substring mode here — a caller that needs one hydrates the hits from
//! LastDB and filters what it is entitled to read.

use crate::schema::{ResolvedScope, SchemaTable};
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
    /// Structural allow-list of **caller-facing** scope terms — a readable
    /// descriptive name (`Card`), an identity hash, or a registry `name`.
    /// Empty means every schema. Resolution happens in `Searcher::search`
    /// against the schema table, and an unresolvable term is an error there,
    /// never a silently dropped filter term.
    pub schemas: Vec<String>,
    pub min_score: Option<f32>,
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
    /// Schema table used to resolve caller-facing scope terms. Default-empty,
    /// which still resolves any schema the store has interned — so an
    /// unscoped or hash-scoped query needs no table — while a readable name
    /// requires one and says so instead of matching nothing.
    table: SchemaTable,
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
        Ok(Self {
            store,
            table: SchemaTable::default(),
        })
    }

    /// Attach the Schema Service table, which is what lets a caller scope by a
    /// readable name.
    pub fn with_schema_table(mut self, table: SchemaTable) -> Self {
        self.table = table;
        self
    }

    /// Resolve `opts.schemas` to the identity keys the index is filtered on.
    ///
    /// Every term must resolve — via the SS table or as a schema this store has
    /// interned — or the whole query fails. Terms that resolve to identities
    /// this store holds no rows for are kept: they scope to nothing and the
    /// query legitimately returns no hits, which is the observable that must
    /// stay distinct from an unknown name.
    fn resolve_scope(&self, opts: &QueryOpts) -> Result<Vec<ResolvedScope>, LastSeekError> {
        if opts.schemas.is_empty() {
            return Ok(Vec::new());
        }
        self.table
            .resolve_all(&opts.schemas, self.store.interned_schemas())
    }

    /// Score `query` against the corpus and return the best `k`.
    ///
    /// Reads vectors and row metadata straight out of their mappings and
    /// touches a record's identity line only for rows that survive into the
    /// result set.
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

        // Resolve first, so an unresolvable scope term fails the query before
        // any scoring happens. This replaces a `filter_map` that dropped an
        // unresolved term from the filter set, leaving the query to report
        // `Ok(vec![])` — indistinguishable from an empty schema.
        let resolved = self.resolve_scope(opts)?;
        let scope_identities: HashSet<&str> = resolved
            .iter()
            .flat_map(|r| r.identities.iter().map(String::as_str))
            .collect();
        // Identities that resolved but aren't interned in this generation
        // simply contribute no ids: the schema is real and holds no rows here,
        // which is a legitimate empty answer rather than a config error.
        let scope_ids: Option<HashSet<u32>> = scoped.then(|| {
            scope_identities
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
            self.consider(&mut heap, k, s, row.id_hash, ScoredRef::Committed(i));
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
            // Pending ops are scoped against the resolved identity set, not
            // the raw terms: a pending row carries the identity key ingest
            // wrote, so comparing it to a readable name would exclude every
            // uncompacted row from a name-scoped query.
            if scoped && !scope_identities.contains(op.record.schema_identity.as_str()) {
                continue;
            }
            let s = dot(query, v);
            if s < min {
                continue;
            }
            let idh = id_hash(&op.record.id());
            self.consider(&mut heap, k, s, idh, ScoredRef::Pending(j));
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
                let record = self.record_of(&c.r)?;
                Some(Hit {
                    score: c.score,
                    record,
                })
            })
            .collect())
    }

    /// Admit `r` into the bounded top-k heap if it beats the current worst
    /// kept candidate (or the heap isn't full yet). Nothing here reads
    /// `meta.jsonl`; hydration happens once, for the final `k`, in `search`.
    fn consider(
        &self,
        heap: &mut BinaryHeap<Candidate>,
        k: usize,
        score: f32,
        id_hash: u64,
        r: ScoredRef,
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

        if heap.len() >= k {
            heap.pop();
        }
        heap.push(Candidate { score, id_hash, r });
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

/// One kept top-k slot.
struct Candidate {
    score: f32,
    id_hash: u64,
    r: ScoredRef,
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
