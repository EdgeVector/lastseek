//! Ingest — turning LastDB's index-change feed into indexable records.
//!
//! # There is no backfill scan, and there cannot be one
//!
//! LastDB is Dynamo-shaped: point get, range under one hash, **no scan**. So
//! "page every schema to build the first index" is not a path LastSeek is
//! allowed to have. Ingest is therefore two things, neither of which enumerates
//! LastDB:
//!
//! - **Drain** (this module): the host's outbox sink writes one
//!   `IndexChangeBatch` JSON file per mutation into
//!   `{LASTDB_HOME}/apps/search/inbox/`. LastSeek reads that directory. It is a
//!   push feed off the write path, which is why this crate has no LastDB client
//!   in `Cargo.toml` — the "app on top, from the outside" boundary is literal.
//! - **Backfill**: the owning app feeds its own corpus to `lastseek add` as
//!   JSONL. brain and kanban can enumerate their records without a scan because
//!   they dual-write the secondaries that make enumeration keyed. A central
//!   backfill would have to guess, for every app forever, which fields are
//!   prose.
//!
//! # Why the drain is bounded and non-destructive
//!
//! **Bounded** because the predecessor's was not: on 2026-08-07 an unbounded
//! drain over a large backlog wrote 549.80 GB in twenty minutes and took the
//! machine down. A drain here is a repeatable unit of work with a file cap that
//! reports what it left behind. A capped run that stays silent reads as
//! "finished", which is how that backlog reached 99k files unnoticed.
//!
//! **Non-destructive by default** because the incumbent Search app is still
//! live and reads the same inbox. LastSeek advances a watermark over the batch
//! filenames rather than deleting them; only `--consume` removes files.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::schema::SchemaTable;
use crate::store::Record;
use crate::LastSeekError;

/// Per-run file cap. See the module doc — the number matters less than the
/// existence of a bound and a report of what was left.
pub const DEFAULT_MAX_FILES: usize = 2000;

/// Fields whose classification bars them from an index. Mirrors the host's
/// native-index classifications, so a field the node considers secret does not
/// become searchable prose just because it travelled through a different app.
const BLOCKED_CLASSIFICATIONS: &[&str] = &["secret", "no_index", "no-index"];

#[derive(Debug, Clone, Deserialize)]
pub struct KeyValue {
    #[serde(default)]
    pub hash: Option<String>,
    #[serde(default)]
    pub range: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IndexChange {
    #[serde(default)]
    pub mutation_id: Option<String>,
    /// `upsert` or `tombstone`. Anything else is treated as an upsert only if
    /// it carries fields; see `Batch::records`.
    #[serde(default)]
    pub kind: String,
    pub key_value: KeyValue,
    #[serde(default)]
    pub fields_and_values: Option<BTreeMap<String, serde_json::Value>>,
}

/// One host-written batch.
///
/// `schema_name` here is the registry `name`, **not** an identity hash — that
/// is measured, not assumed (see `crate::schema`). Translating it is the whole
/// job of `Batch::records`.
#[derive(Debug, Clone, Deserialize)]
pub struct Batch {
    pub schema_name: String,
    #[serde(default)]
    pub searchable_fields: Option<Vec<String>>,
    #[serde(default)]
    pub classifications: Option<BTreeMap<String, Vec<String>>>,
    #[serde(default)]
    pub changes: Vec<IndexChange>,
}

/// True when a field's classifications permit indexing.
///
/// No classifications means allow: the host omits them for most schemas, and
/// defaulting to deny would empty the index. `searchable_fields`, when the batch
/// carries it, is the narrower gate.
fn field_is_indexable(classes: Option<&Vec<String>>) -> bool {
    let Some(classes) = classes else {
        return true;
    };
    if classes.is_empty() {
        return true;
    }
    let lower: Vec<String> = classes.iter().map(|c| c.to_lowercase()).collect();
    if lower
        .iter()
        .any(|c| BLOCKED_CLASSIFICATIONS.contains(&c.as_str()))
    {
        return false;
    }
    lower.iter().any(|c| c == "word")
}

/// Render a record's indexable fields as the text that gets embedded.
///
/// Values are joined by newline in key order. Key order — rather than the map's
/// arrival order — because the same record re-ingested must produce the same
/// text and therefore the same vector; an order that varies would make
/// re-embedding produce spurious diffs.
fn fields_to_text(
    fields: &BTreeMap<String, serde_json::Value>,
    searchable: Option<&Vec<String>>,
    classifications: Option<&BTreeMap<String, Vec<String>>>,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (k, v) in fields {
        if let Some(allow) = searchable {
            // `Some([])` is the host's explicit "this schema has no
            // searchable fields" signal. Treating it like `None` indexed
            // every field of internal bookkeeping schemas instead.
            if !allow.contains(k) {
                continue;
            }
        }
        if !field_is_indexable(classifications.and_then(|c| c.get(k))) {
            continue;
        }
        match v {
            serde_json::Value::Null => {}
            serde_json::Value::String(s) => parts.push(s.clone()),
            serde_json::Value::Bool(b) => parts.push(b.to_string()),
            serde_json::Value::Number(n) => parts.push(n.to_string()),
            other => parts.push(other.to_string()),
        }
    }
    parts.join("\n")
}

/// Why a batch produced no records. Reported rather than swallowed: a drain that
/// silently skipped everything looks exactly like a drain with nothing to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// The Schema Service table does not know this batch's registry `name`.
    SchemaUnknownToSchemaService,
}

#[derive(Debug, Clone)]
pub struct BatchRecords {
    pub records: Vec<Record>,
    pub tombstones: Vec<Record>,
    pub skipped: Option<SkipReason>,
    /// Upserts suppressed because the owner declared zero searchable fields.
    /// Count rows, not batches, so status remains meaningful when batches are
    /// coalesced differently by the host.
    pub non_searchable_records_skipped: usize,
}

impl Batch {
    /// Translate this batch into records keyed on a canonical identity.
    ///
    /// The registry `name` becomes the Schema Service `identity_hash`. When SS
    /// does not know the schema, the batch is still indexed — under the registry
    /// `name` — and reported as `SchemaUnknownToSchemaService`. Dropping it would
    /// make "any app can search its own records" false for every app that has
    /// not published to SS, of which there were 71 on the live node when this
    /// was written.
    pub fn records(&self, table: &SchemaTable) -> BatchRecords {
        let (identity, skipped) = match table.identity_for_name(&self.schema_name) {
            Some(id) => (id.to_string(), None),
            None => (
                self.schema_name.clone(),
                Some(SkipReason::SchemaUnknownToSchemaService),
            ),
        };

        let mut records = Vec::new();
        let mut tombstones = Vec::new();
        let mut non_searchable_records_skipped = 0usize;
        let schema_is_non_searchable = self
            .searchable_fields
            .as_ref()
            .is_some_and(|fields| fields.is_empty());
        for change in &self.changes {
            let base = Record {
                schema_identity: identity.clone(),
                key_hash: change.key_value.hash.clone(),
                key_range: change.key_value.range.clone(),
                // One vector per record. `fragment_key` exists so a future
                // per-field or per-chunk split does not need a new identity
                // scheme; "body" is the whole record, which is what the
                // predecessor also wrote.
                fragment_key: "body".to_string(),
                text: String::new(),
                mutation_id: change.mutation_id.clone(),
                deleted: false,
            };
            if change.kind == "tombstone" {
                tombstones.push(Record {
                    deleted: true,
                    ..base
                });
                continue;
            }
            if schema_is_non_searchable {
                // Delete any copy indexed before the owner applied the
                // policy. Replaying an upsert may therefore clean old corpus
                // state without a LastDB scan or a LastSeek-wide rebuild.
                tombstones.push(Record {
                    deleted: true,
                    ..base
                });
                non_searchable_records_skipped += 1;
                continue;
            }
            let Some(fields) = change.fields_and_values.as_ref() else {
                continue;
            };
            let text = fields_to_text(
                fields,
                self.searchable_fields.as_ref(),
                self.classifications.as_ref(),
            );
            if text.trim().is_empty() {
                // Nothing to embed. Not an error and not a skip worth
                // reporting: a record whose only fields are numbers and ids
                // legitimately has no prose.
                continue;
            }
            records.push(Record { text, ..base });
        }
        BatchRecords {
            records,
            tombstones,
            skipped,
            non_searchable_records_skipped,
        }
    }
}

/// How far the drain has read.
///
/// A watermark rather than deletion, because the incumbent Search app reads the
/// same inbox and is still live. Batch filenames are `<epoch_ms>_<uuid>.json`,
/// so the time prefix gives a total order that is stable across runs — and
/// stays stable while the millisecond count is 13 digits, i.e. until 2286.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Watermark {
    /// Highest batch filename applied. Empty means nothing has been applied.
    #[serde(default)]
    pub applied_through: String,
    #[serde(default)]
    pub files_applied: u64,
    /// Cumulative owner-declared non-searchable upserts suppressed by drain.
    /// Persisted beside the watermark so `lastseek status` can expose the
    /// policy even when no drain is currently running.
    #[serde(default)]
    pub non_searchable_records_skipped: u64,
}

impl Watermark {
    pub fn path(home: &Path) -> PathBuf {
        home.join("drain.watermark.json")
    }

    pub fn load(home: &Path) -> Self {
        std::fs::read(Self::path(home))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    /// Lives under the store root, so it is written the way every other
    /// file there is: owner-only at creation, never at the umask.
    pub fn save(&self, home: &Path) -> Result<(), LastSeekError> {
        crate::store::create_private_dir_all(home)?;
        crate::store::write_private_atomic(&Self::path(home), &serde_json::to_vec_pretty(self)?)
    }
}

/// Default inbox: the one the host's outbox sink already writes to.
///
/// Deliberately `apps/search/inbox` and not a LastSeek-specific path. The sink's
/// path lives in fold; changing it is a fold change and a separate slice, and
/// until then a private inbox would simply never receive anything.
pub fn default_inbox() -> PathBuf {
    if let Ok(v) = std::env::var("LASTSEEK_INBOX") {
        return v.into();
    }
    let home = std::env::var("LASTDB_HOME")
        .or_else(|_| std::env::var("FOLDDB_HOME"))
        .unwrap_or_else(|_| {
            let base = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            format!("{base}/.lastdb")
        });
    Path::new(&home).join("apps").join("search").join("inbox")
}

/// The epoch-millisecond prefix of a batch filename.
///
/// Batch files are named `<epoch_ms>_<uuid>.json`, which is what makes the
/// watermark a total order. The same prefix is the only clock LastSeek has for
/// how old the work it has applied is — the host writes the file when it
/// captures the mutation, so `now - prefix` is the age of the newest change in
/// the index, without asking the node anything.
pub fn batch_epoch_ms(name: &str) -> Option<u64> {
    name.split('_').next()?.parse::<u64>().ok()
}

fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// How far behind the node the applied corpus is, in milliseconds.
///
/// `None` when nothing has been applied yet, or when the watermark does not
/// carry a parseable time prefix — reported as absent rather than as zero,
/// because "fresh" and "unknown" are different answers and a consumer deciding
/// whether to trust a semantic read depends on telling them apart.
///
/// # Why this field exists
///
/// It is the signal `remaining` could never be. `remaining` is "files left
/// after this bounded pass", so a drain tick slow enough to absorb its own
/// interval always empties the inbox and always reports zero — however far
/// behind it is. Measured on 2026-08-17: every one of 2,132 logged ticks
/// reported `remaining=0` while the index served `brain ask` from a corpus
/// 30–59 minutes stale.
pub fn freshness_lag_ms(applied_through: &str) -> Option<u64> {
    let applied = batch_epoch_ms(applied_through)?;
    Some(now_epoch_ms().saturating_sub(applied))
}

/// Inbox backlog, for the opt-in `--queue` half of `status`.
#[derive(Debug, Clone, Serialize)]
pub struct QueueStats {
    /// Batch files strictly after the watermark — real work still to apply.
    pub pending: usize,
    /// Age of the oldest such file. This is the lag a consumer will still be
    /// served once the current pass finishes, so it is the forward-looking
    /// number where `freshness_lag_ms` is the current one.
    pub oldest_pending_age_ms: Option<u64>,
    /// Files strictly BELOW the watermark that are still on disk. The drain
    /// selects `name > applied_through`, so it will never look at these again
    /// — they are stranded, not queued, and are counted apart rather than
    /// folded into `pending`.
    ///
    /// Only meaningful for a drain that runs `--consume`, which is what the
    /// host tick does: without it, every applied file legitimately remains and
    /// this counts the whole applied history. The file exactly AT the
    /// watermark is the last one applied and is never counted.
    ///
    /// On the primary host this reads 17 — a band of batches from
    /// 2026-08-16T04:01Z that a rewound watermark skipped past, invisible to
    /// `remaining` because `remaining` only ever looks forward.
    pub stranded_below_watermark: usize,
}

/// Count the inbox without reading any batch body.
///
/// Kept opt-in behind `--queue`: this is one `read_dir`, but the plane clients
/// in brain and kanban call `lastseek status` to decide whether the plane is
/// available, and a directory that has held 127,000 entries on this host is not
/// something a health check should walk on every semantic query. A health check
/// has to be cheaper than the thing it checks.
pub fn queue_stats(inbox: &Path, after: &str) -> Result<QueueStats, LastSeekError> {
    if !inbox.is_dir() {
        return Ok(QueueStats {
            pending: 0,
            oldest_pending_age_ms: None,
            stranded_below_watermark: 0,
        });
    }
    let now = now_epoch_ms();
    let mut pending = 0usize;
    let mut stranded = 0usize;
    let mut oldest: Option<u64> = None;
    for entry in std::fs::read_dir(inbox)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".json") || name.starts_with('.') {
            continue;
        }
        // `== after` is the file the watermark names — the last one applied,
        // not something skipped.
        if name.as_str() == after {
            continue;
        }
        if name.as_str() < after {
            stranded += 1;
            continue;
        }
        pending += 1;
        if let Some(ms) = batch_epoch_ms(&name) {
            oldest = Some(oldest.map_or(ms, |o: u64| o.min(ms)));
        }
    }
    Ok(QueueStats {
        pending,
        oldest_pending_age_ms: oldest.map(|ms| now.saturating_sub(ms)),
        stranded_below_watermark: stranded,
    })
}

/// Batch files strictly after `after`, in order, at most `max` of them.
///
/// Returns `(files, remaining)`. `remaining` is what the cap left behind, so a
/// caller can report it instead of implying the inbox is empty.
pub fn pending_batches(
    inbox: &Path,
    after: &str,
    max: usize,
) -> Result<(Vec<PathBuf>, usize), LastSeekError> {
    if !inbox.is_dir() {
        return Ok((Vec::new(), 0));
    }
    let mut names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(inbox)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".json") || name.starts_with('.') {
            continue;
        }
        if name.as_str() <= after {
            continue;
        }
        names.push(name);
    }
    names.sort();
    let total = names.len();
    let taken = names.len().min(max);
    let files = names
        .into_iter()
        .take(taken)
        .map(|n| inbox.join(n))
        .collect();
    Ok((files, total - taken))
}
