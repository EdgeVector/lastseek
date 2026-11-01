//! Corpus coverage: vectors held compared with the live records that exist.
//!
//! The node catalog and the Schema Service describe the same schemas with
//! different identity hashes on the live host. The stable join is the registry
//! `name`, so coverage follows the ingest path exactly:
//!
//! `node catalog.name -> Schema Service identity_hash -> LastSeek schema id`
//!
//! The node's locally recomputed `identity_hash` is deliberately not
//! deserialized. Using it here produced the predecessor's false 2.7% coverage
//! alarm over a working index.
//!
//! # The denominator is declared, not counted here
//!
//! The catalog's `record_count` is **rows in the partition**, which includes
//! every superseded version of every row. Dividing by it reported `degraded` on
//! a fully-covered corpus: `Reference` measured 1,788 vectors over a catalog
//! count of 2,835 (63%) when brain's own census is 1,788 — 100%. See
//! `crate::source_counts` for the measurement and the rule it enforces.
//!
//! So a schema is **tracked** only when its owning app has declared a live
//! count. Tracked schemas are rated against the floor and drive `state`.
//! Everything else is reported with what is held and labelled `not_tracked` —
//! visible, but never red. A signal that is always red is not a signal, and the
//! thing this report exists to make visible (a stopped drain) would go straight
//! back to being invisible.
//!
//! The catalog count is still carried, as `partition_rows`, because it is
//! genuinely useful — it is just not a denominator.
//!
//! # A catalog count can be unknown
//!
//! A format 7 LastDB home sends `record_count: null` when it did not count a
//! schema: an unfiltered count fails above 10,000 records, and any count
//! failure is unknown, never zero. LastSeek reports such a schema with
//! `partition_rows: null` and `partition_rows_note: "unknown"`. It does not
//! page the schema to count it: that is one cursor walk per schema on every
//! `--coverage` call. Coverage does not divide by this count, so the rating
//! does not change.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::schema::SchemaTable;
use crate::source_counts::SourceCounts;
use crate::LastSeekError;

pub const DEFAULT_COVERAGE_FLOOR: f64 = 0.95;

/// Per-schema verdict. Only `Healthy`/`Degraded` are rated; the rest describe
/// why no ratio was computed, so an unrated schema is never mistaken for a
/// covered one.
pub const STATE_HEALTHY: &str = "healthy";
pub const STATE_DEGRADED: &str = "degraded";
/// No app has declared a live count for this schema, so there is nothing
/// honest to divide by. Reported, not rated.
pub const STATE_NOT_TRACKED: &str = "not_tracked";
/// Vectors exist for an identity the source catalog does not know. A real
/// anomaly worth surfacing, but not a coverage shortfall.
pub const STATE_UNKNOWN_SOURCE: &str = "unknown_source";

#[derive(Debug, Deserialize)]
struct CatalogEnvelope {
    #[serde(default)]
    counts_included: bool,
    #[serde(default)]
    schemas: Vec<CatalogEntry>,
}

#[derive(Debug, Deserialize)]
struct CatalogEntry {
    #[serde(default)]
    descriptive_name: Option<String>,
    name: String,
    /// `None` for a missing key or a JSON `null`: the node did not count the
    /// schema (see the module doc). Unknown, never zero.
    #[serde(default)]
    record_count: Option<u64>,
}

#[derive(Debug, Default)]
struct JoinedSchema {
    label: Option<String>,
    registry_names: BTreeSet<String>,
    /// The largest known catalog count over the aliases of this identity.
    source_records: u64,
    /// True when any alias reported no count. The joined count is then
    /// unknown: a known alias count is not a count of the others.
    source_records_unknown: bool,
}

/// `partition_rows_note` for a schema whose catalog count is unknown.
pub const PARTITION_ROWS_UNKNOWN: &str = "unknown";

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SchemaCoverage {
    pub schema: String,
    pub identity: String,
    pub registry_names: Vec<String>,
    pub vectors_held: u64,
    /// The declared live-record count — the ONLY thing `ratio` divides by.
    /// `None` means no app has declared one, and then `ratio` is `None` too.
    pub live_records: Option<u64>,
    /// The node catalog's `record_count`: rows in the partition, including
    /// superseded versions. Carried because it is informative, and named so it
    /// cannot be mistaken for the denominator again. `None` when the source
    /// catalog does not know the identity, or did not count it.
    pub partition_rows: Option<u64>,
    /// `"unknown"` when the source catalog lists the schema but sent a null
    /// count. Absent when `partition_rows` holds a count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partition_rows_note: Option<&'static str>,
    /// Where the declared count came from, in the declaring app's own words.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declared_by: Option<String>,
    pub ratio: Option<f64>,
    pub state: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CoverageReport {
    pub available: bool,
    pub source: String,
    pub floor: f64,
    /// Schemas with a declared live count — the population the floor applies to.
    pub tracked_schemas: usize,
    pub degraded_schemas: usize,
    /// Schemas holding vectors with no declaration. Not a fault; a to-do for
    /// whoever owns them.
    pub untracked_schemas: usize,
    /// Identities holding vectors that the source catalog does not know.
    pub unknown_source_schemas: usize,
    /// Listed schemas whose catalog count is unknown (`partition_rows_note`).
    pub uncounted_schemas: usize,
    pub schemas: Vec<SchemaCoverage>,
}

impl CoverageReport {
    pub fn from_catalog_json(
        bytes: &[u8],
        source: impl Into<String>,
        table: &SchemaTable,
        vectors_by_identity: &BTreeMap<String, usize>,
        declared: &SourceCounts,
        floor: f64,
    ) -> Result<Self, LastSeekError> {
        let catalog: CatalogEnvelope = serde_json::from_slice(bytes)?;
        if !catalog.counts_included {
            return Err(LastSeekError::Embedder(
                "LastDB schema catalog omitted record counts; status requires include_counts=true"
                    .into(),
            ));
        }

        let mut joined: BTreeMap<String, JoinedSchema> = BTreeMap::new();
        for entry in catalog.schemas {
            // This is the load-bearing join. Never substitute the catalog's
            // identity_hash: the node recomputes it, while ingest keys on SS.
            let identity = table
                .identity_for_name(&entry.name)
                .unwrap_or(&entry.name)
                .to_string();
            let row = joined.entry(identity.clone()).or_default();
            row.registry_names.insert(entry.name);
            // Several live catalog identities expose the same logical Card
            // rows. Collapsing aliases by canonical identity and taking max
            // prevents one corpus from multiplying the denominator.
            match entry.record_count {
                Some(count) => row.source_records = row.source_records.max(count),
                None => row.source_records_unknown = true,
            }
            if row.label.is_none() {
                row.label = table
                    .label_for_identity(&identity)
                    .map(str::to_string)
                    .or(entry.descriptive_name);
            }
        }

        let mut schemas = Vec::new();
        for (identity, row) in &joined {
            let vectors = vectors_by_identity.get(identity).copied().unwrap_or(0) as u64;
            let decl = declared.get(identity);
            // An unknown count may hide records, so only a counted zero skips.
            if !row.source_records_unknown
                && row.source_records == 0
                && vectors == 0
                && decl.is_none()
            {
                continue;
            }

            // The whole fix in four lines: divide by what the owning app
            // declared, or do not divide at all. `partition_rows` is reported
            // alongside so the difference is visible rather than mysterious.
            let (live_records, ratio, state) = match decl {
                Some(d) if d.live_records > 0 => {
                    let r = vectors as f64 / d.live_records as f64;
                    let s = if r < floor {
                        STATE_DEGRADED
                    } else {
                        STATE_HEALTHY
                    };
                    (Some(d.live_records), Some(r), s)
                }
                // Declared zero: nothing to hold, so holding nothing is correct
                // and holding something is not a shortfall either.
                Some(_) => (Some(0), None, STATE_HEALTHY),
                None => (None, None, STATE_NOT_TRACKED),
            };

            schemas.push(SchemaCoverage {
                schema: row.label.clone().unwrap_or_else(|| identity.clone()),
                identity: identity.clone(),
                registry_names: row.registry_names.iter().cloned().collect(),
                vectors_held: vectors,
                live_records,
                partition_rows: (!row.source_records_unknown).then_some(row.source_records),
                partition_rows_note: row.source_records_unknown.then_some(PARTITION_ROWS_UNKNOWN),
                declared_by: decl.map(|d| d.source.clone()).filter(|s| !s.is_empty()),
                ratio,
                state,
            });
        }

        // A vector identity absent from the source catalog is a real anomaly —
        // something was indexed under a key the catalog cannot account for. Keep
        // it visible under its own state. It does NOT force the overall verdict
        // red: previously this failed closed, which meant one orphan identity
        // pinned `state` to degraded forever and buried the schemas that matter.
        for (identity, vectors) in vectors_by_identity {
            if *vectors == 0 || joined.contains_key(identity) {
                continue;
            }
            schemas.push(SchemaCoverage {
                schema: table
                    .label_for_identity(identity)
                    .unwrap_or(identity)
                    .to_string(),
                identity: identity.clone(),
                registry_names: Vec::new(),
                vectors_held: *vectors as u64,
                live_records: None,
                partition_rows: None,
                partition_rows_note: None,
                declared_by: None,
                ratio: None,
                state: STATE_UNKNOWN_SOURCE,
            });
        }

        schemas.sort_by(|a, b| a.schema.cmp(&b.schema).then(a.identity.cmp(&b.identity)));
        let count_of = |want: &str| schemas.iter().filter(|s| s.state == want).count();
        let degraded_schemas = count_of(STATE_DEGRADED);
        let tracked_schemas = degraded_schemas + count_of(STATE_HEALTHY);
        Ok(Self {
            available: true,
            source: source.into(),
            floor,
            tracked_schemas,
            degraded_schemas,
            untracked_schemas: count_of(STATE_NOT_TRACKED),
            unknown_source_schemas: count_of(STATE_UNKNOWN_SOURCE),
            uncounted_schemas: schemas
                .iter()
                .filter(|s| s.partition_rows_note.is_some())
                .count(),
            schemas,
        })
    }

    /// Overall verdict, over the tracked population only.
    ///
    /// `unknown` when nothing has opted in — deliberately not `healthy`, because
    /// "no schema declared a denominator" and "every schema is covered" are
    /// different facts and the first must not read as the second.
    pub fn state(&self) -> &'static str {
        if self.tracked_schemas == 0 {
            "unknown"
        } else if self.degraded_schemas == 0 {
            STATE_HEALTHY
        } else {
            STATE_DEGRADED
        }
    }
}
