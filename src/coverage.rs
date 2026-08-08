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
    #[serde(default)]
    record_count: u64,
}

#[derive(Debug, Default)]
struct JoinedSchema {
    label: Option<String>,
    registry_names: BTreeSet<String>,
    source_records: u64,
}

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
    /// cannot be mistaken for the denominator again.
    pub partition_rows: Option<u64>,
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
            row.source_records = row.source_records.max(entry.record_count);
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
            if row.source_records == 0 && vectors == 0 && decl.is_none() {
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
                partition_rows: Some(row.source_records),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> SchemaTable {
        SchemaTable::from_available_json(
            br#"{"schemas":[
              {"descriptive_name":"Card","name":"card-registry","identity_hash":"card-ss"},
              {"descriptive_name":"Card","name":"card-alias","identity_hash":"card-ss"},
              {"descriptive_name":"Reference","name":"ref-registry","identity_hash":"ref-ss"},
              {"descriptive_name":"RoutineRunSummary","name":"noise-registry","identity_hash":"noise-ss"}
            ]}"#,
            "test",
        )
        .unwrap()
    }

    fn declared(pairs: &[(&str, u64)]) -> SourceCounts {
        let mut c = SourceCounts::default();
        for (id, n) in pairs {
            c.declare(id, *n, "test census", "2026-08-08T16:00:00Z");
        }
        c
    }

    fn report(catalog: &[u8], vectors: &[(&str, usize)], decl: &SourceCounts) -> CoverageReport {
        let v: BTreeMap<String, usize> = vectors.iter().map(|(k, n)| (k.to_string(), *n)).collect();
        CoverageReport::from_catalog_json(
            catalog,
            "test",
            &table(),
            &v,
            decl,
            DEFAULT_COVERAGE_FLOOR,
        )
        .unwrap()
    }

    fn find<'a>(r: &'a CoverageReport, identity: &str) -> &'a SchemaCoverage {
        r.schemas
            .iter()
            .find(|s| s.identity == identity)
            .expect("schema in report")
    }

    /// The measurement that opened the papercut, as a regression.
    ///
    /// brain's own census of Reference is 1,788 and LastSeek holds 1,788 — that
    /// is 100%. The catalog says 2,835 because a partition holds every
    /// superseded version. Dividing by the catalog reported 63% and `degraded`.
    #[test]
    fn a_partition_row_count_larger_than_the_live_count_does_not_make_a_covered_schema_degraded() {
        let catalog = br#"{"counts_included":true,"schemas":[
          {"descriptive_name":"Reference","name":"ref-registry","record_count":2835}
        ]}"#;
        let r = report(catalog, &[("ref-ss", 1788)], &declared(&[("ref-ss", 1788)]));

        assert_eq!(r.state(), STATE_HEALTHY);
        let s = find(&r, "ref-ss");
        assert_eq!(s.ratio, Some(1.0));
        assert_eq!(s.live_records, Some(1788));
        // Still reported, under a name that cannot be mistaken for a denominator.
        assert_eq!(s.partition_rows, Some(2835));
    }

    /// The other half: machine schemas nobody opted into must not be able to
    /// hold `state` red. 139 of them did.
    #[test]
    fn an_undeclared_schema_is_reported_but_never_degraded() {
        let catalog = br#"{"counts_included":true,"schemas":[
          {"descriptive_name":"Reference","name":"ref-registry","record_count":1788},
          {"descriptive_name":"RoutineRunSummary","name":"noise-registry","record_count":5465}
        ]}"#;
        let r = report(catalog, &[("ref-ss", 1788)], &declared(&[("ref-ss", 1788)]));

        assert_eq!(
            r.state(),
            STATE_HEALTHY,
            "one untracked schema must not turn the verdict red"
        );
        assert_eq!(r.tracked_schemas, 1);
        assert_eq!(r.untracked_schemas, 1);
        let noise = find(&r, "noise-ss");
        assert_eq!(noise.state, STATE_NOT_TRACKED);
        assert_eq!(
            noise.ratio, None,
            "no denominator means no ratio, not a zero one"
        );
    }

    /// The acceptance test the shipped version could not pass.
    #[test]
    fn state_can_read_healthy_on_a_covered_machine() {
        let catalog = br#"{"counts_included":true,"schemas":[
          {"descriptive_name":"Reference","name":"ref-registry","record_count":2835},
          {"descriptive_name":"Card","name":"card-registry","record_count":1986},
          {"descriptive_name":"RoutineRunSummary","name":"noise-registry","record_count":5465}
        ]}"#;
        let r = report(
            catalog,
            &[("ref-ss", 1788), ("card-ss", 167)],
            &declared(&[("ref-ss", 1788), ("card-ss", 167)]),
        );
        assert_eq!(r.state(), STATE_HEALTHY);
        assert_eq!(r.degraded_schemas, 0);
    }

    /// And a real shortfall still reads red, or the fix would be a whitewash.
    #[test]
    fn a_genuine_shortfall_against_a_declared_count_is_still_degraded() {
        let catalog = br#"{"counts_included":true,"schemas":[
          {"descriptive_name":"Reference","name":"ref-registry","record_count":2835}
        ]}"#;
        let r = report(catalog, &[("ref-ss", 900)], &declared(&[("ref-ss", 1788)]));
        assert_eq!(r.state(), STATE_DEGRADED);
        let s = find(&r, "ref-ss");
        assert_eq!(s.state, STATE_DEGRADED);
        assert!(s.ratio.unwrap() < 0.51);
    }

    #[test]
    fn nothing_declared_reads_unknown_rather_than_healthy() {
        // "No app declared a denominator" and "every schema is covered" are
        // different facts. The first must not be reported as the second.
        let catalog = br#"{"counts_included":true,"schemas":[
          {"descriptive_name":"Reference","name":"ref-registry","record_count":1788}
        ]}"#;
        let r = report(catalog, &[("ref-ss", 1788)], &SourceCounts::default());
        assert_eq!(r.state(), "unknown");
        assert_eq!(r.tracked_schemas, 0);
    }

    #[test]
    fn source_records_join_through_the_registry_name_not_the_node_identity() {
        // Unchanged behaviour, kept under test: the node's recomputed
        // identity_hash must never be the join key.
        let catalog = br#"{"counts_included":true,"schemas":[
          {"descriptive_name":"Card","name":"card-registry","identity_hash":"node-recomputed","record_count":12}
        ]}"#;
        let r = report(catalog, &[], &declared(&[("card-ss", 12)]));
        let s = find(&r, "card-ss");
        assert_eq!(s.vectors_held, 0);
        assert_eq!(s.ratio, Some(0.0));
        assert_eq!(s.state, STATE_DEGRADED);
    }

    #[test]
    fn duplicate_catalog_aliases_do_not_multiply_the_denominator() {
        let catalog = br#"{"counts_included":true,"schemas":[
          {"descriptive_name":"Card","name":"card-registry","record_count":12},
          {"descriptive_name":"Card","name":"card-alias","record_count":12}
        ]}"#;
        let r = report(catalog, &[("card-ss", 12)], &declared(&[("card-ss", 12)]));
        assert_eq!(r.state(), STATE_HEALTHY);
        let s = find(&r, "card-ss");
        assert_eq!(s.partition_rows, Some(12));
        assert_eq!(s.ratio, Some(1.0));
        assert_eq!(s.registry_names.len(), 2);
    }

    #[test]
    fn vectors_for_an_identity_the_catalog_does_not_know_are_flagged_not_counted_as_a_shortfall() {
        let catalog = br#"{"counts_included":true,"schemas":[
          {"descriptive_name":"Reference","name":"ref-registry","record_count":1788}
        ]}"#;
        let r = report(
            catalog,
            &[("ref-ss", 1788), ("orphan-ss", 4)],
            &declared(&[("ref-ss", 1788)]),
        );
        assert_eq!(find(&r, "orphan-ss").state, STATE_UNKNOWN_SOURCE);
        assert_eq!(r.unknown_source_schemas, 1);
        // Visible, but it does not pin the verdict red — which is what buried
        // the real schemas before.
        assert_eq!(r.state(), STATE_HEALTHY);
    }

    #[test]
    fn a_catalog_without_counts_is_refused() {
        let err = CoverageReport::from_catalog_json(
            br#"{"schemas":[]}"#,
            "test",
            &table(),
            &BTreeMap::new(),
            &SourceCounts::default(),
            DEFAULT_COVERAGE_FLOOR,
        );
        assert!(err.is_err());
    }
}
