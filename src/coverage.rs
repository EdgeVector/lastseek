//! Corpus coverage: vectors held compared with source records available.
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

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::schema::SchemaTable;
use crate::LastSeekError;

pub const DEFAULT_COVERAGE_FLOOR: f64 = 0.95;

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
    pub source_records: Option<u64>,
    pub ratio: Option<f64>,
    pub state: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CoverageReport {
    pub available: bool,
    pub source: String,
    pub floor: f64,
    pub degraded_schemas: usize,
    pub schemas: Vec<SchemaCoverage>,
}

impl CoverageReport {
    pub fn from_catalog_json(
        bytes: &[u8],
        source: impl Into<String>,
        table: &SchemaTable,
        vectors_by_identity: &BTreeMap<String, usize>,
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
            if row.source_records == 0 && vectors == 0 {
                continue;
            }
            let ratio = if row.source_records == 0 {
                None
            } else {
                Some(vectors as f64 / row.source_records as f64)
            };
            let state = if ratio.is_some_and(|value| value < floor) {
                "degraded"
            } else {
                "healthy"
            };
            schemas.push(SchemaCoverage {
                schema: row.label.clone().unwrap_or_else(|| identity.clone()),
                identity: identity.clone(),
                registry_names: row.registry_names.iter().cloned().collect(),
                vectors_held: vectors,
                source_records: Some(row.source_records),
                ratio,
                state,
            });
        }

        // A vector identity absent from the source catalog cannot be called
        // covered. Keep it visible and make the overall state fail closed.
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
                source_records: None,
                ratio: None,
                state: "degraded",
            });
        }

        schemas.sort_by(|a, b| a.schema.cmp(&b.schema).then(a.identity.cmp(&b.identity)));
        let degraded_schemas = schemas
            .iter()
            .filter(|schema| schema.state == "degraded")
            .count();
        Ok(Self {
            available: true,
            source: source.into(),
            floor,
            degraded_schemas,
            schemas,
        })
    }

    pub fn state(&self) -> &'static str {
        if self.degraded_schemas == 0 {
            "healthy"
        } else {
            "degraded"
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
              {"descriptive_name":"Card","name":"card-alias","identity_hash":"card-ss"}
            ]}"#,
            "test",
        )
        .unwrap()
    }

    #[test]
    fn source_records_without_vectors_are_degraded_and_join_through_name() {
        let catalog = br#"{"counts_included":true,"schemas":[
          {"descriptive_name":"Card","name":"card-registry","identity_hash":"node-recomputed","record_count":12}
        ]}"#;
        let report = CoverageReport::from_catalog_json(
            catalog,
            "test",
            &table(),
            &BTreeMap::new(),
            DEFAULT_COVERAGE_FLOOR,
        )
        .unwrap();
        assert_eq!(report.state(), "degraded");
        assert_eq!(report.schemas.len(), 1);
        assert_eq!(report.schemas[0].identity, "card-ss");
        assert_eq!(report.schemas[0].vectors_held, 0);
        assert_eq!(report.schemas[0].source_records, Some(12));
        assert_eq!(report.schemas[0].ratio, Some(0.0));
    }

    #[test]
    fn duplicate_catalog_aliases_do_not_multiply_the_denominator() {
        let catalog = br#"{"counts_included":true,"schemas":[
          {"descriptive_name":"Card","name":"card-registry","record_count":12},
          {"descriptive_name":"Card","name":"card-alias","record_count":12}
        ]}"#;
        let vectors = BTreeMap::from([("card-ss".to_string(), 12usize)]);
        let report = CoverageReport::from_catalog_json(
            catalog,
            "test",
            &table(),
            &vectors,
            DEFAULT_COVERAGE_FLOOR,
        )
        .unwrap();
        assert_eq!(report.state(), "healthy");
        assert_eq!(report.schemas[0].source_records, Some(12));
        assert_eq!(report.schemas[0].ratio, Some(1.0));
        assert_eq!(report.schemas[0].registry_names.len(), 2);
    }
}
