//! Searchable Brain schemas LastSeek indexes.
//!
//! A restricted app cannot call `GET /api/schemas`. The field list below is
//! the `word` classification from Brain's manifest (`EdgeVector/brain`
//! `lastdb-app.json`). `lastdb-app.json` `access.read` in this repo names the
//! same schemas. The node's definition id, availability, and schema type are
//! not on this list: the changes entry already carries the key.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::live_source::{invalid, Projection};
use crate::schema::SchemaTable;
use crate::LastSeekError;

#[derive(Debug, Clone, Deserialize)]
struct CatalogFile {
    schemas: Vec<SearchSchema>,
}

#[derive(Debug, Clone, Deserialize)]
struct SearchSchema {
    name: String,
    fields: Vec<String>,
}

fn catalog() -> &'static [SearchSchema] {
    static CELL: OnceLock<Vec<SearchSchema>> = OnceLock::new();
    CELL.get_or_init(|| {
        let file: CatalogFile = serde_json::from_str(include_str!("../searchable-schemas.json"))
            .expect("searchable-schemas.json ships with this binary and must parse");
        file.schemas
    })
}

/// Projections for `names`. An empty `names` selects the whole catalog.
pub(crate) fn projections(
    names: &[String],
    table: &SchemaTable,
) -> Result<BTreeMap<String, Projection>, LastSeekError> {
    let known: BTreeMap<&str, &[String]> = catalog()
        .iter()
        .map(|schema| (schema.name.as_str(), schema.fields.as_slice()))
        .collect();
    if let Some(name) = names.iter().find(|name| !known.contains_key(name.as_str())) {
        return Err(invalid(format!(
            "schema {name} is not in searchable-schemas.json"
        )));
    }
    let selected: Vec<&str> = if names.is_empty() {
        catalog()
            .iter()
            .map(|schema| schema.name.as_str())
            .collect()
    } else {
        names.iter().map(String::as_str).collect()
    };
    let mut projections = BTreeMap::new();
    for name in selected {
        let mut fields = known[name].to_vec();
        fields.sort();
        fields.dedup();
        if fields.is_empty() {
            return Err(invalid(format!(
                "configured schema {name} has no searchable fields"
            )));
        }
        let identity = table.identity_for_name(name).unwrap_or(name).to_string();
        projections.insert(name.to_string(), Projection::configured(identity, fields));
    }
    if projections.is_empty() {
        return Err(invalid(
            "rebuild requires configured searchable Brain schemas",
        ));
    }
    Ok(projections)
}
