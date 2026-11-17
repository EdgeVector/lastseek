//! The schemas one index covers.
//!
//! The app that wants search names its schemas (`lastseek rebuild --schema`).
//! LastSeek keeps no list of its own and reads nothing the caller did not
//! name, so a schema no app asked for is never indexed. Every field of a named
//! schema is embedded.

use std::collections::BTreeMap;

use crate::live_source::{invalid, Projection};
use crate::schema::SchemaTable;
use crate::LastSeekError;

pub(crate) fn projections(
    names: &[String],
    table: &SchemaTable,
) -> Result<BTreeMap<String, Projection>, LastSeekError> {
    if names.is_empty() {
        return Err(invalid(
            "name the schemas to index: lastseek rebuild --schema NAME...",
        ));
    }
    Ok(names
        .iter()
        .map(|name| {
            let identity = table.identity_for_name(name).unwrap_or(name).to_string();
            (name.clone(), Projection { identity })
        })
        .collect())
}
