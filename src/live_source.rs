//! Read the local LastDB owner socket using existing catalog, key and feed APIs.

use crate::ingest::{field_is_indexable, KeyValue};
use crate::schema::SchemaTable;
use crate::LastSeekError;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

pub(crate) const KEY_PAGE: usize = 32;
pub(crate) const FEED_PAGE: usize = 500;

pub(crate) fn invalid(message: impl Into<String>) -> LastSeekError {
    LastSeekError::Online(message.into())
}

pub(crate) struct Source<'a>(pub &'a Path);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SourceIdentity {
    pid: u32,
    process_start_ts: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Projection {
    pub identity: String,
    definition: String,
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub(crate) struct Key {
    pub hash: String,
    #[serde(default)]
    pub range: Option<String>,
}

impl Key {
    fn normalized(mut self) -> Self {
        self.range = self.range.filter(|range| !range.is_empty());
        self
    }
}

#[derive(Deserialize)]
pub(crate) struct Keys {
    schema_id: String,
    pub keys: Vec<Key>,
    pub next_cursor: Option<String>,
    pub has_more: bool,
}

#[derive(Deserialize)]
pub(crate) struct Change {
    cursor: String,
    pub schema_name: String,
    key: KeyValue,
}

#[derive(Deserialize)]
pub(crate) struct FeedPage {
    pub changes: Vec<Change>,
    pub next_cursor: String,
    pub has_more: bool,
    gap: bool,
}

fn sequence(cursor: &str) -> Result<u64, LastSeekError> {
    cursor
        .strip_prefix("v1-")
        .filter(|raw| raw.len() == 16)
        .and_then(|raw| u64::from_str_radix(raw, 16).ok())
        .ok_or_else(|| invalid("LastDB returned an invalid change cursor"))
}

fn encode_query_value(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[(byte >> 4) as usize]));
            encoded.push(char::from(HEX[(byte & 15) as usize]));
        }
    }
    encoded
}

impl Source<'_> {
    fn request<T: DeserializeOwned>(
        &self,
        route: &str,
        query: &[(&str, &str)],
        body: Option<Value>,
    ) -> Result<T, LastSeekError> {
        let mut url = format!("http://localhost{route}");
        for (index, (name, value)) in query.iter().enumerate() {
            url.push(if index == 0 { '?' } else { '&' });
            url.push_str(name);
            url.push('=');
            // A physical-key cursor can contain NUL. Encode it before it
            // reaches an OS argument and pass the opaque bytes back unchanged.
            url.push_str(&encode_query_value(value));
        }
        let mut command = Command::new("curl");
        command
            .args(["--disable", "--fail", "--silent", "--max-time", "30"])
            .args(["-H", "X-LastDB-Client: lastseek"])
            .arg("--unix-socket")
            .arg(self.0)
            .arg(url);
        let mut body_bytes = None;
        if let Some(body) = body {
            body_bytes = Some(serde_json::to_vec(&body)?);
            command
                .args([
                    "-H",
                    "Content-Type: application/json",
                    "--data-binary",
                    "@-",
                ])
                .stdin(Stdio::piped());
        }
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let written = match body_bytes {
            Some(bytes) => child
                .stdin
                .take()
                .expect("piped curl input")
                .write_all(&bytes),
            None => Ok(()),
        };
        let out = child.wait_with_output()?;
        if !out.status.success() {
            return Err(invalid(format!(
                "LastDB {route} request failed ({})",
                out.status
            )));
        }
        written?;
        // Parser errors can include input values. Keep record values out of errors.
        serde_json::from_slice(&out.stdout)
            .map_err(|_| invalid(format!("LastDB {route} returned an invalid response")))
    }

    pub(crate) fn identity(&self) -> Result<SourceIdentity, LastSeekError> {
        self.request("/api/system/boot-identity", &[], None)
    }

    pub(crate) fn projections(
        &self,
        names: &[String],
        table: &SchemaTable,
    ) -> Result<BTreeMap<String, Projection>, LastSeekError> {
        #[derive(Deserialize)]
        struct Catalog {
            ready: bool,
            schemas: Vec<Schema>,
        }
        #[derive(Deserialize)]
        struct Schema {
            name: String,
            owner_app_id: Option<String>,
            identity_hash: Option<String>,
            state: Option<String>,
            schema_type: String,
            #[serde(default)]
            field_classifications: BTreeMap<String, Vec<String>>,
        }
        let catalog: Catalog = self.request("/api/schemas", &[("include_full", "true")], None)?;
        if !catalog.ready {
            return Err(invalid("LastDB schema catalog is not ready"));
        }
        let mut projections = BTreeMap::new();
        for schema in catalog.schemas {
            if schema.owner_app_id.as_deref() != Some("brain")
                || !schema.name.starts_with("brain/")
                || schema.state.as_deref() != Some("Available")
                || !matches!(schema.schema_type.as_str(), "Hash" | "HashRange")
                || (!names.is_empty() && !names.contains(&schema.name))
            {
                continue;
            }
            let fields: Vec<_> = schema
                .field_classifications
                .into_iter()
                .filter(|(_, classes)| !classes.is_empty() && field_is_indexable(Some(classes)))
                .map(|(field, _)| field)
                .collect();
            if fields.is_empty() {
                continue;
            }
            let definition = schema
                .identity_hash
                .filter(|identity| !identity.is_empty())
                .ok_or_else(|| invalid("a searchable Brain schema has no definition identity"))?;
            let identity = table
                .identity_for_name(&schema.name)
                .unwrap_or(&schema.name)
                .into();
            projections.insert(
                schema.name,
                Projection {
                    identity,
                    definition,
                    fields,
                },
            );
        }
        Ok(projections)
    }

    pub(crate) fn keys(&self, schema: &str, cursor: Option<&str>) -> Result<Keys, LastSeekError> {
        #[derive(Deserialize)]
        struct Envelope {
            list: Keys,
        }
        let limit = KEY_PAGE.to_string();
        let mut params = vec![("schema", schema), ("limit", limit.as_str())];
        if let Some(cursor) = cursor {
            params.push(("cursor", cursor));
        }
        let mut page = self.request::<Envelope>("/api/list", &params, None)?.list;
        if page.schema_id != schema
            || page.keys.len() > KEY_PAGE
            || (page.has_more
                && (page.next_cursor.is_none() || page.next_cursor.as_deref() == cursor))
        {
            return Err(invalid("LastDB returned an invalid primary key page"));
        }
        page.keys = page.keys.into_iter().map(Key::normalized).collect();
        Ok(page)
    }

    pub(crate) fn records(
        &self,
        schema: &str,
        keys: &[Key],
        projection: &Projection,
    ) -> Result<BTreeMap<Key, BTreeMap<String, Value>>, LastSeekError> {
        #[derive(Deserialize)]
        struct Envelope {
            results: Vec<Row>,
            has_more: bool,
        }
        #[derive(Deserialize)]
        struct Row {
            key: Key,
            fields: BTreeMap<String, Value>,
        }
        let pairs: Vec<_> = keys
            .iter()
            .map(|key| (&key.hash, key.range.as_deref().unwrap_or("")))
            .collect();
        let body =
            json!({"schema_name":schema,"filter":{"HashRangeKeys":pairs},"limit":keys.len()});
        let response: Envelope = self.request("/api/query", &[], Some(body))?;
        if response.has_more {
            return Err(invalid("LastDB truncated a primary-key batch read"));
        }
        let mut records = BTreeMap::new();
        for row in response.results {
            let key = row.key.normalized();
            if !keys.contains(&key) || records.contains_key(&key) {
                return Err(invalid(
                    "LastDB returned an unexpected or repeated primary key",
                ));
            }
            let mut fields = row.fields;
            fields.retain(|field, _| projection.fields.contains(field));
            records.insert(key, fields);
        }
        Ok(records)
    }

    pub(crate) fn changes(
        &self,
        after: Option<&str>,
        limit: usize,
    ) -> Result<FeedPage, LastSeekError> {
        let page: FeedPage = self.request(
            "/api/app/changes",
            &[],
            Some(json!({"since_cursor":after,"limit":limit})),
        )?;
        if page.gap {
            return Err(LastSeekError::OnlineFeedGap);
        }
        let mut prior = after.map(sequence).transpose()?;
        for change in &page.changes {
            let next = sequence(&change.cursor)?;
            if prior.is_some_and(|prior| prior.checked_add(1) != Some(next)) {
                return Err(LastSeekError::OnlineFeedGap);
            }
            prior = Some(next);
        }
        let next = sequence(&page.next_cursor)?;
        if next != prior.unwrap_or(0) || (page.has_more && page.changes.is_empty()) {
            return Err(LastSeekError::OnlineFeedGap);
        }
        Ok(page)
    }

    pub(crate) fn head(&self) -> Result<String, LastSeekError> {
        let mut cursor = None;
        for _ in 0..1_000 {
            let page = self.changes(cursor.as_deref(), FEED_PAGE)?;
            if !page.has_more {
                return Ok(page.next_cursor);
            }
            cursor = Some(page.next_cursor);
        }
        Err(invalid(
            "could not reach the LastDB change feed head within the page limit",
        ))
    }
}

/// Coalesce the latest hints. Each key is read from LastDB after the page arrived.
pub(crate) fn changed_keys(
    page: &FeedPage,
    projections: &BTreeMap<String, Projection>,
) -> Result<BTreeMap<String, BTreeSet<Key>>, LastSeekError> {
    let mut keys: BTreeMap<String, BTreeSet<Key>> = BTreeMap::new();
    for change in &page.changes {
        if projections.contains_key(&change.schema_name) {
            let hash = change
                .key
                .hash
                .clone()
                .ok_or_else(|| invalid("a Brain change has no primary hash"))?;
            let key = Key {
                hash,
                range: change.key.range.clone(),
            }
            .normalized();
            keys.entry(change.schema_name.clone())
                .or_default()
                .insert(key);
        }
    }
    Ok(keys)
}
