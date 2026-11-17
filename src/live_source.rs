//! Read LastDB through the app API.
//!
//! The live feed is `POST /v1/x/changes`. Current records are `POST /v1/read`.
//! A rebuild lists a schema's keys with a `keys_only` read that names no key,
//! because the changes list keeps only its newest entries and so does not
//! yield every live key.

use crate::ingest::KeyValue;
use crate::LastSeekError;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

pub(crate) const KEY_PAGE: usize = 32;
pub(crate) const READ_BATCH: usize = 64;
pub(crate) const FEED_PAGE: usize = 500;

pub(crate) fn invalid(message: impl Into<String>) -> LastSeekError {
    LastSeekError::Online(message.into())
}

pub(crate) struct Source<'a>(pub &'a Path);

/// Identity of one changes list. A different `id` means a restore replaced
/// the database and any saved position is void. This replaces the owner
/// boot-identity check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SourceIdentity {
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Projection {
    pub identity: String,
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

    fn json(&self) -> Value {
        match &self.range {
            Some(range) => json!({"hash": self.hash, "range": range}),
            None => json!({"hash": self.hash}),
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct Keys {
    #[serde(default)]
    pub keys: Vec<Key>,
    #[serde(default)]
    pub next_cursor: Option<String>,
    #[serde(default)]
    error: Option<ReadError>,
}

#[derive(Deserialize)]
struct ReadError {
    code: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Change {
    pub seq: u64,
    pub schema: String,
    pub key: KeyValue,
    /// The following read decides whether the key is still live.
    #[serde(default)]
    #[allow(dead_code)]
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FeedPage {
    pub id: String,
    pub head: u64,
    pub through: u64,
    pub gap: bool,
    pub changes: Vec<Change>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct FeedPosition {
    pub through: u64,
}

impl FeedPage {
    pub(crate) fn caught_up(&self) -> bool {
        !self.gap && self.through >= self.head
    }
}

/// `X-LastDB-App-Session` when the launcher set `LASTDB_APP_SESSION`.
pub fn session_header() -> Result<Option<String>, LastSeekError> {
    match std::env::var("LASTDB_APP_SESSION") {
        Err(_) => Ok(None),
        Ok(value) if value.is_empty() => Ok(None),
        Ok(value) if value.bytes().any(|byte| byte == b'\r' || byte == b'\n') => {
            Err(invalid("LASTDB_APP_SESSION contains a line break"))
        }
        Ok(value) => Ok(Some(format!("X-LastDB-App-Session: {value}"))),
    }
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
            .args(["--disable", "--silent", "--show-error", "--max-time", "30"])
            .args(["-H", "X-LastDB-Client: lastseek"])
            .args(["-w", "\n%{http_code}"])
            .arg("--unix-socket")
            .arg(self.0)
            .arg(url);
        if let Some(header) = session_header()? {
            command.args(["-H", &header]);
        }
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
        written?;
        if !out.status.success() {
            return Err(invalid(format!(
                "LastDB {route} request failed ({})",
                out.status
            )));
        }
        let (payload, status) = split_http_status(&out.stdout)?;
        if status != 200 {
            return Err(route_failure(route, status));
        }
        // Parser errors can include input values. Keep record values out of errors.
        serde_json::from_slice(payload)
            .map_err(|_| invalid(format!("LastDB {route} returned an invalid response")))
    }

    /// One page of a schema's primary keys. The last page has no cursor.
    pub(crate) fn keys(&self, schema: &str, cursor: Option<&str>) -> Result<Keys, LastSeekError> {
        #[derive(Deserialize)]
        struct Envelope {
            results: Vec<Keys>,
        }
        let mut read = json!({"schema": schema, "keys_only": true, "limit": KEY_PAGE});
        if let Some(cursor) = cursor {
            read["cursor"] = json!(cursor);
        }
        let response: Envelope = self.request("/v1/read", &[], Some(json!({"reads": [read]})))?;
        let mut pages = response.results.into_iter();
        let (Some(mut page), None) = (pages.next(), pages.next()) else {
            return Err(invalid("LastDB returned a key list of the wrong length"));
        };
        if let Some(error) = &page.error {
            return Err(invalid(format!(
                "LastDB refused a key list of {schema} ({})",
                error.code
            )));
        }
        if page.keys.len() > KEY_PAGE
            || page.keys.iter().any(|key| key.hash.is_empty())
            || (page.next_cursor.is_some() && page.next_cursor.as_deref() == cursor)
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
    ) -> Result<BTreeMap<Key, BTreeMap<String, Value>>, LastSeekError> {
        #[derive(Deserialize)]
        struct Envelope {
            results: Vec<ReadResult>,
        }
        #[derive(Deserialize)]
        struct ReadResult {
            #[serde(default)]
            records: Vec<ReadRecord>,
            #[serde(default)]
            error: Option<ReadError>,
        }
        #[derive(Deserialize)]
        struct ReadRecord {
            key: Key,
            record: BTreeMap<String, Value>,
        }
        if keys.is_empty() {
            return Ok(BTreeMap::new());
        }
        if keys.len() > READ_BATCH {
            return Err(invalid("a record read exceeds 64 keys"));
        }
        let reads: Vec<Value> = keys
            .iter()
            .map(|key| json!({"schema": schema, "key": key.json()}))
            .collect();
        let response: Envelope = self.request("/v1/read", &[], Some(json!({"reads": reads})))?;
        if response.results.len() != keys.len() {
            return Err(invalid(
                "LastDB returned a record batch of the wrong length",
            ));
        }
        let mut records = BTreeMap::new();
        for (key, result) in keys.iter().zip(response.results) {
            if let Some(error) = result.error {
                return Err(invalid(format!(
                    "LastDB refused a record read ({})",
                    error.code
                )));
            }
            if result.records.is_empty() {
                continue;
            }
            if result.records.len() != 1 {
                return Err(invalid("LastDB returned more than one record for a key"));
            }
            let row = result.records.into_iter().next().expect("one record");
            let found = row.key.normalized();
            if &found != key || records.contains_key(&found) {
                return Err(invalid(
                    "LastDB returned an unexpected or repeated primary key",
                ));
            }
            records.insert(found, row.record);
        }
        Ok(records)
    }

    /// One page of `POST /v1/x/changes`. `gap` stays on the page so the caller
    /// can tell a replaced list (`id`) from a trimmed window.
    pub(crate) fn changes(
        &self,
        after: u64,
        limit: usize,
        schemas: &[String],
    ) -> Result<FeedPage, LastSeekError> {
        if schemas.is_empty() {
            return Err(invalid("a changes read needs the schemas LastSeek indexes"));
        }
        let page: FeedPage = self.request(
            "/v1/x/changes",
            &[],
            Some(json!({"after": after, "limit": limit, "schemas": schemas})),
        )?;
        if page.id.is_empty() {
            return Err(invalid("LastDB returned a changes list with no id"));
        }
        if page.gap {
            return Ok(page);
        }
        if page.through < after {
            return Err(invalid(
                "LastDB returned a changes position before the request",
            ));
        }
        if limit == 0 {
            if page.through != after || !page.changes.is_empty() {
                return Err(invalid("LastDB returned changes for a head-only read"));
            }
            return Ok(page);
        }
        let mut prior = after;
        for change in &page.changes {
            if change.seq <= prior || change.schema.is_empty() {
                return Err(invalid("LastDB returned a change outside the page"));
            }
            if change.key.hash.as_deref().unwrap_or("").is_empty() {
                return Err(invalid("a change has no primary hash"));
            }
            prior = change.seq;
        }
        if page.through == after && page.through < page.head {
            return Err(invalid(
                "LastDB returned no progress toward the changes head",
            ));
        }
        Ok(page)
    }

    /// `limit: 0` reads `id` and `head` without entries. `gap` on `after: 0`
    /// is normal once retention has trimmed the list; `head` is still current.
    pub(crate) fn changes_head(&self, schemas: &[String]) -> Result<FeedPage, LastSeekError> {
        self.changes(0, 0, schemas)
    }
}

fn split_http_status(stdout: &[u8]) -> Result<(&[u8], u16), LastSeekError> {
    let Some(split) = stdout.iter().rposition(|byte| *byte == b'\n') else {
        return Err(invalid("LastDB returned a response without a status"));
    };
    let code = std::str::from_utf8(&stdout[split + 1..])
        .ok()
        .filter(|code| code.len() == 3 && code.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| invalid("LastDB returned a response without a status"))?;
    Ok((&stdout[..split], code))
}

fn route_failure(route: &str, status: u16) -> LastSeekError {
    invalid(format!("LastDB {route} request failed (HTTP {status})"))
}

/// Coalesce the latest keys. Each key is read from LastDB after the page arrived.
pub(crate) fn changed_keys(
    page: &FeedPage,
    projections: &BTreeMap<String, Projection>,
) -> Result<BTreeMap<String, BTreeSet<Key>>, LastSeekError> {
    let mut keys: BTreeMap<String, BTreeSet<Key>> = BTreeMap::new();
    for change in &page.changes {
        if !projections.contains_key(&change.schema) {
            continue;
        }
        let hash = change
            .key
            .hash
            .clone()
            .filter(|hash| !hash.is_empty())
            .ok_or_else(|| invalid("a Brain change has no primary hash"))?;
        let key = Key {
            hash,
            range: change.key.range.clone(),
        }
        .normalized();
        keys.entry(change.schema.clone()).or_default().insert(key);
    }
    Ok(keys)
}

/// A saved position is void when the list id changed. A gap means entries
/// were dropped. Call this only after comparing `id`.
pub(crate) fn page_break(page: &FeedPage, saved: &SourceIdentity) -> Option<LastSeekError> {
    if page.id != saved.id {
        return Some(LastSeekError::ChangesListReplaced);
    }
    if page.gap {
        return Some(LastSeekError::OnlineFeedGap);
    }
    None
}
