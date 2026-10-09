//! Schema scope resolution — readable names in, identity hashes out.
//!
//! # Why this module exists
//!
//! A caller says `--schema Card`. The index stores rows under a 64-hex identity
//! hash. Something has to turn one into the other, and until this module
//! existed nothing did: the scope term was looked up directly against the
//! store's interned schema list, a miss was silently dropped from the filter,
//! and the query returned `Ok(vec![])`. A caller could not tell "you spelled
//! the schema differently than the ingest wrote it" from "that schema has no
//! rows yet" — both were an empty hit list.
//!
//! That is the same confident-but-wrong answer the embedder boundary in
//! `index::Searcher::new` refuses one axis over, and it was live in the
//! predecessor: `search query … --schema Card` returned zero hits while the
//! identical query scoped by the hash returned correctly ranked results.
//!
//! # Where the table comes from
//!
//! The **Schema Service** — directly, not proxied through LastDB. SS holds the
//! definitive names (Tom, 2026-08-08). `GET /v1/schemas/available` serves
//! `{descriptive_name, name, identity_hash}` for every active registry entry,
//! which is the whole table this module needs.
//!
//! # Why the join key is `name` and not either side's `identity_hash`
//!
//! Measured 2026-08-08 against the live primary and deployed prod SS: for the
//! schemas that matter, **the node's `identity_hash` and SS's `identity_hash`
//! disagree**, because the node recomputes the hash with whatever binary is
//! running while SS's is a property of the registry. `fbrain`'s `Reference` is
//! `85fb0452…` on the node and `5c691083…` at SS — one row, joined on `name`,
//! two hashes.
//!
//! `name` is the only field the two agree on and the only one nothing
//! recomputes. So `name` is the join key. It is deliberately **not** the
//! storage key: ingest translates `name` → SS's `identity_hash` and the index
//! interns that, so ingest and query key off the same authority and agree by
//! construction. The node's recomputed hash never enters LastSeek, and
//! therefore cannot split the corpus the way it split the predecessor's
//! coverage metric.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::LastSeekError;

/// Default Schema Service base URL (prod). Overridable with
/// `LASTSEEK_SCHEMA_SERVICE_URL`; the dev stage lives in fold's
/// `folddb_profile/environments.json`, which is the org's single source of
/// truth for these hostnames.
pub const DEFAULT_SCHEMA_SERVICE_URL: &str =
    "https://axo709qs11.execute-api.us-east-1.amazonaws.com";

/// Path within the Schema Service that serves the table.
///
/// `/v1/registry/index` would be preferable — it is signed, ETagged, and
/// compact — but it returns 404 on both deployed stages as of 2026-08-08: the
/// route is in `schema_service/openapi.yaml` on fold `main` and has not been
/// deployed. Switch to it once it ships.
pub const AVAILABLE_PATH: &str = "/v1/schemas/available";

/// One registry entry, in the shape `/v1/schemas/available` serves.
///
/// The response carries far more per entry (field types, classifications,
/// purpose statements); only these three are deserialized, so a registry that
/// grows fields does not break the table.
#[derive(Debug, Clone, Deserialize)]
struct AvailableEntry {
    #[serde(default)]
    descriptive_name: Option<String>,
    name: String,
    #[serde(default)]
    identity_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AvailableEnvelope {
    schemas: Vec<AvailableEntry>,
}

/// The resolution table, as persisted. Flat and explicit rather than the
/// derived maps, so a human can read the cache file and see exactly what the
/// resolver will do with a given term.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchemaTable {
    /// Where it came from — a URL, or a file path for an offline load.
    #[serde(default)]
    pub source: String,
    /// Registry `name` → SS `identity_hash`. The ingest-side translation.
    #[serde(default)]
    by_name: BTreeMap<String, String>,
    /// Readable `descriptive_name` → every identity hash carrying it.
    ///
    /// Set-valued on purpose. `Card` is six distinct identities on the live
    /// registry and `Reference` is three; resolving to one of them would be a
    /// quieter version of the bug this module closes.
    #[serde(default)]
    by_descriptive: BTreeMap<String, BTreeSet<String>>,
    /// Every identity hash the table knows, so an identity-hash scope term can
    /// be told apart from a typo.
    #[serde(default)]
    identities: BTreeSet<String>,
}

/// What resolved a scope term. Reported by `doctor` so a table that has gone
/// stale against SS is visible as a tier shift rather than as silence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeTier {
    /// A readable `descriptive_name` known to the Schema Service.
    Descriptive,
    /// An SS `identity_hash`.
    Identity,
    /// A registry `name` known to the Schema Service.
    RegistryName,
    /// Not in the SS table, but interned in this store — a schema whose owner
    /// has not published to SS. Indexed and searchable by the key ingest saw,
    /// and counted separately so the publish gap stays visible.
    StoreLocal,
}

/// One resolved scope term: the identity keys to filter on, and which tier
/// produced them.
#[derive(Debug, Clone)]
pub struct ResolvedScope {
    pub term: String,
    pub tier: ScopeTier,
    pub identities: Vec<String>,
}

impl SchemaTable {
    /// Build from a `/v1/schemas/available` response body.
    ///
    /// Entries with no `identity_hash` fall back to their `name` as the
    /// identity key. That is not a guess: the registry's own convention for
    /// these rows is that `name` *is* the hash (the 64-hex-in-`name` residue
    /// documented in `design-schema-identity-one-algorithm-versioned-ss-authority`),
    /// and a missing hash would otherwise silently drop the entry from the
    /// table — reintroducing the exact failure this module exists to remove.
    pub fn from_available_json(
        bytes: &[u8],
        source: impl Into<String>,
    ) -> Result<Self, LastSeekError> {
        let env: AvailableEnvelope = serde_json::from_slice(bytes)?;
        let mut table = SchemaTable {
            source: source.into(),
            ..Default::default()
        };
        for e in env.schemas {
            let identity = e
                .identity_hash
                .filter(|h| !h.is_empty())
                .unwrap_or_else(|| e.name.clone());
            table.by_name.insert(e.name, identity.clone());
            if let Some(d) = e.descriptive_name.filter(|d| !d.is_empty()) {
                table
                    .by_descriptive
                    .entry(d)
                    .or_default()
                    .insert(identity.clone());
            }
            table.identities.insert(identity);
        }
        Ok(table)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, LastSeekError> {
        let bytes = std::fs::read(path.as_ref())?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Write atomically: the table is read on every query, so a half-written
    /// file would turn a refresh into an outage of the whole query surface.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), LastSeekError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.lastseek-tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Default cache location: `<lastseek home>/../schema-table.json`, i.e.
    /// beside the index rather than inside it, because the table survives an
    /// index rebuild.
    pub fn default_path() -> std::path::PathBuf {
        if let Ok(v) = std::env::var("LASTSEEK_SCHEMA_TABLE") {
            return v.into();
        }
        let base = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Path::new(&base).join(".lastseek").join("schema-table.json")
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn descriptive_names(&self) -> usize {
        self.by_descriptive.len()
    }

    /// Ingest-side translation: the registry `name` a host batch carries →
    /// the identity key the index should intern.
    ///
    /// `None` means SS does not know this schema. The caller must not invent a
    /// key: it indexes under the registry `name` and reports the schema as
    /// SS-unknown, which is what keeps "any app can search its own records"
    /// true for apps that have not published to SS while still naming the gap.
    pub fn identity_for_name(&self, name: &str) -> Option<&str> {
        self.by_name.get(name).map(String::as_str)
    }

    /// Query-side resolution of one caller-facing scope term.
    ///
    /// Tried in the order a caller is most likely to have meant: the readable
    /// name first (`Card`), then an identity hash, then a registry `name`.
    /// `None` means the SS table has never heard of this term — the caller
    /// must then decide between a store-local schema and a loud error, which
    /// `resolve_all` does.
    pub fn resolve(&self, term: &str) -> Option<ResolvedScope> {
        if let Some(ids) = self.by_descriptive.get(term) {
            return Some(ResolvedScope {
                term: term.to_string(),
                tier: ScopeTier::Descriptive,
                identities: ids.iter().cloned().collect(),
            });
        }
        if self.identities.contains(term) {
            return Some(ResolvedScope {
                term: term.to_string(),
                tier: ScopeTier::Identity,
                identities: vec![term.to_string()],
            });
        }
        if let Some(id) = self.by_name.get(term) {
            return Some(ResolvedScope {
                term: term.to_string(),
                tier: ScopeTier::RegistryName,
                identities: vec![id.clone()],
            });
        }
        None
    }

    /// Resolve every scope term, consulting `interned` for terms SS does not
    /// know.
    ///
    /// A term that resolves nowhere is an `UnknownSchema` error, **never** an
    /// empty filter. A term that resolves to identities the store happens to
    /// hold no rows for resolves fine and yields no hits — that is a legitimate
    /// empty answer, and the whole point is that the two are distinguishable.
    pub fn resolve_all(
        &self,
        terms: &[String],
        interned: &[String],
    ) -> Result<Vec<ResolvedScope>, LastSeekError> {
        let mut out = Vec::with_capacity(terms.len());
        for term in terms {
            if let Some(r) = self.resolve(term) {
                out.push(r);
                continue;
            }
            if interned.iter().any(|s| s == term) {
                out.push(ResolvedScope {
                    term: term.clone(),
                    tier: ScopeTier::StoreLocal,
                    identities: vec![term.clone()],
                });
                continue;
            }
            return Err(LastSeekError::UnknownSchema {
                name: term.clone(),
                known: self.suggestions(term),
            });
        }
        Ok(out)
    }

    /// A short, useful `known:` list for an `UnknownSchema` error. Prefers
    /// readable names whose lowercase form shares a prefix with the term —
    /// `card` should surface `Card` — and falls back to a stable sample so the
    /// error always shows the caller what shape of thing is accepted.
    ///
    /// The anagram tier is what makes this useful rather than decorative. On a
    /// 1075-name registry the alphabetical fallback opens with `3DModel,
    /// APIReference, AboutPage…`, which tells a caller who typed `Crad`
    /// precisely nothing. Comparing sorted characters catches transpositions
    /// and most single-key slips for the price of a sort.
    fn suggestions(&self, term: &str) -> Vec<String> {
        let lower = term.to_lowercase();
        let sorted = |s: &str| {
            let mut c: Vec<char> = s.to_lowercase().chars().collect();
            c.sort_unstable();
            c
        };
        let key = sorted(term);

        let mut near: Vec<String> = self
            .by_descriptive
            .keys()
            .filter(|k| {
                let kl = k.to_lowercase();
                kl.starts_with(&lower) || lower.starts_with(&kl) || kl.contains(&lower)
            })
            .take(8)
            .cloned()
            .collect();
        if near.is_empty() {
            near = self
                .by_descriptive
                .keys()
                .filter(|k| sorted(k) == key)
                .take(8)
                .cloned()
                .collect();
        }
        if near.is_empty() {
            near = self.by_descriptive.keys().take(8).cloned().collect();
        }
        near
    }

    /// Readable name for an identity key, when SS knows one. Used to label
    /// query output, so a hit reads as `Card` rather than 64 hex characters.
    pub fn label_for_identity(&self, identity: &str) -> Option<&str> {
        self.by_descriptive
            .iter()
            .find(|(_, ids)| ids.contains(identity))
            .map(|(d, _)| d.as_str())
    }
}
