//! Declared live-record counts — the denominator coverage is allowed to divide by.
//!
//! # Why the catalog count cannot be the denominator
//!
//! Coverage shipped dividing `vectors_held` by the node catalog's
//! `record_count`, and that reported `degraded` on a fully-covered corpus.
//! Measured 2026-08-08 on the live primary:
//!
//! | schema | catalog `record_count` | the owning app's own census | vectors held |
//! |---|---|---|---|
//! | `Reference` | 2,835 | 1,788 (`brain list --count`) | 1,788 |
//! | `Milestone` | 415 | 67 (`kanban milestone list`) | 10 |
//!
//! `Reference` was at **100%** and reported 63%. The catalog counts **rows in
//! the partition**, and a LastDB partition holds the materialized row alongside
//! every superseded version of it — the property that makes a partition read a
//! *wrong* read rather than merely a slow one. So the ratio was
//! `live / (live + history)`, which cannot reach the floor and falls as the
//! corpus ages.
//!
//! # Why the owning app declares it
//!
//! The rule this module exists to enforce: **a coverage metric must divide by a
//! count the owning app would recognise as its own.** If brain says 1,788 and
//! the metric says 2,835, the metric is not measuring brain's corpus.
//!
//! The owning app is the only thing that knows. It dual-writes the keyed
//! secondary that makes its corpus enumerable without a scan — brain's
//! type-list index, kanban's board. LastSeek cannot derive that for every app,
//! and should not try.
//!
//! ## The declared count is the app's claim, NOT a census this module vouches for
//!
//! An earlier version of this doc said `brain list --count` and `kanban list`
//! are "by construction, the live set." That was wrong, and it contradicted a
//! standing decision (Tom, 2026-08-06,
//! `preference-brain-read-via-search-not-list-enumeration`): `brain list` "does
//! not get a completeness contract … nothing may build a completeness-dependent
//! sweep on it." A coverage denominator is exactly such a sweep.
//!
//! Two measured defects in that instrument, pulling opposite ways: the list
//! projection under-delivers (993 of 999 reference rows), while `--count` — the
//! half actually used — over-counts, because `delete.ts` never patches the type
//! index, so deleted records stay countable (18 phantoms measured). The residual
//! error therefore INFLATES the denominator, which can only make coverage look
//! worse than it is. Biasing toward a false red rather than a false green is the
//! survivable direction, and here it was luck rather than design.
//!
//! This is why `source` is not decoration. **The declaring app owns the method,
//! and records it.** That keeps the fix local to the app when its method
//! improves — a better access pattern changes the declaration, not this module —
//! and it makes a stale or hand-waved denominator auditable instead of
//! anonymous. A reader who distrusts a ratio can read the method that produced
//! its denominator.
//!
//! The intended end state is a named access pattern per
//! `design-brain-no-enumeration-match-array-retrieval`, not `list`.
//!
//! # Declaring is also opting in
//!
//! A declaration doubles as the opt-in, which is what stops the report from
//! being noise. Before this, the floor applied to all 1,137 schemas on the node
//! and produced 139 `degraded` entries dominated by machine schemas nobody
//! wants semantically searchable — `LastgitRepoPolicyEventV2` (12,633 rows),
//! `LastDB Request Ops Rollup` (9,892), `RoutineRunSummary` (5,465). A signal
//! that is always red is not a signal, and the thing it was built to make
//! visible (a stopped drain) goes back to being invisible.
//!
//! So: a schema with a declared count is **tracked** and rated against the
//! floor. Everything else is reported with what is held, and labelled
//! `not_tracked`. It never drives `state`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::LastSeekError;

/// One app's declaration for one schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeclaredCount {
    /// Live records the owning app says exist. Not rows in a partition.
    pub live_records: u64,
    /// How it was obtained, in the app's own words (e.g.
    /// `brain list --type reference --count`). Recorded so a stale or
    /// hand-waved denominator can be audited rather than trusted.
    #[serde(default)]
    pub source: String,
    /// When it was declared. A denominator ages; this is what lets a reader
    /// tell a fresh declaration from one made before a bulk import.
    #[serde(default)]
    pub declared_at: String,
}

/// Declared counts, keyed by canonical schema identity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceCounts {
    #[serde(default)]
    pub schemas: BTreeMap<String, DeclaredCount>,
}

impl SourceCounts {
    /// Beside the index rather than inside it: a declaration outlives a rebuild.
    pub fn default_path() -> PathBuf {
        if let Ok(v) = std::env::var("LASTSEEK_SOURCE_COUNTS") {
            return v.into();
        }
        let base = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Path::new(&base)
            .join(".lastseek")
            .join("source-counts.json")
    }

    /// Absent is not an error: it means nothing has opted in yet, and the report
    /// says so per schema instead of refusing to run.
    pub fn load_or_default(path: impl AsRef<Path>) -> Self {
        std::fs::read(path.as_ref())
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    /// Atomic, because `status` reads this on every call.
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

    pub fn get(&self, identity: &str) -> Option<&DeclaredCount> {
        self.schemas.get(identity)
    }

    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }

    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Declare (or re-declare) one schema's live count.
    ///
    /// Keyed on the canonical identity, so the same declaration is found no
    /// matter which spelling the declarer used at the query surface.
    pub fn declare(&mut self, identity: &str, live_records: u64, source: &str, at: &str) {
        self.schemas.insert(
            identity.to_string(),
            DeclaredCount {
                live_records,
                source: source.to_string(),
                declared_at: at.to_string(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_declaration_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source-counts.json");
        let mut counts = SourceCounts::default();
        counts.declare(
            "5c691083",
            1788,
            "brain list --type reference --count",
            "2026-08-08T16:00:00Z",
        );
        counts.save(&path).unwrap();

        let back = SourceCounts::load_or_default(&path);
        let got = back.get("5c691083").unwrap();
        assert_eq!(got.live_records, 1788);
        assert_eq!(got.source, "brain list --type reference --count");
    }

    #[test]
    fn a_missing_file_is_empty_rather_than_an_error() {
        // Nothing has opted in yet is a normal state, reported per schema. A
        // hard failure here would take the whole status surface down for a
        // fresh install.
        let dir = tempfile::tempdir().unwrap();
        let counts = SourceCounts::load_or_default(dir.path().join("nope.json"));
        assert!(counts.is_empty());
    }

    #[test]
    fn re_declaring_replaces_rather_than_accumulates() {
        let mut counts = SourceCounts::default();
        counts.declare("id", 10, "first", "t1");
        counts.declare("id", 20, "second", "t2");
        assert_eq!(counts.len(), 1);
        assert_eq!(counts.get("id").unwrap().live_records, 20);
    }
}
