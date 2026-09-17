//! LastSeek — memory-mapped semantic vector search for LastDB.
//!
//! Runs **outside** Mini, so a default `lastdbd` still ships no model weights
//! or inference runtime. That boundary is a product commitment, not an
//! accident: see the `north-star-search-native-semantic-parity` record.
//!
//! The index is regenerable from LastDB at any time, which is what lets the
//! on-disk format stay simple and unreplicated.

pub mod cli;
pub mod coverage;
pub mod embed;
pub mod index;
pub mod ingest;
pub mod metrics;
pub mod schema;
pub mod source_counts;
pub mod store;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LastSeekError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported store version {0}")]
    UnsupportedVersion(u32),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("embedder: {0}")]
    Embedder(String),
    #[error("corrupt generation: {0}")]
    CorruptGeneration(String),
    #[error("embedder boundary: {0}")]
    EmbedderBoundary(String),
    /// A scope term that resolves to no schema at all.
    ///
    /// Deliberately an error rather than an empty filter. A caller who
    /// mistypes a schema, or names one under a convention the ingest never
    /// wrote, must not receive the same answer as a caller who correctly
    /// named a schema that legitimately holds no rows. The predecessor
    /// returned `Ok(vec![])` for both, live, on brain's and kanban's own
    /// corpora.
    #[error("unknown schema {name:?}; known names include: {}", known.join(", "))]
    UnknownSchema { name: String, known: Vec<String> },
    /// A flag the verb does not read.
    ///
    /// Deliberately an error rather than a silent skip, and deliberately
    /// modelled on the same distinction as `UnknownSchema`: a caller who
    /// mistypes a flag must not receive the answer a caller who omitted it
    /// would get. The scanning parser this replaced ignored the flag, ran the
    /// verb, and exited 0 — which on `drain --help` meant applying 1,234 batch
    /// files to the primary index instead of printing usage.
    #[error(
        "flag is not accepted by {verb}: {flag}. Run 'lastseek {verb} --help' for the flags this \
         verb accepts ({})",
        known.join(", ")
    )]
    UnknownFlag {
        flag: String,
        verb: String,
        known: Vec<String>,
    },
    #[error("{flag} needs a value; run 'lastseek {verb} --help'")]
    MissingFlagValue { flag: String, verb: String },
    #[error("{verb} takes no argument {arg:?}; run 'lastseek {verb} --help'")]
    UnexpectedArgument { arg: String, verb: String },
}
