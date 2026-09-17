//! Durable operator metrics for the disposable LastSeek projection.
//!
//! These files contain counters and timings only. They never contain query
//! text, record bodies, or LastDB data.

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::LastSeekError;

pub const QUERY_RING_CAPACITY: usize = 256;
pub const QUERY_TIMEOUT_MS: u64 = 1_000;
const MAX_CALLER_LEN: usize = 64;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct QueryMetricsFile {
    #[serde(default)]
    samples: Vec<QuerySample>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct QuerySample {
    ms: u64,
    timeout: bool,
    #[serde(default = "default_caller")]
    caller: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct LastDrain {
    pub files: u64,
    pub records: u64,
    pub tombstones: u64,
    pub compacted: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuerySummary {
    pub query_count: usize,
    pub p95_ms: u64,
    pub timeout_count: usize,
    pub callers: Vec<CallerSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CallerSummary {
    pub caller: String,
    pub query_count: usize,
    pub p95_ms: u64,
    pub timeout_count: usize,
}

fn default_caller() -> String {
    "unknown".to_string()
}

/// Keep attribution useful without allowing argv, paths, or query text to
/// create unbounded metric labels.
pub fn normalize_caller(raw: Option<&str>) -> String {
    let mut caller: String = raw
        .unwrap_or("unknown")
        .trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '/' | '-'))
        .take(MAX_CALLER_LEN)
        .collect();
    if caller.is_empty() {
        caller = default_caller();
    }
    caller
}

fn p95_ms(samples: &[&QuerySample]) -> u64 {
    let mut durations: Vec<u64> = samples.iter().map(|sample| sample.ms).collect();
    durations.sort_unstable();
    if durations.is_empty() {
        0
    } else {
        let rank = (durations.len() * 95).div_ceil(100).max(1) - 1;
        durations[rank]
    }
}

fn query_metrics_path(home: &Path) -> PathBuf {
    home.join("query-metrics.json")
}

fn query_metrics_lock_path(home: &Path) -> PathBuf {
    home.join("query-metrics.lock")
}

fn last_drain_path(home: &Path) -> PathBuf {
    home.join("last-drain.json")
}

fn last_drain_lock_path(home: &Path) -> PathBuf {
    home.join("last-drain.lock")
}

fn private_file(path: &Path) -> Result<File, LastSeekError> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(path)?)
}

fn write_private_atomic(path: &Path, bytes: &[u8]) -> Result<(), LastSeekError> {
    let tmp = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let mut file = private_file(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(tmp, path)?;
    Ok(())
}

fn private_lock(path: &Path) -> Result<File, LastSeekError> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        options.mode(0o600);
        let file = options.open(path)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        Ok(file)
    }
    #[cfg(not(unix))]
    Ok(options.open(path)?)
}

fn with_lock<T>(
    home: &Path,
    path: &Path,
    f: impl FnOnce() -> Result<T, LastSeekError>,
) -> Result<T, LastSeekError> {
    fs::create_dir_all(home)?;
    let lock = private_lock(path)?;
    lock.lock_exclusive()?;
    let result = f();
    let _ = lock.unlock();
    result
}

fn load_file<T: for<'de> Deserialize<'de> + Default>(path: &Path) -> T {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

pub fn record_query(
    home: &Path,
    elapsed_ms: u64,
    timeout: bool,
    caller: &str,
) -> Result<(), LastSeekError> {
    with_lock(home, &query_metrics_lock_path(home), || {
        let mut metrics: QueryMetricsFile = load_file(&query_metrics_path(home));
        metrics.samples.push(QuerySample {
            ms: elapsed_ms,
            timeout,
            caller: normalize_caller(Some(caller)),
        });
        if metrics.samples.len() > QUERY_RING_CAPACITY {
            let first = metrics.samples.len() - QUERY_RING_CAPACITY;
            metrics.samples.drain(..first);
        }
        write_private_atomic(
            &query_metrics_path(home),
            &serde_json::to_vec_pretty(&metrics)?,
        )
    })
}

pub fn query_summary(home: &Path) -> QuerySummary {
    let metrics: QueryMetricsFile = load_file(&query_metrics_path(home));
    let samples: Vec<&QuerySample> = metrics.samples.iter().collect();
    let mut by_caller: BTreeMap<&str, Vec<&QuerySample>> = BTreeMap::new();
    for sample in &samples {
        by_caller
            .entry(sample.caller.as_str())
            .or_default()
            .push(sample);
    }
    let callers = by_caller
        .into_iter()
        .map(|(caller, samples)| CallerSummary {
            caller: caller.to_string(),
            query_count: samples.len(),
            p95_ms: p95_ms(&samples),
            timeout_count: samples.iter().filter(|sample| sample.timeout).count(),
        })
        .collect();
    QuerySummary {
        query_count: samples.len(),
        p95_ms: p95_ms(&samples),
        timeout_count: metrics
            .samples
            .iter()
            .filter(|sample| sample.timeout)
            .count(),
        callers,
    }
}

pub fn save_last_drain(home: &Path, drain: &LastDrain) -> Result<(), LastSeekError> {
    with_lock(home, &last_drain_lock_path(home), || {
        write_private_atomic(&last_drain_path(home), &serde_json::to_vec_pretty(drain)?)
    })
}

pub fn last_drain(home: &Path) -> LastDrain {
    load_file(&last_drain_path(home))
}

pub fn mark_last_drain_compacted(home: &Path) -> Result<(), LastSeekError> {
    with_lock(home, &last_drain_lock_path(home), || {
        let path = last_drain_path(home);
        let mut drain: LastDrain = load_file(&path);
        if path.exists() {
            drain.compacted = true;
            write_private_atomic(&path, &serde_json::to_vec_pretty(&drain)?)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_uses_the_recent_ring_and_p95() {
        let root = tempfile::tempdir().unwrap();
        for ms in [10, 20, 30, 40, 50] {
            record_query(root.path(), ms, ms == 50, "brain/ask").unwrap();
        }
        let summary = query_summary(root.path());
        assert_eq!(summary.query_count, 5);
        assert_eq!(summary.p95_ms, 50);
        assert_eq!(summary.timeout_count, 1);
        assert_eq!(summary.callers[0].caller, "brain/ask");
        assert_eq!(summary.callers[0].query_count, 5);
    }

    #[test]
    fn query_samples_are_bounded() {
        let root = tempfile::tempdir().unwrap();
        for ms in 0..(QUERY_RING_CAPACITY + 3) as u64 {
            record_query(root.path(), ms, false, "manual").unwrap();
        }
        let summary = query_summary(root.path());
        assert_eq!(summary.query_count, QUERY_RING_CAPACITY);
        assert_eq!(summary.p95_ms, QUERY_RING_CAPACITY as u64 - 10);
    }

    #[test]
    fn caller_labels_are_bounded_and_grouped() {
        let root = tempfile::tempdir().unwrap();
        record_query(root.path(), 12, false, "brain/ask?query=secret").unwrap();
        record_query(root.path(), 20, true, "kanban/search").unwrap();
        let summary = query_summary(root.path());
        assert_eq!(summary.callers.len(), 2);
        assert_eq!(summary.callers[0].caller, "brain/askquerysecret");
        assert_eq!(summary.callers[1].caller, "kanban/search");
        assert_eq!(normalize_caller(Some(&"x".repeat(100))), "x".repeat(64));
    }
}
