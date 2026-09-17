//! Durable operator metrics for the disposable LastSeek projection.
//!
//! These files contain counters and timings only. They never contain query
//! text, record bodies, or LastDB data.

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::LastSeekError;

pub const QUERY_RING_CAPACITY: usize = 256;
pub const QUERY_TIMEOUT_MS: u64 = 1_000;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct QueryMetricsFile {
    #[serde(default)]
    samples: Vec<QuerySample>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct QuerySample {
    ms: u64,
    timeout: bool,
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

pub fn record_query(home: &Path, elapsed_ms: u64, timeout: bool) -> Result<(), LastSeekError> {
    with_lock(home, &query_metrics_lock_path(home), || {
        let mut metrics: QueryMetricsFile = load_file(&query_metrics_path(home));
        metrics.samples.push(QuerySample {
            ms: elapsed_ms,
            timeout,
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
    let mut durations: Vec<u64> = metrics.samples.iter().map(|sample| sample.ms).collect();
    durations.sort_unstable();
    let p95_ms = if durations.is_empty() {
        0
    } else {
        let rank = (durations.len() * 95).div_ceil(100).max(1) - 1;
        durations[rank]
    };
    QuerySummary {
        query_count: metrics.samples.len(),
        p95_ms,
        timeout_count: metrics
            .samples
            .iter()
            .filter(|sample| sample.timeout)
            .count(),
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
            record_query(root.path(), ms, ms == 50).unwrap();
        }
        let summary = query_summary(root.path());
        assert_eq!(summary.query_count, 5);
        assert_eq!(summary.p95_ms, 50);
        assert_eq!(summary.timeout_count, 1);
    }

    #[test]
    fn query_samples_are_bounded() {
        let root = tempfile::tempdir().unwrap();
        for ms in 0..(QUERY_RING_CAPACITY + 3) as u64 {
            record_query(root.path(), ms, false).unwrap();
        }
        let summary = query_summary(root.path());
        assert_eq!(summary.query_count, QUERY_RING_CAPACITY);
        assert_eq!(summary.p95_ms, QUERY_RING_CAPACITY as u64 - 10);
    }
}
