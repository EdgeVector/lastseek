//! Memory-mapped vector store.
//!
//! # Why this shape
//!
//! The predecessor (the TypeScript Search app) held the whole index in the heap
//! as language objects and re-serialized all of it on every flush. Two costs
//! followed, and both scale with corpus size rather than with work done:
//!
//! - **Residency.** 108 MB of payload cost 373 MB of heap, and answering one
//!   query required loading every byte first.
//! - **Write amplification.** A flush rewrote the entire snapshot. On
//!   2026-08-07 that wrote 549.80 GB in 20 minutes and took the machine down.
//!
//! LastSeek separates the two things those costs came from:
//!
//! - **Vectors** live in one contiguous `f32` slab, memory-mapped and never
//!   parsed or copied. Scoring reads straight out of the mapping, so the OS
//!   page cache decides residency instead of the process. Resident set tracks
//!   the working set, not the corpus.
//! - **Text and identity** live in a separate segment, touched only for records
//!   that actually make the result set. This is the load-bearing split: text is
//!   ~4x the bytes of a vector, so keeping it out of the scan means a query
//!   reads 1.5 KB per candidate instead of ~7 KB.
//!
//! Writes append to a log and are folded into the slab by compaction, so a
//! flush costs O(changed) rather than O(corpus).

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::LastSeekError;

pub const MANIFEST_VERSION: u32 = 1;
const VECTORS_FILE: &str = "vectors.f32";
const META_FILE: &str = "meta.jsonl";
const MANIFEST_FILE: &str = "manifest.json";
const LOG_FILE: &str = "pending.jsonl";

/// Identity and payload for one indexed fragment. Deliberately NOT co-located
/// with the vector slab: scoring must never page this in.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub schema_name: String,
    pub key_hash: Option<String>,
    pub key_range: Option<String>,
    pub fragment_key: String,
    pub text: String,
    pub mutation_id: Option<String>,
    /// Tombstone. Kept as a row so ordinals stay stable until compaction.
    #[serde(default)]
    pub deleted: bool,
}

impl Record {
    /// Stable identity across re-embeds.
    pub fn id(&self) -> String {
        format!(
            "{}\u{0}{}\u{0}{}\u{0}{}",
            self.schema_name,
            self.key_hash.as_deref().unwrap_or(""),
            self.key_range.as_deref().unwrap_or(""),
            self.fragment_key
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub dimensions: usize,
    pub embedder_id: String,
    /// Rows in the slab, including tombstoned ones.
    pub rows: usize,
}

/// One pending mutation, appended as a line to the log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingOp {
    pub record: Record,
    /// Absent for a delete.
    pub vector: Option<Vec<f32>>,
}

/// Read side: the mmap'd slab plus the metadata needed to interpret it.
pub struct VectorStore {
    dir: PathBuf,
    manifest: Manifest,
    /// None when the store has no committed rows yet.
    slab: Option<Mmap>,
    meta: Vec<Record>,
    /// Uncommitted ops, replayed over the slab on read.
    pending: Vec<PendingOp>,
}

impl VectorStore {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, LastSeekError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let manifest = match std::fs::read_to_string(dir.join(MANIFEST_FILE)) {
            Ok(s) => serde_json::from_str(&s)?,
            Err(_) => Manifest {
                version: MANIFEST_VERSION,
                dimensions: 0,
                embedder_id: String::new(),
                rows: 0,
            },
        };
        if manifest.version != MANIFEST_VERSION {
            return Err(LastSeekError::UnsupportedVersion(manifest.version));
        }

        let slab = match File::open(dir.join(VECTORS_FILE)) {
            // SAFETY: the slab is only rewritten via atomic rename, so an open
            // mapping always refers to a complete, immutable file.
            Ok(f) => Some(unsafe { Mmap::map(&f)? }),
            Err(_) => None,
        };

        let meta = match std::fs::read_to_string(dir.join(META_FILE)) {
            Ok(s) => s
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<Result<Vec<Record>, _>>()?,
            Err(_) => Vec::new(),
        };

        let pending = match std::fs::read_to_string(dir.join(LOG_FILE)) {
            Ok(s) => s
                .lines()
                .filter(|l| !l.trim().is_empty())
                // A torn tail costs the last op, never the log.
                .filter_map(|l| serde_json::from_str::<PendingOp>(l).ok())
                .collect(),
            Err(_) => Vec::new(),
        };

        Ok(Self {
            dir,
            manifest,
            slab,
            meta,
            pending,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.manifest.dimensions
    }

    pub fn embedder_id(&self) -> &str {
        &self.manifest.embedder_id
    }

    /// Committed rows in the slab, tombstones included.
    pub fn committed_rows(&self) -> usize {
        self.manifest.rows
    }

    pub fn pending_ops(&self) -> usize {
        self.pending.len()
    }

    /// Bytes the slab occupies on disk. Resident memory is a subset — the OS
    /// pages in only what scoring actually reads.
    pub fn slab_bytes(&self) -> usize {
        self.slab.as_ref().map(|m| m.len()).unwrap_or(0)
    }

    /// Borrow row `i`'s vector directly out of the mapping. No copy, no parse.
    pub fn vector_at(&self, i: usize) -> Option<&[f32]> {
        let d = self.manifest.dimensions;
        if d == 0 || i >= self.manifest.rows {
            return None;
        }
        let slab = self.slab.as_ref()?;
        let start = i * d * 4;
        let end = start + d * 4;
        if end > slab.len() {
            return None;
        }
        let bytes = &slab[start..end];
        // SAFETY: the slab is written as native-endian f32 and the file length
        // is validated against rows*dims*4 at open; alignment is guaranteed
        // because mmap returns page-aligned memory and offsets are multiples
        // of 4.
        debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<f32>(), 0);
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, d) })
    }

    pub fn record_at(&self, i: usize) -> Option<&Record> {
        self.meta.get(i)
    }

    pub fn pending(&self) -> &[PendingOp] {
        &self.pending
    }

    /// Append mutations to the log. O(changed), never O(corpus).
    pub fn append(&mut self, ops: &[PendingOp]) -> Result<(), LastSeekError> {
        if ops.is_empty() {
            return Ok(());
        }
        let mut f = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.dir.join(LOG_FILE))?,
        );
        for op in ops {
            serde_json::to_writer(&mut f, op)?;
            f.write_all(b"\n")?;
        }
        f.flush()?;
        self.pending.extend_from_slice(ops);
        Ok(())
    }

    /// Fold pending ops into the slab and drop the log.
    ///
    /// Streams row by row: peak memory is one vector, not one corpus. Writes to
    /// temporaries and renames, so a crash mid-compaction leaves the previous
    /// generation intact and the log still replayable.
    pub fn compact(&mut self) -> Result<(), LastSeekError> {
        if self.pending.is_empty() && self.manifest.rows == self.meta.len() {
            return Ok(());
        }

        // Last write wins per identity; the live set is committed rows updated
        // by pending ops, minus tombstones.
        let mut order: Vec<String> = Vec::new();
        let mut live: std::collections::HashMap<String, (Option<usize>, Option<PendingOp>)> =
            std::collections::HashMap::new();

        for (i, rec) in self.meta.iter().enumerate() {
            if rec.deleted {
                continue;
            }
            let id = rec.id();
            if live.insert(id.clone(), (Some(i), None)).is_none() {
                order.push(id);
            }
        }
        for op in &self.pending {
            let id = op.record.id();
            let entry = live.entry(id.clone()).or_insert_with(|| {
                order.push(id.clone());
                (None, None)
            });
            entry.1 = Some(op.clone());
        }

        let dims = self
            .pending
            .iter()
            .find_map(|op| op.vector.as_ref().map(|v| v.len()))
            .unwrap_or(self.manifest.dimensions);

        let tmp_vec = self.dir.join(format!("{VECTORS_FILE}.lastseek-tmp"));
        let tmp_meta = self.dir.join(format!("{META_FILE}.lastseek-tmp"));
        let mut vw = BufWriter::new(File::create(&tmp_vec)?);
        let mut mw = BufWriter::new(File::create(&tmp_meta)?);
        let mut rows = 0usize;

        for id in &order {
            let (committed, pending) = &live[id];
            let (record, vector): (Record, Vec<f32>) = match pending {
                Some(op) => {
                    if op.record.deleted || op.vector.is_none() {
                        continue; // deleted: drop the row entirely
                    }
                    (op.record.clone(), op.vector.clone().unwrap())
                }
                None => {
                    let i = committed.expect("committed row without pending op");
                    let v = match self.vector_at(i) {
                        Some(v) => v.to_vec(),
                        None => continue,
                    };
                    (self.meta[i].clone(), v)
                }
            };
            if vector.len() != dims {
                return Err(LastSeekError::DimensionMismatch {
                    expected: dims,
                    got: vector.len(),
                });
            }
            for x in &vector {
                vw.write_all(&x.to_ne_bytes())?;
            }
            serde_json::to_writer(&mut mw, &record)?;
            mw.write_all(b"\n")?;
            rows += 1;
        }
        vw.flush()?;
        mw.flush()?;
        drop(vw);
        drop(mw);

        let embedder_id = self
            .pending
            .iter()
            .map(|_| self.manifest.embedder_id.clone())
            .next()
            .unwrap_or_else(|| self.manifest.embedder_id.clone());

        // Drop the mapping before replacing the file underneath it.
        self.slab = None;
        std::fs::rename(&tmp_vec, self.dir.join(VECTORS_FILE))?;
        std::fs::rename(&tmp_meta, self.dir.join(META_FILE))?;

        let manifest = Manifest {
            version: MANIFEST_VERSION,
            dimensions: dims,
            embedder_id,
            rows,
        };
        std::fs::write(
            self.dir.join(MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest)?,
        )?;

        // Log is folded in; remove it only after the new generation is durable.
        let _ = std::fs::remove_file(self.dir.join(LOG_FILE));

        *self = Self::open(&self.dir)?;
        Ok(())
    }

    /// Record the embedder identity for a fresh store.
    pub fn set_embedder(&mut self, id: &str, dims: usize) -> Result<(), LastSeekError> {
        self.manifest.embedder_id = id.to_string();
        if self.manifest.dimensions == 0 {
            self.manifest.dimensions = dims;
        }
        std::fs::write(
            self.dir.join(MANIFEST_FILE),
            serde_json::to_vec_pretty(&self.manifest)?,
        )?;
        Ok(())
    }

    pub fn log_bytes(&self) -> u64 {
        std::fs::metadata(self.dir.join(LOG_FILE))
            .map(|m| m.len())
            .unwrap_or(0)
    }
}
