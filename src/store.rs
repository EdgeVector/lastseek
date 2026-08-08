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
//!   reads a few bytes per candidate instead of the whole record.
//!
//! Writes append to a log and are folded into the slab by compaction, so a
//! flush costs O(changed) rather than O(corpus).
//!
//! # Generations (crash-safe compaction)
//!
//! A store is one atomic pointer (`CURRENT`) plus a sequence of immutable
//! generation directories (`gen-NNNNNNN/`), each holding a self-consistent
//! `vectors.f32` + `rows.bin` + `meta.jsonl` + `manifest.json` quartet.
//! Compaction builds the *next* generation entirely inside a `.staging`
//! sibling directory, fsyncs it, renames it into place, and only then swaps
//! `CURRENT` (itself write-tmp-fsync-rename) to point at it. Nothing ever
//! reads from a generation until `CURRENT` names it, so a crash at any point
//! before the final rename leaves the previous generation — and the pending
//! log — completely untouched. The slab/meta desync the predecessor could hit
//! (a new slab paired with old, unrelated meta) is structurally impossible
//! here: both live in the same directory, swapped by one rename.
//!
//! `rows.bin` is a fixed-width, mmap'd side-table (`id_hash`, `text_off`,
//! `text_len`, `schema_id`, `flags`) that lets scoring and schema-scope
//! filtering happen without ever touching `meta.jsonl`. `meta.jsonl` is
//! parsed one line at a time, only for rows that survive into the result set
//! — that's what makes the residency property real rather than aspirational.
//!
//! # Durability decision: process-crash, not power-loss
//!
//! Every sync in the generation-swap path above is `File::sync_all`, i.e.
//! `fsync(2)`. On macOS/APFS, `fsync` does **not** flush the drive's write
//! cache — only `fcntl(fd, F_FULLFSYNC)` does. `laststore` (`src/durability.rs`,
//! LastDB's Storage v2 engine) pays that extra barrier on every dirty-file sync,
//! because it holds the data of record and a torn write there is real,
//! unrecoverable data loss.
//!
//! LastSeek makes the opposite call, deliberately: this index holds nothing
//! that doesn't already live elsewhere (LastDB's own storage). A generation
//! torn by an actual power loss costs a rebuild — `lastseek` re-embeds and
//! re-appends from source — never data. So LastSeek does not call
//! `F_FULLFSYNC`, and does not need to. The crash-safety argument two
//! paragraphs up (interior atomicity via staging + rename) holds fully against
//! **process** crashes — panics, `kill -9`, OOM — because the OS page cache
//! survives those regardless of whether the underlying fsync reached the
//! platter. It does not, and is not meant to, hold against power loss.

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::embed::Embedder;
use crate::LastSeekError;

pub const MANIFEST_VERSION: u32 = 2;
const VECTORS_FILE: &str = "vectors.f32";
const ROWS_FILE: &str = "rows.bin";
const META_FILE: &str = "meta.jsonl";
const MANIFEST_FILE: &str = "manifest.json";
const LOG_FILE: &str = "pending.jsonl";
const LOCK_FILE: &str = "writer.lock";
const CURRENT_FILE: &str = "CURRENT";
const EMBEDDER_FILE: &str = "embedder.json";

/// Width in bytes of one `rows.bin` entry: `id_hash`(8) + `text_off`(8) +
/// `text_len`(4) + `schema_id`(4) + `flags`(4) + 4 reserved.
const ROW_WIDTH: usize = 32;

/// Stable, deterministic identity hash used for shadow lookups and tie-break
/// ordering without allocating or comparing the identity string itself.
/// Persisted in `rows.bin`, so this must stay fixed across versions.
pub(crate) fn id_hash(id: &str) -> u64 {
    let mut h: u64 = 1469598103934665603; // FNV-1a offset basis
    for b in id.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211); // FNV-1a prime
    }
    h
}

fn gen_dir_name(generation: u64) -> String {
    format!("gen-{generation:07}")
}

/// Identity and payload for one indexed fragment. Deliberately NOT co-located
/// with the vector slab: scoring must never page this in.
///
/// # The ingest convention, stated
///
/// `schema_identity` holds a **canonical schema identity key** — the
/// `identity_hash` the Schema Service publishes for the schema, resolved from
/// the registry `name` a host batch carries (see `crate::schema`). It is never
/// a readable descriptive name, and never the node's locally recomputed
/// identity hash.
///
/// This used to be an unstated `schema_name: String`, and that omission was the
/// whole of two live defects: the predecessor's host wrote the registry `name`
/// while its coverage metric counted under `identity_hash`, so coverage read
/// 2.7% over a working index; and `--schema Card` matched nothing because
/// nothing on the write path had ever written a descriptive name. The field
/// deserializes from `schema_name` too, so batches and JSONL written against
/// the old shape still load.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Record {
    #[serde(alias = "schema_name")]
    pub schema_identity: String,
    pub key_hash: Option<String>,
    pub key_range: Option<String>,
    pub fragment_key: String,
    pub text: String,
    pub mutation_id: Option<String>,
    /// Tombstone on the pending log only. Compaction drops deleted rows
    /// entirely, so a committed row is never marked deleted.
    #[serde(default)]
    pub deleted: bool,
}

impl Record {
    /// Stable identity across re-embeds.
    pub fn id(&self) -> String {
        format!(
            "{}\u{0}{}\u{0}{}\u{0}{}",
            self.schema_identity,
            self.key_hash.as_deref().unwrap_or(""),
            self.key_range.as_deref().unwrap_or(""),
            self.fragment_key
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    /// Generation number this manifest describes. 0 means "no committed
    /// generation yet" and has no on-disk `gen-*` directory.
    #[serde(default)]
    pub generation: u64,
    pub dimensions: usize,
    pub embedder_id: String,
    #[serde(default)]
    pub embedder_is_neural: bool,
    /// Interned schema names; a row's `schema_id` indexes into this.
    #[serde(default)]
    pub schemas: Vec<String>,
    /// Rows in the slab. Compaction never writes tombstones, so this is also
    /// the live row count.
    pub rows: usize,
    #[serde(default)]
    pub vectors_bytes: u64,
    #[serde(default)]
    pub rows_bytes: u64,
    #[serde(default)]
    pub meta_bytes: u64,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            version: MANIFEST_VERSION,
            generation: 0,
            dimensions: 0,
            embedder_id: String::new(),
            embedder_is_neural: false,
            schemas: Vec::new(),
            rows: 0,
            vectors_bytes: 0,
            rows_bytes: 0,
            meta_bytes: 0,
        }
    }
}

/// One pending mutation, appended as a line to the log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingOp {
    pub record: Record,
    /// Absent for a delete.
    pub vector: Option<Vec<f32>>,
}

/// A `rows.bin` entry, decoded. Everything here is readable without touching
/// `meta.jsonl`, which is what keeps scoring and schema-scope filtering off
/// the text segment.
#[derive(Debug, Clone, Copy)]
pub struct RowMeta {
    pub id_hash: u64,
    pub text_off: u64,
    pub text_len: u32,
    pub schema_id: u32,
    pub flags: u32,
}

impl RowMeta {
    pub fn deleted(&self) -> bool {
        self.flags & 1 != 0
    }

    fn encode(&self) -> [u8; ROW_WIDTH] {
        let mut buf = [0u8; ROW_WIDTH];
        buf[0..8].copy_from_slice(&self.id_hash.to_ne_bytes());
        buf[8..16].copy_from_slice(&self.text_off.to_ne_bytes());
        buf[16..20].copy_from_slice(&self.text_len.to_ne_bytes());
        buf[20..24].copy_from_slice(&self.schema_id.to_ne_bytes());
        buf[24..28].copy_from_slice(&self.flags.to_ne_bytes());
        buf
    }

    fn decode(bytes: &[u8]) -> Self {
        RowMeta {
            id_hash: u64::from_ne_bytes(bytes[0..8].try_into().unwrap()),
            text_off: u64::from_ne_bytes(bytes[8..16].try_into().unwrap()),
            text_len: u32::from_ne_bytes(bytes[16..20].try_into().unwrap()),
            schema_id: u32::from_ne_bytes(bytes[20..24].try_into().unwrap()),
            flags: u32::from_ne_bytes(bytes[24..28].try_into().unwrap()),
        }
    }
}

/// Table-driven crash simulation for `compact()`, gated behind a cargo
/// feature so no fault-injection code links into a default build.
#[cfg(feature = "fault-injection")]
pub mod fault {
    use std::cell::Cell;

    /// Every interior point of `compact()` where a crash is distinguishable
    /// from any other: everything before `AfterStagingDirFsync` is still
    /// invisible garbage (nothing references the staging dir yet), and
    /// everything from `AfterCurrentRename` onward has already committed to
    /// the new generation. Keep `ALL` in lockstep with `compact()` — the
    /// matrix test asserts the count so a new interior step fails the build
    /// until it gets a case.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    pub enum FaultPoint {
        AfterStagingBuild = 0,
        AfterStagingDirFsync = 1,
        AfterGenerationRename = 2,
        AfterGenerationRenameFsync = 3,
        AfterCurrentTmpWritten = 4,
        AfterCurrentRename = 5,
        AfterCurrentRenameFsync = 6,
        AfterPendingRemoved = 7,
    }

    impl FaultPoint {
        pub const ALL: &'static [FaultPoint] = &[
            FaultPoint::AfterStagingBuild,
            FaultPoint::AfterStagingDirFsync,
            FaultPoint::AfterGenerationRename,
            FaultPoint::AfterGenerationRenameFsync,
            FaultPoint::AfterCurrentTmpWritten,
            FaultPoint::AfterCurrentRename,
            FaultPoint::AfterCurrentRenameFsync,
            FaultPoint::AfterPendingRemoved,
        ];
    }

    thread_local! {
        // Thread-local, not a shared global: cargo test runs each #[test] on
        // its own thread, and compact() calls from unrelated tests must
        // never see this test's armed fault point.
        static ARMED: Cell<u8> = const { Cell::new(255) };
        // Counts calls into `durable_sync`, i.e. every fsync that actually ran.
        // A crash (panic) proves ordering, not durability — the OS page cache
        // survives a panic regardless of whether the fsync below it ran. This
        // counter is the thing that catches a future edit silently dropping a
        // sync: the matrix test asserts the exact expected count.
        static SYNC_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    /// Arm the next `compact()` call **on this thread** to panic once it
    /// reaches `point`.
    pub fn arm(point: FaultPoint) {
        ARMED.with(|a| a.set(point as u8));
    }

    pub fn disarm() {
        ARMED.with(|a| a.set(255));
    }

    pub(crate) fn maybe_fire(point: FaultPoint) {
        let armed = ARMED.with(|a| a.get());
        if armed == point as u8 {
            disarm();
            panic!("fault-injection: simulated crash at {point:?}");
        }
    }

    pub(crate) fn record_sync() {
        SYNC_COUNT.with(|c| c.set(c.get() + 1));
    }

    /// Zero the sync counter on this thread. Call before the operation under
    /// test so an unrelated setup call (e.g. `append`) doesn't inflate the count.
    pub fn reset_sync_count() {
        SYNC_COUNT.with(|c| c.set(0));
    }

    /// Durability syncs recorded since the last `reset_sync_count`.
    pub fn sync_count() -> usize {
        SYNC_COUNT.with(|c| c.get())
    }
}

#[cfg(feature = "fault-injection")]
macro_rules! fault_point {
    ($point:expr) => {
        fault::maybe_fire($point)
    };
}
#[cfg(not(feature = "fault-injection"))]
macro_rules! fault_point {
    ($point:expr) => {};
}

/// Advisory cross-process mutual exclusion for the mutating half of the API
/// (`append`, `compact`). Held for the whole call, closing the race where a
/// concurrent append's op could be lost by a compaction that read the log
/// before the append landed and deleted it after.
struct WriterLock(File);

impl Drop for WriterLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.0);
    }
}

fn lock_writer(dir: &Path) -> Result<WriterLock, LastSeekError> {
    let f = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join(LOCK_FILE))?;
    fs2::FileExt::lock_exclusive(&f)?;
    Ok(WriterLock(f))
}

/// `File::sync_all`, counted under `fault-injection` so the compaction fault
/// matrix can assert that every durability-critical sync actually ran instead
/// of only proving crash-ordering (see the module doc's "process-crash vs.
/// power-loss" section).
fn durable_sync(file: &File) -> Result<(), LastSeekError> {
    file.sync_all()?;
    #[cfg(feature = "fault-injection")]
    fault::record_sync();
    Ok(())
}

/// fsync a directory's own entry, so renames/creates inside it are durable.
fn sync_dir(dir: &Path) -> Result<(), LastSeekError> {
    durable_sync(&File::open(dir)?)
}

fn load_pending(dir: &Path) -> Result<Vec<PendingOp>, LastSeekError> {
    match std::fs::read_to_string(dir.join(LOG_FILE)) {
        Ok(s) => Ok(s
            .lines()
            .filter(|l| !l.trim().is_empty())
            // A torn tail costs the last op, never the log.
            .filter_map(|l| serde_json::from_str::<PendingOp>(l).ok())
            .collect()),
        Err(_) => Ok(Vec::new()),
    }
}

/// Read side: the mmap'd slab plus the side-tables needed to interpret it.
pub struct VectorStore {
    dir: PathBuf,
    generation: u64,
    manifest: Manifest,
    /// None when the store has no committed rows yet.
    slab: Option<Mmap>,
    rows_map: Option<Mmap>,
    meta_map: Option<Mmap>,
    /// Uncommitted ops, replayed over the slab on read.
    pending: Vec<PendingOp>,
    /// Counts calls that actually parsed a line out of `meta.jsonl`. Exists so
    /// tests can prove residency as a deterministic count instead of an RSS
    /// measurement.
    meta_reads: Cell<usize>,
}

impl VectorStore {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, LastSeekError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        // A pre-generations (v1) store has a top-level manifest.json and no
        // CURRENT pointer. Refuse it explicitly rather than misreading it as
        // an empty v2 store.
        if !dir.join(CURRENT_FILE).exists() {
            if let Ok(s) = std::fs::read_to_string(dir.join(MANIFEST_FILE)) {
                let v: serde_json::Value = serde_json::from_str(&s)?;
                let version = v.get("version").and_then(|x| x.as_u64()).unwrap_or(1) as u32;
                if version != MANIFEST_VERSION {
                    return Err(LastSeekError::UnsupportedVersion(version));
                }
            }
        }

        let generation = match std::fs::read_to_string(dir.join(CURRENT_FILE)) {
            Ok(s) => s.trim().parse::<u64>().unwrap_or(0),
            Err(_) => 0,
        };

        let (mut manifest, slab, rows_map, meta_map) = if generation == 0 {
            (Manifest::default(), None, None, None)
        } else {
            let gen_dir = dir.join(gen_dir_name(generation));
            let manifest: Manifest =
                serde_json::from_str(&std::fs::read_to_string(gen_dir.join(MANIFEST_FILE))?)?;
            if manifest.version != MANIFEST_VERSION {
                return Err(LastSeekError::UnsupportedVersion(manifest.version));
            }

            let open_checked = |name: &str, want_len: u64| -> Result<Option<Mmap>, LastSeekError> {
                let f = File::open(gen_dir.join(name))?;
                let len = f.metadata()?.len();
                if len != want_len {
                    return Err(LastSeekError::CorruptGeneration(format!(
                        "{name} is {len} bytes, manifest for generation {} expects {want_len}",
                        manifest.generation
                    )));
                }
                if len == 0 {
                    return Ok(None);
                }
                // SAFETY: generation directories are only ever written via a
                // build-in-staging-then-rename protocol, so any file reachable
                // through a CURRENT-named generation is complete and immutable
                // for the lifetime of this mapping.
                Ok(Some(unsafe { Mmap::map(&f)? }))
            };

            let slab = open_checked(VECTORS_FILE, manifest.vectors_bytes)?;
            let rows_map = open_checked(ROWS_FILE, manifest.rows_bytes)?;
            let meta_map = open_checked(META_FILE, manifest.meta_bytes)?;
            (manifest, slab, rows_map, meta_map)
        };

        // Before any generation exists, embedder identity is durable via a
        // small sidecar so `status`/`add` agree across separate processes.
        if generation == 0 {
            if let Ok(s) = std::fs::read_to_string(dir.join(EMBEDDER_FILE)) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                    if let Some(id) = v.get("embedder_id").and_then(|x| x.as_str()) {
                        manifest.embedder_id = id.to_string();
                    }
                    if let Some(d) = v.get("dimensions").and_then(|x| x.as_u64()) {
                        manifest.dimensions = d as usize;
                    }
                    if let Some(n) = v.get("embedder_is_neural").and_then(|x| x.as_bool()) {
                        manifest.embedder_is_neural = n;
                    }
                }
            }
        }

        let pending = load_pending(&dir)?;

        Ok(Self {
            dir,
            generation,
            manifest,
            slab,
            rows_map,
            meta_map,
            pending,
            meta_reads: Cell::new(0),
        })
    }

    pub fn dimensions(&self) -> usize {
        self.manifest.dimensions
    }

    pub fn embedder_id(&self) -> &str {
        &self.manifest.embedder_id
    }

    pub fn embedder_is_neural(&self) -> bool {
        self.manifest.embedder_is_neural
    }

    /// Committed rows in the slab. Compaction never writes tombstones, so
    /// every committed row is live.
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

    /// Bytes the text segment occupies on disk, for residency diagnostics.
    pub fn meta_bytes(&self) -> u64 {
        self.manifest.meta_bytes
    }

    /// How many times a call actually parsed a `meta.jsonl` line. Resets on
    /// reopen; exists for the residency test, not general use.
    pub fn meta_reads(&self) -> usize {
        self.meta_reads.get()
    }

    /// Interned id for a schema identity key, if any committed row uses it.
    pub fn schema_id(&self, identity: &str) -> Option<u32> {
        self.manifest
            .schemas
            .iter()
            .position(|s| s == identity)
            .map(|p| p as u32)
    }

    /// Every schema identity key this generation has interned.
    ///
    /// Scope resolution consults this as its last tier, so a schema whose owner
    /// has not published to the Schema Service stays searchable by the key
    /// ingest saw rather than becoming unaddressable.
    pub fn interned_schemas(&self) -> &[String] {
        &self.manifest.schemas
    }

    /// Logical live vector rows per schema, including the pending overlay.
    ///
    /// Pending mutations shadow committed rows just as they do in search. The
    /// newest pending mutation for one record wins, so repeated upserts do not
    /// inflate coverage and a pending tombstone removes the committed row.
    pub fn schema_row_counts(&self) -> BTreeMap<String, usize> {
        let shadowed: HashSet<u64> = self
            .pending
            .iter()
            .map(|op| id_hash(&op.record.id()))
            .collect();
        let mut counts = BTreeMap::new();
        for i in 0..self.committed_rows() {
            let Some(row) = self.row_at(i) else {
                continue;
            };
            if row.deleted() || shadowed.contains(&row.id_hash) {
                continue;
            }
            let Some(identity) = self.manifest.schemas.get(row.schema_id as usize) else {
                continue;
            };
            *counts.entry(identity.clone()).or_insert(0) += 1;
        }

        let mut seen = HashSet::new();
        for op in self.pending.iter().rev() {
            if !seen.insert(id_hash(&op.record.id())) {
                continue;
            }
            if !op.record.deleted && op.vector.is_some() {
                *counts.entry(op.record.schema_identity.clone()).or_insert(0) += 1;
            }
        }
        counts
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
        // SAFETY: the slab is written as native-endian f32 and the file
        // length is validated against rows*dims*4 at open (via
        // vectors_bytes); alignment is guaranteed because mmap returns
        // page-aligned memory and offsets are multiples of 4.
        debug_assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<f32>(), 0);
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, d) })
    }

    /// Read row `i`'s side-table entry: identity hash, schema id, flags. Never
    /// touches `meta.jsonl`, which is what lets scoring filter by schema and
    /// tombstone state without paging in text.
    pub fn row_at(&self, i: usize) -> Option<RowMeta> {
        if i >= self.manifest.rows {
            return None;
        }
        let map = self.rows_map.as_ref()?;
        let start = i * ROW_WIDTH;
        let end = start + ROW_WIDTH;
        if end > map.len() {
            return None;
        }
        Some(RowMeta::decode(&map[start..end]))
    }

    /// Parse row `i`'s full record out of `meta.jsonl`, using the byte range
    /// `rows.bin` recorded for it. This is the only path that touches the
    /// text segment, and it is O(1) — a direct offset read, not a scan.
    pub fn record_at(&self, i: usize) -> Option<Record> {
        let row = self.row_at(i)?;
        let map = self.meta_map.as_ref()?;
        let start = row.text_off as usize;
        let end = start.checked_add(row.text_len as usize)?;
        if end > map.len() {
            return None;
        }
        self.meta_reads.set(self.meta_reads.get() + 1);
        serde_json::from_slice(&map[start..end]).ok()
    }

    pub fn pending(&self) -> &[PendingOp] {
        &self.pending
    }

    /// Append mutations to the log. O(changed), never O(corpus).
    ///
    /// Every vector's length is checked against the bound embedder's
    /// dimensionality **before** it reaches the log. A wedge that used to be
    /// merely detected later (at compaction, or as a silently truncated
    /// search) is now unrepresentable: a mixed-dimension store can no longer
    /// be built in the first place.
    pub fn append(&mut self, ops: &[PendingOp]) -> Result<(), LastSeekError> {
        if ops.is_empty() {
            return Ok(());
        }
        if self.manifest.dimensions != 0 {
            for op in ops {
                if let Some(v) = &op.vector {
                    if v.len() != self.manifest.dimensions {
                        return Err(LastSeekError::DimensionMismatch {
                            expected: self.manifest.dimensions,
                            got: v.len(),
                        });
                    }
                }
            }
        }
        let _guard = lock_writer(&self.dir)?;
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join(LOG_FILE))?;
        {
            let mut w = BufWriter::new(&f);
            for op in ops {
                serde_json::to_writer(&mut w, op)?;
                w.write_all(b"\n")?;
            }
            w.flush()?;
        }
        durable_sync(&f)?;
        self.pending.extend_from_slice(ops);
        Ok(())
    }

    /// Fold pending ops into a fresh generation and swap `CURRENT` to point at
    /// it. See the module doc for the crash-safety argument; in short: every
    /// byte of the new generation is written, fsynced and closed inside a
    /// `.staging` directory before anything durable references it, so a crash
    /// at any point up to the final `CURRENT` rename leaves the previous
    /// generation, and the pending log, entirely intact and re-compactable.
    pub fn compact(&mut self) -> Result<(), LastSeekError> {
        #[cfg(feature = "fault-injection")]
        use fault::FaultPoint;

        let _guard = lock_writer(&self.dir)?;

        // Re-read fresh under the lock: a concurrent process may have
        // appended since this handle's last open()/compact(), and folding
        // only what was in memory would silently drop that op when the log
        // is removed at the end of this call.
        let pending = load_pending(&self.dir)?;
        if pending.is_empty() {
            return Ok(());
        }

        // Safe because we hold the writer lock: no concurrent compactor could
        // be using a staging dir left over from a prior interrupted run.
        for entry in std::fs::read_dir(&self.dir)?.flatten() {
            if entry.file_name().to_string_lossy().ends_with(".staging") {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }

        // Last write wins per identity; the live set is committed rows
        // updated by pending ops, minus deletes.
        let mut order: Vec<String> = Vec::new();
        let mut live: HashMap<String, (Option<usize>, Option<&PendingOp>)> = HashMap::new();

        for i in 0..self.manifest.rows {
            let rec = self.record_at(i).ok_or_else(|| {
                LastSeekError::CorruptGeneration(format!("row {i} unreadable during compaction"))
            })?;
            let id = rec.id();
            if live.insert(id.clone(), (Some(i), None)).is_none() {
                order.push(id);
            }
        }
        for op in &pending {
            let id = op.record.id();
            let entry = live.entry(id.clone()).or_insert_with(|| {
                order.push(id.clone());
                (None, None)
            });
            entry.1 = Some(op);
        }

        let dims = pending
            .iter()
            .find_map(|op| op.vector.as_ref().map(|v| v.len()))
            .unwrap_or(self.manifest.dimensions);

        let new_generation = self.generation + 1;
        let staging = self
            .dir
            .join(format!("{}.staging", gen_dir_name(new_generation)));
        if staging.exists() {
            std::fs::remove_dir_all(&staging)?;
        }

        if let Err(e) = self.build_generation(&staging, &order, &live, dims, new_generation) {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(e);
        }
        fault_point!(FaultPoint::AfterStagingBuild);

        sync_dir(&staging)?;
        fault_point!(FaultPoint::AfterStagingDirFsync);

        let final_dir = self.dir.join(gen_dir_name(new_generation));
        if final_dir.exists() {
            std::fs::remove_dir_all(&final_dir)?;
        }
        std::fs::rename(&staging, &final_dir)?;
        fault_point!(FaultPoint::AfterGenerationRename);
        sync_dir(&self.dir)?;
        fault_point!(FaultPoint::AfterGenerationRenameFsync);

        // Drop mappings before the pointer swap so this handle never mixes
        // an old view with the new generation once CURRENT moves.
        self.slab = None;
        self.rows_map = None;
        self.meta_map = None;

        let current_tmp = self.dir.join(format!("{CURRENT_FILE}.tmp"));
        {
            let mut f = File::create(&current_tmp)?;
            f.write_all(new_generation.to_string().as_bytes())?;
            durable_sync(&f)?;
        }
        fault_point!(FaultPoint::AfterCurrentTmpWritten);
        std::fs::rename(&current_tmp, self.dir.join(CURRENT_FILE))?;
        fault_point!(FaultPoint::AfterCurrentRename);
        sync_dir(&self.dir)?;
        fault_point!(FaultPoint::AfterCurrentRenameFsync);

        // The generation swap is durable and every op is now represented in
        // it. Remove the log last, so a crash before this point always
        // leaves it fully intact and safely re-foldable.
        let _ = std::fs::remove_file(self.dir.join(LOG_FILE));
        fault_point!(FaultPoint::AfterPendingRemoved);

        self.gc_old_generations(new_generation)?;

        *self = Self::open(&self.dir)?;
        Ok(())
    }

    /// Write one complete, self-consistent generation into `staging`. Nothing
    /// outside `compact()` ever sees this directory before it is renamed into
    /// its final `gen-N` name.
    fn build_generation(
        &self,
        staging: &Path,
        order: &[String],
        live: &HashMap<String, (Option<usize>, Option<&PendingOp>)>,
        dims: usize,
        new_generation: u64,
    ) -> Result<(), LastSeekError> {
        std::fs::create_dir_all(staging)?;

        let vectors_f = File::create(staging.join(VECTORS_FILE))?;
        let rows_f = File::create(staging.join(ROWS_FILE))?;
        let meta_f = File::create(staging.join(META_FILE))?;
        let mut vw = BufWriter::new(&vectors_f);
        let mut rw = BufWriter::new(&rows_f);
        let mut mw = BufWriter::new(&meta_f);

        let mut schemas: Vec<String> = Vec::new();
        let mut schema_ids: HashMap<String, u32> = HashMap::new();
        let mut meta_off: u64 = 0;
        let mut rows = 0usize;

        for id in order {
            let (committed, pend) = &live[id];
            let (record, vector): (Record, Vec<f32>) = match pend {
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
                        None => {
                            return Err(LastSeekError::CorruptGeneration(format!(
                                "row {i} has metadata but no vector during compaction"
                            )));
                        }
                    };
                    let rec = self.record_at(i).ok_or_else(|| {
                        LastSeekError::CorruptGeneration(format!(
                            "row {i} unreadable during compaction"
                        ))
                    })?;
                    (rec, v)
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
            let line = serde_json::to_vec(&record)?;
            let schema_id = *schema_ids
                .entry(record.schema_identity.clone())
                .or_insert_with(|| {
                    schemas.push(record.schema_identity.clone());
                    (schemas.len() - 1) as u32
                });
            let row = RowMeta {
                id_hash: id_hash(&record.id()),
                text_off: meta_off,
                text_len: line.len() as u32,
                schema_id,
                flags: 0,
            };
            rw.write_all(&row.encode())?;
            mw.write_all(&line)?;
            mw.write_all(b"\n")?;
            meta_off += line.len() as u64 + 1;
            rows += 1;
        }

        vw.flush()?;
        rw.flush()?;
        mw.flush()?;
        drop(vw);
        drop(rw);
        drop(mw);
        durable_sync(&vectors_f)?;
        durable_sync(&rows_f)?;
        durable_sync(&meta_f)?;
        drop(vectors_f);
        drop(rows_f);
        drop(meta_f);

        let manifest = Manifest {
            version: MANIFEST_VERSION,
            generation: new_generation,
            dimensions: dims,
            embedder_id: self.manifest.embedder_id.clone(),
            embedder_is_neural: self.manifest.embedder_is_neural,
            schemas,
            rows,
            vectors_bytes: (rows * dims * 4) as u64,
            rows_bytes: (rows * ROW_WIDTH) as u64,
            meta_bytes: meta_off,
        };
        let mf = File::create(staging.join(MANIFEST_FILE))?;
        serde_json::to_writer_pretty(&mf, &manifest)?;
        durable_sync(&mf)?;

        Ok(())
    }

    /// Remove generation directories other than `keep`. Only ever called
    /// immediately after the pointer swap to `keep` is durable, under the
    /// writer lock, so this can never race a reader (readers only ever open
    /// the generation CURRENT names) or another compactor.
    fn gc_old_generations(&self, keep: u64) -> Result<(), LastSeekError> {
        for entry in std::fs::read_dir(&self.dir)?.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(n) = name
                .strip_prefix("gen-")
                .and_then(|s| s.parse::<u64>().ok())
            {
                if n != keep && entry.path().is_dir() {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
        Ok(())
    }

    /// Bind the embedder identity. Durable immediately via a small sidecar
    /// (there may be no generation directory yet to carry it), and carried
    /// forward into every future generation's manifest by `compact()`.
    ///
    /// This is a boundary, not documentation: once the store holds any data
    /// (committed rows or a pending log), rebinding a different id or a
    /// different dimensionality is refused outright — that mismatch is what
    /// let a store accept vectors it could never consistently search or
    /// compact again.
    ///
    /// Neural and fixture embedders are also mutually exclusive once bound:
    /// `manifest.embedder_is_neural` permanently marks a fixture-built store,
    /// and a fixture can only bind onto an already-neural store with an
    /// explicit `LASTSEEK_ALLOW_FIXTURE=1` opt-in (there is no opt-in the
    /// other way — a fixture-marked store never legitimately gains real
    /// vectors).
    pub fn bind_embedder(&mut self, embedder: &dyn Embedder) -> Result<(), LastSeekError> {
        let id = embedder.id();
        let dims = embedder.dimensions();
        let is_neural = embedder.is_neural();
        let already_bound = !self.manifest.embedder_id.is_empty();
        let has_data = self.manifest.rows > 0 || !self.pending.is_empty();

        if already_bound && has_data {
            if self.manifest.embedder_id != id || self.manifest.dimensions != dims {
                return Err(LastSeekError::EmbedderBoundary(format!(
                    "store is bound to embedder {:?} (dims={}); refusing to rebind to {:?} \
                     (dims={}) with data already present",
                    self.manifest.embedder_id, self.manifest.dimensions, id, dims
                )));
            }
            if self.manifest.embedder_is_neural != is_neural {
                if is_neural {
                    return Err(LastSeekError::EmbedderBoundary(format!(
                        "store is permanently fixture-marked (embedder {:?}); refusing to bind \
                         neural embedder {id:?}",
                        self.manifest.embedder_id
                    )));
                }
                if std::env::var("LASTSEEK_ALLOW_FIXTURE").as_deref() != Ok("1") {
                    return Err(LastSeekError::EmbedderBoundary(format!(
                        "store holds neural vectors (embedder {:?}); refusing to bind fixture \
                         embedder {id:?} without LASTSEEK_ALLOW_FIXTURE=1",
                        self.manifest.embedder_id
                    )));
                }
            }
        }

        self.manifest.embedder_id = id.to_string();
        self.manifest.embedder_is_neural = is_neural;
        self.manifest.dimensions = dims;
        let sidecar = serde_json::json!({
            "embedder_id": self.manifest.embedder_id,
            "dimensions": self.manifest.dimensions,
            "embedder_is_neural": self.manifest.embedder_is_neural,
        });
        std::fs::write(
            self.dir.join(EMBEDDER_FILE),
            serde_json::to_vec_pretty(&sidecar)?,
        )?;
        Ok(())
    }

    pub fn log_bytes(&self) -> u64 {
        std::fs::metadata(self.dir.join(LOG_FILE))
            .map(|m| m.len())
            .unwrap_or(0)
    }
}
