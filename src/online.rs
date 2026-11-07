//! Build a private projection, then publish its index and feed checkpoint together.

use crate::embed::Embedder;
use crate::ingest::{Batch, IndexChange, KeyValue};
use crate::live_source::{
    changed_keys, invalid, FeedPage, FeedPosition, Key, Projection, Source, SourceIdentity,
    FEED_PAGE, KEY_PAGE,
};
use crate::metrics::{self, LastDrain};
use crate::schema::SchemaTable;
use crate::store::{
    create_private_dir_all, private_open_options, sync_dir, write_private_atomic, PendingOp,
    VectorStore,
};
use crate::LastSeekError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const ONLINE: &str = ".lastseek-online";
const ACTIVE: &str = "active";
const STATE: &str = "checkpoint.json";
const COMPACT_AT: usize = 4096;
const REBUILD_TIMEOUT: Duration = Duration::from_secs(30 * 60);

#[derive(Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    socket: PathBuf,
    source: SourceIdentity,
    names: Vec<String>,
    projections: BTreeMap<String, Projection>,
    #[serde(flatten)]
    position: FeedPosition,
    needs_rebuild: bool,
    #[serde(default)]
    recovery: Recovery,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct Recovery {
    resumed_restarts: u64,
    gap_rebuilds: u64,
    blocked_by: Option<Block>,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Block {
    FeedGap,
    SchemaChanged,
}

impl Block {
    fn error(self) -> LastSeekError {
        match self {
            Self::FeedGap => LastSeekError::OnlineFeedGap,
            Self::SchemaChanged => invalid("searchable schemas changed; run lastseek rebuild"),
        }
    }
}

impl Checkpoint {
    pub fn needs_rebuild(&self) -> bool {
        self.needs_rebuild
    }

    pub fn cursor(&self) -> &str {
        &self.position.cursor
    }

    pub fn recovery_status(&self) -> Value {
        json!({
            "anchored": self.position.anchor.is_some(),
            "resumed_restarts": self.recovery.resumed_restarts,
            "gap_rebuilds": self.recovery.gap_rebuilds,
            "blocked_by": self.recovery.blocked_by,
        })
    }

    pub fn require_ready(&self) -> Result<(), LastSeekError> {
        if self.needs_rebuild {
            return Err(self.recovery.blocked_by.unwrap_or(Block::FeedGap).error());
        }
        Ok(())
    }

    fn block(&mut self, dir: &Path, reason: Block) -> Result<LastSeekError, LastSeekError> {
        self.needs_rebuild = true;
        self.recovery.blocked_by = Some(reason);
        self.save(dir)?;
        Ok(reason.error())
    }

    pub fn covers(&self, schemas: &[String]) -> bool {
        schemas
            .iter()
            .all(|schema| self.projections.contains_key(schema))
    }

    pub fn include_scopes(&self, table: &mut SchemaTable) {
        for (name, projection) in &self.projections {
            table.include_scope(name, &projection.identity);
        }
    }

    fn save(&self, dir: &Path) -> Result<(), LastSeekError> {
        write_private_atomic(&dir.join(STATE), &serde_json::to_vec(self)?)?;
        sync_dir(dir)
    }
}

pub struct Selection {
    pub dir: PathBuf,
    pub checkpoint: Option<Checkpoint>,
    _guard: File,
}

fn lock(root: &Path, name: &str, exclusive: bool) -> Result<File, LastSeekError> {
    let control = root.join(ONLINE);
    create_private_dir_all(&control)?;
    let lock = private_open_options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(control.join(name))?;
    if exclusive {
        fs2::FileExt::lock_exclusive(&lock)?;
    } else {
        fs2::FileExt::lock_shared(&lock)?;
    }
    Ok(lock)
}

pub fn lock_updates(root: &Path) -> Result<File, LastSeekError> {
    lock(root, "updates.lock", true)
}

fn active(root: &Path) -> Result<Option<&'static str>, LastSeekError> {
    match fs::read_to_string(root.join(ONLINE).join(ACTIVE)) {
        Ok(slot) => match slot.trim() {
            "a" => Ok(Some("a")),
            "b" => Ok(Some("b")),
            _ => Err(invalid("invalid online index selection")),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

pub fn select(root: &Path) -> Result<Selection, LastSeekError> {
    let guard = lock(root, "selection.lock", false)?;
    let (dir, checkpoint) = match active(root)? {
        Some(slot) => {
            let dir = root.join(ONLINE).join(slot);
            let checkpoint = serde_json::from_slice(&fs::read(dir.join(STATE))?)?;
            (dir, Some(checkpoint))
        }
        None => (root.to_owned(), None),
    };
    Ok(Selection {
        dir,
        checkpoint,
        _guard: guard,
    })
}

pub fn ensure_current(dir: &Path) -> Result<(), LastSeekError> {
    let checkpoint: Checkpoint = serde_json::from_slice(&fs::read(dir.join(STATE))?)?;
    checkpoint.require_ready()
}

fn require_scope(
    current: &Path,
    projections: &BTreeMap<String, Projection>,
) -> Result<(), LastSeekError> {
    let identities: BTreeSet<_> = projections.values().map(|p| &p.identity).collect();
    if VectorStore::open(current)?
        .schema_row_counts()
        .keys()
        .any(|schema| !identities.contains(schema))
    {
        return Err(invalid(
            "rebuild would drop an indexed schema outside its scope; select a separate --home",
        ));
    }
    Ok(())
}

#[derive(Default, Serialize)]
struct Applied {
    records: usize,
    tombstones: usize,
}

impl Applied {
    fn include(&mut self, other: Self) {
        self.records += other.records;
        self.tombstones += other.tombstones;
    }
}

struct Importer<'a> {
    source: Source<'a>,
    table: &'a SchemaTable,
    store: &'a mut VectorStore,
    embedder: &'a mut dyn Embedder,
}

impl Importer<'_> {
    fn read(
        &mut self,
        schema: &str,
        projection: &Projection,
        keys: &[Key],
    ) -> Result<Applied, LastSeekError> {
        let ops = self.read_ops(schema, projection, keys)?;
        self.apply(&ops)
    }

    fn read_ops(
        &mut self,
        schema: &str,
        projection: &Projection,
        keys: &[Key],
    ) -> Result<Vec<PendingOp>, LastSeekError> {
        let mut ops = Vec::new();
        for keys in keys.chunks(KEY_PAGE) {
            let mut rows = self.source.records(schema, keys, projection)?;
            let batch = Batch {
                schema_name: schema.to_owned(),
                searchable_fields: Some(projection.fields.clone()),
                classifications: None,
                changes: keys
                    .iter()
                    .map(|key| {
                        let fields = rows.remove(key);
                        IndexChange {
                            mutation_id: None,
                            kind: if fields.is_some() {
                                "upsert"
                            } else {
                                "tombstone"
                            }
                            .into(),
                            key_value: KeyValue {
                                hash: Some(key.hash.clone()),
                                range: key.range.clone(),
                            },
                            record_fields: fields,
                        }
                    })
                    .collect(),
            };
            let batch = batch.records(self.table);
            let texts: Vec<_> = batch
                .records
                .iter()
                .map(|record| record.text.clone())
                .collect();
            let vectors = if texts.is_empty() {
                Vec::new()
            } else {
                self.embedder.embed(&texts)?
            };
            if vectors.len() != texts.len() {
                return Err(invalid("embedder returned an incomplete vector batch"));
            }
            ops.extend(
                batch
                    .records
                    .into_iter()
                    .zip(vectors)
                    .map(|(record, vector)| PendingOp {
                        record,
                        vector: Some(vector),
                    })
                    .chain(batch.tombstones.into_iter().map(|record| PendingOp {
                        record,
                        vector: None,
                    })),
            );
        }
        Ok(ops)
    }

    fn apply(&mut self, ops: &[PendingOp]) -> Result<Applied, LastSeekError> {
        self.store.append(ops)?;
        let records = ops.iter().filter(|op| op.vector.is_some()).count();
        Ok(Applied {
            records,
            tombstones: ops.len() - records,
        })
    }

    fn replay(
        &mut self,
        page: &FeedPage,
        projections: &BTreeMap<String, Projection>,
        boot: &SourceIdentity,
    ) -> Result<Applied, LastSeekError> {
        let mut ops = Vec::new();
        for (schema, keys) in changed_keys(page, projections)? {
            let keys: Vec<_> = keys.into_iter().collect();
            ops.extend(self.read_ops(&schema, &projections[&schema], &keys)?);
        }
        // A restart between the feed read and hydration must not append
        // records from a different source to the active generation.
        self.source.require_boot(boot)?;
        self.apply(&ops)
    }
}

fn within_budget(started: Instant) -> Result<(), LastSeekError> {
    if started.elapsed() >= REBUILD_TIMEOUT {
        return Err(invalid("online rebuild exceeded its 30 minute budget"));
    }
    Ok(())
}

pub fn rebuild(
    root: &Path,
    socket: &Path,
    names: &[String],
    table: &SchemaTable,
    embedder: &mut dyn Embedder,
) -> Result<Value, LastSeekError> {
    let _rebuild = lock(root, "rebuild.lock", true)?;
    let recovery = select(root)?
        .checkpoint
        .map(|state| state.recovery)
        .unwrap_or_default();
    rebuild_locked(root, socket, names, table, embedder, recovery, None)
}

pub fn recover(
    root: &Path,
    table: &SchemaTable,
    embedder: &mut dyn Embedder,
) -> Result<Value, LastSeekError> {
    let _rebuild = lock(root, "rebuild.lock", true)?;
    let state = select(root)?
        .checkpoint
        .ok_or_else(|| invalid("online index is missing"))?;
    if !state.needs_rebuild {
        return Ok(json!({"ok":true, "source":"lastdb-change-feed",
            "applied_through":state.cursor(), "recovery":state.recovery_status()}));
    }
    if state.recovery.blocked_by == Some(Block::SchemaChanged) {
        state.require_ready()?;
    }
    rebuild_locked(
        root,
        &state.socket,
        &state.names,
        table,
        embedder,
        state.recovery,
        Some(&state.projections),
    )
}

fn rebuild_locked(
    root: &Path,
    socket: &Path,
    names: &[String],
    table: &SchemaTable,
    embedder: &mut dyn Embedder,
    mut recovery: Recovery,
    previous_projections: Option<&BTreeMap<String, Projection>>,
) -> Result<Value, LastSeekError> {
    let started = Instant::now();
    let source = Source(socket);
    let identity = source.identity()?;
    let projections = source.projections(names, table)?;
    if previous_projections.is_some_and(|previous| *previous != projections) {
        return Err(invalid("searchable schemas changed; run lastseek rebuild"));
    }
    if projections.is_empty() || names.iter().any(|name| !projections.contains_key(name)) {
        return Err(invalid(
            "rebuild requires available, explicitly searchable local Brain schemas",
        ));
    }
    require_scope(&select(root)?.dir, &projections)?;
    let mut position = source.head()?;

    let control = root.join(ONLINE);
    let slot = if active(root)? == Some("a") { "b" } else { "a" };
    let stage = control.join(slot);
    match fs::remove_dir_all(&stage) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let mut store = VectorStore::open(&stage)?;
    store.bind_embedder(embedder)?;
    let mut base = Applied::default();
    let mut tail = Applied::default();
    let mut key_pages = 0;
    let mut feed_events = 0;
    {
        let mut importer = Importer {
            source,
            table,
            store: &mut store,
            embedder,
        };
        for (schema, projection) in &projections {
            eprintln!("lastseek rebuild: reading {schema}");
            let mut after = None;
            let mut seen = BTreeSet::new();
            loop {
                within_budget(started)?;
                let page = importer.source.keys(schema, after.as_deref())?;
                base.include(importer.read(schema, projection, &page.keys)?);
                key_pages += 1;
                if importer.store.pending_ops() >= COMPACT_AT {
                    importer.store.compact()?;
                }
                if !page.has_more {
                    break;
                }
                after = page.next_cursor;
                if !seen.insert(after.clone()) {
                    return Err(invalid("LastDB repeated a primary key page cursor"));
                }
            }
        }
        loop {
            within_budget(started)?;
            let page = importer.source.changes(Some(&position.cursor), FEED_PAGE)?;
            feed_events += page.changes.len();
            tail.include(importer.replay(&page, &projections, &identity)?);
            position = page.position(Some(&position));
            if importer.store.pending_ops() >= COMPACT_AT {
                importer.store.compact()?;
            }
            if !page.has_more {
                break;
            }
        }
    }
    store.compact()?;
    let committed = store.committed_rows();
    let source = Source(socket);
    if source.identity()? != identity || source.projections(names, table)? != projections {
        return Err(invalid(
            "LastDB or its searchable schemas changed during the build; retry the rebuild",
        ));
    }
    // Verify the next page is still available after compaction. Later changes
    // remain for drain, which starts at the exact published cursor.
    source.changes(Some(&position.cursor), 1)?;
    source.require_boot(&identity)?;
    recovery.blocked_by = None;
    if previous_projections.is_some() {
        recovery.gap_rebuilds = recovery.gap_rebuilds.saturating_add(1);
    }
    let checkpoint = Checkpoint {
        socket: socket.to_owned(),
        source: identity,
        names: names.to_vec(),
        projections,
        position,
        needs_rebuild: false,
        recovery,
    };
    checkpoint.save(&stage)?;
    sync_dir(&control)?;
    let _selection = lock(root, "selection.lock", true)?;
    let current = active(root)?.map_or_else(|| root.to_owned(), |slot| control.join(slot));
    require_scope(&current, &checkpoint.projections)?;
    write_private_atomic(&control.join(ACTIVE), slot.as_bytes())?;
    sync_dir(&control)?;
    sync_dir(root)?;
    Ok(json!({
        "ok":true, "schemas":checkpoint.projections.keys().collect::<Vec<_>>(),
        "key_pages":key_pages, "indexed":base, "replayed":tail, "feed_events":feed_events,
        "source":"lastdb-change-feed", "applied_through":checkpoint.cursor(), "committed_rows":committed,
        "recovery":checkpoint.recovery_status(),
    }))
}

pub fn drain(
    dir: &Path,
    mut checkpoint: Checkpoint,
    table: &SchemaTable,
    embedder: &mut dyn Embedder,
    mut remaining_budget: usize,
) -> Result<Value, LastSeekError> {
    checkpoint.require_ready()?;
    let socket = checkpoint.socket.clone();
    let source = Source(&socket);
    let identity = source.identity()?;
    let projections = source.projections(&checkpoint.names, table)?;
    let restarted = identity != checkpoint.source;
    let observed_anchor = if restarted || checkpoint.position.anchor.is_none() {
        source.anchor_at(checkpoint.cursor())?
    } else {
        None
    };
    source.require_boot(&identity)?;
    if projections != checkpoint.projections {
        return Err(checkpoint.block(dir, Block::SchemaChanged)?);
    }
    if restarted {
        if observed_anchor.is_none() || observed_anchor != checkpoint.position.anchor {
            return Err(checkpoint.block(dir, Block::FeedGap)?);
        }
    } else if checkpoint.position.anchor.is_none() {
        // Existing checkpoints acquire an anchor while their original daemon
        // is still serving. Across a restart an unsigned cursor is ambiguous.
        checkpoint.position.anchor = observed_anchor;
    }
    let mut store = VectorStore::open(dir)?;
    store.bind_embedder(embedder)?;
    let mut importer = Importer {
        source,
        table,
        store: &mut store,
        embedder,
    };
    let mut applied = Applied::default();
    let mut events = 0;
    let mut has_more = true;
    while remaining_budget > 0 && has_more {
        let page = match importer
            .source
            .changes(Some(checkpoint.cursor()), remaining_budget.min(FEED_PAGE))
        {
            Err(LastSeekError::OnlineFeedGap) => {
                importer.source.require_boot(&identity)?;
                return Err(checkpoint.block(dir, Block::FeedGap)?);
            }
            result => result?,
        };
        applied.include(importer.replay(&page, &checkpoint.projections, &identity)?);
        remaining_budget = remaining_budget.saturating_sub(page.changes.len());
        events += page.changes.len();
        has_more = page.has_more;
        if checkpoint.source != identity {
            checkpoint.recovery.resumed_restarts =
                checkpoint.recovery.resumed_restarts.saturating_add(1);
            checkpoint.source = identity.clone();
        }
        checkpoint.position = page.position(Some(&checkpoint.position));
        checkpoint.save(dir)?;
    }
    let compacted = store.pending_ops() >= COMPACT_AT;
    if compacted {
        store.compact()?;
    }
    metrics::save_last_drain(
        dir,
        &LastDrain {
            files: 0,
            records: applied.records as u64,
            tombstones: applied.tombstones as u64,
            compacted,
        },
    )?;
    Ok(json!({
        "ok":true, "source":"lastdb-change-feed", "files":0, "events":events,
        "records":applied.records, "tombstones":applied.tombstones,
        "remaining":usize::from(has_more), "has_more":has_more, "unreadable":[],
        "applied_through":checkpoint.cursor(), "pending_ops":store.pending_ops(),
        "recovery":checkpoint.recovery_status(),
    }))
}
