//! Build a private projection, then publish its index and feed checkpoint together.

use crate::embed::Embedder;
use crate::ingest::{Batch, IndexChange, KeyValue};
use crate::live_source::{
    changed_keys, invalid, FeedPage, Key, Projection, Source, SourceIdentity, FEED_PAGE, KEY_PAGE,
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
    cursor: String,
    needs_rebuild: bool,
}

impl Checkpoint {
    pub fn needs_rebuild(&self) -> bool {
        self.needs_rebuild
    }

    pub fn cursor(&self) -> &str {
        &self.cursor
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
    if checkpoint.needs_rebuild {
        return Err(LastSeekError::OnlineFeedGap);
    }
    Ok(())
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
        let mut applied = Applied::default();
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
            let stats = Applied {
                records: vectors.len(),
                tombstones: batch.tombstones.len(),
            };
            let ops: Vec<_> = batch
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
                }))
                .collect();
            self.store.append(&ops)?;
            applied.include(stats);
        }
        Ok(applied)
    }

    fn replay(
        &mut self,
        page: &FeedPage,
        projections: &BTreeMap<String, Projection>,
    ) -> Result<Applied, LastSeekError> {
        let mut applied = Applied::default();
        for (schema, keys) in changed_keys(page, projections)? {
            let keys: Vec<_> = keys.into_iter().collect();
            applied.include(self.read(&schema, &projections[&schema], &keys)?);
        }
        Ok(applied)
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
    let started = Instant::now();
    let _rebuild = lock(root, "rebuild.lock", true)?;
    let source = Source(socket);
    let identity = source.identity()?;
    let projections = source.projections(names, table)?;
    if projections.is_empty() || names.iter().any(|name| !projections.contains_key(name)) {
        return Err(invalid(
            "rebuild requires available, explicitly searchable local Brain schemas",
        ));
    }
    require_scope(&select(root)?.dir, &projections)?;
    let mut cursor = source.head()?;

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
            let page = importer.source.changes(Some(&cursor), FEED_PAGE)?;
            feed_events += page.changes.len();
            tail.include(importer.replay(&page, &projections)?);
            cursor = page.next_cursor;
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
    source.changes(Some(&cursor), 1)?;
    let checkpoint = Checkpoint {
        socket: socket.to_owned(),
        source: identity,
        names: names.to_vec(),
        projections,
        cursor,
        needs_rebuild: false,
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
        "applied_through":checkpoint.cursor, "committed_rows":committed,
    }))
}

pub fn drain(
    dir: &Path,
    mut checkpoint: Checkpoint,
    table: &SchemaTable,
    embedder: &mut dyn Embedder,
    mut remaining_budget: usize,
) -> Result<Value, LastSeekError> {
    if checkpoint.needs_rebuild {
        return Err(LastSeekError::OnlineFeedGap);
    }
    let source = Source(&checkpoint.socket);
    let identity = source.identity()?;
    if identity != checkpoint.source
        || source.projections(&checkpoint.names, table)? != checkpoint.projections
    {
        checkpoint.needs_rebuild = true;
        checkpoint.save(dir)?;
        return Err(LastSeekError::OnlineFeedGap);
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
            .changes(Some(&checkpoint.cursor), remaining_budget.min(FEED_PAGE))
        {
            Err(LastSeekError::OnlineFeedGap) => {
                checkpoint.needs_rebuild = true;
                checkpoint.save(dir)?;
                return Err(LastSeekError::OnlineFeedGap);
            }
            result => result?,
        };
        applied.include(importer.replay(&page, &checkpoint.projections)?);
        if importer.source.identity()? != checkpoint.source {
            checkpoint.needs_rebuild = true;
            checkpoint.save(dir)?;
            return Err(LastSeekError::OnlineFeedGap);
        }
        remaining_budget = remaining_budget.saturating_sub(page.changes.len());
        events += page.changes.len();
        has_more = page.has_more;
        checkpoint.cursor = page.next_cursor;
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
        "applied_through":checkpoint.cursor, "pending_ops":store.pending_ops(),
    }))
}
