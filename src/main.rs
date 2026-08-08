//! `lastseek` CLI.
//!
//! Output is JSON on stdout so callers (brain, kanban) parse rather than scrape.
//! Every command that bounds its work reports what it left behind — a capped run
//! that stays silent reads as "finished", which is how the predecessor's inbox
//! backlog grew to 99k unnoticed.

use lastseek::coverage::{CoverageReport, DEFAULT_COVERAGE_FLOOR};
use lastseek::embed::{DeterministicEmbedder, Embedder};
use lastseek::index::{QueryOpts, Searcher};
use lastseek::ingest::Watermark;
use lastseek::schema::{SchemaTable, AVAILABLE_PATH, DEFAULT_SCHEMA_SERVICE_URL};
use lastseek::store::{PendingOp, Record, VectorStore};
use lastseek::LastSeekError;
use std::io::Read;

const USAGE: &str = "\
lastseek — memory-mapped semantic vector search for LastDB

usage:
  lastseek status [--home DIR]
  lastseek query <text> [--home DIR] [--k N] [--schema S]... [--min-score F] [--exact]
  lastseek add [--home DIR]           read Records as JSONL on stdin, embed, append
  lastseek compact [--home DIR]       fold the pending log into the slab
  lastseek drain [--home DIR] [--inbox DIR] [--max-files N] [--consume]
                                      apply the host's IndexChangeBatch feed
  lastseek doctor [--home DIR]
  lastseek schema-table [--from PATH|URL] [--table PATH]
                                      refresh the Schema Service resolution table

  --home DIR   store directory (default: $LASTSEEK_HOME or ~/.lastseek/index)
  --table PATH resolution table (default: $LASTSEEK_SCHEMA_TABLE or
               ~/.lastseek/schema-table.json)
  --catalog PATH|URL
               LastDB schema catalog fixture/URL (default: live Unix socket)
  --lastdb-socket PATH
               LastDB Unix socket (default: $LASTSEEK_LASTDB_SOCKET,
               $LASTDB_SOCKET, or $LASTDB_HOME/data/folddb.sock)
  --coverage-floor F
               minimum vectors/source ratio for healthy (default: 0.95)
  --inbox DIR  host batch feed (default: $LASTSEEK_INBOX or
               $LASTDB_HOME/apps/search/inbox)

`drain` is bounded (--max-files, default 2000) and reports `remaining`, because
an unbounded drain over a large backlog is what took the machine down on
2026-08-07. It advances a watermark instead of deleting batch files, so the
incumbent Search app can keep reading the same inbox until cutover; --consume
deletes them.

`--schema` accepts a readable name (Card), a schema identity hash, or a
registry name, and is resolved through the Schema Service table. A schema name
that resolves to nothing is an ERROR, not an empty result — those are different
answers and callers depend on telling them apart.

A store names its own embedder, so LASTSEEK_EMBEDDER is an OVERRIDE, not a
requirement: querying a store built with bge-small-en-v1.5 loads that model
without being told. Only a fresh, unbound store needs telling. `bge` loads
bge-small-en-v1.5 from $LASTSEEK_MODEL_DIR or
~/.lastseek/models/bge-small-en-v1.5 and requires a build with
`--features neural`; `deterministic` is a TEST FIXTURE with no semantics and is
refused for production stores.
";

fn home(args: &[String]) -> std::path::PathBuf {
    if let Some(i) = args.iter().position(|a| a == "--home") {
        if let Some(v) = args.get(i + 1) {
            return v.into();
        }
    }
    if let Ok(v) = std::env::var("LASTSEEK_HOME") {
        return v.into();
    }
    let base = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::Path::new(&base).join(".lastseek").join("index")
}

fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Parse a flag's value as `T`, erroring on a present-but-malformed value
/// rather than silently falling back to a default — `--k abc` used to fall
/// back to 20 and `--min-score foo` used to silently disable the floor.
fn parse_flag<T: std::str::FromStr>(
    args: &[String],
    name: &str,
) -> Result<Option<T>, LastSeekError> {
    match flag_value(args, name) {
        Some(v) => v
            .parse::<T>()
            .map(Some)
            .map_err(|_| LastSeekError::Embedder(format!("{name} value {v:?} is not valid"))),
        None => Ok(None),
    }
}

fn schema_table_path(args: &[String]) -> std::path::PathBuf {
    flag_value(args, "--table")
        .map(Into::into)
        .unwrap_or_else(SchemaTable::default_path)
}

/// Load the resolution table, tolerating its absence.
///
/// A missing table is not an error here: an unscoped query, and a query scoped
/// by an identity hash the store already interns, both work without one. What
/// it costs is readable-name scoping, and `resolve_all` reports that as an
/// unknown schema naming what it does know — which is the actionable message,
/// where refusing to start would not be.
fn load_schema_table(args: &[String]) -> SchemaTable {
    let path = schema_table_path(args);
    SchemaTable::load(&path).unwrap_or_default()
}

/// Fetch the Schema Service table.
///
/// `src` is a local path or an `http(s)` URL. URLs are fetched by shelling out
/// to `curl` rather than linking an HTTP client: this is one cacheable GET on a
/// path a human can also run by hand, and the alternative measured at a rustls
/// dependency tree and minutes of added compile time on every build of a crate
/// whose whole point is a fast local query. A path always works, so an
/// environment without `curl` is never stuck.
fn fetch_schema_table(src: &str) -> Result<SchemaTable, LastSeekError> {
    if !(src.starts_with("http://") || src.starts_with("https://")) {
        let bytes = std::fs::read(src)?;
        return SchemaTable::from_available_json(&bytes, src);
    }
    let out = std::process::Command::new("curl")
        .args(["-sS", "--fail", "-m", "60", src])
        .output()
        .map_err(|e| {
            LastSeekError::Embedder(format!(
                "fetching {src}: {e} (no curl on PATH — pass --from with a downloaded \
                 /v1/schemas/available body instead)"
            ))
        })?;
    if !out.status.success() {
        return Err(LastSeekError::Embedder(format!(
            "fetching {src}: curl exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    SchemaTable::from_available_json(&out.stdout, src)
}

fn lastdb_socket(args: &[String]) -> std::path::PathBuf {
    if let Some(path) = flag_value(args, "--lastdb-socket") {
        return path.into();
    }
    if let Ok(path) = std::env::var("LASTSEEK_LASTDB_SOCKET") {
        return path.into();
    }
    if let Ok(path) = std::env::var("LASTDB_SOCKET") {
        return path.into();
    }
    if let Ok(home) = std::env::var("LASTDB_HOME") {
        return std::path::Path::new(&home).join("data").join("folddb.sock");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::Path::new(&home)
        .join(".lastdb")
        .join("data")
        .join("folddb.sock")
}

/// Read source record counts from LastDB.
///
/// This is the status half of LastSeek's existing LastDB relationship, not a
/// third integration: ingest reads LastDB changes and status reads the node's
/// catalog. The Unix socket remains the default; retired TCP is never probed.
fn fetch_lastdb_catalog(args: &[String]) -> Result<(Vec<u8>, String), LastSeekError> {
    if let Some(src) =
        flag_value(args, "--catalog").or_else(|| std::env::var("LASTSEEK_LASTDB_CATALOG").ok())
    {
        if !(src.starts_with("http://") || src.starts_with("https://")) {
            return Ok((std::fs::read(&src)?, src));
        }
        let out = std::process::Command::new("curl")
            .args([
                "-sS",
                "--fail",
                "-m",
                "180",
                "-H",
                "X-LastDB-Client: lastseek",
                &src,
            ])
            .output()?;
        if out.status.success() {
            return Ok((out.stdout, src));
        }
        return Err(LastSeekError::Embedder(format!(
            "fetching LastDB catalog {src}: curl exited {:?}: {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }

    let socket = lastdb_socket(args);
    let socket_text = socket.to_string_lossy().to_string();
    let url = "http://localhost/api/schemas?include_counts=true";
    let out = std::process::Command::new("curl")
        .args([
            "-sS",
            "--fail",
            "-m",
            "180",
            "-H",
            "X-LastDB-Client: lastseek",
            "--unix-socket",
            &socket_text,
            url,
        ])
        .output()?;
    if !out.status.success() {
        return Err(LastSeekError::Embedder(format!(
            "fetching LastDB catalog over {}: curl exited {:?}: {}",
            socket.display(),
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok((
        out.stdout,
        format!(
            "unix://{}:/api/schemas?include_counts=true",
            socket.display()
        ),
    ))
}

fn flag_values(args: &[String], name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (i, a) in args.iter().enumerate() {
        if a == name {
            if let Some(v) = args.get(i + 1) {
                out.push(v.clone());
            }
        }
    }
    out
}

/// Short selector name for an embedder id, so a store can name its own.
///
/// The store persists a full identity (`fastembed/bge-small-en-v1.5`); the CLI
/// selector is the short form (`bge`). Mapping between them here keeps the
/// persisted identity unabbreviated — it is the thing that must never be
/// ambiguous — while the selector stays typeable.
fn selector_for_embedder_id(id: &str) -> Option<&'static str> {
    match id {
        "fastembed/bge-small-en-v1.5" => Some("bge"),
        "deterministic-test-fixture" => Some("deterministic"),
        _ => None,
    }
}

/// Resolve the embedder for a store.
///
/// `LASTSEEK_EMBEDDER` wins when set. When it is not, **the store names its own
/// embedder**: a store bound to `fastembed/bge-small-en-v1.5` is queried with
/// that model, because `Searcher::new` would refuse anything else anyway — a
/// query embedded with the wrong embedder is a config error, not a corpus with
/// no matches. Requiring the variable added a way to get that error and no way
/// to get a different answer.
///
/// This is not a convenience. Callers reach LastSeek by spawning it (brain and
/// kanban both do), so the subprocess sees whatever environment the parent
/// happened to have. An unset variable made `lastseek query` exit non-zero,
/// which those callers correctly read as "plane unavailable" and answered from
/// the incumbent instead — a working index, silently unused, on every host that
/// had not exported a variable nobody documents needing.
///
/// An empty store (no embedder bound yet) still needs to be told, since there
/// is nothing to infer from; `add` on a fresh store is the one place the
/// variable is genuinely load-bearing.
fn embedder_for(store_embedder_id: &str) -> Result<Box<dyn Embedder>, LastSeekError> {
    let from_env = std::env::var("LASTSEEK_EMBEDDER").ok();
    let selector = match from_env.as_deref() {
        Some(s) => s,
        None => selector_for_embedder_id(store_embedder_id).ok_or_else(|| {
            LastSeekError::Embedder(if store_embedder_id.is_empty() {
                "this store has no embedder bound yet, so there is nothing to infer: set \
                 LASTSEEK_EMBEDDER=bge for neural embedding or =deterministic for fixtures"
                    .to_string()
            } else {
                format!(
                    "store is bound to embedder {store_embedder_id:?}, which this binary has no \
                     selector for; set LASTSEEK_EMBEDDER explicitly"
                )
            })
        })?,
    };

    match selector {
        "deterministic" => Ok(Box::new(DeterministicEmbedder::new(384))),
        #[cfg(feature = "neural")]
        "bge" => {
            let dir = lastseek::embed::BgeSmallEmbedder::default_model_dir();
            Ok(Box::new(lastseek::embed::BgeSmallEmbedder::load(&dir)?))
        }
        #[cfg(not(feature = "neural"))]
        "bge" => Err(LastSeekError::Embedder(
            "bge requires a build with --features neural".into(),
        )),
        other => Err(LastSeekError::Embedder(format!(
            "unknown embedder {other:?}; set LASTSEEK_EMBEDDER=deterministic for fixtures \
             or LASTSEEK_EMBEDDER=bge for neural embedding"
        ))),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("lastseek: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), LastSeekError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first().cloned() else {
        print!("{USAGE}");
        return Ok(());
    };
    let dir = home(&args);

    match cmd.as_str() {
        "schema-table" => {
            let src = flag_value(&args, "--from")
                .unwrap_or_else(|| format!("{DEFAULT_SCHEMA_SERVICE_URL}{AVAILABLE_PATH}"));
            let path = schema_table_path(&args);
            let table = fetch_schema_table(&src)?;
            table.save(&path)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "ok": true,
                    "source": src,
                    "table": path.to_string_lossy(),
                    "schemas": table.len(),
                    "descriptive_names": table.descriptive_names(),
                }))?
            );
        }
        "status" | "doctor" => {
            let table = load_schema_table(&args);
            let table_path = schema_table_path(&args);
            let coverage_floor =
                parse_flag::<f64>(&args, "--coverage-floor")?.unwrap_or(DEFAULT_COVERAGE_FLOOR);
            if !(0.0..=1.0).contains(&coverage_floor) {
                return Err(LastSeekError::Embedder(format!(
                    "--coverage-floor must be between 0 and 1, got {coverage_floor}"
                )));
            }
            match VectorStore::open(&dir) {
                Ok(store) => {
                    // Schemas the index holds that the Schema Service table
                    // cannot name. These are indexed and searchable by their
                    // ingest key, and counted here so the gap reads as "their
                    // owner has not published to SS" instead of as a hole
                    // nobody can see.
                    let ss_unknown: Vec<&String> = store
                        .interned_schemas()
                        .iter()
                        .filter(|s| table.resolve(s).is_none())
                        .collect();
                    let coverage = fetch_lastdb_catalog(&args).and_then(|(bytes, source)| {
                        CoverageReport::from_catalog_json(
                            &bytes,
                            source,
                            &table,
                            &store.schema_row_counts(),
                            coverage_floor,
                        )
                    });
                    let state = coverage
                        .as_ref()
                        .map(CoverageReport::state)
                        .unwrap_or("degraded");
                    let coverage_json = match coverage {
                        Ok(report) => serde_json::to_value(report)?,
                        Err(error) => serde_json::json!({
                            "available": false,
                            "floor": coverage_floor,
                            "error": error.to_string(),
                            "schemas": [],
                        }),
                    };
                    let out = serde_json::json!({
                        "ok": true,
                        "state": state,
                        "home": dir.to_string_lossy(),
                        "needs_rebuild": false,
                        "embedder_id": store.embedder_id(),
                        "dimensions": store.dimensions(),
                        "committed_rows": store.committed_rows(),
                        "pending_ops": store.pending_ops(),
                        "slab_bytes": store.slab_bytes(),
                        "log_bytes": store.log_bytes(),
                        "schema_table": {
                            "path": table_path.to_string_lossy(),
                            "present": !table.is_empty(),
                            "source": table.source,
                            "schemas": table.len(),
                            "descriptive_names": table.descriptive_names(),
                        },
                        "interned_schemas": store.interned_schemas().len(),
                        "ss_unknown_schemas": ss_unknown.len(),
                        "ss_unknown_sample": ss_unknown.iter().take(10).collect::<Vec<_>>(),
                        "non_searchable_records_skipped": Watermark::load(&dir)
                            .non_searchable_records_skipped,
                        "coverage": coverage_json,
                    });
                    println!("{}", serde_json::to_string_pretty(&out)?);
                }
                // The index is local-only and regenerable, so an
                // unsupported (e.g. pre-generations) store format is
                // something callers can act on, not a hard error.
                Err(LastSeekError::UnsupportedVersion(v)) => {
                    let out = serde_json::json!({
                        "ok": true,
                        "home": dir.to_string_lossy(),
                        "needs_rebuild": true,
                        "reason": format!(
                            "store format version {v} is no longer supported; \
                             delete {} and rebuild",
                            dir.to_string_lossy()
                        ),
                    });
                    println!("{}", serde_json::to_string_pretty(&out)?);
                }
                Err(e) => return Err(e),
            }
        }
        "query" => {
            let text = args
                .get(1)
                .cloned()
                .filter(|t| !t.starts_with("--"))
                .unwrap_or_default();
            if text.is_empty() {
                return Err(LastSeekError::Embedder("query requires text".into()));
            }
            let store = VectorStore::open(&dir)?;
            let table = load_schema_table(&args);
            let mut emb = embedder_for(store.embedder_id())?;
            let q = emb.embed_query(std::slice::from_ref(&text))?.remove(0);
            let opts = QueryOpts {
                k: parse_flag(&args, "--k")?,
                schemas: flag_values(&args, "--schema"),
                min_score: parse_flag(&args, "--min-score")?,
                exact: args.iter().any(|a| a == "--exact").then_some(text),
            };
            let searcher = Searcher::new(&store, emb.id())?.with_schema_table(table.clone());
            let hits = searcher.search(&q, &opts)?;
            let rows: Vec<_> = hits
                .iter()
                .map(|h| {
                    serde_json::json!({
                        "score": h.score,
                        // Both the identity the row is keyed on and the
                        // readable label, when the table knows one. The hash
                        // alone made every result unreadable; the label alone
                        // would hide which of several same-named identities
                        // actually answered.
                        "schema_identity": h.record.schema_identity,
                        "schema": table.label_for_identity(&h.record.schema_identity),
                        "key_hash": h.record.key_hash,
                        "key_range": h.record.key_range,
                        "fragment_key": h.record.fragment_key,
                        "text": h.record.text,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "ok": true, "hits": rows.len(), "results": rows
                }))?
            );
        }
        "add" => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            let records: Vec<Record> = buf
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(serde_json::from_str)
                .collect::<Result<_, _>>()?;
            let mut store = VectorStore::open(&dir)?;
            let mut emb = embedder_for(store.embedder_id())?;
            store.bind_embedder(emb.as_ref())?;
            let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
            let vectors = emb.embed(&texts)?;
            let ops: Vec<PendingOp> = records
                .into_iter()
                .zip(vectors)
                .map(|(record, v)| PendingOp {
                    record,
                    vector: Some(v),
                })
                .collect();
            let n = ops.len();
            store.append(&ops)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "ok": true, "appended": n, "pending_ops": store.pending_ops(),
                    "log_bytes": store.log_bytes()
                }))?
            );
        }
        "drain" => {
            let inbox = flag_value(&args, "--inbox")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(lastseek::ingest::default_inbox);
            let max = parse_flag::<usize>(&args, "--max-files")?
                .unwrap_or(lastseek::ingest::DEFAULT_MAX_FILES);
            let consume = args.iter().any(|a| a == "--consume");
            let table = load_schema_table(&args);
            let mut mark = Watermark::load(&dir);
            let (files, remaining) =
                lastseek::ingest::pending_batches(&inbox, &mark.applied_through, max)?;

            let mut store = VectorStore::open(&dir)?;
            let mut emb = embedder_for(store.embedder_id())?;
            store.bind_embedder(emb.as_ref())?;

            let mut applied_records = 0usize;
            let mut applied_tombstones = 0usize;
            let mut ss_unknown_batches = 0usize;
            let mut non_searchable_records_skipped = 0usize;
            let mut unreadable: Vec<String> = Vec::new();

            for path in &files {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();
                let bytes = match std::fs::read(path) {
                    Ok(b) => b,
                    // A file that raced away between listing and reading is
                    // normal while the incumbent is also draining. A file that
                    // will not parse is reported, and the watermark still
                    // advances past it: a single malformed batch must not wedge
                    // the feed forever, which is how the predecessor's backlog
                    // became unrecoverable.
                    Err(e) => {
                        unreadable.push(format!("{name}: {e}"));
                        mark.applied_through = name;
                        continue;
                    }
                };
                let batch: lastseek::ingest::Batch = match serde_json::from_slice(&bytes) {
                    Ok(b) => b,
                    Err(e) => {
                        unreadable.push(format!("{name}: {e}"));
                        mark.applied_through = name;
                        continue;
                    }
                };
                let out = batch.records(&table);
                if out.skipped.is_some() {
                    ss_unknown_batches += 1;
                }
                non_searchable_records_skipped += out.non_searchable_records_skipped;
                if !out.records.is_empty() {
                    let texts: Vec<String> = out.records.iter().map(|r| r.text.clone()).collect();
                    let vectors = emb.embed(&texts)?;
                    let ops: Vec<PendingOp> = out
                        .records
                        .into_iter()
                        .zip(vectors)
                        .map(|(record, v)| PendingOp {
                            record,
                            vector: Some(v),
                        })
                        .collect();
                    applied_records += ops.len();
                    store.append(&ops)?;
                }
                if !out.tombstones.is_empty() {
                    let ops: Vec<PendingOp> = out
                        .tombstones
                        .into_iter()
                        .map(|record| PendingOp {
                            record,
                            vector: None,
                        })
                        .collect();
                    applied_tombstones += ops.len();
                    store.append(&ops)?;
                }
                mark.applied_through = name;
                mark.files_applied += 1;
                mark.non_searchable_records_skipped += out.non_searchable_records_skipped as u64;
                if consume {
                    // Best-effort: the incumbent may already have taken it.
                    let _ = std::fs::remove_file(path);
                }
            }
            mark.save(&dir)?;

            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "ok": true,
                    "inbox": inbox.to_string_lossy(),
                    "files": files.len(),
                    // What the cap left behind. A bounded run that reports
                    // nothing reads as "inbox empty", which is the specific
                    // way a 99k-file backlog stayed invisible.
                    "remaining": remaining,
                    "records": applied_records,
                    "tombstones": applied_tombstones,
                    "ss_unknown_batches": ss_unknown_batches,
                    "non_searchable_records_skipped": non_searchable_records_skipped,
                    "non_searchable_records_skipped_total":
                        mark.non_searchable_records_skipped,
                    "unreadable": unreadable,
                    "applied_through": mark.applied_through,
                    "pending_ops": store.pending_ops(),
                }))?
            );
        }
        "compact" => {
            let mut store = VectorStore::open(&dir)?;
            let before = store.pending_ops();
            store.compact()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "ok": true, "folded_ops": before,
                    "committed_rows": store.committed_rows(),
                    "slab_bytes": store.slab_bytes()
                }))?
            );
        }
        "--help" | "-h" | "help" => print!("{USAGE}"),
        other => {
            eprintln!("lastseek: unknown command {other:?}\n");
            print!("{USAGE}");
            std::process::exit(2);
        }
    }
    Ok(())
}
