//! `lastseek` CLI.
//!
//! Output is JSON on stdout so callers (brain, kanban) parse rather than scrape.
//! Every command that bounds its work reports what it left behind — a capped run
//! that stays silent reads as "finished", which is how the predecessor's inbox
//! backlog grew to 99k unnoticed.

use lastseek::embed::{DeterministicEmbedder, Embedder};
use lastseek::index::{QueryOpts, Searcher};
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
  lastseek doctor [--home DIR]

  --home DIR   store directory (default: $LASTSEEK_HOME or ~/.lastseek/index)

The embedder is selected by LASTSEEK_EMBEDDER. `deterministic` is a TEST
FIXTURE with no semantics and is refused for production stores.
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

/// Resolve the embedder. Only the fixture exists today; the neural embedder
/// lands behind the same trait, which is why this returns a boxed `Embedder`
/// rather than a concrete type.
fn embedder() -> Result<Box<dyn Embedder>, LastSeekError> {
    match std::env::var("LASTSEEK_EMBEDDER").as_deref() {
        Ok("deterministic") => Ok(Box::new(DeterministicEmbedder::new(384))),
        Ok(other) => Err(LastSeekError::Embedder(format!(
            "unknown embedder {other:?}; neural embedding is not wired up yet — \
             set LASTSEEK_EMBEDDER=deterministic for fixtures"
        ))),
        Err(_) => Err(LastSeekError::Embedder(
            "no embedder configured; neural embedding is not wired up yet — \
             set LASTSEEK_EMBEDDER=deterministic for fixtures"
                .into(),
        )),
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
        "status" | "doctor" => {
            match VectorStore::open(&dir) {
                Ok(store) => {
                    let out = serde_json::json!({
                        "ok": true,
                        "home": dir.to_string_lossy(),
                        "needs_rebuild": false,
                        "embedder_id": store.embedder_id(),
                        "dimensions": store.dimensions(),
                        "committed_rows": store.committed_rows(),
                        "pending_ops": store.pending_ops(),
                        "slab_bytes": store.slab_bytes(),
                        "log_bytes": store.log_bytes(),
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
            let mut emb = embedder()?;
            let q = emb.embed(std::slice::from_ref(&text))?.remove(0);
            let opts = QueryOpts {
                k: flag_value(&args, "--k").and_then(|v| v.parse().ok()),
                schemas: flag_values(&args, "--schema"),
                min_score: flag_value(&args, "--min-score").and_then(|v| v.parse().ok()),
                exact: args.iter().any(|a| a == "--exact").then_some(text),
            };
            let hits = Searcher::new(&store).search(&q, &opts);
            let rows: Vec<_> = hits
                .iter()
                .map(|h| {
                    serde_json::json!({
                        "score": h.score,
                        "schema_name": h.record.schema_name,
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
            let mut emb = embedder()?;
            let mut store = VectorStore::open(&dir)?;
            store.set_embedder(emb.id(), emb.dimensions(), emb.is_neural())?;
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
