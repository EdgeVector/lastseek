//! The three-command app interface.
//!
//! An app names itself. It does not name a directory. The index lives at
//! `$LASTSEEK_HOME/apps/<name>`, or `~/.lastseek/apps/<name>` when that
//! variable is unset. `LASTDB_APP_DATA` is ignored on purpose: the caller's
//! app data would otherwise put the same app's index in a different place
//! for every caller.
//!
//! `index` and `search` build and catch up through `online::rebuild_app` and
//! `online::drain`. Search never rebuilds in-process. A gap starts the same
//! detached `index --wait` that `index` starts.

use lastseek::index::{QueryOpts, Searcher};
use lastseek::online::{self, Published};
use lastseek::store::{self, VectorStore};
use lastseek::LastSeekError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const BUILDING: &str = "building";
const READY: &str = "ready";
const FAILED: &str = "failed";

#[derive(Clone, Serialize, Deserialize)]
struct IndexState {
    state: String,
    schemas: Vec<String>,
    started_at: String,
    finished_at: Option<String>,
    error: Option<String>,
}

pub fn run(args: &[String]) -> Result<(), LastSeekError> {
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let result = match cmd {
        "index" => index(args),
        "search" => search(args),
        "status" => status(args),
        other => Err(LastSeekError::Online(format!(
            "unknown app command {other:?}"
        ))),
    };
    match result {
        Ok(()) => Ok(()),
        Err(error @ LastSeekError::UnknownSchema { .. }) => {
            refuse(
                2,
                json!({
                    "ok": false,
                    "code": "unknown_schema",
                    "error": one_line(&error),
                }),
            );
        }
        Err(error) => {
            refuse(
                1,
                json!({
                    "ok": false,
                    "code": "error",
                    "error": one_line(&error),
                }),
            );
        }
    }
}

fn index(args: &[String]) -> Result<(), LastSeekError> {
    let (app, dir) = open_app(args);
    let schemas = normalized_schemas(&super::flag_values(args, "--schema"));
    if schemas.is_empty() {
        refuse(
            2,
            json!({
                "ok": false,
                "code": "missing_schema",
                "error": "index requires --schema NAME",
            }),
        );
    }
    if args.iter().any(|arg| arg == "--wait") {
        return build_foreground(&app, &dir, &schemas);
    }
    if online::build_running(&dir)? {
        return print_json(&building_json(&app));
    }
    if let Some(published) = exact_ready(&dir, &schemas)? {
        note_ready(&dir, &published)?;
        return print_json(&ready_json(&app, &published));
    }
    store::ensure_private_dir(&dir)?;
    write_state(
        &dir,
        &IndexState {
            state: BUILDING.into(),
            schemas: schemas.clone(),
            started_at: rfc3339_now(),
            finished_at: None,
            error: None,
        },
    )?;
    if let Err(error) = spawn_build(&dir, &app, &schemas) {
        let line = one_line(&error);
        write_state(
            &dir,
            &IndexState {
                state: FAILED.into(),
                schemas,
                started_at: rfc3339_now(),
                finished_at: Some(rfc3339_now()),
                error: Some(line.clone()),
            },
        )?;
        refuse(
            1,
            json!({"ok": false, "app": app, "state": FAILED, "error": line}),
        );
    }
    print_json(&building_json(&app))
}

fn status(args: &[String]) -> Result<(), LastSeekError> {
    let (app, dir) = open_app(args);
    print_json(&report(&app, &dir)?.body)
}

fn search(args: &[String]) -> Result<(), LastSeekError> {
    let (app, dir) = open_app(args);
    let text = search_text(args);
    if text.is_empty() {
        refuse(
            2,
            json!({
                "ok": false,
                "code": "missing_text",
                "error": "search requires the query text",
            }),
        );
    }
    let report = report(&app, &dir)?;
    if report.state != READY {
        refuse(
            3,
            json!({
                "ok": false,
                "code": "index_not_ready",
                "state": report.state,
            }),
        );
    }
    let Some(published) = online::published(&dir)? else {
        refuse(
            3,
            json!({"ok": false, "code": "index_not_ready", "state": "absent"}),
        );
    };
    let filters = super::flag_values(args, "--schema");
    for schema in &filters {
        if !published.checkpoint.holds(schema) {
            refuse(
                2,
                json!({
                    "ok": false,
                    "code": "unknown_schema",
                    "schema": schema,
                    "error": format!("the index does not hold schema {schema}"),
                }),
            );
        }
    }
    let max_changes = flag_usize(args, "--max-changes")?.unwrap_or(200);
    if max_changes == 0 {
        refuse(
            2,
            json!({
                "ok": false,
                "code": "bad_flag",
                "error": "--max-changes must be positive",
            }),
        );
    }
    let k = flag_usize(args, "--k")?;
    let min_score = flag_f32(args, "--min-score")?;

    let probe = VectorStore::open(&published.dir)?;
    let mut embedder = super::embedder_for(probe.embedder_id())?;
    drop(probe);
    let (stale, applied_through) =
        catch_up(&app, &dir, &published, embedder.as_mut(), max_changes)?;

    let store = VectorStore::open(&published.dir)?;
    let mut table = super::load_schema_table(args);
    published.checkpoint.include_scopes(&mut table);
    let query = embedder
        .embed_query(std::slice::from_ref(&text))?
        .into_iter()
        .next()
        .ok_or_else(|| LastSeekError::Embedder("embedder returned no query vector".into()))?;
    let opts = QueryOpts {
        k,
        schemas: filters,
        min_score,
    };
    let searcher = Searcher::new(&store, embedder.id())?.with_schema_table(table.clone());
    let hits = searcher.search(&query, &opts)?;
    let rows: Vec<Value> = hits
        .iter()
        .map(|hit| {
            let schema = table
                .label_for_identity(&hit.record.schema_identity)
                .unwrap_or(hit.record.schema_identity.as_str());
            json!({
                "score": hit.score,
                "schema": schema,
                "schema_identity": hit.record.schema_identity,
                "key_hash": hit.record.key_hash,
                "key_range": hit.record.key_range,
                "fragment_key": hit.record.fragment_key,
            })
        })
        .collect();
    print_json(&json!({
        "ok": true,
        "hits": rows.len(),
        "results": rows,
        "applied_through": applied_through,
        "stale": stale,
    }))
}

struct Report {
    state: String,
    body: Value,
}

fn report(app: &str, dir: &Path) -> Result<Report, LastSeekError> {
    let file = read_state(dir)?;
    let published = online::published(dir)?;
    let running = online::build_running(dir)?;
    let (state, error) = if running {
        (BUILDING, None)
    } else if file.as_ref().is_some_and(|saved| saved.state == BUILDING) {
        (FAILED, Some("the build process is gone".to_string()))
    } else if published
        .as_ref()
        .is_some_and(|item| item.checkpoint.needs_rebuild())
    {
        ("rebuild_required", None)
    } else if published.is_some() {
        (READY, None)
    } else if let Some(saved) = file.as_ref().filter(|saved| saved.state == FAILED) {
        (
            FAILED,
            Some(
                saved
                    .error
                    .clone()
                    .unwrap_or_else(|| "the build failed".to_string()),
            ),
        )
    } else {
        ("absent", None)
    };
    let (schemas, records, applied_through) = if state == "absent" {
        (Vec::new(), 0, String::new())
    } else if let Some(item) = &published {
        (
            item.checkpoint.names().to_vec(),
            live_records(&item.dir),
            item.checkpoint.cursor(),
        )
    } else if let Some(saved) = &file {
        (saved.schemas.clone(), 0, String::new())
    } else {
        (Vec::new(), 0, String::new())
    };
    let body = json!({
        "ok": true,
        "app": app,
        "state": state,
        "schemas": schemas,
        "records": records,
        "applied_through": applied_through,
        "error": error,
    });
    Ok(Report {
        state: state.to_string(),
        body,
    })
}

fn build_foreground(app: &str, dir: &Path, schemas: &[String]) -> Result<(), LastSeekError> {
    store::ensure_private_dir(dir)?;
    // Held until this function returns, including the rebuild. Status treats
    // a `building` record with nobody holding this lock as a dead build.
    let _lock = online::lock_build(dir)?;
    if let Some(published) = exact_ready(dir, schemas)? {
        note_ready(dir, &published)?;
        return print_json(&ready_json(app, &published));
    }
    let started = rfc3339_now();
    write_state(
        dir,
        &IndexState {
            state: BUILDING.into(),
            schemas: schemas.to_vec(),
            started_at: started.clone(),
            finished_at: None,
            error: None,
        },
    )?;
    match run_rebuild(dir, schemas) {
        Ok(()) => {
            let finished = rfc3339_now();
            write_state(
                dir,
                &IndexState {
                    state: READY.into(),
                    schemas: schemas.to_vec(),
                    started_at: started,
                    finished_at: Some(finished),
                    error: None,
                },
            )?;
            let published = online::published(dir)?.ok_or_else(|| {
                LastSeekError::Online("the build finished without a published index".into())
            })?;
            print_json(&ready_json(app, &published))
        }
        Err(error) => {
            let line = one_line(&error);
            write_state(
                dir,
                &IndexState {
                    state: FAILED.into(),
                    schemas: schemas.to_vec(),
                    started_at: started,
                    finished_at: Some(rfc3339_now()),
                    error: Some(line.clone()),
                },
            )?;
            refuse(
                1,
                json!({
                    "ok": false,
                    "app": app,
                    "state": FAILED,
                    "schemas": schemas,
                    "error": line,
                }),
            );
        }
    }
}

fn run_rebuild(dir: &Path, schemas: &[String]) -> Result<(), LastSeekError> {
    let table = super::load_schema_table(&[]);
    let socket = std::path::absolute(super::lastdb_socket(&[])?)?;
    let mut embedder = super::embedder_for("fastembed/bge-small-en-v1.5")?;
    if embedder.id() != "fastembed/bge-small-en-v1.5" {
        return Err(LastSeekError::Online(
            "rebuild requires the BGE embedder".into(),
        ));
    }
    online::rebuild_app(dir, &socket, schemas, &table, embedder.as_mut())?;
    Ok(())
}

fn catch_up(
    app: &str,
    root: &Path,
    published: &Published,
    embedder: &mut dyn lastseek::embed::Embedder,
    max_changes: usize,
) -> Result<(bool, String), LastSeekError> {
    let fallback = published.checkpoint.cursor();
    let Some(_guard) = online::try_lock_updates(root)? else {
        return Ok((true, fallback));
    };
    let table = super::load_schema_table(&[]);
    let socket = std::path::absolute(super::lastdb_socket(&[])?)?;
    let drained = online::drain(
        &published.dir,
        published.checkpoint.clone(),
        &socket,
        &table,
        embedder,
        max_changes,
    );
    drop(_guard);
    match drained {
        Ok(value) => {
            let stale = value
                .get("has_more")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let applied = value
                .get("applied_through")
                .and_then(Value::as_str)
                .unwrap_or(&fallback)
                .to_string();
            Ok((stale, applied))
        }
        Err(error) if feed_needs_rebuild(&error) => {
            // The published index stays readable. Rebuild in another process,
            // the same way `index` does. This search does not call `recover`.
            if !online::build_running(root)? {
                let schemas = published.checkpoint.names().to_vec();
                eprintln!(
                    "lastseek: the changes list needs a rebuild; indexing {app} in the background"
                );
                if let Err(error) = write_state(
                    root,
                    &IndexState {
                        state: BUILDING.into(),
                        schemas: schemas.clone(),
                        started_at: rfc3339_now(),
                        finished_at: None,
                        error: None,
                    },
                ) {
                    eprintln!("lastseek: could not record the rebuild: {error}");
                }
                if let Err(error) = spawn_build(root, app, &schemas) {
                    let line = one_line(&error);
                    eprintln!("lastseek: could not start the rebuild: {error}");
                    if let Err(error) = write_state(
                        root,
                        &IndexState {
                            state: FAILED.into(),
                            schemas,
                            started_at: rfc3339_now(),
                            finished_at: Some(rfc3339_now()),
                            error: Some(line),
                        },
                    ) {
                        eprintln!("lastseek: could not record the rebuild failure: {error}");
                    }
                }
            }
            Ok((true, cursor_of(&published.dir, &fallback)))
        }
        Err(error) if lastdb_unreachable(&error) => {
            Ok((true, cursor_of(&published.dir, &fallback)))
        }
        Err(error) => Err(error),
    }
}

fn feed_needs_rebuild(error: &LastSeekError) -> bool {
    matches!(
        error,
        LastSeekError::OnlineFeedGap | LastSeekError::ChangesListReplaced
    ) || matches!(error, LastSeekError::Online(message) if message.contains("indexed schemas changed"))
}

fn lastdb_unreachable(error: &LastSeekError) -> bool {
    match error {
        LastSeekError::Io(_) => true,
        LastSeekError::Online(message) => message.contains("request failed"),
        _ => false,
    }
}

fn cursor_of(slot: &Path, fallback: &str) -> String {
    online::read_checkpoint(slot)
        .map(|checkpoint| checkpoint.cursor())
        .unwrap_or_else(|_| fallback.to_string())
}

/// Re-exec this binary as `index --wait` in a new process group.
///
/// The parent does not wait. Stdin is closed. Stdout and stderr append to
/// `index.log`. The child is LastSeek, so it keeps this app's `LASTDB_APP`
/// and `LASTDB_SOCKET_PATH`. LastSeek starts no other app's command. If it
/// did, that command would be started without those two, and that app's
/// launcher would set its own. Another app's values are removed here.
fn spawn_build(dir: &Path, app: &str, schemas: &[String]) -> Result<(), LastSeekError> {
    let exe = std::env::current_exe().map_err(|error| {
        LastSeekError::Online(format!("finding the current executable: {error}"))
    })?;
    store::ensure_private_dir(dir)?;
    let mut cmd = Command::new(exe);
    cmd.arg("index").arg("--app").arg(app).arg("--wait");
    for schema in schemas {
        cmd.arg("--schema").arg(schema);
    }
    lastseek::retain_own_launcher_env(&mut cmd)?;
    cmd.stdin(Stdio::null());
    let log = open_log(dir)?;
    cmd.stdout(Stdio::from(log.try_clone()?));
    cmd.stderr(Stdio::from(log));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let child = cmd.spawn()?;
    // Dropping the handle does not wait and does not kill the child.
    drop(child);
    Ok(())
}

fn open_log(dir: &Path) -> Result<std::fs::File, LastSeekError> {
    let mut options = std::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    Ok(options.open(dir.join("index.log"))?)
}

fn exact_ready(dir: &Path, schemas: &[String]) -> Result<Option<Published>, LastSeekError> {
    let Some(published) = online::published(dir)? else {
        return Ok(None);
    };
    if published.checkpoint.needs_rebuild() || !published.checkpoint.same_schemas(schemas) {
        return Ok(None);
    }
    Ok(Some(published))
}

/// Record a ready outcome when the published index is already the requested
/// set, without starting a build. Leave an existing ready record alone so a
/// second `index` does not rewrite it.
fn note_ready(dir: &Path, published: &Published) -> Result<(), LastSeekError> {
    if let Some(saved) = read_state(dir)? {
        if saved.state == READY && published.checkpoint.same_schemas(&saved.schemas) {
            return Ok(());
        }
    }
    let started = read_state(dir)?
        .map(|saved| saved.started_at)
        .filter(|started| !started.is_empty())
        .unwrap_or_else(rfc3339_now);
    write_state(
        dir,
        &IndexState {
            state: READY.into(),
            schemas: published.checkpoint.names().to_vec(),
            started_at: started,
            finished_at: Some(rfc3339_now()),
            error: None,
        },
    )
}

fn ready_json(app: &str, published: &Published) -> Value {
    json!({
        "ok": true,
        "app": app,
        "state": READY,
        "schemas": published.checkpoint.names(),
        "records": live_records(&published.dir),
        "applied_through": published.checkpoint.cursor(),
        "error": null,
    })
}

fn building_json(app: &str) -> Value {
    json!({"ok": true, "app": app, "state": BUILDING})
}

fn live_records(dir: &Path) -> u64 {
    VectorStore::open(dir)
        .map(|store| {
            store
                .schema_row_counts()
                .values()
                .copied()
                .map(|count| count as u64)
                .sum()
        })
        .unwrap_or(0)
}

fn open_app(args: &[String]) -> (String, PathBuf) {
    let Some(name) = super::flag_value(args, "--app") else {
        refuse(
            2,
            json!({
                "ok": false,
                "code": "bad_app_name",
                "error": "--app NAME is required",
            }),
        );
    };
    if !valid_app_name(&name) {
        refuse(
            2,
            json!({
                "ok": false,
                "code": "bad_app_name",
                "error": "app name must be 1-64 characters of [a-z0-9_-]",
            }),
        );
    }
    let base = match std::env::var("LASTSEEK_HOME") {
        Ok(value) if !value.is_empty() => PathBuf::from(value),
        _ => {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".lastseek")
        }
    };
    // LASTDB_APP_DATA is not consulted. The caller's app data would put one
    // app's index in a different directory for every caller.
    (name.clone(), base.join("apps").join(name))
}

fn valid_app_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|byte| matches!(byte, b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-'))
}

fn normalized_schemas(schemas: &[String]) -> Vec<String> {
    let mut names: Vec<String> = schemas
        .iter()
        .filter(|schema| !schema.is_empty())
        .cloned()
        .collect();
    names.sort();
    names.dedup();
    names
}

fn search_text(args: &[String]) -> String {
    const VALUES: &[&str] = &["--app", "--schema", "--k", "--min-score", "--max-changes"];
    let mut index = 1usize;
    while index < args.len() {
        let token = args[index].as_str();
        if VALUES.contains(&token) {
            index += 2;
            continue;
        }
        if token.starts_with('-') && token != "-" && token.parse::<f64>().is_err() {
            index += 1;
            continue;
        }
        return token.to_string();
    }
    String::new()
}

fn flag_usize(args: &[String], name: &str) -> Result<Option<usize>, LastSeekError> {
    match super::parse_flag::<usize>(args, name) {
        Ok(value) => Ok(value),
        Err(error) => refuse(
            2,
            json!({"ok": false, "code": "bad_flag", "error": one_line(&error)}),
        ),
    }
}

fn flag_f32(args: &[String], name: &str) -> Result<Option<f32>, LastSeekError> {
    match super::parse_flag::<f32>(args, name) {
        Ok(value) => Ok(value),
        Err(error) => refuse(
            2,
            json!({"ok": false, "code": "bad_flag", "error": one_line(&error)}),
        ),
    }
}

fn state_path(dir: &Path) -> PathBuf {
    dir.join("index-state.json")
}

fn read_state(dir: &Path) -> Result<Option<IndexState>, LastSeekError> {
    match std::fs::read(state_path(dir)) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn write_state(dir: &Path, state: &IndexState) -> Result<(), LastSeekError> {
    store::write_private_file(&state_path(dir), &serde_json::to_vec_pretty(state)?)
}

fn print_json(value: &Value) -> Result<(), LastSeekError> {
    println!("{}", serde_json::to_string_pretty(value)?);
    std::io::stdout().flush()?;
    Ok(())
}

fn refuse(code: i32, body: Value) -> ! {
    if let Ok(text) = serde_json::to_string_pretty(&body) {
        println!("{text}");
    } else {
        println!("{{\"ok\":false}}");
    }
    let _ = std::io::stdout().flush();
    std::process::exit(code);
}

fn one_line(error: &impl std::fmt::Display) -> String {
    let text = error.to_string().replace(['\r', '\n'], " ");
    let mut chars = text.chars();
    let mut short: String = chars.by_ref().take(400).collect();
    if chars.next().is_some() {
        short.push_str("...");
    }
    short
}

fn rfc3339_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let tod = (secs % 86_400) as u32;
    // Howard Hinnant's civil_from_days. Days are the Unix epoch.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let year_of_era = yoe as i64 + era * 400;
    let day_of_year = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let month_part = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_part + 2) / 5 + 1;
    let month = if month_part < 10 {
        month_part + 3
    } else {
        month_part - 9
    };
    let year = if month <= 2 {
        year_of_era + 1
    } else {
        year_of_era
    };
    let hour = tod / 3600;
    let minute = (tod % 3600) / 60;
    let second = tod % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}
