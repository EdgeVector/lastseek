//! Compound probe for the two invariants that stop the index from being a
//! second, weaker copy of LastDB:
//!
//! 1. No artifact under the store root, and no query response, carries a
//!    record body. Credential material that reaches the index through an
//!    ordinary record (an `em_<48hex>` API key, an `EXM-XXXX-XXXX` invite code,
//!    a PEM header) must not be findable in `pending.jsonl`, in any
//!    generation's `meta.jsonl`, or in `lastseek query` stdout.
//! 2. No artifact under the store root is group- or world-readable — set at
//!    creation time in the writer, so it holds across compactions, not only
//!    on the one path a `chmod` was pointed at.
//!
//! Red case: on 2026-08-17 the live `~/.lastseek/index/gen-*/meta.jsonl` was
//! `0644` and 38 MB, held a record slugged `references-fkanban-fbrain-prod-owner-key`
//! in full, and `lastseek query "fkanban fbrain prod owner key"` printed the
//! key shape on stdout at exit 0. Re-verified still live on 2026-09-14 at
//! 133 MB. (`papercut-lastseek-serves-credential-material-from-a-world-readable-plaintext-corpus`)
//!
//! The probe drives the real binary through the real ingest path (`drain`,
//! then `add`) under `umask 000` — the most permissive umask there is — so a
//! writer that leaves file modes to the process umask fails here, and only a
//! writer that sets the mode itself passes. It takes the store across two
//! compaction ticks because a generation directory is replaced by the next
//! compaction, which is exactly why a one-time `chmod` on a live path was
//! never a fix.

#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The three credential shapes the 2026-08-17 sweep found in the live index.
/// Synthetic values in the same shape; nothing here is a real credential.
const API_KEY: &str = "em_0123456789abcdef0123456789abcdef0123456789abcdef";
const INVITE_CODE: &str = "EXM-QX7B-M2KD";
const PEM_HEADER: &str = "-----BEGIN PRIVATE KEY-----";

/// Run the binary under `umask 000` so every created path would be
/// `0666`/`0777` unless the writer says otherwise.
fn run(args: &[&str], stdin: Option<&str>) -> std::process::Output {
    let mut cmd = Command::new("sh");
    cmd.arg("-c")
        .arg("umask 000; exec \"$0\" \"$@\"")
        .arg(env!("CARGO_BIN_EXE_lastseek"))
        .args(args)
        .env("LASTSEEK_EMBEDDER", "deterministic");
    if stdin.is_some() {
        cmd.stdin(Stdio::piped());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().unwrap();
    if let Some(input) = stdin {
        use std::io::Write;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
    }
    child.wait_with_output().unwrap()
}

fn run_ok(args: &[&str], stdin: Option<&str>) -> serde_json::Value {
    let out = run(args, stdin);
    assert!(
        out.status.success(),
        "lastseek {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

/// Every path under `root` (inclusive), depth-first. Bounded by the store
/// layout: one level of generation directories.
fn walk(root: &Path, out: &mut Vec<PathBuf>) {
    out.push(root.to_path_buf());
    if root.is_dir() {
        for entry in std::fs::read_dir(root).unwrap().flatten() {
            walk(&entry.path(), out);
        }
    }
}

fn assert_owner_only(root: &Path, tick: &str) {
    let mut paths = Vec::new();
    walk(root, &mut paths);
    assert!(
        paths.len() > 1,
        "{tick}: store root {} is empty; the probe proved nothing",
        root.display()
    );
    for p in &paths {
        let mode = std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode & 0o077,
            0,
            "{tick}: {} is mode {mode:04o}; the index must be owner-only \
             (LastDB's store is 0600 and this is a projection of it)",
            p.display()
        );
    }
}

fn assert_no_credential_shape(haystack: &[u8], what: &str) {
    for (shape, name) in [
        (API_KEY, "em_<48hex> API key"),
        (INVITE_CODE, "EXM-XXXX-XXXX invite code"),
        (PEM_HEADER, "PEM header"),
    ] {
        assert!(
            !haystack.windows(shape.len()).any(|w| w == shape.as_bytes()),
            "{what} carries the {name} that was ingested; record bodies must not be retained"
        );
    }
}

/// Bytes of every regular file under `root`, concatenated, so the search for
/// a shape covers the pending log, every generation's meta.jsonl, the
/// manifest, the sidecars — everything the writer produced.
fn all_bytes_under(root: &Path) -> Vec<u8> {
    let mut paths = Vec::new();
    walk(root, &mut paths);
    let mut bytes = Vec::new();
    for p in paths.iter().filter(|p| p.is_file()) {
        bytes.extend_from_slice(&std::fs::read(p).unwrap());
        bytes.push(b'\n');
    }
    bytes
}

#[test]
fn credential_bearing_records_leave_no_body_on_disk_or_in_a_response_and_nothing_is_world_readable()
{
    let root = tempfile::tempdir().unwrap();
    let inbox = root.path().join("inbox");
    let home = root.path().join("index");
    std::fs::create_dir_all(&inbox).unwrap();

    // Tick 1: the host's IndexChangeBatch feed delivers a record whose body
    // holds an API key and an invite code. This is the path the live corpus
    // was populated through.
    std::fs::write(
        inbox.join("0001.json"),
        format!(
            r#"{{"schema_name":"reference","changes":[{{
              "mutation_id":"m1","kind":"upsert",
              "key_value":{{"hash":"references-fkanban-fbrain-prod-owner-key","range":null}},
              "fields_and_values":{{
                "title":"fkanban fbrain prod owner key",
                "body":"owner key {API_KEY}\ndogfood invite {INVITE_CODE}"
              }}}}]}}"#
        ),
    )
    .unwrap();
    let home_arg = home.to_string_lossy().to_string();
    let inbox_arg = inbox.to_string_lossy().to_string();
    let drained = run_ok(&["drain", "--home", &home_arg, "--inbox", &inbox_arg], None);
    assert_eq!(drained["records"], 1);

    // The pending log exists now and already must not carry the body.
    assert!(home.join("pending.jsonl").is_file());
    assert_no_credential_shape(&all_bytes_under(&home), "the uncompacted store");
    assert_owner_only(&home, "after drain, before any compaction");

    run_ok(&["compact", "--home", &home_arg], None);
    let gen1 = current_generation(&home);
    assert!(gen1.join("meta.jsonl").is_file());
    assert_no_credential_shape(&all_bytes_under(&home), "generation 1");
    assert_owner_only(&home, "after compaction 1");

    // Tick 2: a second record, this time through `add`, carrying a PEM
    // header. A second compaction replaces the generation directory, which
    // is the moment a post-hoc chmod on the first one would stop mattering.
    let pem_record = format!(
        r#"{{"schema_identity":"reference","key_hash":"schema-resolver-pack-release-key","key_range":null,"fragment_key":"body","text":"schema resolver pack release key public\n{PEM_HEADER}\nMIIE...","mutation_id":"m2"}}"#
    );
    let added = run_ok(&["add", "--home", &home_arg], Some(&pem_record));
    assert_eq!(added["appended"], 1);
    assert_no_credential_shape(&all_bytes_under(&home), "the pending log after add");

    run_ok(&["compact", "--home", &home_arg], None);
    let gen2 = current_generation(&home);
    assert_ne!(
        gen1, gen2,
        "the second compaction must publish a new generation"
    );
    assert!(
        !gen1.exists(),
        "the first generation should have been garbage-collected"
    );
    assert_no_credential_shape(&all_bytes_under(&home), "generation 2");
    assert_owner_only(&home, "after compaction 2");

    // The exact live reproductions, against the fixed writer: each query
    // still finds its record by key, and none of them echo a body.
    for (query, expect_key) in [
        (
            "fkanban fbrain prod owner key",
            "references-fkanban-fbrain-prod-owner-key",
        ),
        (
            "schema resolver pack release key public",
            "schema-resolver-pack-release-key",
        ),
    ] {
        let out = run(&["query", query, "--home", &home_arg, "--k", "5"], None);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_no_credential_shape(&out.stdout, &format!("query {query:?} stdout"));
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let results = v["results"].as_array().unwrap();
        assert_eq!(results[0]["key_hash"], expect_key, "query {query:?}");
        assert!(
            results[0].get("text").is_none(),
            "a hit must carry keys and a score, never a body: {}",
            results[0]
        );
    }
}

#[test]
fn exact_mode_is_refused_with_the_reason_not_silently_ignored() {
    // `--exact` filtered on retained text. With no text retained the flag
    // cannot mean anything, and a caller who passes it must not get the
    // unfiltered answer at exit 0 as if the filter applied.
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("index");
    let home_arg = home.to_string_lossy().to_string();
    let out = run(&["query", "anything", "--home", &home_arg, "--exact"], None);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--exact") && stderr.contains("LastDB"),
        "refusal must name the flag and where the text lives: {stderr}"
    );
}

#[test]
fn a_generation_that_still_retains_bodies_is_rewritten_once_even_when_nothing_is_pending() {
    // A store built by the previous binary carries bodies in its current
    // generation and a manifest without `bodies_retained`. The fixed binary
    // must not wait for new ops to arrive before purging that generation;
    // a quiet index would otherwise keep the copy forever.
    use lastseek::embed::{DeterministicEmbedder, Embedder};
    use lastseek::store::{PendingOp, Record, VectorStore};

    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("index");
    let mut emb = DeterministicEmbedder::new(64);
    let mut store = VectorStore::open(&home).unwrap();
    store.bind_embedder(&emb).unwrap();
    let text = format!("owner key {API_KEY}");
    let v = emb.embed(std::slice::from_ref(&text)).unwrap().remove(0);
    store
        .append(&[PendingOp {
            record: Record {
                schema_identity: "reference".into(),
                key_hash: Some("k".into()),
                key_range: None,
                fragment_key: "body".into(),
                text,
                mutation_id: None,
                deleted: false,
            },
            vector: Some(v),
        }])
        .unwrap();
    store.compact().unwrap();
    drop(store);

    // Forge the previous binary's on-disk state: a body in meta.jsonl and a
    // manifest that predates the field.
    let gen1 = current_generation(&home);
    let meta = gen1.join("meta.jsonl");
    let line = std::fs::read_to_string(&meta).unwrap();
    let mut row: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
    row["text"] = serde_json::Value::String(format!("owner key {API_KEY}"));
    let forged = format!("{}\n", serde_json::to_string(&row).unwrap());
    std::fs::write(&meta, &forged).unwrap();
    let rows_bin = gen1.join("rows.bin");
    let mut rows = std::fs::read(&rows_bin).unwrap();
    // text_len lives at bytes 16..20 of the single 32-byte row.
    rows[16..20].copy_from_slice(&((forged.len() - 1) as u32).to_ne_bytes());
    std::fs::write(&rows_bin, rows).unwrap();
    let manifest_path = gen1.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("bodies_retained");
    manifest["meta_bytes"] = serde_json::Value::from(forged.len() as u64);
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    assert_body_present(&all_bytes_under(&home));

    // Nothing pending. The old compact() returned early here.
    let mut store = VectorStore::open(&home).unwrap();
    assert_eq!(store.pending_ops(), 0);
    store.compact().unwrap();
    let gen2 = current_generation(&home);
    assert_ne!(gen1, gen2, "a body-bearing generation must be rewritten");
    assert_no_credential_shape(&all_bytes_under(&home), "the rewritten store");
    assert_eq!(store.committed_rows(), 1);
    assert_eq!(store.record_at(0).unwrap().key_hash.as_deref(), Some("k"));

    // And only once: the rewritten manifest says the bodies are gone.
    store.compact().unwrap();
    assert_eq!(current_generation(&home), gen2);
}

/// Inverse helper for the forged pre-fix state: the probe must actually be
/// looking at a body-bearing generation, or the rewrite it then asserts is
/// vacuous.
fn assert_body_present(haystack: &[u8]) {
    assert!(
        haystack
            .windows(API_KEY.len())
            .any(|w| w == API_KEY.as_bytes()),
        "the forged pre-fix generation must carry the body for this probe to mean anything"
    );
}

fn current_generation(home: &Path) -> PathBuf {
    let n: u64 = std::fs::read_to_string(home.join("CURRENT"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    home.join(format!("gen-{n:07}"))
}
