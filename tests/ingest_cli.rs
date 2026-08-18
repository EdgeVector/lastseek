use std::process::Command;

fn run(args: &[&str]) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_lastseek"))
        .args(args)
        .env("LASTSEEK_EMBEDDER", "deterministic")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "lastseek failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn drain_and_status_expose_owner_declared_non_searchable_rows() {
    let root = tempfile::tempdir().unwrap();
    let inbox = root.path().join("inbox");
    let home = root.path().join("index");
    std::fs::create_dir_all(&inbox).unwrap();
    std::fs::write(
        inbox.join("0001.json"),
        r#"{
          "schema_name":"internal-schema",
          "searchable_fields":[],
          "changes":[{
            "mutation_id":"proof-1",
            "kind":"upsert",
            "key_value":{"hash":"__internal__proof","range":null},
            "fields_and_values":{"title":"this must never be embedded"}
          }]
        }"#,
    )
    .unwrap();

    let home_arg = home.to_string_lossy();
    let inbox_arg = inbox.to_string_lossy();
    let drained = run(&[
        "drain",
        "--home",
        &home_arg,
        "--inbox",
        &inbox_arg,
        "--max-files",
        "10",
    ]);
    assert_eq!(drained["records"], 0);
    assert_eq!(drained["tombstones"], 1);
    assert_eq!(drained["non_searchable_records_skipped"], 1);
    assert_eq!(drained["non_searchable_records_skipped_total"], 1);

    let catalog = root.path().join("catalog.json");
    std::fs::write(&catalog, r#"{"counts_included":true,"schemas":[]}"#).unwrap();
    let catalog_arg = catalog.to_string_lossy();
    let status = run(&["status", "--home", &home_arg, "--catalog", &catalog_arg]);
    assert_eq!(status["non_searchable_records_skipped"], 1);
}

#[test]
fn an_undeclared_schema_is_not_rated_and_the_partition_count_is_not_a_denominator() {
    // The catalog's `record_count` is rows in the partition, including every
    // superseded version. This test used to assert `degraded` / `ratio: 0.0`
    // from that number alone, which is exactly the defect: on the live host it
    // reported 63% for a schema that was at 100%, and `degraded` for 139
    // schemas nobody had opted into indexing.
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("index");
    let available = root.path().join("available.json");
    let table = root.path().join("schema-table.json");
    let catalog = root.path().join("catalog.json");
    let counts = root.path().join("source-counts.json");
    std::fs::write(
        &available,
        r#"{"schemas":[{"descriptive_name":"Card","name":"card-registry","identity_hash":"card-ss"}]}"#,
    )
    .unwrap();
    std::fs::write(
        &catalog,
        r#"{"counts_included":true,"schemas":[{"descriptive_name":"Card","name":"card-registry","identity_hash":"node-recomputed","record_count":7}]}"#,
    )
    .unwrap();

    let home_arg = home.to_string_lossy();
    let available_arg = available.to_string_lossy();
    let table_arg = table.to_string_lossy();
    let catalog_arg = catalog.to_string_lossy();
    let counts_arg = counts.to_string_lossy();
    run(&[
        "schema-table",
        "--from",
        &available_arg,
        "--table",
        &table_arg,
    ]);

    let status = run(&[
        "status",
        "--coverage",
        "--home",
        &home_arg,
        "--table",
        &table_arg,
        "--catalog",
        &catalog_arg,
        "--source-counts",
        &counts_arg,
    ]);

    // Nothing declared: reported, not rated, and the overall verdict says so
    // rather than claiming a shortfall.
    assert_eq!(status["state"], "unknown");
    assert_eq!(status["coverage"]["tracked_schemas"], 0);
    assert_eq!(status["coverage"]["untracked_schemas"], 1);
    let row = &status["coverage"]["schemas"][0];
    assert_eq!(row["schema"], "Card");
    // The join is still through the registry name, never the node's hash.
    assert_eq!(row["identity"], "card-ss");
    assert_eq!(row["state"], "not_tracked");
    assert_eq!(row["vectors_held"], 0);
    assert!(row["ratio"].is_null(), "no denominator means no ratio");
    assert!(row["live_records"].is_null());
    // Carried, under a name that cannot be mistaken for a denominator again.
    assert_eq!(row["partition_rows"], 7);
}

#[test]
fn declaring_a_live_count_makes_a_schema_tracked_and_rated() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("index");
    let available = root.path().join("available.json");
    let table = root.path().join("schema-table.json");
    let catalog = root.path().join("catalog.json");
    let counts = root.path().join("source-counts.json");
    std::fs::write(
        &available,
        r#"{"schemas":[{"descriptive_name":"Card","name":"card-registry","identity_hash":"card-ss"}]}"#,
    )
    .unwrap();
    // Partition rows deliberately much larger than the live count, mirroring
    // the live host (1,986 partition rows over ~167 live cards).
    std::fs::write(
        &catalog,
        r#"{"counts_included":true,"schemas":[{"descriptive_name":"Card","name":"card-registry","record_count":1986}]}"#,
    )
    .unwrap();

    let home_arg = home.to_string_lossy();
    let available_arg = available.to_string_lossy();
    let table_arg = table.to_string_lossy();
    let catalog_arg = catalog.to_string_lossy();
    let counts_arg = counts.to_string_lossy();
    run(&[
        "schema-table",
        "--from",
        &available_arg,
        "--table",
        &table_arg,
    ]);

    // An empty index still accepts a declaration: the schema resolves through
    // the table even before any row is held.
    let declared = run(&[
        "coverage-declare",
        "--home",
        &home_arg,
        "--table",
        &table_arg,
        "--source-counts",
        &counts_arg,
        "--schema",
        "Card",
        "--live-records",
        "0",
        "--source",
        "kanban list --all",
    ]);
    assert_eq!(declared["ok"], true);
    assert_eq!(declared["identities"][0], "card-ss");

    let status = run(&[
        "status",
        "--coverage",
        "--home",
        &home_arg,
        "--table",
        &table_arg,
        "--catalog",
        &catalog_arg,
        "--source-counts",
        &counts_arg,
    ]);
    // Declared zero live records, holding zero: that is covered, and the
    // 1,986 partition rows must not turn it red.
    assert_eq!(status["state"], "healthy");
    assert_eq!(status["coverage"]["tracked_schemas"], 1);
    assert_eq!(status["coverage"]["degraded_schemas"], 0);
    let row = &status["coverage"]["schemas"][0];
    assert_eq!(row["state"], "healthy");
    assert_eq!(row["live_records"], 0);
    assert_eq!(row["partition_rows"], 1986);
    assert_eq!(row["declared_by"], "kanban list --all");
}

#[test]
fn declaring_an_unresolvable_schema_fails_loudly() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("index");
    let available = root.path().join("available.json");
    let table = root.path().join("schema-table.json");
    let counts = root.path().join("source-counts.json");
    std::fs::write(
        &available,
        r#"{"schemas":[{"descriptive_name":"Card","name":"card-registry","identity_hash":"card-ss"}]}"#,
    )
    .unwrap();
    let home_arg = home.to_string_lossy();
    let available_arg = available.to_string_lossy();
    let table_arg = table.to_string_lossy();
    let counts_arg = counts.to_string_lossy();
    run(&[
        "schema-table",
        "--from",
        &available_arg,
        "--table",
        &table_arg,
    ]);

    // Same strictness as the query surface: a declaration for a schema that
    // does not exist would otherwise sit in the file forever, tracking nothing.
    let out = Command::new(env!("CARGO_BIN_EXE_lastseek"))
        .env("LASTSEEK_EMBEDDER", "deterministic")
        .args([
            "coverage-declare",
            "--home",
            &home_arg,
            "--table",
            &table_arg,
            "--source-counts",
            &counts_arg,
            "--schema",
            "Crad",
            "--live-records",
            "5",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown schema"), "{stderr}");
    assert!(
        !counts.exists(),
        "a failed declaration must not write the file"
    );
}

/// The property that keeps a health check cheaper than the thing it checks.
///
/// `status` used to fetch the node's schema catalog WITH row counts on every
/// invocation. brain's and kanban's plane client calls `lastseek status` to
/// decide whether the plane is up, so every `brain ask` triggered a count over
/// ~1,141 schemas. Measured with `lastdb ops`, `client=lastseek kind=schema`
/// became the worst consumer on the primary: 244 calls, 5,519,430ms total,
/// 194 cold shard loads per call, one call at 749,904ms.
#[test]
fn default_status_never_reads_the_node_catalog() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("index");
    let home_arg = home.to_string_lossy();

    // A catalog path that cannot be read. If `status` touches it at all, this
    // fails — which is the whole point: the assertion is about the ABSENCE of
    // a read, and a passing status here can only mean it never looked.
    let missing = root.path().join("no-such-catalog.json");
    let missing_arg = missing.to_string_lossy();

    let status = run(&["status", "--home", &home_arg, "--catalog", &missing_arg]);
    assert_eq!(status["ok"], true);
    // Local facts are still reported — this is a useful status, not a stub.
    assert!(status["embedder_id"].is_string());
    assert!(status["committed_rows"].is_number());
    // And it says coverage was not measured, rather than implying all-clear.
    assert_eq!(status["state"], "unknown");
    assert_eq!(status["coverage"]["available"], false);
    let reason = status["coverage"]["reason"].as_str().unwrap_or("");
    assert!(reason.contains("not requested"), "{reason}");
    assert!(reason.contains("--coverage"), "{reason}");
}

#[test]
fn coverage_is_reported_when_explicitly_requested() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("index");
    let catalog = root.path().join("catalog.json");
    std::fs::write(&catalog, r#"{"counts_included":true,"schemas":[]}"#).unwrap();
    let home_arg = home.to_string_lossy();
    let catalog_arg = catalog.to_string_lossy();

    let status = run(&[
        "status",
        "--coverage",
        "--home",
        &home_arg,
        "--catalog",
        &catalog_arg,
    ]);
    assert_eq!(status["coverage"]["available"], true);
}

/// The regression that made this test file's harness dangerous to use.
///
/// On 2026-08-17 a scheduled owner run typed `lastseek drain --help` to read
/// the flag list. The CLI scanned for flags it knew and silently discarded the
/// rest, so `drain` RAN — with its default `--max-files 2000`, against the live
/// inbox — applying 1,234 batch files and 903 records to the primary index and
/// advancing the watermark, exit 0. It happened twice in three runs; the second
/// run had read the warning and was avoiding it, because `--help` does not read
/// as a flag you choose to type.
///
/// The bar is byte-identical, not "mostly harmless": help must be answered
/// before any side effect exists to measure.
#[test]
fn help_on_a_mutating_verb_prints_usage_and_touches_nothing() {
    let root = tempfile::tempdir().unwrap();
    let inbox = root.path().join("inbox");
    let home = root.path().join("index");
    std::fs::create_dir_all(&inbox).unwrap();
    std::fs::write(
        inbox.join("1786150242984_a.json"),
        r#"{"schema_name":"s","changes":[{"mutation_id":"m","kind":"upsert",
            "key_value":{"hash":"k","range":null},
            "fields_and_values":{"body":"must not be indexed by a help request"}}]}"#,
    )
    .unwrap();
    let home_arg = home.to_string_lossy().to_string();
    let inbox_arg = inbox.to_string_lossy().to_string();

    for args in [
        vec![
            "drain", "--home", &home_arg, "--inbox", &inbox_arg, "--help",
        ],
        vec!["drain", "--home", &home_arg, "--inbox", &inbox_arg, "-h"],
        vec![
            "drain",
            "--home",
            &home_arg,
            "--inbox",
            &inbox_arg,
            "--consume",
            "--help",
        ],
        vec!["compact", "--home", &home_arg, "--help"],
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_lastseek"))
            .args(&args)
            .env("LASTSEEK_EMBEDDER", "deterministic")
            .output()
            .unwrap();
        assert!(out.status.success(), "{args:?} should exit 0");
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            stdout.contains("usage") || stdout.starts_with("lastseek "),
            "{args:?} printed no usage: {stdout}"
        );
        assert!(
            !stdout.contains("\"applied_through\"") && !stdout.contains("\"folded_ops\""),
            "{args:?} EXECUTED THE VERB: {stdout}"
        );
        // The store was never created, so nothing was written at all.
        assert!(!home.exists(), "{args:?} created the store directory");
        // And the batch file the drain would have consumed is still there.
        assert!(inbox.join("1786150242984_a.json").exists());
    }
}

/// `--limit` is the flag brain and kanban both spell by muscle memory, and the
/// one this CLI does not have. It used to return the default 20 and exit 0 —
/// silent under-return to a recall-sensitive consumer.
#[test]
fn an_unrecognized_flag_is_refused_by_name_instead_of_ignored() {
    let root = tempfile::tempdir().unwrap();
    let home_arg = root.path().join("index").to_string_lossy().to_string();
    let out = Command::new(env!("CARGO_BIN_EXE_lastseek"))
        .args(["status", "--home", &home_arg, "--limit", "50"])
        .env("LASTSEEK_EMBEDDER", "deterministic")
        .output()
        .unwrap();
    assert!(!out.status.success(), "an unknown flag must not exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--limit"), "names the flag: {stderr}");
    assert!(stderr.contains("status"), "names the verb: {stderr}");
    assert!(
        out.stdout.is_empty(),
        "nothing on stdout, so a caller parsing JSON sees a clean failure"
    );
}

/// Freshness is the signal `remaining` could never carry: `remaining` is files
/// left after a bounded pass, so a tick slow enough to absorb its own interval
/// always reports zero however stale the corpus is.
#[test]
fn status_reports_how_far_behind_the_applied_corpus_is() {
    let root = tempfile::tempdir().unwrap();
    let inbox = root.path().join("inbox");
    let home = root.path().join("index");
    std::fs::create_dir_all(&inbox).unwrap();

    let home_arg = home.to_string_lossy().to_string();
    let inbox_arg = inbox.to_string_lossy().to_string();

    // Nothing applied yet: absent, NOT zero. "fresh" and "unknown" are
    // different answers and a consumer deciding whether to trust a semantic
    // read depends on telling them apart.
    let fresh = run(&["status", "--home", &home_arg]);
    assert!(fresh["freshness_lag_ms"].is_null());

    // A batch stamped 2023-11-14, so the lag is unambiguously large and does
    // not depend on how long the test took to run.
    std::fs::write(
        inbox.join("1700000000000_old.json"),
        r#"{"schema_name":"s","searchable_fields":["body"],"changes":[
            {"mutation_id":"m","kind":"upsert","key_value":{"hash":"k","range":null},
             "fields_and_values":{"body":"stale"}}]}"#,
    )
    .unwrap();
    // A second batch that stays queued, so `--queue` has something to count.
    std::fs::write(
        inbox.join("9700000000000_future.json"),
        r#"{"schema_name":"s","changes":[]}"#,
    )
    .unwrap();

    run(&[
        "drain",
        "--home",
        &home_arg,
        "--inbox",
        &inbox_arg,
        "--max-files",
        "1",
    ]);

    let status = run(&["status", "--home", &home_arg, "--inbox", &inbox_arg]);
    let lag = status["freshness_lag_ms"].as_u64().expect("a lag");
    assert!(
        lag > 60_000_000_000,
        "a batch from 2023 must read as years behind, got {lag}"
    );
    // The queue costs a directory read, so it stays opt-in and says so.
    assert_eq!(status["queue"]["available"], false);

    let queued = run(&[
        "status", "--home", &home_arg, "--inbox", &inbox_arg, "--queue",
    ]);
    assert_eq!(queued["queue"]["pending"], 1);
    // The file the watermark names is the last one APPLIED, not one skipped.
    assert_eq!(queued["queue"]["stranded_below_watermark"], 0);

    // A batch that lands with a timestamp BELOW the watermark is invisible to
    // the drain forever — `pending_batches` selects `name > applied_through`,
    // and `remaining` only ever looks forward, so nothing reports it. This is
    // the live 17-file band on the primary host, reproduced.
    std::fs::write(
        inbox.join("1600000000000_late_arrival.json"),
        r#"{"schema_name":"s","changes":[]}"#,
    )
    .unwrap();
    let stranded = run(&[
        "status", "--home", &home_arg, "--inbox", &inbox_arg, "--queue",
    ]);
    assert_eq!(stranded["queue"]["stranded_below_watermark"], 1);
    assert_eq!(
        stranded["queue"]["pending"], 1,
        "a stranded file must not be counted as queued work"
    );
}
