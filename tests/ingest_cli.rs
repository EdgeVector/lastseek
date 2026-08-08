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
