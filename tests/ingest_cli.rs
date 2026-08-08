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
fn status_reports_source_records_without_vectors_as_degraded() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("index");
    let available = root.path().join("available.json");
    let table = root.path().join("schema-table.json");
    let catalog = root.path().join("catalog.json");
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
    ]);
    assert_eq!(status["state"], "degraded");
    assert_eq!(status["coverage"]["schemas"][0]["schema"], "Card");
    assert_eq!(status["coverage"]["schemas"][0]["identity"], "card-ss");
    assert_eq!(status["coverage"]["schemas"][0]["vectors_held"], 0);
    assert_eq!(status["coverage"]["schemas"][0]["source_records"], 7);
    assert_eq!(status["coverage"]["schemas"][0]["ratio"], 0.0);
}
