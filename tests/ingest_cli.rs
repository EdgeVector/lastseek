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

    let status = run(&["status", "--home", &home_arg]);
    assert_eq!(status["non_searchable_records_skipped"], 1);
}
