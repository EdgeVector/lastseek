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
fn query_and_drain_metrics_are_visible_in_status() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("index");
    let inbox = root.path().join("inbox");
    std::fs::create_dir_all(&inbox).unwrap();
    let home_arg = home.to_string_lossy();
    let inbox_arg = inbox.to_string_lossy();

    run(&[
        "drain",
        "--home",
        &home_arg,
        "--inbox",
        &inbox_arg,
        "--max-files",
        "10",
    ]);
    run(&["query", "metrics-probe", "--home", &home_arg, "--k", "1"]);

    let status = run(&["status", "--home", &home_arg]);
    assert_eq!(status["query_count"], 1);
    assert!(status["p95_ms"].is_number());
    assert_eq!(status["timeout_count"], 0);
    assert_eq!(status["last_drain"]["files"], 0);
    assert_eq!(status["last_drain"]["records"], 0);
    assert_eq!(status["last_drain"]["tombstones"], 0);
    assert_eq!(status["last_drain"]["compacted"], false);

    let metrics = std::fs::read_to_string(home.join("query-metrics.json")).unwrap();
    assert!(!metrics.contains("metrics-probe"));
}
