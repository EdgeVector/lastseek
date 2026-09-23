//! `lastseek --version` / `-V` print `lastseek <Cargo version>`.
//!
//! Release tooling and the Homebrew formula test parse the LAST whitespace
//! token as the semver, so the shape is part of the contract.

use std::process::Command;

fn version_output(flag: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_lastseek"))
        .arg(flag)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "lastseek {flag} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn version_flags_print_name_and_cargo_version() {
    let want = format!("lastseek {}\n", env!("CARGO_PKG_VERSION"));
    for flag in ["--version", "-V"] {
        let got = version_output(flag);
        assert_eq!(got, want, "{flag}");
        let last = got.split_whitespace().last().unwrap();
        assert_eq!(last, env!("CARGO_PKG_VERSION"), "{flag}");
        assert_eq!(last.split('.').count(), 3, "{flag}: {last} is not x.y.z");
    }
}
