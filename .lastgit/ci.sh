#!/usr/bin/env bash
# Required status gate for LastSeek (GitHub ci-required calls this script).
set -euo pipefail
cd "$(dirname "$0")/.."

# LastGit runs under launchd, whose minimal PATH does not include rustup's
# user-local toolchain by default.
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
command -v cargo >/dev/null

test "$(head -n 1 .last-stack/pr-venue)" = "github"
test -f .github/workflows/ci-required.yml

# Product boundary: LastSeek runs outside Mini and stores nothing durable that
# LastDB owns. These greps fail loudly if that framing is edited away.
grep -q "https://github.com/EdgeVector/lastseek" README.md
grep -q "outside Mini" README.md
grep -q "local-only and regenerable" README.md
grep -q "not CloudSync product data" README.md

# LastSeek must not vendor LastDB's storage layer — it is a separate plane.
test ! -d vendor/laststore

bash -n .lastgit/ci.sh
bash -n .lastgit/artifact-release.sh

# The feature-complete binary is built only by the separate `publish` job of
# ci-required.yml (macOS runner, last-stack reusable workflow), which packs
# these paths. The builder skips an entry whose context is not ci-required, so
# the entry must carry no context. Keep this gate cheap while proving the
# publication route cannot silently fall back to the slow build.
jq -e '
  [.artifacts[]
    | select(
        .app == "lastseek"
        and .platform == "darwin-arm64"
        and (has("context") | not)
        and .paths == ["dist/lastseek", "LICENSE", "THIRD_PARTY_NOTICES"]
      )]
  | length == 1
' .lastgit/artifacts.json >/dev/null

# Public Homebrew binaries are Apache-2.0 (decision 2026-09-23). The license
# files the release artifact carries must exist and agree with Cargo.toml.
grep -qx 'license = "Apache-2.0"' Cargo.toml
grep -q 'Apache License' LICENSE
grep -q 'Version 2.0, January 2004' LICENSE
grep -q 'ONNX Runtime' THIRD_PARTY_NOTICES
grep -q 'fastembed' THIRD_PARTY_NOTICES
grep -q 'Install with Homebrew' README.md
if grep -qE '^[[:space:]]*(bash[[:space:]]+)?\.lastgit/artifact-release\.sh([[:space:]]|$)' .lastgit/ci.sh; then
  echo "ci-required must not invoke the artifact-release context" >&2
  exit 1
fi

# The gate builds an explicit feature set, NOT --all-features.
#
# --all-features turns on `neural`, and that broke this gate two ways at once:
#
#   1. It downloads. The vendored fastembed dep enables `ort-download-binaries`,
#      so building `neural` fetches ONNX Runtime binaries at BUILD time. The
#      gate's own contract — stated in Cargo.toml and README — is that CI links
#      no inference runtime and fetches no weights. --all-features contradicted
#      the thing it was supposed to protect.
#   2. It times out. Compiling ort + the vendored fastembed graph in release
#      takes ~11.5 minutes on this host, against LastGit's 600000ms CI timeout.
#      Measured: cr-msjufz3p-b283 died exactly there
#      (papercut-pipeline-stuck-cr-lastseek-cr-msjufz3p, P0), and every future
#      Rust CR in this repo would have died the same way.
#
# `neural` is proven by `tests/neural.rs` against real bge weights, which only a
# machine that HAS the weights can run — see README. Keep it out of the required
# gate and run it locally:
#
#   cargo test --release --features neural,fault-injection
#
# `fault-injection` stays in the gate: it is pure Rust, costs nothing, and the
# compaction fault matrix is the durability proof.
FEATURES=fault-injection

# Guard the reason above against a well-meaning re-broadening.
if grep -qE '^[[:space:]]*cargo (clippy|test).*--all-features' .lastgit/ci.sh; then
  echo "ci-required must not build all features" >&2
  exit 1
fi

cargo fmt --all -- --check
cargo clippy --all-targets --features "$FEATURES" -- -D warnings
cargo test --release --features "$FEATURES"

echo "lastgit ci gate PASSED"
