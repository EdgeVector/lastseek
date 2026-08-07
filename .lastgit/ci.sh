#!/usr/bin/env bash
# Required LastGit status gate for LastSeek.
set -euo pipefail
cd "$(dirname "$0")/.."

test "$(head -n 1 .last-stack/pr-venue)" = "lastgit"
test ! -e .github/workflows

# Product boundary: LastSeek runs outside Mini and stores nothing durable that
# LastDB owns. These greps fail loudly if that framing is edited away.
grep -q "lastdb:///lastseek" README.md
grep -q "outside Mini" README.md
grep -q "local-only and regenerable" README.md
grep -q "not CloudSync product data" README.md

# LastSeek must not vendor LastDB's storage layer — it is a separate plane.
test ! -d vendor/laststore

bash -n .lastgit/ci.sh

cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release

echo "lastgit ci gate PASSED"
