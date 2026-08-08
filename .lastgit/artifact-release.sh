#!/usr/bin/env bash
# Slow, feature-complete artifact producer. This is intentionally separate from
# ci-required: neural compilation downloads ONNX Runtime and exceeds that gate's
# ten-minute budget.
set -euo pipefail
cd "$(dirname "$0")/.."

# LastGit runs under launchd, whose minimal PATH does not include rustup's
# user-local toolchain by default.
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
command -v cargo >/dev/null

if [ "${LASTGIT_CI_CONTEXT:-artifact-release}" != "artifact-release" ]; then
  echo "artifact-release.sh must run as the artifact-release LastGit context" >&2
  exit 2
fi

cargo build --locked --release --features neural
mkdir -p dist
install -m 0755 target/release/lastseek dist/lastseek
test -x dist/lastseek
dist/lastseek --help >/dev/null

echo "lastseek artifact release PASSED"
