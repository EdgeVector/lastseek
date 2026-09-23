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

# Ask cargo where it put the binary instead of assuming `target/`.
#
# `CARGO_TARGET_DIR` redirects the output tree, and this script hardcoded
# `target/release/lastseek` — so under it the build SUCCEEDED and the install
# failed with "No such file or directory", failing the whole release for a
# reason that has nothing to do with the code.
#
# That variable is not hypothetical here: the watcher builds each commit in a
# fresh temp checkout, so without a shared target dir it recompiles ort +
# tokenizers + fastembed from scratch every time — ~11.5 minutes of identical
# work, which is what blew the CI timeout in the first place. Pointing it at a
# persistent directory takes a rebuild to ~30s, and this script has to work
# when it does.
target_dir="${CARGO_TARGET_DIR:-target}"
built="$target_dir/release/lastseek"
test -x "$built" || {
  echo "expected a release binary at $built (CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-unset})" >&2
  exit 1
}

# `dist/` is what artifacts.json publishes, and it is always repo-relative
# regardless of where cargo built.
mkdir -p dist
install -m 0755 "$built" dist/lastseek
test -x dist/lastseek
# Prove the shipped binary runs, and that it is the FEATURE-COMPLETE one — a
# default-features build would install and pass `--help` while being unable to
# embed anything, which is exactly the silent-useless-artifact case.
dist/lastseek --help >/dev/null
dist/lastseek --help | grep -q 'bge-small-en-v1.5'

# The release tarball and the Homebrew formula read the version from
# `--version`: the LAST whitespace token must be the Cargo semver.
cargo_version="$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)"
shipped_version="$(dist/lastseek --version | awk '{print $NF}')"
if [ "$shipped_version" != "$cargo_version" ]; then
  echo "dist/lastseek --version says '$shipped_version', Cargo.toml says '$cargo_version'" >&2
  exit 1
fi

# The public binary is Apache-2.0 and statically links third-party code, so
# the license and the third-party notices travel with it. artifacts.json
# publishes them from the tree root, next to dist/ (not inside it), so the
# dist/lastseek path that Host Track links stays unchanged.
for f in LICENSE THIRD_PARTY_NOTICES; do
  test -s "$f" || {
    echo "release artifact file $f is missing or empty" >&2
    exit 1
  }
done
grep -q 'Apache License' LICENSE
grep -q 'ONNX Runtime' THIRD_PARTY_NOTICES

echo "lastseek artifact release PASSED"
