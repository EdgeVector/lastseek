#!/usr/bin/env bash
# The bge-small-en-v1.5 model files LastSeek embeds with.
#
#   registry-models.sh           check that the five files are present (install probe)
#   registry-models.sh --fetch   download the missing files, then check (install prepare)
#
# The files are host state under ~/.lastseek/models, shared by every installed
# version, so an upgrade does not download them again. Each download comes
# from one pinned revision of the public model repository and must match its
# pinned SHA-256 before it is moved into place. These are the same revision
# and digests the Homebrew formula uses. A file that is already present is
# kept as it is.
set -euo pipefail

model_dir="${LASTSEEK_MODEL_DIR:-$HOME/.lastseek/models/bge-small-en-v1.5}"
# bge-small-en-v1.5 (BAAI, MIT), ONNX export by Xenova.
base="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/ea104dacec62c0de699686887e3f920caeb4f3e3"
# local name | path under the revision | SHA-256
files=(
  "model.onnx|onnx/model.onnx|828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
  "tokenizer.json|tokenizer.json|d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
  "config.json|config.json|fa73f90bf92c8cace1fbcb709626306f2bdbc9ea3e5b5f94b440df9b6aa56350"
  "special_tokens_map.json|special_tokens_map.json|b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"
  "tokenizer_config.json|tokenizer_config.json|9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3"
)

fetch=0
case "${1:-}" in
  "") ;;
  --fetch) fetch=1 ;;
  *)
    echo "usage: registry-models.sh [--fetch]" >&2
    exit 2
    ;;
esac

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

if [ "$fetch" -eq 1 ]; then
  mkdir -p "$model_dir"
  for entry in "${files[@]}"; do
    IFS='|' read -r name path expected <<<"$entry"
    if [ -s "$model_dir/$name" ]; then
      continue
    fi
    partial="$model_dir/.$name.download.$$"
    printf 'lastseek install: downloading %s\n' "$name" >&2
    if ! curl --disable --fail --silent --show-error --location --proto '=https' \
      --max-time 540 --max-filesize 268435456 --output "$partial" "$base/$path"; then
      rm -f -- "$partial"
      printf 'lastseek install: could not download %s from %s\n' "$name" "$base/$path" >&2
      exit 1
    fi
    actual="$(sha256_of "$partial")"
    if [ "$actual" != "$expected" ]; then
      rm -f -- "$partial"
      printf 'lastseek install: %s has SHA-256 %s, expected %s; nothing was installed\n' \
        "$name" "$actual" "$expected" >&2
      exit 1
    fi
    mv -- "$partial" "$model_dir/$name"
  done
fi

missing_files=()
for entry in "${files[@]}"; do
  name="${entry%%|*}"
  if [ ! -s "$model_dir/$name" ]; then
    missing_files+=("$name")
  fi
done

if [ "${#missing_files[@]}" -ne 0 ]; then
  printf 'lastseek install: missing model files under %s: %s\n' \
    "$model_dir" "${missing_files[*]}" >&2
  printf 'Run this from the LastSeek source to download them: bash scripts/registry-models.sh --fetch\n' >&2
  exit 1
fi

printf 'lastseek install: five bge-small-en-v1.5 model files are present under %s\n' \
  "$model_dir"
