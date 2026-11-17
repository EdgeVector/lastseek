#!/usr/bin/env bash
# Put the bge-small-en-v1.5 files in a directory and verify every checksum.
# CI caches the directory, so the download happens only when the cache is empty.
set -euo pipefail

dir="${1:?usage: ci-fetch-model.sh DIR}"
base="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main"
sums="
828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35  model.onnx
d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66  tokenizer.json
fa73f90bf92c8cace1fbcb709626306f2bdbc9ea3e5b5f94b440df9b6aa56350  config.json
b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3  special_tokens_map.json
9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3  tokenizer_config.json
"

mkdir -p "$dir"
cd "$dir"
if ! sha256sum --check --status <<< "${sums:1}" 2> /dev/null; then
  curl --fail --location --silent --show-error --retry 3 "$base/onnx/model.onnx" -o model.onnx
  for file in tokenizer.json config.json special_tokens_map.json tokenizer_config.json; do
    curl --fail --location --silent --show-error --retry 3 "$base/$file" -o "$file"
  done
fi
sha256sum --check <<< "${sums:1}"
