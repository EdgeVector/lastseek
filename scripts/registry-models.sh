#!/usr/bin/env bash
set -euo pipefail

model_dir="${LASTSEEK_MODEL_DIR:-$HOME/.lastseek/models/bge-small-en-v1.5}"
required_files=(
  model.onnx
  tokenizer.json
  config.json
  special_tokens_map.json
  tokenizer_config.json
)
missing_files=()

for file in "${required_files[@]}"; do
  if [ ! -s "$model_dir/$file" ]; then
    missing_files+=("$file")
  fi
done

if [ "${#missing_files[@]}" -ne 0 ]; then
  printf 'lastseek post-install: missing required model files under %s: %s\n' \
    "$model_dir" "${missing_files[*]}" >&2
  cat >&2 <<'EOF'
Fetch the model exactly as documented in the LastSeek README:

dir="${LASTSEEK_MODEL_DIR:-$HOME/.lastseek/models/bge-small-en-v1.5}"
mkdir -p "$dir"
base="https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main"
curl -L "$base/onnx/model.onnx" -o "$dir/model.onnx"
for f in tokenizer.json config.json special_tokens_map.json tokenizer_config.json; do
  curl -L "$base/$f" -o "$dir/$f"
done
EOF
  exit 1
fi

printf 'lastseek post-install: verified five bge-small-en-v1.5 model files under %s\n' \
  "$model_dir"
