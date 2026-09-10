#!/bin/bash
set -euo pipefail

# Downloads the raw sharded TimeCapsuleLLM v3 corpus from Hugging Face.
# Requires: pip install huggingface_hub
#
# Usage:
#   HF_TOKEN=... ./download_dataset_from_hf.sh /scratch/$USER/timecapsule_v3/raw_shards
#
# Public repos may not need HF_TOKEN, but setting it is harmless.

DEST="${1:-/scratch/$USER/timecapsule_v3/raw_shards}"
REPO_ID="${REPO_ID:-haykgrigorian/TimeCapsuleLLM-World-English-1800-1875}"
TMP_DIR="${DEST}.hf_download"

mkdir -p "$DEST" "$TMP_DIR"

huggingface-cli download "$REPO_ID" \
  --repo-type dataset \
  --include "shards/*.jsonl.gz" \
  --local-dir "$TMP_DIR"

if [ -d "$TMP_DIR/shards" ]; then
  cp -n "$TMP_DIR"/shards/*.jsonl.gz "$DEST"/
else
  cp -n "$TMP_DIR"/*.jsonl.gz "$DEST"/
fi

echo "Downloaded raw shard files to: $DEST"
echo "Shard count:"
find "$DEST" -maxdepth 1 -name '*.jsonl.gz' | wc -l
