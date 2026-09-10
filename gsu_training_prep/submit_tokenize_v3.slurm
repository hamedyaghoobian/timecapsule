#!/bin/bash
#SBATCH --job-name=tcv3_tokenize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/tokenize_v3_%j.out
#SBATCH --error=logs/tokenize_v3_%j.err

set -euo pipefail

# Edit these paths for the GSU cluster.
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/$USER/timecapsule_v3}"
SRC_DIR="${SRC_DIR:-$PROJECT_ROOT/src}"
RAW_SHARDS="${RAW_SHARDS:-$PROJECT_ROOT/raw_shards}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$PROJECT_ROOT/tokenizer_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/packed_arrow_v3_4096}"

mkdir -p "$PROJECT_ROOT/logs" "$OUTPUT_DIR"
cd "$SRC_DIR"

python preprocess_raw_jsonl_to_arrow_v3.py \
  --input_dir "$RAW_SHARDS" \
  --output_dir "$OUTPUT_DIR" \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --sequence_length 4096 \
  --target_sequences_per_shard 8192
