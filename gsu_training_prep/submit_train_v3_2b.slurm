#!/bin/bash
#SBATCH --job-name=tcv3_2b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/train_v3_2b_%j.out
#SBATCH --error=logs/train_v3_2b_%j.err

set -euo pipefail

# Edit these paths for the GSU cluster.
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/$USER/timecapsule_v3}"
SRC_DIR="${SRC_DIR:-$PROJECT_ROOT/src}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/packed_arrow_v3_4096}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs}"

mkdir -p "$PROJECT_ROOT/logs" "$OUTPUT_DIR"
cd "$SRC_DIR"

torchrun --nproc_per_node=3 train_model_cuda_v3.py \
  --config_module config_v3 \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --run_prefix timecapsule_v3_2b_4096 \
  --max_steps 100000
