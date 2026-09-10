# TimeCapsuleLLM v3 GSU Tokenization And Training Runbook

Updated: 2026-09-10

This runbook is written for the GSU team. The important current fact is simple:
the corpus is cleaned and sharded, the tokenizer is trained, but the full
corpus still needs to be tokenized and packed before training.

## 1. Inputs

Required inputs on the cluster:

- Raw corpus shards from `haykgrigorian/TimeCapsuleLLM-World-English-1800-1875`
  or from the transferred local folder:
  `corpus_final/sharded_jsonl_v1/shards/*.jsonl.gz`
- Tokenizer folder from this bundle:
  `tokenizer_v3/`
- Training scripts from this bundle.

Each raw corpus record is JSONL with:

```json
{"id": "...", "text": "...", "metadata": {...}}
```

The raw dataset is not tokenized and does not contain fixed-length rows.

## 2. Recommended Cluster Layout

Use fast local scratch if available. Avoid tokenizing directly on slow network
storage if the cluster policy allows local scratch staging.

Example layout:

```text
/scratch/$USER/timecapsule_v3/
  raw_shards/
  tokenizer_v3/
  packed_arrow_v3_4096/
  outputs/
  logs/
  src/
```

## 3. Environment Setup

Recommended baseline:

- Python 3.10 or 3.11
- PyTorch with CUDA matching the cluster image
- `transformers`
- `datasets`
- `tokenizers`
- `accelerate`
- `safetensors`
- `tqdm`
- `numpy`
- Optional: `flash-attn` if supported by the CUDA image

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `flash-attn` fails to install, continue without it for the first smoke test.
The training script automatically falls back when FlashAttention is unavailable.

## 4. Tokenization And Packing

Run tokenization before training:

```bash
python preprocess_raw_jsonl_to_arrow_v3.py \
  --input_dir /scratch/$USER/timecapsule_v3/raw_shards \
  --output_dir /scratch/$USER/timecapsule_v3/packed_arrow_v3_4096 \
  --tokenizer_dir /scratch/$USER/timecapsule_v3/tokenizer_v3 \
  --sequence_length 4096 \
  --target_sequences_per_shard 8192
```

Expected result:

- Packed Arrow shard directories named `packed_arrow_00001`,
  `packed_arrow_00002`, etc.
- A `preprocessing_summary.json` file in the output directory.
- Rows with exactly two columns:
  - `input_ids`
  - `attention_mask`
- Every row should be exactly 4096 tokens.

Expected full-corpus scale:

- Approved input estimate: about 39.0B tokens.
- Expected packed rows: about 9.53 million rows at 4096 tokens each.
- Tokenization time depends heavily on CPU, storage, and gzip throughput.
- Reserve generous disk space. Arrow output with `input_ids` and
  `attention_mask` can be much larger than a compact raw uint16 token stream.
  A safe planning range is 200-350 GB until measured on the cluster.

## 5. Tokenization Smoke Test

Before full tokenization, run one small smoke test on one or two raw shards:

```bash
mkdir -p /scratch/$USER/timecapsule_v3/raw_smoke
cp /scratch/$USER/timecapsule_v3/raw_shards/shard_000*.jsonl.gz /scratch/$USER/timecapsule_v3/raw_smoke/

python preprocess_raw_jsonl_to_arrow_v3.py \
  --input_dir /scratch/$USER/timecapsule_v3/raw_smoke \
  --output_dir /scratch/$USER/timecapsule_v3/packed_arrow_smoke_4096 \
  --tokenizer_dir /scratch/$USER/timecapsule_v3/tokenizer_v3 \
  --sequence_length 4096 \
  --target_sequences_per_shard 256
```

Inspect the summary:

```bash
cat /scratch/$USER/timecapsule_v3/packed_arrow_smoke_4096/preprocessing_summary.json
```

The smoke output should contain nonzero packed sequences, no schema errors, and
no token IDs above 31,999.

## 6. Training Smoke Test

Once packed Arrow exists, run a short smoke test:

```bash
torchrun --nproc_per_node=3 train_model_cuda_v3.py \
  --config_module config_v3 \
  --dataset_dir /scratch/$USER/timecapsule_v3/packed_arrow_v3_4096 \
  --output_dir /scratch/$USER/timecapsule_v3/outputs \
  --run_prefix timecapsule_v3_smoke \
  --max_steps 100
```

The training script performs prechecks:

- Confirms CUDA is available.
- Rejects raw `.jsonl.gz` input by design.
- Loads Arrow shards from disk.
- Validates fixed 4096-token row length.
- Checks sampled token IDs are within vocab size.
- Builds the model from `config_v3.py`.

If this OOMs on 3 x L40 48 GB, first try keeping batch size at 1 and reducing
checkpoint pressure or increasing gradient accumulation:

```bash
--batch_size 1 --grad_accum 64
```

If it still OOMs, use `config_1b_2048.py` as a fallback diagnostic only. The
preferred v3 target remains the 4096-context roughly 2B model.

## 7. Full Training Run

For the full run:

```bash
torchrun --nproc_per_node=3 train_model_cuda_v3.py \
  --config_module config_v3 \
  --dataset_dir /scratch/$USER/timecapsule_v3/packed_arrow_v3_4096 \
  --output_dir /scratch/$USER/timecapsule_v3/outputs \
  --run_prefix timecapsule_v3_2b_4096 \
  --max_steps 100000
```

Current default training assumptions:

- Model: roughly 2B parameters
- Context length: 4096
- Per-device batch size: 1
- Gradient accumulation: 32
- Effective batch: 96 sequences across 3 GPUs
- Tokens per optimizer step: 393,216
- 100,000 steps: about 39.3B token-slots, roughly one corpus pass
- BF16 enabled
- Gradient checkpointing enabled
- Checkpoints every 2,000 steps

Resume:

```bash
torchrun --nproc_per_node=3 train_model_cuda_v3.py \
  --config_module config_v3 \
  --dataset_dir /scratch/$USER/timecapsule_v3/packed_arrow_v3_4096 \
  --output_dir /scratch/$USER/timecapsule_v3/outputs \
  --run_prefix timecapsule_v3_2b_4096 \
  --resume latest
```

## 8. Notes For GSU

- The included scripts are conservative and intentionally simple.
- They are suitable for a first smoke test and baseline run.
- If the cluster team prefers FSDP, DeepSpeed, Megatron, or a custom binary
  token format, use the tokenizer and corpus manifests here as the source of
  truth and adapt the packing stage accordingly.
- Question-answer fine-tuning is separate from pretraining. The v3 base model
  should be pretrained first; instruction/Q&A fine-tuning can follow.

## 9. Known Non-Blocking Documentation Issue

The sharded dataset README in the corpus package may contain stale per-source
breakdown lines from before the final July 2026 post-1875 cleanup. The current
source of truth is `corpus_final/corpus_manifest.json`, with approximately
159.0 GB represented source text and 39.031B estimated tokens.
