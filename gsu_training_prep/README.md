# GSU Training Prep Bundle

Updated: 2026-09-10

This folder is the current handoff bundle for Georgia State University to
tokenize and train TimeCapsuleLLM v3. It supersedes the older June 2026 prep
notes.

## Current Status

- The v3 corpus is cleaned, validated, sharded, and ready for cluster transfer.
- The v3 tokenizer is already trained and included here.
- The corpus is still raw compressed JSONL text, not tokenized and not packed.
- GSU should run tokenization/packing before model training.
- The intended first full model target is a roughly 2B parameter causal LM with
  4096-token context.

## Final Corpus Snapshot

- Dataset: TimeCapsuleLLM World English 1800-1875
- Hugging Face dataset repo: `haykgrigorian/TimeCapsuleLLM-World-English-1800-1875`
- Local source package: `D:\TimeCapsuleLLM_v3\corpus_final`
- Local sharded corpus: `D:\TimeCapsuleLLM_v3\corpus_final\sharded_jsonl_v1\shards`
- Shards: 164 compressed `.jsonl.gz` files
- Records: 6,755,308
- Corrected source text represented: 159,036,129,719 bytes, about 159.0 GB
- Compressed shard size: 65,161,674,284 bytes, about 65.2 GB
- Approved token estimate with tokenizer v3: 39,031,191,729 tokens, about 39.0B
- Date range: 1800-1875, except Chronicling America restricted to 1800-1859

The final manifest is stored in the main corpus package at
`corpus_final/corpus_manifest.json`. A copy should be kept with any cluster
transfer.

## Tokenizer Snapshot

- Path in this bundle: `tokenizer_v3/`
- Algorithm: HuggingFace Tokenizers ByteLevel BPE
- Vocabulary size: 32,000
- Minimum frequency during training: 2
- Special token IDs:
  - `<s>`: 0
  - `</s>`: 1
  - `<unk>`: 2
  - `<pad>`: 3
  - `<|year|>`: 4
  - `<|title|>`: 5

## Hardware Target

The expected GSU target discussed for the first run is:

- 3 x NVIDIA L40
- 48 GB VRAM per GPU
- Single node preferred
- BF16 training
- 4096-token context

With 3 GPUs, per-device batch size 1, gradient accumulation 32, and sequence
length 4096, each optimizer step covers about 393,216 tokens. A 100,000 step
run is therefore approximately one pass over the 39.0B-token corpus.

## Bundle Contents

- `config_v3.py`: current roughly 2B / 4096-context training config.
- `config_1b_2048.py`: preserved smaller legacy fallback config.
- `preprocess_raw_jsonl_to_arrow_v3.py`: raw `.jsonl.gz` to packed Arrow.
- `train_model_cuda_v3.py`: trains from already-packed Arrow shards.
- `requirements.txt`: Python dependencies.
- `TRAINING_README.md`: detailed cluster runbook.
- `submit_tokenize_v3.slurm`: SLURM template for tokenization/packing.
- `submit_smoke_train_v3.slurm`: SLURM template for a short smoke test.
- `submit_train_v3_2b.slurm`: SLURM template for the full 2B run.
- `tokenizer_v3/`: frozen tokenizer reference.

## Critical Order Of Operations

1. Transfer or download the raw sharded corpus.
2. Install dependencies in a clean environment.
3. Run tokenization/packing to fixed 4096-token Arrow shards.
4. Confirm `preprocessing_summary.json` and run a small decode/schema check.
5. Run a 100-step training smoke test.
6. If memory and throughput are acceptable, launch the full training run.

Do not point `train_model_cuda_v3.py` directly at raw `.jsonl.gz` shards. It
expects packed Arrow shards produced by `preprocess_raw_jsonl_to_arrow_v3.py`.
