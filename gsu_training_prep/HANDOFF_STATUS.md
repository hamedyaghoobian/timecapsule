# TimeCapsuleLLM v3 Handoff Status

Updated: 2026-09-10

## Ready

- Cleaned final corpus exists and is packaged as raw `.jsonl.gz` shards.
- v3 tokenizer is trained and frozen.
- 5B and 10B representative sample datasets exist for smaller experiments.
- A 500M evaluation model was trained successfully from a 5B-token sample.
- Full pretraining scripts and configs are present in this bundle.

## Not Yet Done

- The full 39.0B-token corpus has not been tokenized.
- The corpus has not been packed into fixed 4096-token Arrow rows.
- No full 2B v3 pretraining run has started.

## Immediate Next Step For GSU

Tokenize and pack the raw `.jsonl.gz` shards:

```bash
sbatch submit_tokenize_v3.slurm
```

Then run:

```bash
sbatch submit_smoke_train_v3.slurm
```

Only after the smoke test passes should the full training job be submitted.

## Current Source Of Truth

- Corpus manifest: `corpus_manifest.json`
- Tokenizer: `tokenizer_v3/`
- Training config: `config_v3.py`
- Tokenization script: `preprocess_raw_jsonl_to_arrow_v3.py`
- Training script: `train_model_cuda_v3.py`
