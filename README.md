# TimeCapsule: Generative Hallucination as a Method for Historical Sensemaking

**Authors**: Hayk Grigorian and Hamed Yaghoobian (Muhlenberg College)

This repository contains the official code, dataset, and models for our paper **"TimeCapsule: Generative Hallucination as a Method for Historical Sensemaking"**, published in the **Proceedings of the 2026 ACM Conference on Creativity and Cognition (C&C '26)** ([https://dl.acm.org/doi/10.1145/3803784.3807554](https://dl.acm.org/doi/10.1145/3803784.3807554)). It includes the training pipeline for a 1.5B parameter language model trained on 90GB of historical English text (1800-1875).

## Current v3 / GSU Training Handoff

The current TimeCapsuleLLM v3 handoff materials for Georgia State University are
in [`gsu_training_prep/`](gsu_training_prep/). This folder is the starting point
for the next training run.

Current v3 status:

- The v3 corpus is cleaned, validated, sharded, and uploaded as raw text.
- The v3 tokenizer is trained and included in `gsu_training_prep/tokenizer_v3/`.
- The full corpus is **not tokenized yet** and **not packed yet**.
- GSU should run tokenization/packing first, then a smoke test, then training.
- Intended first full target: roughly 2B parameters, 4096-token context,
  3 x NVIDIA L40 48GB.

Current v3 dataset links:

- Full v3 corpus: [haykgrigorian/english-historical-corpus-1800-1875](https://huggingface.co/datasets/haykgrigorian/english-historical-corpus-1800-1875)
- 15GB sample: [haykgrigorian/english-historical-corpus-1800-1875-15GB-sample](https://huggingface.co/datasets/haykgrigorian/english-historical-corpus-1800-1875-15GB-sample)

Final v3 corpus snapshot:

- 164 compressed `.jsonl.gz` shards
- 6,755,308 records
- 159.0GB corrected source text represented
- 65.2GB compressed shard release
- Approximately 39.0B tokens with the approved v3 tokenizer

For GSU, begin with:

```bash
cd gsu_training_prep
sbatch submit_tokenize_v3.slurm
sbatch submit_smoke_train_v3.slurm
```

Only submit `submit_train_v3_2b.slurm` after tokenization and the smoke test
both pass.

## Replication & Paper Resources

All code, tokenizer artifacts, cleaned corpus documentation, and model checkpoints are available at [https://github.com/hamedyaghoobian/timecapsule.git](https://github.com/hamedyaghoobian/timecapsule.git). Random seeds, dataset versions, and training configurations are fixed to support replication.

### Artifact Locations:
- **Code**: [`src/`](src/)
- **Tokenizer Artifacts**: [`outputs/tokenizer/`](outputs/tokenizer/)
- **Cleaned Corpus Documentation**: [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md)
- **Local Model Checkpoints**: [`outputs/checkpoints/`](outputs/checkpoints/)

### Hosted Assets (Hugging Face):
- **Current v3 Raw Corpus**: [haykgrigorian/english-historical-corpus-1800-1875](https://huggingface.co/datasets/haykgrigorian/english-historical-corpus-1800-1875)
- **Current v3 15GB Sample**: [haykgrigorian/english-historical-corpus-1800-1875-15GB-sample](https://huggingface.co/datasets/haykgrigorian/english-historical-corpus-1800-1875-15GB-sample)
- **Legacy v2 Model (0.5 Epochs)**: [haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B](https://huggingface.co/haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B)
- **Legacy v2 Tokenized Dataset**: [postgrammar/london-llm-1800](https://huggingface.co/datasets/postgrammar/london-llm-1800)

## RunPod Setup Guide

To train this model efficiently, we recommend using **RunPod**. Follow these steps to set up your environment.

### 1. Select a GPU Instance
- **Recommended**: **1x A100 80GB** (Best performance/price balance)
- **Alternative**: 1x H100 80GB (Faster, more expensive)
- **Template**: Select `PyTorch 2.1` (or newer) with CUDA 12.1+.

### 2. Configure Storage (CRITICAL)
You effectively need **200GB+** of storage to hold the dataset (120GB) and checkpoints.
- **Container Disk**: 50 GB
- **Volume Disk**: **200 GB** (Mount path: `/workspace`)
- *Note: If you don't add a Volume Disk, you will run out of space immediately.*

### 3. Setup Environment
Once your pod is running, open the **Web Terminal** or SSH in.

```bash
# 1. Clone this repository
cd /workspace
git clone https://github.com/hamedyaghoobian/timecapsule.git
cd timecapsule

# 2. Install dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Data Download
For v3, use the GSU handoff folder:

```bash
cd gsu_training_prep
./download_dataset_from_hf.sh /scratch/$USER/timecapsule_v3/raw_shards
```

The full v3 corpus is hosted on Hugging Face at
[haykgrigorian/english-historical-corpus-1800-1875](https://huggingface.co/datasets/haykgrigorian/english-historical-corpus-1800-1875).

The 15GB sample is hosted at
[haykgrigorian/english-historical-corpus-1800-1875-15GB-sample](https://huggingface.co/datasets/haykgrigorian/english-historical-corpus-1800-1875-15GB-sample).

The older `postgrammar/london-llm-1800` dataset is a legacy v2 London-only
asset and is not the v3 corpus.

```bash
# Run this script to download ~120GB of data
# Ensure you have a stable internet connection in the pod
python src/download_data.py
```
This will create a `data/` directory with:
- `data/dataset/`: The tokenized Arrow dataset
- `data/tokenizer/`: The custom BPE tokenizer

## Training
To start training the 1.5B parameter model:

```bash
python src/05_train_model_cuda.py --data_dir data --output_dir outputs
```

### Monitoring
- **Checkpoints**: Saved to `outputs/checkpoints`
- **Logs**: Saved to `outputs/logs` (view with `tail -f outputs/logs/training.log`)
- **Loss**: Watch the loss curve. It should decrease from ~10.0 to <3.0 over time.

## Advanced Info
- **Model Config**: 1.5B params, 2048 ctx length, Flash Attention 2 enabled.
- **Dataset**: ~240 shards of tokenized historical text.
- **Tokenizer**: Custom BPE trained on 100M+ tokens of the corpus.

## License & Citation

This project is licensed under the [MIT License](LICENSE).

If you use this work in your research, please cite our paper. The publication is available in the ACM Digital Library: [https://dl.acm.org/doi/10.1145/3803784.3807554](https://dl.acm.org/doi/10.1145/3803784.3807554)
