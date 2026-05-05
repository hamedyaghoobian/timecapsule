# TimeCapsule: Generative Hallucination as a Method for Historical Sensemaking

**Authors**: Hayk Grigorian and Hamed Yaghoobian (Muhlenberg College)

This repository contains the official code, dataset, and models for our paper **"TimeCapsule: Generative Hallucination as a Method for Historical Sensemaking"**, prepared for the **ACM Creativity and Cognition (C&C) 2026 Conference** in the UK ([https://cc.acm.org/2026/](https://cc.acm.org/2026/)). It includes the training pipeline for a 1.5B parameter language model trained on 90GB of historical English text (1800-1875).

## Replication & Paper Resources

All code, tokenizer artifacts, cleaned corpus documentation, and model checkpoints are available at [https://github.com/hamedyaghoobian/timecapsule.git](https://github.com/hamedyaghoobian/timecapsule.git). Random seeds, dataset versions, and training configurations are fixed to support replication.

### Artifact Locations:
- **Code**: [`src/`](src/)
- **Tokenizer Artifacts**: [`outputs/tokenizer/`](outputs/tokenizer/)
- **Cleaned Corpus Documentation**: [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md)
- **Local Model Checkpoints**: [`outputs/checkpoints/`](outputs/checkpoints/)

### Hosted Assets (Hugging Face):
- **Trained Model (0.5 Epochs)**: [haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B](https://huggingface.co/haykgrigorian/TimeCapsuleLLM-v2-llama-1.2B)
- **Tokenized Dataset**: [postgrammar/london-llm-1800](https://huggingface.co/datasets/postgrammar/london-llm-1800)

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
The dataset and tokenizer are hosted on Hugging Face at [postgrammar/london-llm-1800](https://huggingface.co/datasets/postgrammar/london-llm-1800). We have provided a script to download them automatically.

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
