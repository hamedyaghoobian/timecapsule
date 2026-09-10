#!/usr/bin/env python3
"""
Future training entrypoint for TimeCapsuleLLM v3.

Important:
- this script expects already-tokenized, fixed-length Arrow shards
- it is not meant to consume raw JSONL.GZ text directly
- it is kept conservative for future smoke tests and cluster runs
"""

import argparse
import importlib
import json
import os
from datetime import datetime
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from datasets import concatenate_datasets, load_from_disk
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments


DEFAULT_CONFIG_MODULE = os.environ.get("TRAIN_CONFIG_MODULE", "config_v3")
DEFAULT_DATASET_DIR = os.environ.get("TRAIN_DATASET_DIR", "/workspace/packed_arrow")
DEFAULT_OUTPUT_DIR = os.environ.get("TRAIN_OUTPUT_DIR", "/workspace/outputs")
DEFAULT_RUN_PREFIX = os.environ.get("TRAIN_RUN_PREFIX", "timecapsule_v3")
EXPECTED_SEQUENCE_LENGTH = 4096

try:
    import pynvml  # noqa: F401

    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False


def print_gpu_utilization():
    if not _HAS_PYNVML:
        print("GPU memory: pynvml not installed")
        return

    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = info.used // (1024**2)
    total = info.total // (1024**2)
    print(f"GPU memory: {used}/{total} MB ({used / total * 100:.1f}%)")


def flash_attn_available():
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def load_training_config(module_name: str):
    module = importlib.import_module(module_name)
    return module


def build_llama_config(module, use_flash: bool) -> LlamaConfig:
    cfg = dict(module.MODEL_CONFIG)
    if use_flash:
        cfg["attn_implementation"] = "flash_attention_2"
    else:
        cfg.pop("attn_implementation", None)
    return LlamaConfig(**cfg)


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except Exception:
        return -1


def resolve_resume_checkpoint(resume_arg: str, checkpoint_root: Path, run_prefix: str) -> Path | None:
    if not resume_arg:
        return None

    if resume_arg.lower() == "latest":
        candidates = [p for p in checkpoint_root.iterdir() if p.is_dir() and p.name.startswith(run_prefix)]
        candidates.sort(key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError(f"No prior runs found under {checkpoint_root}")
        run_dir = candidates[-1]
        ckpts = sorted([p for p in run_dir.glob("checkpoint-*") if p.is_dir()], key=checkpoint_step)
        if not ckpts:
            raise RuntimeError(f"No checkpoint-* folders found in {run_dir}")
        return ckpts[-1]

    path = Path(resume_arg)
    if path.is_dir() and path.name.startswith("checkpoint-"):
        return path
    if path.is_dir():
        ckpts = sorted([p for p in path.glob("checkpoint-*") if p.is_dir()], key=checkpoint_step)
        if ckpts:
            return ckpts[-1]
    raise RuntimeError(f"Unsupported --resume value: {resume_arg}")


def load_packed_arrow_dataset(dataset_dir: Path):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir does not exist: {dataset_dir}")

    if list(dataset_dir.glob("*.jsonl.gz")):
        raise RuntimeError(
            "Dataset directory appears to contain raw JSONL.GZ shards.\n"
            "This training script requires pre-tokenized, fixed-length Arrow shards.\n"
            "Run preprocess_raw_jsonl_to_arrow_v3.py first."
        )

    shard_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and (p / "dataset_info.json").exists()])
    if not shard_dirs:
        raise RuntimeError(
            f"No Arrow shard directories with dataset_info.json found in {dataset_dir}.\n"
            "Expected output from preprocess_raw_jsonl_to_arrow_v3.py."
        )

    datasets = [load_from_disk(str(path)) for path in shard_dirs]
    train_ds = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    return train_ds, shard_dirs


def validate_dataset_schema(train_ds, expected_length: int, vocab_size: int):
    features = list(train_ds.features)
    required = {"input_ids", "attention_mask"}
    missing = required.difference(features)
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {sorted(missing)}")

    sample_count = min(64, len(train_ds))
    if sample_count == 0:
        raise RuntimeError("Dataset contains zero rows")

    max_token_id = -1
    bad_lengths = 0
    for i in range(sample_count):
        row = train_ds[i]
        ids = row["input_ids"]
        mask = row["attention_mask"]
        if len(ids) != expected_length or len(mask) != expected_length:
            bad_lengths += 1
        if ids:
            max_token_id = max(max_token_id, max(ids))

    if bad_lengths:
        raise RuntimeError(
            f"Found {bad_lengths} sampled rows with sequence length != {expected_length}. "
            "Packed Arrow shards must contain fixed-length rows."
        )

    if max_token_id >= vocab_size:
        raise RuntimeError(
            f"Observed token id {max_token_id}, which exceeds tokenizer vocab size {vocab_size}."
        )

    return max_token_id


def build_training_config(module, args) -> dict:
    cfg = dict(module.TRAINING_CONFIG)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
        cfg.pop("num_train_epochs", None)
    if args.num_train_epochs is not None:
        cfg["num_train_epochs"] = args.num_train_epochs
        cfg.pop("max_steps", None)
    if args.batch_size is not None:
        cfg["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum is not None:
        cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.warmup_steps is not None:
        cfg["warmup_steps"] = args.warmup_steps
    if args.save_steps is not None:
        cfg["save_steps"] = args.save_steps
    if args.save_total_limit is not None:
        cfg["save_total_limit"] = args.save_total_limit
    cfg["report_to"] = args.report_to
    cfg["bf16"] = True
    cfg["tf32"] = True
    cfg["remove_unused_columns"] = False
    return cfg


def data_collator(features):
    input_ids = torch.as_tensor([f["input_ids"] for f in features], dtype=torch.long)
    attention_mask = torch.as_tensor([f["attention_mask"] for f in features], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.clone(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_module", type=str, default=DEFAULT_CONFIG_MODULE)
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_prefix", type=str, default=DEFAULT_RUN_PREFIX)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--report_to", type=str, default="none")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    module = load_training_config(args.config_module)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    checkpoint_root = output_dir / "checkpoints"
    log_root = output_dir / "logs"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TIMECAPSULELLM V3 TRAINING PRECHECK")
    print("=" * 60)
    print(f"config_module: {args.config_module}")
    print(f"dataset_dir:   {dataset_dir}")
    print_gpu_utilization()

    train_ds, shard_dirs = load_packed_arrow_dataset(dataset_dir)
    print(f"Loaded {len(shard_dirs)} packed Arrow shards")
    print(f"Total rows: {len(train_ds):,}")
    print(f"Features: {list(train_ds.features)}")

    max_token_id = validate_dataset_schema(
        train_ds,
        expected_length=module.CONTEXT_LENGTH,
        vocab_size=module.VOCAB_SIZE,
    )
    print(f"Validated sample rows at sequence length {module.CONTEXT_LENGTH}")
    print(f"Max sampled token id: {max_token_id}")

    use_flash = flash_attn_available()
    config = build_llama_config(module, use_flash=use_flash)
    model = LlamaForCausalLM(config)
    model = model.to(dtype=torch.bfloat16, device="cuda")

    total_params = sum(param.numel() for param in model.parameters())
    print(f"Model params: {total_params / 1e9:.2f}B")

    resume_ckpt = resolve_resume_checkpoint(args.resume, checkpoint_root, args.run_prefix) if args.resume else None
    if resume_ckpt:
        print(f"Resuming from: {resume_ckpt}")
        run_dir = resume_ckpt.parent
        run_name = run_dir.name
    else:
        run_name = f"{args.run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = checkpoint_root / run_name

    training_config = build_training_config(module, args)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_name,
        logging_dir=str(log_root / run_name),
        **training_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    effective_batch = (
        training_config.get("per_device_train_batch_size", 1)
        * training_config.get("gradient_accumulation_steps", 1)
        * max(1, torch.cuda.device_count())
    )
    print(f"Effective batch size: {effective_batch}")
    print(
        "Note: this script is for future packed-Arrow smoke tests and full runs only. "
        "Do not point it at raw JSONL.GZ shards."
    )

    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
    trainer.save_model(str(run_dir / "final"))

    if _HAS_PYNVML:
        import pynvml

        pynvml.nvmlShutdown()

    print(f"Saved final model to {run_dir / 'final'}")


if __name__ == "__main__":
    main()
