#!/usr/bin/env python3
"""
Training script for v3 

Still undecided for v3:
- exact model architecture / parameter target
- max_steps vs num_train_epochs
- batch size / grad accumulation
- learning-rate schedule
- dataset location and final run naming convention

Conifg file has not been made yet
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

# Keep the training flow fully offline unless you explicitly change it.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments


DEFAULT_CONFIG_MODULE = os.environ.get("TRAIN_CONFIG_MODULE", "config_v3")
DEFAULT_DATASET_DIR = os.environ.get(
    "TRAIN_DATASET_DIR",
    "/workspace/hf/datasets/postgrammar___london-llm-1800/train",
)
DEFAULT_OUTPUT_DIR = os.environ.get("TRAIN_OUTPUT_DIR", "/workspace/outputs")
DEFAULT_RUN_PREFIX = os.environ.get("TRAIN_RUN_PREFIX", "v3_h100")

# Placeholder overrides for the future v3 model.
# Leave commented until the architecture is finalized.
V3_MODEL_OVERRIDES = {
    # "hidden_size": ...,
    # "intermediate_size": ...,
    # "num_hidden_layers": ...,
    # "num_attention_heads": ...,
    # "num_key_value_heads": ...,
    # "max_position_embeddings": ...,
}

# Placeholder overrides for the future v3 training recipe.
# Use either max_steps or num_train_epochs once the plan is finalized.
V3_TRAINING_OVERRIDES = {
    # "max_steps": ...,
    # "num_train_epochs": ...,
    # "per_device_train_batch_size": ...,
    # "gradient_accumulation_steps": ...,
    # "learning_rate": ...,
    # "warmup_steps": ...,
    # "save_steps": ...,
    # "save_total_limit": ...,
}

# Optional hard guard once v3's expected size is known.
# Example: EXPECTED_PARAM_BOUNDS = (2_900_000_000, 3_100_000_000)
EXPECTED_PARAM_BOUNDS = None


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
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not import config module '{module_name}'. "
            "Make sure it is in the same folder or on PYTHONPATH.\n"
            f"Original error:\n{exc}"
        ) from exc

    if not hasattr(module, "MODEL_CONFIG") or not hasattr(module, "TRAINING_CONFIG"):
        raise RuntimeError(
            f"Config module '{module_name}' must define MODEL_CONFIG and TRAINING_CONFIG."
        )

    return module.MODEL_CONFIG, module.TRAINING_CONFIG


def infer_safe_vocab_size(ds, sample_n=2000, round_to=64, min_vocab=32000):
    """Sample the first N rows and infer a safe vocab ceiling."""
    n = min(sample_n, len(ds))
    max_token_id = 0
    for i in range(n):
        input_ids = ds[i]["input_ids"]
        if input_ids:
            max_token_id = max(max_token_id, max(input_ids))

    safe_vocab = max(min_vocab, max_token_id + 1)
    safe_vocab = int(((safe_vocab + round_to - 1) // round_to) * round_to)
    return max_token_id, safe_vocab


def build_llama_config(base_model_config: dict, vocab_size: int, use_flash: bool) -> LlamaConfig:
    cfg = dict(base_model_config)
    cfg.update(V3_MODEL_OVERRIDES)
    cfg["vocab_size"] = vocab_size

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


def run_dirs_with_checkpoints(checkpoint_root: Path, run_prefix: str):
    run_dirs = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and any(child.is_dir() for child in path.glob("checkpoint-*"))
    ]
    prefixed = [path for path in run_dirs if path.name.startswith(run_prefix)]
    return prefixed or run_dirs


def validate_checkpoint(ckpt: Path):
    required = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
    ]
    missing = [name for name in required if not (ckpt / name).exists()]
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing required files: {missing}\n"
            f"Checkpoint: {ckpt}\n"
            "If optimizer/scheduler are missing, LR will reset."
        )


def resolve_resume_checkpoint(resume_arg: str, checkpoint_root: Path, run_prefix: str) -> Path:
    """
    Accepts:
    - "latest" (latest run + latest checkpoint)
    - a checkpoint dir (.../checkpoint-60000)
    - a run dir containing checkpoint-*
    - a run name under checkpoint_root
    """
    if resume_arg is None:
        return None

    resume_value = resume_arg.strip()

    if resume_value.lower() == "latest":
        runs = run_dirs_with_checkpoints(checkpoint_root, run_prefix)
        if not runs:
            raise RuntimeError(
                f'No run directories with checkpoints found under "{checkpoint_root}"'
            )
        runs.sort(key=lambda path: path.stat().st_mtime)
        run_dir = runs[-1]
        ckpts = [path for path in run_dir.glob("checkpoint-*") if path.is_dir()]
        ckpts.sort(key=checkpoint_step)
        ckpt = ckpts[-1]
        validate_checkpoint(ckpt)
        return ckpt

    path = Path(resume_value)
    if not path.exists() and (checkpoint_root / resume_value).exists():
        path = checkpoint_root / resume_value

    if not path.exists():
        raise FileNotFoundError(f"--resume path does not exist: {path}")

    if path.is_dir() and path.name.startswith("checkpoint-"):
        validate_checkpoint(path)
        return path

    if path.is_dir():
        ckpts = [child for child in path.glob("checkpoint-*") if child.is_dir()]
        if not ckpts:
            raise RuntimeError(f'No checkpoint-* found in resume directory: "{path}"')
        ckpts.sort(key=checkpoint_step)
        ckpt = ckpts[-1]
        validate_checkpoint(ckpt)
        return ckpt

    raise RuntimeError(f"Unsupported --resume value: {resume_arg} (resolved to {path})")


def read_trainer_state(ckpt: Path) -> dict:
    with open(ckpt / "trainer_state.json", "r") as file:
        return json.load(file)


def summarize_resume_state(state: dict):
    global_step = state.get("global_step")
    epoch = state.get("epoch")
    max_steps = state.get("max_steps")
    last_lr = None

    for item in reversed(state.get("log_history", [])):
        if isinstance(item, dict) and "learning_rate" in item:
            last_lr = item["learning_rate"]
            break

    print("Resume state from trainer_state.json")
    print(f"  global_step: {global_step}")
    print(f"  epoch:       {epoch}")
    print(f"  max_steps:   {max_steps}")
    if last_lr is not None:
        print(f"  last_lr:     {last_lr}")


def maybe_validate_param_count(total_params: int, expected_min: int = None, expected_max: int = None):
    if expected_min is None or expected_max is None:
        return

    if not (expected_min <= total_params <= expected_max):
        raise RuntimeError(
            f"ABORT: model is {total_params / 1e9:.2f}B params. "
            f"Expected range: {expected_min / 1e9:.2f}B - {expected_max / 1e9:.2f}B."
        )


def build_training_config(base_training_config: dict, args) -> dict:
    cfg = dict(base_training_config)
    cfg.update(V3_TRAINING_OVERRIDES)

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
    parser.add_argument(
        "--dataset_glob",
        type=str,
        default="*.arrow",
        help="Glob used to discover local Arrow shards inside dataset_dir.",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_prefix", type=str, default=DEFAULT_RUN_PREFIX)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            'Resume from checkpoint/run. Examples: ".../checkpoint-60000", '
            '".../v3_h100_YYYYMMDD_HHMMSS", or "latest".'
        ),
    )
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--sample_vocab_n", type=int, default=2000)
    parser.add_argument("--report_to", type=str, default="none", help="none|tensorboard|wandb...")
    parser.add_argument("--expected_params_min", type=int, default=None)
    parser.add_argument("--expected_params_max", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    model_config, training_config = load_training_config(args.config_module)

    output_dir = Path(args.output_dir)
    checkpoint_root = output_dir / "checkpoints"
    log_root = output_dir / "logs"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V3 TRAINING INIT (LOCAL ARROW SHARDS, OFFLINE)")
    print("=" * 60)
    print(f"config_module: {args.config_module}")
    print(f"run_prefix:    {args.run_prefix}")
    print_gpu_utilization()

    resume_ckpt = None
    resume_state = None
    if args.resume:
        resume_ckpt = resolve_resume_checkpoint(args.resume, checkpoint_root, args.run_prefix)
        resume_state = read_trainer_state(resume_ckpt)
        print("\nResume enabled")
        print(f"  resuming from: {resume_ckpt}")
        summarize_resume_state(resume_state)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir does not exist: {dataset_dir}")

    train_shards = sorted(dataset_dir.glob(args.dataset_glob))
    if not train_shards:
        raise RuntimeError(
            f"No shards matching {args.dataset_glob!r} found in {dataset_dir}.\n"
            "Update --dataset_dir or --dataset_glob for the final v3 dataset layout."
        )

    print("\nLoading local Arrow shards")
    print(f"  dataset_dir:  {dataset_dir}")
    print(f"  shard_count:  {len(train_shards)}")
    print(f"  first_shard:  {train_shards[0].name}")

    ds = load_dataset("arrow", data_files={"train": [str(path) for path in train_shards]})
    train_ds = ds["train"]
    print(f"Loaded train split with {len(train_ds):,} rows")
    print(f"Features: {list(train_ds.features)}")

    print("\nChecking token id range")
    max_id, vocab_size = infer_safe_vocab_size(train_ds, sample_n=args.sample_vocab_n)
    print(f"  max_token_id: {max_id}")
    print(f"  vocab_size:   {vocab_size}")

    use_flash = flash_attn_available()
    print("\nAttention backend")
    print(f"  flash_attention_2: {'enabled' if use_flash else 'not available'}")

    print("\nBuilding model")
    config = build_llama_config(model_config, vocab_size=vocab_size, use_flash=use_flash)
    model = LlamaForCausalLM(config)
    model = model.to(dtype=torch.bfloat16, device="cuda")

    total_params = sum(param.numel() for param in model.parameters())
    print(f"  params: {total_params / 1e9:.2f}B")
    print_gpu_utilization()

    expected_min = args.expected_params_min
    expected_max = args.expected_params_max
    if expected_min is None and expected_max is None and EXPECTED_PARAM_BOUNDS:
        expected_min, expected_max = EXPECTED_PARAM_BOUNDS
    maybe_validate_param_count(total_params, expected_min, expected_max)

    if resume_ckpt:
        run_dir = resume_ckpt.parent
        run_name = run_dir.name
        print(f"\nContinuing in-place in run directory: {run_dir}")
    else:
        run_name = f"{args.run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = checkpoint_root / run_name

    resolved_training_config = build_training_config(training_config, args)

    if resume_state is not None:
        resumed_step = int(resume_state.get("global_step", 0))
        max_steps = resolved_training_config.get("max_steps")
        if max_steps is not None and int(max_steps) <= resumed_step:
            raise RuntimeError(
                f"max_steps ({max_steps}) must be greater than resumed global_step "
                f"({resumed_step}). Increase --max_steps or switch to --num_train_epochs."
            )

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_name,
        logging_dir=str(log_root / run_name),
        **resolved_training_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    effective_batch = (
        resolved_training_config.get("per_device_train_batch_size", 1)
        * resolved_training_config.get("gradient_accumulation_steps", 1)
        * max(1, torch.cuda.device_count())
    )

    print("\nStarting training")
    print(f"  effective_batch: {effective_batch}")
    print(f"  output_dir:      {run_dir}")
    print(f"  max_steps:       {resolved_training_config.get('max_steps')}")
    print(f"  num_epochs:      {resolved_training_config.get('num_train_epochs')}")
    print(f"  resume_from:     {resume_ckpt or '(none)'}")

    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)

    print("\nSaving final model")
    trainer.save_model(str(run_dir / "final"))

    if _HAS_PYNVML:
        import pynvml

        pynvml.nvmlShutdown()

    print(f"\nComplete: {run_dir / 'final'}")


if __name__ == "__main__":
    main()
