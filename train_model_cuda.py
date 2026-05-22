#!/usr/bin/env python3
"""
H100 SXM Optimized LLaMA Training Script (LOCAL-DISK, NO DOWNLOADS)
==================================================================
- This was used to train v2, DO NOT USE AS IS TO TRAIN V3
Fixes:
- Uses the already-prepared Arrow shards on disk (NO hub, NO downloads)
- Single source of truth: imports MODEL_CONFIG / TRAINING_CONFIG from config.py
- Hard abort if params drift outside ~1.2B range
- Forces bf16 dtype explicitly for FlashAttention2 stability
- Disables tensorboard by default (report_to="none")

RESUME FIXES (NEW):
- Robust resume: accepts checkpoint dir OR run dir OR "latest"
- Hard validation for optimizer/scheduler/trainer_state
- Continues in-place (same run folder) when resuming (no accidental fresh run)
- Auto-bumps max_steps to target ~0.5 epochs total when resuming (unless you override)
- Prints resumed global_step + last LR so you can confirm it truly resumed

TARGET:
- Train to ~0.5 epochs total. Based on your current run (60k steps ≈ 0.165 epoch),
  0.5 epoch ≈ 181,800 steps → we set default TARGET_MAX_STEPS = 182,000.

FINAL ADJUSTMENTS (NEW):
- Your downloaded shards are named: train/data-00000-of-00239.arrow
  so we now glob "*.arrow" instead of "london-llm-1800-train-*.arrow".
- Slightly stronger shard detection + clearer error messages.
"""

import os
import argparse
import json
from datetime import datetime
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# HARD OFFLINE LOCK (prevents surprise downloads)
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

# -----------------------------------------------------------
# Import config from same folder
# -----------------------------------------------------------
try:
    from config import MODEL_CONFIG, TRAINING_CONFIG
except Exception as e:
    raise RuntimeError(
        "Could not import config.py. Make sure train_model_cuda.py and config.py "
        "are in the same folder (/workspace/src/train). Original error:\n"
        f"{e}"
    )

# Optional GPU monitoring (won't crash if pynvml not installed)
try:
    import pynvml  # noqa: F401
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False


def print_gpu_utilization():
    if not _HAS_PYNVML:
        print("📊 GPU Memory: (pynvml not installed)")
        return
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    used = info.used // (1024**2)
    total = info.total // (1024**2)
    print(f"📊 GPU Memory: {used}/{total} MB ({used/total*100:.1f}%)")


def flash_attn_available():
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def infer_safe_vocab_size(ds, sample_n=2000, round_to=64, min_vocab=32000):
    """Sample first N rows and find max token id."""
    n = min(sample_n, len(ds))
    mx = 0
    for i in range(n):
        ids = ds[i]["input_ids"]
        if ids:
            v = max(ids)
            if v > mx:
                mx = v
    safe = max(min_vocab, mx + 1)
    safe = int(((safe + round_to - 1) // round_to) * round_to)
    return mx, safe


def build_llama_config(vocab_size: int, use_flash: bool) -> LlamaConfig:
    cfg = dict(MODEL_CONFIG)  # copy
    cfg["vocab_size"] = vocab_size

    if use_flash:
        cfg["attn_implementation"] = "flash_attention_2"
    else:
        cfg.pop("attn_implementation", None)

    return LlamaConfig(**cfg)


def _checkpoint_step(p: Path) -> int:
    # checkpoint-60000 -> 60000
    try:
        return int(p.name.split("-")[-1])
    except Exception:
        return -1


def resolve_resume_checkpoint(resume_arg: str, checkpoint_root: Path) -> Path:
    """
    Accepts:
      - "latest" (find latest run + latest checkpoint)
      - a checkpoint dir (.../checkpoint-60000)
      - a run dir (.../london_h100_YYYYMMDD_HHMMSS) containing checkpoint-*
      - a run name (london_h100_...) under checkpoint_root
    Returns a validated checkpoint path.
    """
    if resume_arg is None:
        return None

    r = resume_arg.strip()

    # "latest" => find latest run folder under checkpoint_root, then latest checkpoint within it
    if r.lower() == "latest":
        runs = [p for p in checkpoint_root.glob("london_h100_*") if p.is_dir()]
        if not runs:
            raise RuntimeError(f'No run dirs found under "{checkpoint_root}" for --resume latest')
        # sort by folder mtime for "latest run"
        runs.sort(key=lambda p: p.stat().st_mtime)
        run_dir = runs[-1]
        ckpts = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
        if not ckpts:
            raise RuntimeError(f'No checkpoint-* found in latest run dir: "{run_dir}"')
        ckpts.sort(key=_checkpoint_step)
        ckpt = ckpts[-1]
        validate_checkpoint(ckpt)
        return ckpt

    p = Path(r)

    # If they passed a run-name (not a path), interpret under checkpoint_root
    if not p.exists() and (checkpoint_root / r).exists():
        p = checkpoint_root / r

    if not p.exists():
        raise FileNotFoundError(f"--resume path does not exist: {p}")

    # If they passed a checkpoint dir directly
    if p.is_dir() and p.name.startswith("checkpoint-"):
        validate_checkpoint(p)
        return p

    # If they passed a run dir (contains checkpoint-*)
    if p.is_dir():
        ckpts = [c for c in p.glob("checkpoint-*") if c.is_dir()]
        if not ckpts:
            raise RuntimeError(f'No checkpoint-* found in resume directory: "{p}"')
        ckpts.sort(key=_checkpoint_step)
        ckpt = ckpts[-1]
        validate_checkpoint(ckpt)
        return ckpt

    raise RuntimeError(f"Unsupported --resume value: {resume_arg} (resolved to {p})")


def validate_checkpoint(ckpt: Path):
    # Hard requirements for a REAL resume (optimizer + scheduler)
    required = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
    ]
    missing = [f for f in required if not (ckpt / f).exists()]
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing required files: {missing}\n"
            f"Checkpoint: {ckpt}\n"
            f"If optimizer/scheduler are missing, LR will reset (bad)."
        )


def read_trainer_state(ckpt: Path) -> dict:
    state_path = ckpt / "trainer_state.json"
    with open(state_path, "r") as f:
        return json.load(f)


def summarize_resume_state(state: dict):
    gs = state.get("global_step", None)
    epoch = state.get("epoch", None)
    max_steps = state.get("max_steps", None)
    last_lr = None
    if state.get("log_history"):
        # scan backwards for most recent LR entry
        for item in reversed(state["log_history"]):
            if isinstance(item, dict) and "learning_rate" in item:
                last_lr = item.get("learning_rate")
                break
    print("📌 Resume state from trainer_state.json")
    print(f"   global_step: {gs}")
    print(f"   epoch:       {epoch}")
    print(f"   max_steps:   {max_steps}")
    if last_lr is not None:
        print(f"   last_lr:     {last_lr}")


def main():
    parser = argparse.ArgumentParser()

    # This is the directory that contains *.arrow shards
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/workspace/hf/datasets/postgrammar___london-llm-1800/train",
        help="Directory containing local *.arrow shards (offline).",
    )

    parser.add_argument("--output_dir", type=str, default="/workspace/outputs")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help='Resume from checkpoint/run. Examples: '
             '".../checkpoint-60000" OR ".../london_h100_YYYYMMDD_HHMMSS" OR "latest".',
    )

    # Overrides (optional)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--sample_vocab_n", type=int, default=2000)
    parser.add_argument("--report_to", type=str, default="none", help="none|tensorboard|wandb...")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    OUTPUT_DIR = Path(args.output_dir)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("H100 TRAINING INIT (LOCAL ARROW SHARDS, NO DOWNLOADS)")
    print("=" * 60)
    print_gpu_utilization()

    # ===========================================================
    # 0) RESOLVE RESUME CHECKPOINT (BEFORE SETTING output_dir)
    # ===========================================================
    RESUME_CKPT = None
    RESUME_STATE = None
    if args.resume:
        RESUME_CKPT = resolve_resume_checkpoint(args.resume, CHECKPOINT_DIR)
        RESUME_STATE = read_trainer_state(RESUME_CKPT)
        print(f"\n🔁 RESUME ENABLED")
        print(f"   Resuming from: {RESUME_CKPT}")
        summarize_resume_state(RESUME_STATE)

    # ===========================================================
    # 1) LOAD DATASET FROM LOCAL .arrow SHARDS (OFFLINE)
    # ===========================================================
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir does not exist: {dataset_dir}")

    # FINAL FIX: your shards are named "data-xxxxx-of-00239.arrow"
    # so we glob all .arrow shards in the directory.
    train_shards = sorted(dataset_dir.glob("*.arrow"))
    if not train_shards:
        raise RuntimeError(
            f"No .arrow shards found in {dataset_dir}.\n"
            f"Expected files like data-00000-of-00239.arrow\n"
            f"Tip: run `ls {dataset_dir}` and confirm *.arrow files exist."
        )

    print("\n📂 Loading local Arrow shards (no network)...")
    print(f"   Dataset dir: {dataset_dir}")
    print(f"   Train shards: {len(train_shards)} (example: {train_shards[0].name})")

    ds = load_dataset(
        "arrow",
        data_files={"train": [str(p) for p in train_shards]},
    )
    train_ds = ds["train"]
    print(f"✅ Loaded train | rows={len(train_ds):,} | features={list(train_ds.features)}")

    # ===========================================================
    # 2) SAFE VOCAB CHECK
    # ===========================================================
    print("\n🧪 Checking token id range (sample)...")
    max_id, vocab_size = infer_safe_vocab_size(train_ds, sample_n=args.sample_vocab_n)
    print(f"✅ Max token id (sample): {max_id}")
    print(f"✅ Using vocab_size:       {vocab_size}")

    # ===========================================================
    # 3) BUILD MODEL (~1.2B) + DTYPE SAFETY
    # ===========================================================
    use_flash = flash_attn_available()
    if use_flash:
        print("⚡ FlashAttention2 detected: enabled")
    else:
        print("⚠️ FlashAttention2 not detected: using standard attention")

    print("\n🔧 Building model from config.py ...")
    config = build_llama_config(vocab_size=vocab_size, use_flash=use_flash)

    model = LlamaForCausalLM(config)
    # IMPORTANT: force bf16 explicitly
    model = model.to(dtype=torch.bfloat16, device="cuda")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Params: {total_params/1e9:.2f}B")
    print_gpu_utilization()

    # HARD GUARD: keep you in the ~1.2B band
    if not (1.10e9 <= total_params <= 1.30e9):
        raise RuntimeError(
            f"ABORT: model is {total_params/1e9:.2f}B params. "
            f"Expected ~1.2B. Check config.py MODEL_CONFIG (layers/hidden sizes)."
        )

    # ===========================================================
    # 4) TRAINING ARGS (CONFIG + CLI OVERRIDES)
    # ===========================================================
    # If resuming, continue IN-PLACE in the same run folder to avoid accidental fresh-start runs.
    if RESUME_CKPT:
        out_run = RESUME_CKPT.parent  # .../run_name
        run_name = out_run.name
        print(f"\n🧷 Continuing in-place in run directory: {out_run}")
    else:
        run_name = f"london_h100_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_run = CHECKPOINT_DIR / run_name

    tcfg = dict(TRAINING_CONFIG)

    # --- max_steps logic ---
    # Target ~0.5 epochs total. Using your current run estimate:
    # 0.5 epoch ≈ 181,800 steps -> round to 182,000.
    TARGET_MAX_STEPS = 182_000

    if args.max_steps is not None:
        tcfg["max_steps"] = args.max_steps
    else:
        if RESUME_CKPT:
            # default bump to ~0.5 epoch total when resuming
            tcfg["max_steps"] = TARGET_MAX_STEPS

    if args.batch_size is not None:
        tcfg["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum is not None:
        tcfg["gradient_accumulation_steps"] = args.grad_accum

    # Force safe defaults
    tcfg["report_to"] = args.report_to  # default "none"
    tcfg["bf16"] = True
    tcfg["tf32"] = True
    tcfg["remove_unused_columns"] = False

    # If resuming, ensure we will actually run more steps
    if RESUME_STATE is not None:
        resumed_step = int(RESUME_STATE.get("global_step", 0))
        if int(tcfg.get("max_steps", 0)) <= resumed_step:
            raise RuntimeError(
                f"max_steps ({tcfg.get('max_steps')}) must be > resumed global_step ({resumed_step}).\n"
                f"Set --max_steps {TARGET_MAX_STEPS} (or higher) to continue."
            )

    training_args = TrainingArguments(
        output_dir=str(out_run),
        run_name=run_name,
        logging_dir=str(LOG_DIR / run_name),
        **tcfg,
    )

    def data_collator(features):
        input_ids = torch.as_tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.as_tensor([f["attention_mask"] for f in features], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    # ===========================================================
    # 5) TRAIN
    # ===========================================================
    eff_batch = (
        tcfg.get("per_device_train_batch_size", 1)
        * tcfg.get("gradient_accumulation_steps", 1)
        * max(1, torch.cuda.device_count())
    )

    print("\n🚀 Starting training")
    print(f"   Effective batch: {eff_batch}")
    print(f"   Output dir: {out_run}")
    print(f"   max_steps: {tcfg.get('max_steps')}")
    if RESUME_CKPT:
        print(f"   resume_from: {RESUME_CKPT}\n")
    else:
        print("   resume_from: (none)\n")

    trainer.train(resume_from_checkpoint=str(RESUME_CKPT) if RESUME_CKPT else None)

    print("\n💾 Saving final model...")
    trainer.save_model(str(out_run / "final"))

    if _HAS_PYNVML:
        import pynvml
        pynvml.nvmlShutdown()

    print(f"\n✅ COMPLETE: {out_run / 'final'}")


if __name__ == "__main__":
    main()
