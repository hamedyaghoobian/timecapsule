#!/usr/bin/env python3
"""
CUDA-Optimized LLaMA Training Script for RunPod
================================================
Designed for A100/H100 GPUs with Flash Attention 2

Usage:
    python 05_train_model_cuda.py --max_steps 100000
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# ===========================================================
# CUDA CONFIGURATION (set before importing torch)
# ===========================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
)
from datasets import load_from_disk

# ===========================================================
# PATHS - Adjust these for your RunPod setup
# ===========================================================
# Default paths assume data is in /workspace/data/
DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/outputs"))

TOKENIZER_DIR = DATA_DIR / "tokenizer"
DATASET_DIR = DATA_DIR / "dataset"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# ===========================================================
# TOKENIZER SETTINGS
# ===========================================================
VOCAB_SIZE = 32000
CONTEXT_LENGTH = 2048

# ===========================================================
# MODEL ARCHITECTURE (~300M params, optimized for A100)
# ===========================================================
MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_hidden_layers": 20,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,  # GQA: 2:1 ratio
    "hidden_act": "silu",
    "max_position_embeddings": CONTEXT_LENGTH,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "use_cache": True,
    "pad_token_id": 3,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    # Flash Attention 2 support
    "attn_implementation": "flash_attention_2",
}

# ===========================================================
# TRAINING PARAMETERS (optimized for A100 80GB)
# ===========================================================
TRAINING_CONFIG = {
    # Batch settings - A100 can handle larger batches
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,  # Effective batch = 32 per GPU
    
    # Learning rate
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "warmup_steps": 2000,
    "lr_scheduler_type": "cosine",
    
    # Training duration
    "max_steps": 100000,  # Adjust based on dataset size
    "num_train_epochs": 1,  # Will be overridden by max_steps
    
    # Precision and optimization
    "bf16": True,
    "tf32": True,
    "optim": "adamw_torch_fused",  # Fused AdamW for CUDA
    "gradient_checkpointing": True,
    
    # Logging and saving
    "logging_steps": 50,
    "save_steps": 2500,
    "save_total_limit": 5,
    "eval_strategy": "steps",
    "eval_steps": 2500,
    
    # DataLoader
    "dataloader_num_workers": 8,
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 2,
    
    # Other
    "remove_unused_columns": False,
    "report_to": "tensorboard",
    "seed": 42,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Historical LLM on RunPod")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    return parser.parse_args()


def check_flash_attention():
    """Check if Flash Attention 2 is available."""
    try:
        import flash_attn
        print(f"   ✅ Flash Attention 2: v{flash_attn.__version__}")
        return True
    except ImportError:
        print("   ⚠️  Flash Attention 2 not installed, using standard attention")
        return False


def get_device_info():
    """Print GPU information."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
            print(f"          Memory: {props.total_memory / 1e9:.1f} GB")
        return "cuda"
    else:
        print("   ⚠️  No CUDA GPU found!")
        return "cpu"


def main():
    args = parse_args()
    
    # Override paths if specified
    global DATA_DIR, OUTPUT_DIR, TOKENIZER_DIR, DATASET_DIR, CHECKPOINT_DIR, LOG_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
        TOKENIZER_DIR = DATA_DIR / "tokenizer"
        DATASET_DIR = DATA_DIR / "dataset"
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
        LOG_DIR = OUTPUT_DIR / "logs"
    
    print("=" * 60)
    print("HISTORICAL LLM TRAINER (1800-1875)")
    print("RunPod CUDA-Optimized Version")
    print("=" * 60)
    
    # Device info
    print("\n🖥️  Device Configuration:")
    device = get_device_info()
    has_flash_attn = check_flash_attention()
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.version.cuda}")
    
    # Create output directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("\n📂 Loading components...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    print(f"   ✅ Tokenizer loaded (vocab: {len(tokenizer):,})")
    
    # Load dataset
    dataset = load_from_disk(str(DATASET_DIR))
    print(f"   ✅ Dataset loaded:")
    for split in dataset:
        print(f"      {split}: {len(dataset[split]):,} samples")
    
    # Create model
    print("\n🔧 Creating model...")
    
    # Adjust config based on Flash Attention availability
    config_dict = MODEL_CONFIG.copy()
    if not has_flash_attn:
        config_dict.pop("attn_implementation", None)
    
    config = LlamaConfig(**config_dict)
    model = LlamaForCausalLM(config)
    
    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters:     {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Layers:               {config.num_hidden_layers}")
    print(f"   Hidden size:          {config.hidden_size}")
    print(f"   Attention heads:      {config.num_attention_heads}")
    print(f"   KV heads (GQA):       {config.num_key_value_heads}")
    if has_flash_attn:
        print(f"   Attention:            Flash Attention 2 ⚡")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Setup training
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = CHECKPOINT_DIR / run_name
    
    # Update training config with args
    training_config = TRAINING_CONFIG.copy()
    training_config["max_steps"] = args.max_steps
    training_config["per_device_train_batch_size"] = args.batch_size
    training_config["gradient_accumulation_steps"] = args.gradient_accum
    
    effective_batch = (
        training_config["per_device_train_batch_size"] * 
        training_config["gradient_accumulation_steps"] *
        max(1, torch.cuda.device_count())
    )
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Max steps:              {training_config['max_steps']:,}")
    print(f"   Batch size per GPU:     {training_config['per_device_train_batch_size']}")
    print(f"   Gradient accumulation:  {training_config['gradient_accumulation_steps']}")
    print(f"   Effective batch size:   {effective_batch}")
    print(f"   Learning rate:          {training_config['learning_rate']}")
    print(f"   Warmup steps:           {training_config['warmup_steps']}")
    print(f"   Output dir:             {output_path}")
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        run_name=run_name,
        logging_dir=str(LOG_DIR / run_name),
        **training_config,
    )
    
    # Data collator for causal LM
    def data_collator(features):
        batch = {
            "input_ids": torch.stack([
                torch.tensor(f["input_ids"]) for f in features
            ]),
            "attention_mask": torch.stack([
                torch.tensor(f["attention_mask"]) for f in features
            ]),
        }
        batch["labels"] = batch["input_ids"].clone()
        return batch
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"] if "val" in dataset else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Resume from checkpoint if specified
    resume_checkpoint = args.resume
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")
    
    # Train
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save final model
    print("\n💾 Saving model...")
    trainer.save_model(str(output_path / "final"))
    tokenizer.save_pretrained(str(output_path / "final"))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n✅ Model saved to: {output_path / 'final'}")


if __name__ == "__main__":
    main()
