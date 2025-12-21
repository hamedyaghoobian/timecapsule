#!/usr/bin/env python3
"""
05_train_model.py - MPS-Optimized Historical LLM Training

Trains a 500M LLaMA-style model for diachronic analysis on Apple Silicon.

Key features:
- MPS backend with bfloat16 precision
- Gradient checkpointing for memory efficiency
- Configured for M3 Max with 128GB unified memory
- Optimized for semantic representation learning

Usage: python src/05_train_model.py [--max_steps N] [--resume PATH]
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TOKENIZER_DIR, DATASET_DIR, CHECKPOINT_DIR, LOG_DIR,
    MODEL_CONFIG, TRAINING_CONFIG, CONTEXT_LENGTH,
    get_device, get_dtype
)

# ============================================================
# MPS CONFIGURATION
# ============================================================
# Prevent MPS memory fragmentation issues
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Historical LLM on MPS")
    
    parser.add_argument(
        "--max_steps", type=int, default=TRAINING_CONFIG["max_steps"],
        help="Maximum training steps"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--batch_size", type=int, default=TRAINING_CONFIG["per_device_train_batch_size"],
        help="Per-device batch size"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=TRAINING_CONFIG["eval_steps"],
        help="Evaluation frequency"
    )
    
    return parser.parse_args()


def load_tokenizer() -> PreTrainedTokenizerFast:
    """Load the trained tokenizer."""
    
    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print("   Run 03_train_tokenizer.py first")
        sys.exit(1)
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.model_max_length = CONTEXT_LENGTH
    
    return tokenizer


def load_dataset():
    """Load the prepared dataset."""
    
    if not DATASET_DIR.exists():
        print(f"❌ Dataset not found: {DATASET_DIR}")
        print("   Run 04_prepare_dataset.py first")
        sys.exit(1)
    
    ds = load_from_disk(str(DATASET_DIR))
    return ds


def create_model(tokenizer: PreTrainedTokenizerFast) -> LlamaForCausalLM:
    """Create the LLaMA model with proper configuration."""
    
    device = get_device()
    dtype = get_dtype()
    
    print(f"🔧 Creating model...")
    print(f"   Device: {device}")
    print(f"   Dtype:  {dtype}")
    
    # Create config
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=MODEL_CONFIG["hidden_size"],
        intermediate_size=MODEL_CONFIG["intermediate_size"],
        num_hidden_layers=MODEL_CONFIG["num_hidden_layers"],
        num_attention_heads=MODEL_CONFIG["num_attention_heads"],
        num_key_value_heads=MODEL_CONFIG["num_key_value_heads"],
        max_position_embeddings=MODEL_CONFIG["max_position_embeddings"],
        rms_norm_eps=MODEL_CONFIG["rms_norm_eps"],
        rope_theta=MODEL_CONFIG["rope_theta"],
        use_cache=False,  # Disable for training
        torch_dtype=dtype,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Initialize model
    model = LlamaForCausalLM(config)
    
    # Ensure embedding size matches vocab
    model.resize_token_embeddings(tokenizer.vocab_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters:     {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Layers:               {config.num_hidden_layers}")
    print(f"   Hidden size:          {config.hidden_size}")
    print(f"   Attention heads:      {config.num_attention_heads}")
    print(f"   KV heads (GQA):       {config.num_key_value_heads}")
    
    return model


def create_training_args(args) -> TrainingArguments:
    """Create training arguments optimized for MPS."""
    
    device = get_device()
    
    # Create unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = CHECKPOINT_DIR / f"run_{timestamp}"
    log_dir = LOG_DIR / f"run_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        
        # Batch settings
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        
        # Learning rate
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        max_steps=args.max_steps,
        
        # Precision - use bf16 for M3
        bf16=(device.type == "mps"),
        fp16=False,  # Don't use fp16 on MPS
        
        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Evaluation
        eval_strategy="steps" if "val" in str(DATASET_DIR) else "no",
        eval_steps=args.eval_steps,
        
        # Checkpointing
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        
        # Logging
        logging_dir=str(log_dir),
        logging_steps=TRAINING_CONFIG["logging_steps"],
        report_to="none",  # Disable wandb etc.
        
        # MPS-specific settings
        dataloader_pin_memory=False,  # Not supported on MPS
        dataloader_num_workers=0,      # Safer for MPS
        
        # Other settings
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    return training_args


def main():
    args = parse_args()
    
    print("=" * 60)
    print("HISTORICAL LLM TRAINER (1800-1875)")
    print("MPS-Optimized for Apple Silicon M3 Max")
    print("=" * 60)
    
    # Device info
    device = get_device()
    dtype = get_dtype()
    print(f"\n🖥️  Device Configuration:")
    print(f"   Device:    {device}")
    print(f"   Dtype:     {dtype}")
    print(f"   MPS:       {'✅ Available' if torch.backends.mps.is_available() else '❌ Not available'}")
    
    # Load components
    print(f"\n📂 Loading components...")
    tokenizer = load_tokenizer()
    print(f"   ✅ Tokenizer loaded (vocab: {tokenizer.vocab_size:,})")
    
    ds = load_dataset()
    print(f"   ✅ Dataset loaded:")
    for split, dataset in ds.items():
        print(f"      {split}: {len(dataset):,} samples")
    
    # Create model
    model = create_model(tokenizer)
    
    # Move model to device
    model = model.to(device)
    
    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = create_training_args(args)
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Max steps:              {args.max_steps:,}")
    print(f"   Batch size:             {args.batch_size}")
    print(f"   Gradient accumulation:  {training_args.gradient_accumulation_steps}")
    print(f"   Effective batch size:   {args.batch_size * training_args.gradient_accumulation_steps}")
    print(f"   Learning rate:          {training_args.learning_rate}")
    print(f"   Warmup steps:           {training_args.warmup_steps}")
    print(f"   Output dir:             {training_args.output_dir}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("val"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")
    
    if args.resume:
        print(f"📂 Resuming from: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    # Save final model
    print("\n💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n✅ Model saved to: {training_args.output_dir}")
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Run: python src/06_evaluate_model.py --checkpoint {training_args.output_dir}")


if __name__ == "__main__":
    main()
