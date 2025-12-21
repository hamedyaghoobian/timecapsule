#!/usr/bin/env python3
"""
04_prepare_dataset.py - Dataset Preparation for Historical LLM Training

Converts cleaned text corpus to tokenized Arrow format for efficient training.
Uses memory-mapped datasets suitable for 128GB unified memory.

Key features:
- Tokenizes entire corpus with trained BPE tokenizer
- Creates fixed-length chunks (2048 tokens)
- Saves as HuggingFace Arrow format
- Memory-efficient processing

Usage: 
    python src/04_prepare_dataset.py
    python src/04_prepare_dataset.py --input_dir data/02_cleaned --output_dir outputs_full
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Iterator

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, TOKENIZER_DIR, DATASET_DIR, CONTEXT_LENGTH
)

# ============================================================
# CLI ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Arrow dataset from cleaned corpus")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory with cleaned text files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for dataset")
    parser.add_argument("--tokenizer_dir", type=str, default=None,
                        help="Directory containing trained tokenizer")
    return parser.parse_args()

# ============================================================
# CONFIGURATION
# ============================================================
CLEANED_DIR = OUTPUT_DIR / "cleaned_corpus"
CHUNK_SIZE = CONTEXT_LENGTH  # 2048 tokens per sample


def load_tokenizer(tokenizer_dir: Path = None) -> PreTrainedTokenizerFast:
    """Load the trained tokenizer."""
    
    if tokenizer_dir:
        tokenizer_path = tokenizer_dir / "tokenizer.json"
    else:
        tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found at: {tokenizer_path}")
        print("   Please run 03_train_tokenizer.py first")
        sys.exit(1)
    
    # Load as HuggingFace tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.model_max_length = CONTEXT_LENGTH
    
    print(f"✅ Loaded tokenizer from: {tokenizer_path}")
    print(f"   Vocab size: {tokenizer.vocab_size:,}")
    
    return tokenizer


def load_texts(input_dir: Path, split: str = None) -> list[str]:
    """Load all texts from input directory."""
    
    # If split is specified, look in subdirectory
    if split:
        split_dir = input_dir / split
        if split_dir.exists():
            files = list(split_dir.glob("*.txt"))
        else:
            return []
    else:
        # Load directly from input_dir (flat structure)
        files = list(input_dir.glob("*.txt"))
    
    if not files:
        return []
    
    texts = []
    
    desc = f"Loading {split}" if split else "Loading files"
    for filepath in tqdm(files, desc=desc):
        try:
            text = filepath.read_text(encoding='utf-8', errors='replace')
            if text.strip():
                texts.append(text)
        except Exception as e:
            print(f"   Warning: Could not read {filepath}: {e}")
    
    return texts


def tokenize_and_chunk(
    texts: list[str], 
    tokenizer: PreTrainedTokenizerFast,
    chunk_size: int = CHUNK_SIZE
) -> Iterator[dict]:
    """
    Tokenize texts and yield fixed-size chunks.
    
    This creates training samples by:
    1. Concatenating all tokenized texts with EOS separator
    2. Splitting into non-overlapping chunks of chunk_size
    """
    
    all_tokens = []
    eos_id = tokenizer.eos_token_id
    
    for text in tqdm(texts, desc="Tokenizing"):
        # Tokenize
        encoded = tokenizer.encode(text, add_special_tokens=False)
        
        # Add tokens with EOS separator
        all_tokens.extend(encoded)
        all_tokens.append(eos_id)
    
    print(f"   Total tokens: {len(all_tokens):,}")
    
    # Create chunks
    num_chunks = len(all_tokens) // chunk_size
    print(f"   Creating {num_chunks:,} chunks of {chunk_size} tokens")
    
    for i in tqdm(range(0, len(all_tokens) - chunk_size + 1, chunk_size), desc="Chunking"):
        chunk = all_tokens[i:i + chunk_size]
        
        yield {
            "input_ids": chunk,
            "attention_mask": [1] * len(chunk),
        }


def create_dataset(texts: list[str], tokenizer: PreTrainedTokenizerFast) -> Dataset:
    """Create a HuggingFace Dataset from texts."""
    
    # Collect all chunks
    chunks = list(tokenize_and_chunk(texts, tokenizer))
    
    if not chunks:
        return None
    
    # Create dataset
    dataset = Dataset.from_list(chunks)
    
    return dataset


def main():
    args = parse_args()
    
    # Determine directories
    input_dir = Path(args.input_dir) if args.input_dir else CLEANED_DIR
    tokenizer_dir = Path(args.tokenizer_dir) if args.tokenizer_dir else None
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        dataset_output = output_dir / "dataset"
        if not tokenizer_dir:
            tokenizer_dir = output_dir / "tokenizer"
    else:
        dataset_output = DATASET_DIR
    
    print("=" * 60)
    print("HISTORICAL DATASET PREPARATION (1800-1875)")
    print("=" * 60)
    print(f"\n📂 Input directory: {input_dir}")
    print(f"📂 Tokenizer directory: {tokenizer_dir or TOKENIZER_DIR}")
    print(f"📂 Output directory: {dataset_output}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_dir)
    
    # Check if input has splits or is flat
    has_splits = any((input_dir / split).exists() for split in ["train", "val", "test"])
    
    # Process data
    datasets = {}
    
    if has_splits:
        # Process each split separately
        for split in ["train", "val", "test"]:
            print(f"\n{'='*40}")
            print(f"Processing {split.upper()}")
            print(f"{'='*40}")
            
            # Load texts
            texts = load_texts(input_dir, split)
            
            if not texts:
                print(f"   No texts found for {split}")
                continue
            
            print(f"   Loaded {len(texts):,} documents")
            
            # Create dataset
            dataset = create_dataset(texts, tokenizer)
            
            if dataset:
                datasets[split] = dataset
                print(f"   Created dataset with {len(dataset):,} samples")
    else:
        # Process all files as single "train" split
        print(f"\n{'='*40}")
        print(f"Processing ALL FILES (no splits detected)")
        print(f"{'='*40}")
        
        texts = load_texts(input_dir, split=None)
        
        if not texts:
            print("   No texts found!")
        else:
            print(f"   Loaded {len(texts):,} documents")
            dataset = create_dataset(texts, tokenizer)
            
            if dataset:
                datasets["train"] = dataset
                print(f"   Created dataset with {len(dataset):,} samples")
    
    if not datasets:
        print("\n❌ No datasets created!")
        sys.exit(1)
    
    # Create DatasetDict
    dataset_dict = DatasetDict(datasets)
    
    # Save
    dataset_output.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(dataset_output))
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Dataset Summary:")
    for split, ds in dataset_dict.items():
        total_tokens = len(ds) * CHUNK_SIZE
        print(f"   {split}: {len(ds):,} samples ({total_tokens:,} tokens)")
    
    print(f"\n✅ Saved dataset to: {dataset_output}")
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Upload to RunPod:")
    print(f"      rsync -avz --progress {tokenizer_dir or TOKENIZER_DIR} root@<POD_IP>:/workspace/data/")
    print(f"      rsync -avz --progress {dataset_output} root@<POD_IP>:/workspace/data/")
    print(f"   2. Train on RunPod:")
    print(f"      python 05_train_model_cuda.py --data_dir /workspace/data --output_dir /workspace/outputs")


if __name__ == "__main__":
    main()
