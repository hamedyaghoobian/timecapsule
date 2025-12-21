#!/usr/bin/env python3
"""
Data Preparation Script for RunPod Upload

Prepares the 90GB full dataset for upload to RunPod:
1. Cleans the corpus (if not already done)
2. Trains tokenizer (if not already done)
3. Prepares dataset in Arrow format
4. Packages everything for upload

Usage:
    python prepare_for_runpod.py --data_dir /path/to/90gb/data
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the full 90GB dataset")
    parser.add_argument("--output_dir", type=str, default="outputs_full",
                        help="Output directory for processed data")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Tokenizer vocabulary size")
    parser.add_argument("--context_length", type=int, default=2048,
                        help="Context length for chunking")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of workers (default: CPU count - 2)")
    return parser.parse_args()


# ===========================================================
# CLEANING CONFIG
# ===========================================================
CHAR_NORMALIZATION = {
    'ſ': 's',  # Long s → regular s
    'ﬀ': 'ff', 'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
    '\u00AD': '',  # Soft hyphen
    '\u200B': '',  # Zero-width space
    '\u200C': '',  # Zero-width non-joiner
    '\u200D': '',  # Zero-width joiner
    '\uFEFF': '',  # BOM
}

CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
OCR_ESCAPE_PATTERN = re.compile(r'M-\^[@-Z\[\]\\^_]')
WHITESPACE_PATTERN = re.compile(r'\n{4,}')
MULTI_SPACE_PATTERN = re.compile(r'[ \t]{3,}')


def clean_text(text: str) -> str:
    """Clean a single text string."""
    # Character normalization
    for old, new in CHAR_NORMALIZATION.items():
        text = text.replace(old, new)
    
    # Remove OCR escape sequences
    text = OCR_ESCAPE_PATTERN.sub('', text)
    
    # Remove control characters
    text = CONTROL_CHAR_PATTERN.sub('', text)
    
    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub('\n\n\n', text)
    text = MULTI_SPACE_PATTERN.sub('  ', text)
    
    return text.strip()


def clean_file(args):
    """Clean a single file (for multiprocessing)."""
    input_path, output_path = args
    try:
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        cleaned = clean_text(text)
        
        if len(cleaned) >= 100:  # Minimum length
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            return True, len(text), len(cleaned)
        return False, len(text), 0
    except Exception as e:
        return False, 0, 0


def clean_corpus(data_dir: Path, output_dir: Path, workers: int):
    """Clean the full corpus."""
    print("\n" + "=" * 60)
    print("CLEANING CORPUS")
    print("=" * 60)
    
    cleaned_dir = output_dir / "cleaned_corpus"
    
    # Find all text files
    all_files = []
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            for f in split_dir.glob("**/*.txt"):
                rel_path = f.relative_to(data_dir)
                out_path = cleaned_dir / rel_path
                all_files.append((f, out_path))
    
    if not all_files:
        # Try without splits
        for f in data_dir.glob("**/*.txt"):
            rel_path = f.relative_to(data_dir)
            out_path = cleaned_dir / rel_path
            all_files.append((f, out_path))
    
    print(f"📂 Found {len(all_files):,} files to clean")
    
    # Process in parallel
    success_count = 0
    total_original = 0
    total_cleaned = 0
    
    with Pool(workers) as pool:
        results = list(tqdm(
            pool.imap(clean_file, all_files),
            total=len(all_files),
            desc="Cleaning"
        ))
    
    for success, orig_size, clean_size in results:
        if success:
            success_count += 1
            total_original += orig_size
            total_cleaned += clean_size
    
    print(f"\n✅ Cleaned {success_count:,} files")
    print(f"   Original: {total_original/1e9:.2f} GB")
    print(f"   Cleaned:  {total_cleaned/1e9:.2f} GB")
    
    return cleaned_dir


def train_tokenizer(cleaned_dir: Path, output_dir: Path, vocab_size: int):
    """Train BPE tokenizer."""
    print("\n" + "=" * 60)
    print("TRAINING TOKENIZER")
    print("=" * 60)
    
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all text files
    text_files = list(cleaned_dir.glob("**/*.txt"))
    print(f"📂 Training on {len(text_files):,} files")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<|year|>", "<|title|>"],
        show_progress=True,
    )
    
    tokenizer.train(files=[str(f) for f in text_files], trainer=trainer)
    
    # Save
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    
    # Convert to HuggingFace format
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tokenizer.save_pretrained(str(tokenizer_dir))
    
    print(f"\n✅ Tokenizer saved to: {tokenizer_dir}")
    return tokenizer_dir


def prepare_dataset(cleaned_dir: Path, tokenizer_dir: Path, output_dir: Path, 
                   context_length: int, workers: int):
    """Prepare Arrow dataset."""
    print("\n" + "=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)
    
    from transformers import PreTrainedTokenizerFast
    from datasets import Dataset
    
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
    
    # Process each split
    for split in ["train", "val", "test"]:
        split_dir = cleaned_dir / split
        if not split_dir.exists():
            split_dir = cleaned_dir  # No splits, use all
        
        if not split_dir.exists():
            continue
            
        print(f"\n📂 Processing {split}...")
        text_files = list(split_dir.glob("**/*.txt"))
        
        if not text_files:
            continue
        
        # Tokenize all files
        all_tokens = []
        for f in tqdm(text_files, desc=f"Tokenizing {split}"):
            with open(f, 'r', encoding='utf-8') as file:
                text = file.read()
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        print(f"   Total tokens: {len(all_tokens):,}")
        
        # Create chunks
        chunks = []
        for i in range(0, len(all_tokens) - context_length + 1, context_length):
            chunk = all_tokens[i:i + context_length]
            chunks.append({
                "input_ids": chunk,
                "attention_mask": [1] * len(chunk),
            })
        
        print(f"   Created {len(chunks):,} chunks")
        
        # Save as Dataset
        ds = Dataset.from_list(chunks)
        ds.save_to_disk(str(dataset_dir / split))
        
        print(f"   ✅ Saved {split} to {dataset_dir / split}")
    
    return dataset_dir


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    workers = args.workers or max(1, cpu_count() - 2)
    
    print("=" * 60)
    print("PREPARING DATA FOR RUNPOD")
    print("=" * 60)
    print(f"\n📂 Input:  {data_dir}")
    print(f"📂 Output: {output_dir}")
    print(f"🔧 Workers: {workers}")
    
    # Step 1: Clean corpus
    cleaned_dir = clean_corpus(data_dir, output_dir, workers)
    
    # Step 2: Train tokenizer
    tokenizer_dir = train_tokenizer(cleaned_dir, output_dir, args.vocab_size)
    
    # Step 3: Prepare dataset
    dataset_dir = prepare_dataset(
        cleaned_dir, tokenizer_dir, output_dir, 
        args.context_length, workers
    )
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\n📦 Ready for upload:")
    print(f"   {tokenizer_dir}")
    print(f"   {dataset_dir}")
    print(f"\n💡 To upload to RunPod:")
    print(f"   rsync -avz --progress {output_dir}/tokenizer root@<POD_IP>:/workspace/data/")
    print(f"   rsync -avz --progress {output_dir}/dataset root@<POD_IP>:/workspace/data/")


if __name__ == "__main__":
    main()
