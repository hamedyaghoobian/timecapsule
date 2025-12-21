#!/usr/bin/env python3
"""
04_prepare_dataset_efficient.py - True Streaming Dataset Preparation

Optimized for 90GB+ datasets on limited RAM.
Processes file-by-file and streams directly to Arrow format.
Memory usage: <1GB constant.
"""

import sys
import argparse
import gc
from pathlib import Path
from tqdm import tqdm
from typing import Iterator

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from transformers import PreTrainedTokenizerFast

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, TOKENIZER_DIR, DATASET_DIR, CONTEXT_LENGTH
)

# ============================================================
# CLI ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Arrow dataset via streaming")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory with cleaned text files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for dataset")
    return parser.parse_args()

# ============================================================
# STREAMING GENERATOR
# ============================================================
def chunk_generator(files: list[Path], tokenizer_path: str, chunk_size: int) -> Iterator[dict]:
    """
    Generator that processes files one by one and yields chunks.
    This runs inside the Dataset worker process.
    """
    
    # Re-load tokenizer inside generator (needed for multiprocessing/pickling)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    eos_id = tokenizer.eos_token_id
    
    current_chunk = []
    
    # Iterate through files
    for filepath in files:
        try:
            # Read file (file is closed immediately after read)
            text = filepath.read_text(encoding='utf-8', errors='replace')
            if not text.strip():
                continue
                
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Add EOS
            tokens.append(eos_id)
            
            # Stream tokens into chunks
            chunks_processed = 0
            
            # If current buffer + new tokens > chunk_size, we can yield chunks
            combined = current_chunk + tokens
            
            while len(combined) >= chunk_size:
                # Yield full chunk
                yield {
                    "input_ids": combined[:chunk_size],
                    "attention_mask": [1] * chunk_size
                }
                # Move to next
                combined = combined[chunk_size:]
                chunks_processed += 1
            
            # Keep remainder in buffer
            current_chunk = combined
            
        except Exception as e:
            print(f"Warning: Failed to process {filepath}: {e}")
            continue

    # Yield final leftover chunk if substantial
    if len(current_chunk) > 0:
        # Pad the last chunk
        pad_id = tokenizer.pad_token_id or eos_id
        padding = [pad_id] * (chunk_size - len(current_chunk))
        yield {
            "input_ids": current_chunk + padding,
            "attention_mask": [1] * len(current_chunk) + [0] * len(padding)
        }

def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tokenizer_dir = output_dir / "tokenizer"
    dataset_output = output_dir / "dataset"
    tokenizer_json = str(tokenizer_dir / "tokenizer.json")
    
    print("=" * 60)
    print("STREAMING DATASET PREPARATION")
    print("Optimization: File-by-file streaming (Low RAM usage)")
    print("=" * 60)
    
    # Verify paths
    if not Path(tokenizer_json).exists():
        print(f"❌ Tokenizer not found at: {tokenizer_json}")
        sys.exit(1)
        
    # Get file list
    print(f"📂 Scanning input directory: {input_dir}")
    files = sorted(list(input_dir.glob("*.txt")))
    if not files:
        print("❌ No .txt files found!")
        sys.exit(1)
        
    print(f"✅ Found {len(files):,} files")
    
    # Create Dataset from generator
    print("\n🚀 Starting streaming generation...")
    print("   This will process files and write directly to disk.")
    
    # We pass the generator function and its arguments
    dataset = Dataset.from_generator(
        chunk_generator,
        gen_kwargs={
            "files": files, 
            "tokenizer_path": tokenizer_json,
            "chunk_size": 2048
        },
        num_proc=4,  # Use 4 cores for parallel processing!
        writer_batch_size=1000,
    )
    
    print("\n💾 Saving dataset to disk...")
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(str(dataset_output))
    
    print("\n" + "=" * 60)
    print("✅ PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Output: {dataset_output}")
    print(f"Total samples: {len(dataset):,}")

if __name__ == "__main__":
    main()
