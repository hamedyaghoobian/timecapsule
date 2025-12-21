#!/usr/bin/env python3
"""
02_clean_corpus.py - Corpus Cleaning for Historical LLM Training

Cleans the 1800-1875 English corpus with minimal normalization:
- Normalizes long-s (ſ) to 's' for semantic consistency
- Removes OCR control characters
- Preserves historical spellings and ligatures
- Outputs cleaned corpus for tokenizer training

Usage: python src/02_clean_corpus.py
"""

import os
import sys
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, OUTPUT_DIR,
    CHAR_NORMALIZATION, MIN_TEXT_LENGTH,
    NORMALIZE_LONG_S, REMOVE_CONTROL_CHARS
)

# ============================================================
# CLEANING CONFIGURATION
# ============================================================
CLEANED_DIR = OUTPUT_DIR / "cleaned_corpus"
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# Control character pattern (keep newlines and tabs)
CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# OCR escape sequence pattern (like M-^@, M-^X)
OCR_ESCAPE_PATTERN = re.compile(r'M-\^[@-Z\[\]\\^_]')

# Multiple spaces/newlines normalization
MULTI_SPACE_PATTERN = re.compile(r'[ \t]+')
MULTI_NEWLINE_PATTERN = re.compile(r'\n{3,}')


def clean_text(text: str) -> str:
    """
    Clean a single text document with minimal normalization.
    
    Preserves:
    - Historical spellings (colour, connexion, etc.)
    - Case sensitivity
    - Paragraph structure
    - Ligatures (æ, œ)
    
    Removes/Normalizes:
    - Long-s (ſ → s)
    - Control characters
    - OCR escape sequences
    - Excessive whitespace
    """
    # 1. Character normalization (ſ → s, ligatures, etc.)
    for old_char, new_char in CHAR_NORMALIZATION.items():
        text = text.replace(old_char, new_char)
    
    # 2. Remove OCR escape sequences (M-^@ patterns)
    text = OCR_ESCAPE_PATTERN.sub('', text)
    
    # 3. Remove control characters (except newlines and tabs)
    if REMOVE_CONTROL_CHARS:
        text = CONTROL_CHAR_PATTERN.sub('', text)
    
    # 4. Normalize whitespace (but preserve paragraph structure)
    text = MULTI_SPACE_PATTERN.sub(' ', text)
    text = MULTI_NEWLINE_PATTERN.sub('\n\n', text)
    
    # 5. Strip leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # 6. Remove any remaining non-printable Unicode
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    return text.strip()


def process_file(args: tuple) -> tuple[str, str, int, int]:
    """
    Process a single file. Returns (filename, cleaned_text, original_len, cleaned_len).
    """
    filepath, output_dir = args
    
    try:
        # Read original
        original_text = filepath.read_text(encoding='utf-8', errors='replace')
        original_len = len(original_text)
        
        # Clean
        cleaned_text = clean_text(original_text)
        cleaned_len = len(cleaned_text)
        
        # Skip if too short after cleaning
        if cleaned_len < MIN_TEXT_LENGTH:
            return (filepath.name, None, original_len, 0)
        
        # Save cleaned
        output_path = output_dir / filepath.name
        output_path.write_text(cleaned_text, encoding='utf-8')
        
        return (filepath.name, "success", original_len, cleaned_len)
    
    except Exception as e:
        return (filepath.name, f"error: {e}", 0, 0)


def clean_split(split_name: str) -> dict:
    """Clean all files in a split directory."""
    input_dir = DATA_DIR / split_name
    output_dir = CLEANED_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob("*.txt"))
    if not files:
        print(f"   No files found in {input_dir}")
        return {}
    
    # Prepare arguments
    args = [(f, output_dir) for f in files]
    
    # Process files in parallel
    results = {
        "total_files": len(files),
        "success": 0,
        "skipped": 0,
        "errors": 0,
        "original_bytes": 0,
        "cleaned_bytes": 0,
    }
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_file, arg): arg for arg in args}
        
        for future in tqdm(as_completed(futures), total=len(files), desc=f"Cleaning {split_name}"):
            filename, status, orig_len, clean_len = future.result()
            
            results["original_bytes"] += orig_len
            
            if status == "success":
                results["success"] += 1
                results["cleaned_bytes"] += clean_len
            elif status is None:
                results["skipped"] += 1
            else:
                results["errors"] += 1
    
    return results


def print_results(split_name: str, results: dict):
    """Print cleaning results for a split."""
    print(f"\n   📂 {split_name.upper()}")
    print(f"      Total files:     {results['total_files']:,}")
    print(f"      Successful:      {results['success']:,}")
    print(f"      Skipped (short): {results['skipped']:,}")
    print(f"      Errors:          {results['errors']:,}")
    
    if results['original_bytes'] > 0:
        reduction = (1 - results['cleaned_bytes'] / results['original_bytes']) * 100
        print(f"      Original size:   {results['original_bytes'] / 1e9:.2f} GB")
        print(f"      Cleaned size:    {results['cleaned_bytes'] / 1e9:.2f} GB")
        print(f"      Size reduction:  {reduction:.1f}%")


def main():
    print("=" * 60)
    print("HISTORICAL CORPUS CLEANER (1800-1875)")
    print("=" * 60)
    print(f"\n📝 Cleaning Configuration:")
    print(f"   Long-s normalization: {'ſ → s' if NORMALIZE_LONG_S else 'Disabled'}")
    print(f"   Control char removal: {'Enabled' if REMOVE_CONTROL_CHARS else 'Disabled'}")
    print(f"   Min text length:      {MIN_TEXT_LENGTH} chars")
    print(f"   Workers:              {NUM_WORKERS}")
    print(f"\n📂 Output directory: {CLEANED_DIR}")
    
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for split in ["train", "val", "test"]:
        if not (DATA_DIR / split).exists():
            print(f"\n⚠️  Skipping {split} (directory not found)")
            continue
        
        results = clean_split(split)
        if results:
            all_results[split] = results
            print_results(split, results)
    
    # Summary
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    
    total_original = sum(r['original_bytes'] for r in all_results.values())
    total_cleaned = sum(r['cleaned_bytes'] for r in all_results.values())
    total_files = sum(r['success'] for r in all_results.values())
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total files cleaned: {total_files:,}")
    print(f"   Original corpus:     {total_original / 1e9:.2f} GB")
    print(f"   Cleaned corpus:      {total_cleaned / 1e9:.2f} GB")
    
    print(f"\n✅ Cleaned corpus saved to: {CLEANED_DIR}")
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Run: python src/03_train_tokenizer.py")


if __name__ == "__main__":
    main()
