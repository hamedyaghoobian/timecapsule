#!/usr/bin/env python3
"""
01_analyze_corpus.py - Corpus Analysis for Historical LLM Training

Analyzes the 1800-1875 English corpus for:
- OCR artifact detection
- Character frequency distribution
- Historical spelling patterns
- Data quality metrics

Usage: python src/01_analyze_corpus.py
"""

import os
import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import re

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, OUTPUT_DIR, LOG_DIR

# ============================================================
# ANALYSIS CONFIGURATION
# ============================================================
SAMPLE_SIZE = 1000          # Number of files to sample for detailed analysis
MAX_CHAR_DISPLAY = 50       # Top N characters to display

# Known OCR artifacts and control characters
CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f]')
SUSPECT_PATTERN = re.compile(r'M-\^[@-Z]')  # Common OCR escape sequences


def load_text_files(directory: Path, limit: int = None) -> list[tuple[str, str]]:
    """Load text files from directory. Returns list of (filename, content)."""
    files = list(directory.glob("*.txt"))
    if limit:
        files = files[:limit]
    
    texts = []
    for filepath in tqdm(files, desc=f"Loading from {directory.name}"):
        try:
            content = filepath.read_text(encoding='utf-8', errors='replace')
            texts.append((filepath.name, content))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    return texts


def analyze_characters(texts: list[tuple[str, str]]) -> dict:
    """Analyze character frequency distribution."""
    char_counter = Counter()
    total_chars = 0
    
    for _, content in texts:
        char_counter.update(content)
        total_chars += len(content)
    
    return {
        "total_chars": total_chars,
        "unique_chars": len(char_counter),
        "char_counter": char_counter,
    }


def detect_ocr_artifacts(texts: list[tuple[str, str]]) -> dict:
    """Detect OCR artifacts and control characters."""
    control_char_files = []
    suspect_sequence_files = []
    control_chars_found = Counter()
    
    for filename, content in texts:
        # Check for control characters
        control_matches = CONTROL_CHAR_PATTERN.findall(content)
        if control_matches:
            control_char_files.append(filename)
            control_chars_found.update(control_matches)
        
        # Check for OCR escape sequences like M-^@
        if SUSPECT_PATTERN.search(content):
            suspect_sequence_files.append(filename)
    
    return {
        "files_with_control_chars": len(control_char_files),
        "files_with_ocr_sequences": len(suspect_sequence_files),
        "control_chars_found": control_chars_found,
        "sample_control_files": control_char_files[:10],
        "sample_ocr_files": suspect_sequence_files[:10],
    }


def detect_historical_chars(texts: list[tuple[str, str]]) -> dict:
    """Detect historical characters and ligatures."""
    historical_chars = {
        'ſ': 'long-s',
        'ꝛ': 'r-rotunda',
        'æ': 'ae-ligature',
        'œ': 'oe-ligature',
        'ﬀ': 'ff-ligature',
        'ﬁ': 'fi-ligature',
        'ﬂ': 'fl-ligature',
        'ﬃ': 'ffi-ligature',
        'ﬄ': 'ffl-ligature',
        '—': 'em-dash',
        '–': 'en-dash',
        '"': 'left-double-quote',
        '"': 'right-double-quote',
        ''': 'left-single-quote',
        ''': 'right-single-quote',
    }
    
    char_counts = Counter()
    
    for _, content in texts:
        for char in historical_chars:
            count = content.count(char)
            if count > 0:
                char_counts[char] += count
    
    return {
        "historical_char_counts": {
            historical_chars.get(char, char): count 
            for char, count in char_counts.most_common()
        },
        "total_historical_chars": sum(char_counts.values()),
    }


def analyze_text_quality(texts: list[tuple[str, str]]) -> dict:
    """Analyze text quality metrics."""
    lengths = []
    empty_files = []
    short_files = []
    
    for filename, content in texts:
        length = len(content)
        lengths.append(length)
        
        if length == 0:
            empty_files.append(filename)
        elif length < 100:
            short_files.append(filename)
    
    return {
        "total_files": len(texts),
        "empty_files": len(empty_files),
        "short_files": len(short_files),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "total_bytes": sum(lengths),
    }


def print_report(results: dict, split_name: str):
    """Print analysis report."""
    print(f"\n{'='*60}")
    print(f"CORPUS ANALYSIS REPORT: {split_name.upper()}")
    print(f"{'='*60}")
    
    # Quality metrics
    q = results["quality"]
    print(f"\n📊 TEXT QUALITY METRICS")
    print(f"   Total files:     {q['total_files']:,}")
    print(f"   Total size:      {q['total_bytes'] / 1e9:.2f} GB")
    print(f"   Empty files:     {q['empty_files']}")
    print(f"   Short files:     {q['short_files']} (<100 chars)")
    print(f"   Avg length:      {q['avg_length']:,.0f} chars")
    print(f"   Max length:      {q['max_length']:,} chars")
    
    # Character analysis
    c = results["chars"]
    print(f"\n🔤 CHARACTER ANALYSIS")
    print(f"   Total characters: {c['total_chars']:,}")
    print(f"   Unique characters: {c['unique_chars']}")
    
    # Top characters
    print(f"\n   Top 20 characters:")
    for char, count in c["char_counter"].most_common(20):
        if char == '\n':
            display = '\\n'
        elif char == ' ':
            display = 'SPACE'
        elif char == '\t':
            display = '\\t'
        else:
            display = char
        pct = count / c['total_chars'] * 100
        print(f"      '{display}': {count:,} ({pct:.2f}%)")
    
    # OCR artifacts
    o = results["ocr"]
    print(f"\n⚠️  OCR ARTIFACT DETECTION")
    print(f"   Files with control chars: {o['files_with_control_chars']}")
    print(f"   Files with OCR sequences: {o['files_with_ocr_sequences']}")
    if o['control_chars_found']:
        print(f"   Control chars found: {dict(o['control_chars_found'])}")
    
    # Historical characters
    h = results["historical"]
    print(f"\n📜 HISTORICAL CHARACTERS")
    print(f"   Total historical chars: {h['total_historical_chars']:,}")
    if h['historical_char_counts']:
        for name, count in list(h['historical_char_counts'].items())[:10]:
            print(f"      {name}: {count:,}")


def save_report(all_results: dict, output_path: Path):
    """Save detailed report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Corpus Analysis Report\n\n")
        f.write(f"Generated by: 01_analyze_corpus.py\n\n")
        
        for split_name, results in all_results.items():
            f.write(f"## {split_name.upper()}\n\n")
            
            q = results["quality"]
            f.write(f"- Total files: {q['total_files']:,}\n")
            f.write(f"- Total size: {q['total_bytes'] / 1e9:.2f} GB\n")
            f.write(f"- Average length: {q['avg_length']:,.0f} chars\n")
            
            o = results["ocr"]
            f.write(f"- Files with control chars: {o['files_with_control_chars']}\n")
            f.write(f"- Files with OCR sequences: {o['files_with_ocr_sequences']}\n\n")
    
    print(f"\n✅ Report saved to: {output_path}")


def main():
    print("=" * 60)
    print("HISTORICAL CORPUS ANALYZER (1800-1875)")
    print("=" * 60)
    
    all_results = {}
    
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"⚠️  Directory not found: {split_dir}")
            continue
        
        print(f"\n📂 Analyzing {split}...")
        
        # Load texts (sample for train, all for val/test)
        limit = SAMPLE_SIZE if split == "train" else None
        texts = load_text_files(split_dir, limit=limit)
        
        if not texts:
            print(f"   No files found in {split_dir}")
            continue
        
        # Run analyses
        results = {
            "quality": analyze_text_quality(texts),
            "chars": analyze_characters(texts),
            "ocr": detect_ocr_artifacts(texts),
            "historical": detect_historical_chars(texts),
        }
        
        all_results[split] = results
        print_report(results, split)
    
    # Save report
    report_path = LOG_DIR / "corpus_analysis_report.md"
    save_report(all_results, report_path)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Summary recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("   1. Run 02_clean_corpus.py to remove OCR artifacts")
    print("   2. Long-s (ſ) will be normalized to 's'")
    print("   3. Control characters will be stripped")


if __name__ == "__main__":
    main()
