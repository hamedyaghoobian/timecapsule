#!/usr/bin/env python3
"""
03_train_tokenizer.py - Custom BPE Tokenizer Training

Trains a new BPE tokenizer from scratch on the 1800-1875 corpus.
Does NOT use any modern tokenizer as a base.

Key features:
- 32K vocabulary optimized for 500M model
- Byte-level BPE for handling unknown characters
- Special tokens for temporal metadata
- Trained exclusively on historical text

Usage: 
    python src/03_train_tokenizer.py
    python src/03_train_tokenizer.py --input_dir data/02_cleaned --output_dir outputs_full
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import Sequence, NFD, StripAccents

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, VOCAB_SIZE, MIN_FREQUENCY,
    SPECIAL_TOKENS, TOKENIZER_DIR
)

# ============================================================
# CLI ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on historical corpus")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory with cleaned text files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for tokenizer")
    return parser.parse_args()

# ============================================================
# TOKENIZER CONFIGURATION
# ============================================================
CLEANED_DIR = OUTPUT_DIR / "cleaned_corpus"


def get_corpus_files(input_dir: Path = None) -> list[str]:
    """Get all cleaned text files for tokenizer training."""
    files = []
    
    # If input_dir is specified, use it directly (flat or with splits)
    if input_dir and input_dir.exists():
        # First try direct .txt files in the directory
        direct_files = list(input_dir.glob("*.txt"))
        if direct_files:
            files.extend([str(f) for f in direct_files])
        
        # Also check for splits subdirectories
        for split in ["train", "val", "test"]:
            split_dir = input_dir / split
            if split_dir.exists():
                files.extend([str(f) for f in split_dir.glob("*.txt")])
        
        if files:
            return files
    
    # Use all splits for tokenizer training
    for split in ["train", "val", "test"]:
        split_dir = CLEANED_DIR / split
        if split_dir.exists():
            files.extend([str(f) for f in split_dir.glob("*.txt")])
    
    if not files:
        # Fallback to original data if not cleaned
        from config import DATA_DIR
        for split in ["train", "val", "test"]:
            split_dir = DATA_DIR / split
            if split_dir.exists():
                files.extend([str(f) for f in split_dir.glob("*.txt")])
    
    return files


def create_tokenizer() -> Tokenizer:
    """Create a new BPE tokenizer with proper configuration."""
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Pre-tokenizer: ByteLevel handles any UTF-8 input
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder: ByteLevel to reverse the encoding
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor: Add special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    return tokenizer


def train_tokenizer(tokenizer: Tokenizer, files: list[str]) -> Tokenizer:
    """Train the tokenizer on corpus files."""
    
    # Create trainer with our vocabulary settings
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    
    print(f"\n🔧 Training tokenizer...")
    print(f"   Vocab size:       {VOCAB_SIZE:,}")
    print(f"   Min frequency:    {MIN_FREQUENCY}")
    print(f"   Special tokens:   {SPECIAL_TOKENS}")
    print(f"   Training files:   {len(files):,}")
    
    # Train on files
    tokenizer.train(files, trainer)
    
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer):
    """Save the tokenizer in multiple formats."""
    
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save raw tokenizer.json
    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"\n✅ Saved tokenizer to: {tokenizer_path}")
    
    # Also create HuggingFace-compatible files
    from transformers import PreTrainedTokenizerFast
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tokenizer.model_max_length = 2048
    
    hf_tokenizer.save_pretrained(str(TOKENIZER_DIR))
    print(f"✅ Saved HuggingFace tokenizer to: {TOKENIZER_DIR}")


def test_tokenizer(tokenizer: Tokenizer):
    """Test the tokenizer with sample historical text."""
    
    test_texts = [
        "The connexion between the two parties was not satisfactory.",
        "I shew you the manner in which it was done.",
        "The colour of the leaves in autumn is beautiful.",
        "His behaviour at the assembly was most commendable.",
        "Parliament met in the year 1832 to discuss reform.",
    ]
    
    print("\n" + "=" * 60)
    print("TOKENIZER TEST")
    print("=" * 60)
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        
        print(f"\n📝 Input:  \"{text}\"")
        print(f"   Tokens: {tokens}")
        print(f"   IDs:    {encoded.ids[:15]}{'...' if len(encoded.ids) > 15 else ''}")
        print(f"   Count:  {len(tokens)}")
    
    # Test decode
    decoded = tokenizer.decode(encoded.ids)
    print(f"\n🔄 Decode test:")
    print(f"   Original: \"{test_texts[-1]}\"")
    print(f"   Decoded:  \"{decoded}\"")
    print(f"   Match:    {'✅' if decoded.strip() == test_texts[-1] else '❌'}")


def print_vocab_stats(tokenizer: Tokenizer):
    """Print vocabulary statistics."""
    
    vocab = tokenizer.get_vocab()
    
    print("\n" + "=" * 60)
    print("VOCABULARY STATISTICS")
    print("=" * 60)
    print(f"   Total tokens:     {len(vocab):,}")
    print(f"   Special tokens:   {len(SPECIAL_TOKENS)}")
    
    # Sample vocabulary entries
    print("\n   Sample tokens (alphabetically):")
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[0])
    for token, idx in sorted_tokens[100:110]:
        print(f"      {idx:5d}: '{token}'")
    
    # Check for historical tokens
    historical_samples = ["Ġthe", "Ġand", "Ġof", "ĠParliament", "Ġconnexion", "Ġshew"]
    print("\n   Historical token check:")
    for token in historical_samples:
        if token in vocab:
            print(f"      ✅ '{token}' (id: {vocab[token]})")
        else:
            print(f"      ❌ '{token}' not in vocab")


def main():
    args = parse_args()
    
    # Determine input directory
    input_dir = Path(args.input_dir) if args.input_dir else None
    
    # Determine output tokenizer directory
    global TOKENIZER_DIR
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_output = output_dir / "tokenizer"
    else:
        tokenizer_output = TOKENIZER_DIR
    
    print("=" * 60)
    print("HISTORICAL BPE TOKENIZER TRAINER (1800-1875)")
    print("=" * 60)
    
    if input_dir:
        print(f"\n📂 Input directory: {input_dir}")
    
    # Get training files
    files = get_corpus_files(input_dir)
    
    if not files:
        print("\n❌ No corpus files found!")
        print("   Please run 02_clean_corpus.py first, or ensure data is in data/subset_split/")
        sys.exit(1)
    
    print(f"\n📂 Found {len(files):,} files for training")
    
    # Calculate total size
    total_bytes = sum(Path(f).stat().st_size for f in files)
    print(f"   Total corpus size: {total_bytes / 1e9:.2f} GB")
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Train
    tokenizer = train_tokenizer(tokenizer, files)
    
    # Test
    test_tokenizer(tokenizer)
    
    # Stats
    print_vocab_stats(tokenizer)
    
    # Save
    save_tokenizer_to(tokenizer, tokenizer_output)
    
    print("\n" + "=" * 60)
    print("TOKENIZER TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Run: python src/04_prepare_dataset.py --input_dir {input_dir or 'data/02_cleaned'} --output_dir {args.output_dir or 'outputs'}")


def save_tokenizer_to(tokenizer: Tokenizer, output_dir: Path):
    """Save the tokenizer in multiple formats."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw tokenizer.json
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"\n✅ Saved tokenizer to: {tokenizer_path}")
    
    # Also create HuggingFace-compatible files
    from transformers import PreTrainedTokenizerFast
    
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tokenizer.model_max_length = 2048
    
    hf_tokenizer.save_pretrained(str(output_dir))
    print(f"✅ Saved HuggingFace tokenizer to: {output_dir}")


if __name__ == "__main__":
    main()

