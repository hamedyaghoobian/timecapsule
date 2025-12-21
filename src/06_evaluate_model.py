#!/usr/bin/env python3
"""
06_evaluate_model.py - Diachronic Analysis Evaluation

Evaluates the trained historical LLM for:
- Perplexity on held-out test set
- Semantic similarity analysis
- Word embedding extraction for diachronic studies
- Nearest neighbor analysis

Usage: python src/06_evaluate_model.py --checkpoint PATH
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    LlamaForCausalLM,
    AutoConfig,
)

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TOKENIZER_DIR, DATASET_DIR, CHECKPOINT_DIR, LOG_DIR,
    get_device, get_dtype
)

# ============================================================
# MPS CONFIGURATION
# ============================================================
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Historical LLM")
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--max_samples", type=int, default=1000,
        help="Maximum samples to evaluate"
    )
    
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path: str):
    """Load model and tokenizer from checkpoint."""
    
    device = get_device()
    dtype = get_dtype()
    
    print(f"📂 Loading model from: {checkpoint_path}")
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_path)
    
    # Load model
    model = LlamaForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
        device_map={"": device},
        local_files_only=True,
    )
    model.eval()
    
    print(f"   ✅ Model loaded on {device}")
    
    return model, tokenizer


def compute_perplexity(model, tokenizer, dataset, batch_size: int = 4, max_samples: int = 1000):
    """Compute perplexity on dataset."""
    
    device = get_device()
    
    # Limit samples
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    total_loss = 0
    total_tokens = 0
    
    print(f"\n🔢 Computing perplexity on {len(dataset)} samples...")
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            loss = outputs.loss
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss


def extract_word_embeddings(
    model, 
    tokenizer, 
    words: list[str],
    contexts: list[str] = None
) -> dict[str, np.ndarray]:
    """
    Extract contextual embeddings for words.
    
    For diachronic analysis, we extract embeddings from the last hidden layer.
    """
    
    device = get_device()
    embeddings = {}
    
    if contexts is None:
        # Use simple contexts that position the word
        contexts = [f"The {word} was important." for word in words]
    
    print(f"\n📊 Extracting embeddings for {len(words)} words...")
    
    for word, context in zip(tqdm(words), contexts):
        # Tokenize
        inputs = tokenizer(context, return_tensors="pt").to(device)
        
        # Get hidden states
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )
        
        # Get last hidden state
        last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
        
        # Find word token position
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        input_tokens = inputs["input_ids"][0].tolist()
        
        # Find first occurrence of word tokens in input
        word_start = -1
        for i in range(len(input_tokens) - len(word_tokens) + 1):
            if input_tokens[i:i+len(word_tokens)] == word_tokens:
                word_start = i
                break
        
        if word_start >= 0:
            # Average embeddings across word tokens
            word_embedding = last_hidden[0, word_start:word_start+len(word_tokens)].mean(dim=0)
            embeddings[word] = word_embedding.cpu().numpy()
        else:
            print(f"   Warning: Could not find '{word}' in context")
    
    return embeddings


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def analyze_semantic_relationships(embeddings: dict[str, np.ndarray]):
    """Analyze semantic relationships between embeddings."""
    
    words = list(embeddings.keys())
    n = len(words)
    
    if n < 2:
        print("   Need at least 2 words for similarity analysis")
        return
    
    print(f"\n📈 Semantic Similarity Matrix:")
    print(f"   {'':12}", end="")
    for w in words[:5]:  # Limit display
        print(f"{w[:10]:12}", end="")
    print()
    
    for i, w1 in enumerate(words[:5]):
        print(f"   {w1[:10]:12}", end="")
        for j, w2 in enumerate(words[:5]):
            if i <= j:
                sim = compute_cosine_similarity(embeddings[w1], embeddings[w2])
                print(f"{sim:12.3f}", end="")
            else:
                print(f"{'':12}", end="")
        print()


def find_nearest_neighbors(
    target_word: str,
    embeddings: dict[str, np.ndarray],
    k: int = 5
) -> list[tuple[str, float]]:
    """Find k nearest neighbors to a target word."""
    
    if target_word not in embeddings:
        return []
    
    target_emb = embeddings[target_word]
    similarities = []
    
    for word, emb in embeddings.items():
        if word != target_word:
            sim = compute_cosine_similarity(target_emb, emb)
            similarities.append((word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def run_diachronic_demo(model, tokenizer):
    """
    Demonstrate diachronic analysis capabilities.
    
    Shows how word embeddings can capture historical semantic differences.
    """
    
    print("\n" + "=" * 60)
    print("DIACHRONIC ANALYSIS DEMO")
    print("=" * 60)
    
    # Historical words with known semantic shifts
    historical_words = [
        "gay",           # Originally meant "happy, carefree"
        "awful",         # Originally meant "awe-inspiring"
        "nice",          # Originally meant "ignorant, foolish"
        "meat",          # Originally meant "food" in general
        "want",          # Originally meant "lack"
        "virtue",        # Moral excellence
        "commerce",      # Trade
        "parliament",    # Government
        "science",       # Knowledge (broader meaning in 19th century)
        "manufacture",   # Literally "made by hand"
    ]
    
    # Historical contexts
    historical_contexts = [
        "The gay festivities brought joy to all.",
        "The awful majesty of the cathedral inspired reverence.",
        "His nice distinction between the two was quite foolish.",
        "The meat and drink were provided generously.",
        "The people want for basic necessities.",
        "Her virtue was beyond reproach.",
        "The commerce between nations flourishes.",
        "Parliament convened to discuss the matter.",
        "He devoted himself to natural science.",
        "The manufacture of cotton was done by hand.",
    ]
    
    # Extract embeddings
    embeddings = extract_word_embeddings(
        model, tokenizer,
        historical_words,
        historical_contexts
    )
    
    if len(embeddings) < 2:
        print("   Not enough embeddings extracted for analysis")
        return
    
    # Similarity analysis
    analyze_semantic_relationships(embeddings)
    
    # Nearest neighbors for key words
    print("\n🔍 Nearest Neighbors Analysis:")
    for word in ["gay", "awful", "nice"]:
        if word in embeddings:
            neighbors = find_nearest_neighbors(word, embeddings)
            print(f"\n   '{word}':")
            for neighbor, sim in neighbors:
                print(f"      {neighbor}: {sim:.3f}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("HISTORICAL LLM EVALUATION")
    print("Diachronic Analysis Toolkit")
    print("=" * 60)
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(str(checkpoint_path))
    
    # Load dataset
    if DATASET_DIR.exists():
        ds = load_from_disk(str(DATASET_DIR))
        
        # Compute perplexity on test set
        if "test" in ds:
            ppl, loss = compute_perplexity(
                model, tokenizer, ds["test"],
                batch_size=args.batch_size,
                max_samples=args.max_samples
            )
            print(f"\n📊 Test Set Metrics:")
            print(f"   Perplexity: {ppl:.2f}")
            print(f"   Avg Loss:   {loss:.4f}")
    else:
        print(f"\n⚠️  Dataset not found at {DATASET_DIR}")
        print("   Skipping perplexity evaluation")
    
    # Run diachronic demo
    run_diachronic_demo(model, tokenizer)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    print("\n📋 This model can be used for:")
    print("   • Semantic shift detection across time periods")
    print("   • Historical word sense disambiguation")
    print("   • Period-specific text understanding")
    print("   • Comparative linguistic analysis")


if __name__ == "__main__":
    main()
