#!/usr/bin/env python3
"""
Study 1: N-Gram Overlap / Memorization Test
============================================
Proves the model synthesizes Victorian style rather than memorizing training data.

Method:
- Generate 100 diverse sentences from the model
- Extract all 5-grams from generated text
- Compare against 5-grams from training corpus
- Calculate overlap percentage

Target result: Low overlap (<5%) proves generalization.
"""

import os
import sys
import json
import random
import torch
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from datasets import load_from_disk

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Model paths
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "run_20251219_212817"
HF_MODEL_ID = "haykgrigorian/TimeCapsuleLLM-v2-1800-1875"
DATASET_DIR = PROJECT_ROOT / "outputs" / "dataset"

# Diverse prompts to generate varied content
GENERATION_PROMPTS = [
    "The Parliament convened to discuss",
    "In the year eighteen hundred and",
    "The railway has transformed",
    "The working classes of England",
    "The factory system has produced",
    "The condition of the poor",
    "A gentleman of considerable means",
    "The British Empire extends",
    "Science has demonstrated that",
    "The natural philosophy of",
    "Commerce and trade have",
    "The agricultural labourers",
    "The city of London presents",
    "The steam engine has",
    "The manufacturing districts",
    "Religion and morality require",
    "The education of the young",
    "The laws of nature",
    "The principles of political economy",
    "The history of England shows",
    "The condition of Ireland",
    "The colonies require",
    "The progress of civilization",
    "The duties of a Christian",
    "The improvement of the land",
    "The state of the nation",
    "The character of the English",
    "The customs and manners",
    "The government has determined",
    "The ancient constitution",
    "In the matter of",
    "The question before us",
    "It is well known that",
    "The evidence clearly shows",
    "We may therefore conclude",
    "The following observations",
    "The result of this",
    "It appears from the",
    "The effect of this",
    "According to the principles",
]


def load_model():
    """Load TimeCapsule model from HuggingFace or local checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    
    try:
        print(f"   Trying HuggingFace: {HF_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"   ✓ Loaded from HuggingFace")
        return model, tokenizer
    except Exception as e:
        print(f"   HuggingFace failed: {e}")
    
    if LOCAL_MODEL_PATH.exists():
        print(f"   Trying local checkpoint: {LOCAL_MODEL_PATH}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(LOCAL_MODEL_PATH))
        model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_MODEL_PATH),
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        print(f"   ✓ Loaded from local checkpoint")
        return model, tokenizer
    
    raise RuntimeError("Could not load TimeCapsule model")


def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    """Generate text continuation from a prompt."""
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def extract_ngrams(text, n=5):
    """Extract n-grams from text."""
    # Tokenize by words
    words = text.lower().split()
    
    # Remove punctuation from words
    cleaned_words = []
    for word in words:
        cleaned = ''.join(c for c in word if c.isalnum())
        if cleaned:
            cleaned_words.append(cleaned)
    
    # Extract n-grams
    ngrams = []
    for i in range(len(cleaned_words) - n + 1):
        ngram = tuple(cleaned_words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams


def load_training_corpus_ngrams(tokenizer, max_samples=1000, n=5):
    """Load training corpus and extract all n-grams."""
    print(f"\n📂 Loading training corpus n-grams...")
    
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_DIR}")
    
    ds = load_from_disk(str(DATASET_DIR))
    
    if "train" not in ds:
        raise ValueError("No 'train' split in dataset")
    
    train_data = ds["train"]
    
    # Sample if dataset is large
    indices = list(range(len(train_data)))
    if len(indices) > max_samples:
        indices = random.sample(indices, max_samples)
    
    all_ngrams = set()
    
    for idx in tqdm(indices, desc="Extracting corpus n-grams"):
        input_ids = train_data[idx]["input_ids"]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        ngrams = extract_ngrams(text, n)
        all_ngrams.update(ngrams)
    
    print(f"   Extracted {len(all_ngrams):,} unique {n}-grams from {len(indices)} training samples")
    
    return all_ngrams


def run_memorization_test(output_dir):
    """Run the complete memorization test."""
    print("\n" + "="*60)
    print("STUDY 1: N-GRAM OVERLAP / MEMORIZATION TEST")
    print("="*60)
    
    # Load model
    print("\n📂 Loading model...")
    model, tokenizer = load_model()
    
    if not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   Model loaded on {next(model.parameters()).device}")
    
    # Load training corpus n-grams
    corpus_ngrams = load_training_corpus_ngrams(tokenizer, max_samples=500, n=5)
    
    # Generate diverse sentences
    print(f"\n📝 Generating 100 sentences...")
    
    generated_texts = []
    all_generated_ngrams = []
    
    # Use varied prompts (cycle through if needed)
    prompts_to_use = (GENERATION_PROMPTS * 3)[:100]  # Ensure we have 100
    
    for i, prompt in enumerate(tqdm(prompts_to_use, desc="Generating")):
        text = generate_text(model, tokenizer, prompt, max_new_tokens=80)
        generated_texts.append({
            "prompt": prompt,
            "generation": text
        })
        
        ngrams = extract_ngrams(text, n=5)
        all_generated_ngrams.extend(ngrams)
    
    # Calculate overlap
    print(f"\n📊 Calculating n-gram overlap...")
    
    generated_ngram_set = set(all_generated_ngrams)
    overlapping_ngrams = generated_ngram_set.intersection(corpus_ngrams)
    
    total_generated = len(generated_ngram_set)
    total_overlapping = len(overlapping_ngrams)
    overlap_percentage = (total_overlapping / total_generated * 100) if total_generated > 0 else 0
    
    # Also calculate instance-level overlap
    instance_overlaps = sum(1 for ng in all_generated_ngrams if ng in corpus_ngrams)
    instance_total = len(all_generated_ngrams)
    instance_overlap_pct = (instance_overlaps / instance_total * 100) if instance_total > 0 else 0
    
    results = {
        "n_gram_size": 5,
        "num_generations": len(generated_texts),
        "corpus_samples_checked": 500,
        "unique_corpus_ngrams": len(corpus_ngrams),
        "unique_generated_ngrams": total_generated,
        "overlapping_ngrams": total_overlapping,
        "unique_overlap_percentage": round(overlap_percentage, 2),
        "total_generated_ngram_instances": instance_total,
        "overlapping_instances": instance_overlaps,
        "instance_overlap_percentage": round(instance_overlap_pct, 2),
        "interpretation": "LOW - Model is synthesizing, not memorizing" if overlap_percentage < 5 else "HIGH - Possible memorization",
        "sample_overlapping_ngrams": [" ".join(ng) for ng in list(overlapping_ngrams)[:20]],
    }
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"   Unique 5-grams in corpus:     {len(corpus_ngrams):,}")
    print(f"   Unique 5-grams generated:     {total_generated:,}")
    print(f"   Overlapping 5-grams:          {total_overlapping:,}")
    print(f"   Unique Overlap Rate:          {overlap_percentage:.2f}%")
    print(f"   Instance Overlap Rate:        {instance_overlap_pct:.2f}%")
    print(f"\n   📋 Interpretation: {results['interpretation']}")
    
    if overlapping_ngrams:
        print(f"\n   Sample overlapping 5-grams:")
        for ng in list(overlapping_ngrams)[:5]:
            print(f"      '{' '.join(ng)}'")
    
    # Save results
    output_path = output_dir / "memorization_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to: {output_path}")
    
    # Save generated texts
    texts_path = output_dir / "memorization_test_generations.json"
    with open(texts_path, "w") as f:
        json.dump(generated_texts, f, indent=2)
    print(f"📊 Generations saved to: {texts_path}")
    
    return results


def main():
    print("="*60)
    print("STUDY 1: MEMORIZATION TEST")
    print("Time Capsule Paper - ACM C&C 2026")
    print("="*60)
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    results = run_memorization_test(output_dir)
    
    print("\n" + "="*60)
    print("STUDY 1 COMPLETE")
    print("="*60)
    print(f"\n✅ Key Finding: {results['unique_overlap_percentage']}% n-gram overlap")
    print(f"   {results['interpretation']}")


if __name__ == "__main__":
    main()
