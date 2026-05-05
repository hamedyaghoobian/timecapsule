#!/usr/bin/env python3
"""
Module 2: Diachronic Semantics (Core Finding)
=============================================
Analysis C: Nearest Neighbor Extraction (Cleaned)
Analysis D: Semantic Axis Projection

Visualizes how word meanings have changed between Victorian and modern era.
"""

import os
import sys
import json
import csv
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Local model path fallback
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "run_20251219_212817"
HF_MODEL_ID = "haykgrigorian/TimeCapsuleLLM-v2-1800-1875"


# ============================================================
# ANALYSIS C: CLEANED NEAREST NEIGHBORS
# ============================================================

TARGET_WORDS = ["labor", "time", "poor", "machine", "woman", "nature", "empire"]

# Historical contexts to elicit Victorian meanings (lowercase for tokenizer matching)
HISTORICAL_CONTEXTS = {
    "labor": "The labor of the working classes in the manufactories was arduous.",
    "time": "The time of the working day was determined by the factory bell.",
    "poor": "The poor deserve our charity and compassion, for they suffer greatly.",
    "machine": "The machine revolutionized the production of cotton in the mills.",
    "woman": "The woman kept the house and raised the children in virtue.",
    "nature": "The nature of man is to seek improvement and progress in all things.",
    "empire": "The British empire extends her dominion over vast territories.",
}


def is_clean_token(token_str, model_type="causal"):
    """Filter out special tokens, punctuation, and sub-word fragments."""
    # Reject empty or whitespace-only
    if not token_str or not token_str.strip():
        return False
    
    token_str = token_str.strip()
    
    # Reject special tokens
    special_tokens = [
        "<pad>", "<s>", "</s>", "<unk>", "<|endoftext|>", 
        "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]",
        "<|year|>", "<|title|>", "▁"
    ]
    if token_str in special_tokens:
        return False
    
    # Reject single punctuation
    if len(token_str) == 1 and not token_str.isalnum():
        return False
    
    # Reject single characters (except 'a', 'I')
    if len(token_str) == 1 and token_str.lower() not in ['a', 'i']:
        return False
    
    # For causal LM (BPE), be very strict about what constitutes a real word
    if model_type == "causal":
        # Must be at least 4 characters
        if len(token_str) < 4:
            return False
        
        # Must be all alphabetic (a real word)
        if not token_str.isalpha():
            return False
        
        # Must have at least one vowel
        vowels = set('aeiouAEIOU')
        if not any(c in vowels for c in token_str):
            return False
        
        # Reject tokens that start with lowercase but have random caps (likely fragments)
        if token_str[0].islower() and any(c.isupper() for c in token_str[1:]):
            return False
        
        # Reject common subword fragments
        fragment_patterns = [
            'aker', 'acher', 'imper', 'arius', 'iche', 'ommod', 'iends',
            'ifts', 'ces', 'rel', 'ches', 'ness', 'ment', 'tion',
        ]
        if token_str.lower() in fragment_patterns:
            return False
    else:
        # BERT is simpler - just reject short tokens
        if len(token_str) < 3:
            return False
    
    # Reject weird artifacts
    if "(nan)" in token_str.lower() or "nan" == token_str.lower():
        return False
    
    # Reject tokens that are just numbers
    if token_str.isdigit():
        return False
    
    # Reject subword markers
    if token_str.startswith("##") or token_str.startswith("Ġ"):
        return False
    
    return True


def load_timecapsule_model():
    """Load TimeCapsule model from HuggingFace or local checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    
    # Try HuggingFace first
    try:
        print(f"   Trying HuggingFace: {HF_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"   ✓ Loaded from HuggingFace")
        return model, tokenizer, "causal"
    except Exception as e:
        print(f"   HuggingFace failed: {e}")
    
    # Fallback to local checkpoint
    if LOCAL_MODEL_PATH.exists():
        print(f"   Trying local checkpoint: {LOCAL_MODEL_PATH}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(LOCAL_MODEL_PATH))
        model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_MODEL_PATH),
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        print(f"   ✓ Loaded from local checkpoint")
        return model, tokenizer, "causal"
    
    raise RuntimeError("Could not load TimeCapsule model from HuggingFace or local checkpoint")


def get_word_embedding_from_model(model, tokenizer, word, context=None, model_type="causal"):
    """
    Extract word embedding from a model.
    For causal LMs, uses the embedding layer directly.
    For BERT-style, uses contextual embeddings.
    """
    device = next(model.parameters()).device
    
    # Use lowercase for matching
    word_lower = word.lower()
    
    if context is None:
        context = f"The {word_lower} is important."
    else:
        # Ensure word appears in lowercase in context
        context = context.replace(word, word_lower).replace(word.upper(), word_lower)
    
    # Tokenize
    inputs = tokenizer(context, return_tensors="pt")
    # Remove token_type_ids if present (not used by Llama models)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
    
    with torch.no_grad():
        if model_type == "bert":
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
        else:
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
    
    input_ids = inputs["input_ids"][0].tolist()
    
    # Try multiple tokenization strategies for BPE tokenizers
    # Strategy 1: Try with space prefix (how word appears mid-sentence)
    word_tokens = tokenizer.encode(f" {word_lower}", add_special_tokens=False)
    
    # Find the word in the input
    word_start = -1
    for i in range(len(input_ids) - len(word_tokens) + 1):
        if input_ids[i:i+len(word_tokens)] == word_tokens:
            word_start = i
            break
    
    # Strategy 2: Try without space prefix
    if word_start < 0:
        word_tokens = tokenizer.encode(word_lower, add_special_tokens=False)
        for i in range(len(input_ids) - len(word_tokens) + 1):
            if input_ids[i:i+len(word_tokens)] == word_tokens:
                word_start = i
                break
    
    # Strategy 3: Look for the decoded word in each token
    if word_start < 0:
        for i, tok_id in enumerate(input_ids):
            tok_str = tokenizer.decode([tok_id]).strip().lower()
            if tok_str == word_lower:
                word_start = i
                word_tokens = [tok_id]
                break
    
    if word_start >= 0:
        # Average over word tokens
        word_emb = hidden_states[0, word_start:word_start+len(word_tokens)].mean(dim=0)
        return word_emb.cpu().numpy()
    
    return None


def find_nearest_neighbors_in_vocab(model, tokenizer, target_word, context=None, k=10, model_type="causal"):
    """
    Find nearest neighbors by comparing to all vocabulary embeddings.
    """
    device = next(model.parameters()).device
    
    # Get target word embedding
    target_emb = get_word_embedding_from_model(model, tokenizer, target_word, context, model_type)
    if target_emb is None:
        print(f"   Warning: Could not extract embedding for '{target_word}'")
        return []
    
    target_emb = torch.tensor(target_emb).to(device)
    
    # Get embedding matrix
    if model_type == "bert":
        embedding_matrix = model.embeddings.word_embeddings.weight
    else:
        embedding_matrix = model.model.embed_tokens.weight
    
    # Compute cosine similarities
    target_norm = target_emb / target_emb.norm()
    emb_norms = embedding_matrix / embedding_matrix.norm(dim=1, keepdim=True)
    
    similarities = torch.matmul(emb_norms, target_norm)
    
    # Get top-k indices (get many more to filter through BPE fragments)
    top_k = min(k * 100, len(similarities))  # Get extra to filter
    scores, indices = torch.topk(similarities, top_k)
    
    # Filter and collect clean neighbors
    neighbors = []
    for score, idx in zip(scores, indices):
        token_str = tokenizer.decode([idx.item()]).strip()
        
        # Apply cleaning filter with model type
        if is_clean_token(token_str, model_type):
            # Skip if too similar to target (same word)
            if target_word.lower() in token_str.lower() or token_str.lower() in target_word.lower():
                continue
            
            neighbors.append({
                "word": token_str,
                "score": score.item()
            })
        
        if len(neighbors) >= k:
            break
    
    return neighbors


def run_neighbor_analysis(output_dir):
    """
    Extract cleaned nearest neighbors for target words.
    Compares TimeCapsule vs BERT.
    """
    from transformers import BertModel, BertTokenizer
    
    results = {}
    
    # ---- TimeCapsule Model ----
    print("\n" + "="*60)
    print("Loading TimeCapsule Model for Neighbor Analysis")
    print("="*60)
    
    try:
        tc_model, tc_tokenizer, tc_type = load_timecapsule_model()
        
        if not torch.cuda.is_available():
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            tc_model = tc_model.to(device)
        tc_model.eval()
        
        print("\nExtracting neighbors for target words...")
        tc_neighbors = {}
        for word in tqdm(TARGET_WORDS):
            context = HISTORICAL_CONTEXTS.get(word.lower())
            neighbors = find_nearest_neighbors_in_vocab(
                tc_model, tc_tokenizer, word.lower(), context, k=5, model_type=tc_type
            )
            tc_neighbors[word.upper()] = neighbors
            print(f"   {word.upper()}: {[n['word'] for n in neighbors[:3]]}")
        
        results["TimeCapsule"] = tc_neighbors
        
        del tc_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"✗ Error with TimeCapsule: {e}")
        import traceback
        traceback.print_exc()
        results["TimeCapsule"] = {"error": str(e)}
    
    # ---- BERT Baseline ----
    print("\n" + "="*60)
    print("Loading BERT-base-uncased for Comparison")
    print("="*60)
    
    try:
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                             "cuda" if torch.cuda.is_available() else "cpu")
        bert_model = bert_model.to(device)
        bert_model.eval()
        
        print("\nExtracting neighbors for target words...")
        bert_neighbors = {}
        for word in tqdm(TARGET_WORDS):
            # Use neutral context for BERT (modern baseline)
            context = f"The {word.lower()} is significant."
            neighbors = find_nearest_neighbors_in_vocab(
                bert_model, bert_tokenizer, word.lower(), context, k=5, model_type="bert"
            )
            bert_neighbors[word.upper()] = neighbors
            print(f"   {word.upper()}: {[n['word'] for n in neighbors[:3]]}")
        
        results["BERT"] = bert_neighbors
        
    except Exception as e:
        print(f"✗ Error with BERT: {e}")
        results["BERT"] = {"error": str(e)}
    
    # Save as JSON
    json_path = output_dir / "neighbor_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save as clean CSV
    csv_path = output_dir / "neighbor_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Word", "TimeCapsule_Top5", "BERT_Top5"])
        
        for word in TARGET_WORDS:
            word_key = word.upper()
            tc_top5 = results.get("TimeCapsule", {}).get(word_key, [])
            bert_top5 = results.get("BERT", {}).get(word_key, [])
            
            tc_str = ", ".join([f"{n['word']} ({n['score']:.2f})" for n in tc_top5[:5]])
            bert_str = ", ".join([f"{n['word']} ({n['score']:.2f})" for n in bert_top5[:5]])
            
            writer.writerow([word_key, tc_str, bert_str])
    
    print(f"\n📊 Results saved to:")
    print(f"   {json_path}")
    print(f"   {csv_path}")
    
    return results


# ============================================================
# ANALYSIS D: SEMANTIC AXIS PROJECTION
# ============================================================

def compute_semantic_axis(model, tokenizer, pole_a, pole_b, context_template, model_type="causal"):
    """
    Compute a semantic axis vector from pole_a to pole_b.
    Returns the normalized direction vector.
    """
    # Get embeddings for both poles
    context_a = context_template.format(word=pole_a)
    context_b = context_template.format(word=pole_b)
    
    emb_a = get_word_embedding_from_model(model, tokenizer, pole_a, context_a, model_type)
    emb_b = get_word_embedding_from_model(model, tokenizer, pole_b, context_b, model_type)
    
    if emb_a is None or emb_b is None:
        return None
    
    # Compute axis direction
    axis = emb_b - emb_a
    axis = axis / np.linalg.norm(axis)
    
    return axis, emb_a, emb_b


def project_onto_axis(embedding, axis, origin):
    """
    Project an embedding onto a semantic axis.
    Returns the scalar projection (distance along axis from origin).
    """
    diff = embedding - origin
    projection = np.dot(diff, axis)
    return projection


def run_semantic_axis_analysis(output_dir):
    """
    Project TARGET word "TIME" onto the Nature-Factory axis.
    Compare TimeCapsule vs BERT.
    """
    from transformers import BertModel, BertTokenizer
    
    results = {}
    
    # Define the axis
    POLE_A = "nature"    # One end
    POLE_B = "factory"   # Other end
    TARGET = "time"
    
    context_template = "The {word} is central to our understanding."
    
    print("\n" + "="*60)
    print("SEMANTIC AXIS ANALYSIS: Nature ←→ Factory")
    print("="*60)
    print(f"Target word: {TARGET.upper()}")
    
    # ---- TimeCapsule ----
    print("\n--- TimeCapsule Model ---")
    try:
        tc_model, tc_tokenizer, tc_type = load_timecapsule_model()
        
        if not torch.cuda.is_available():
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            tc_model = tc_model.to(device)
        tc_model.eval()
        
        # Compute axis
        axis_result = compute_semantic_axis(
            tc_model, tc_tokenizer, POLE_A, POLE_B, context_template, tc_type
        )
        
        if axis_result:
            axis, origin, _ = axis_result
            
            # Get target embedding
            target_context = HISTORICAL_CONTEXTS.get("TIME", f"The {TARGET} is important.")
            target_emb = get_word_embedding_from_model(
                tc_model, tc_tokenizer, TARGET, target_context, tc_type
            )
            
            if target_emb is not None:
                projection = project_onto_axis(target_emb, axis, origin)
                results["TimeCapsule"] = {
                    "target": TARGET,
                    "axis": f"{POLE_A} → {POLE_B}",
                    "projection": float(projection),
                    "interpretation": "Closer to Factory" if projection > 0 else "Closer to Nature"
                }
                print(f"   TIME projection: {projection:.4f}")
                print(f"   Interpretation: {results['TimeCapsule']['interpretation']}")
        
        del tc_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"✗ Error: {e}")
        results["TimeCapsule"] = {"error": str(e)}
    
    # ---- BERT ----
    print("\n--- BERT Model ---")
    try:
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                             "cuda" if torch.cuda.is_available() else "cpu")
        bert_model = bert_model.to(device)
        bert_model.eval()
        
        # Compute axis
        axis_result = compute_semantic_axis(
            bert_model, bert_tokenizer, POLE_A, POLE_B, context_template, "bert"
        )
        
        if axis_result:
            axis, origin, _ = axis_result
            
            # Get target embedding
            target_emb = get_word_embedding_from_model(
                bert_model, bert_tokenizer, TARGET, f"The {TARGET} is important.", "bert"
            )
            
            if target_emb is not None:
                projection = project_onto_axis(target_emb, axis, origin)
                results["BERT"] = {
                    "target": TARGET,
                    "axis": f"{POLE_A} → {POLE_B}",
                    "projection": float(projection),
                    "interpretation": "Closer to Factory" if projection > 0 else "Closer to Nature"
                }
                print(f"   TIME projection: {projection:.4f}")
                print(f"   Interpretation: {results['BERT']['interpretation']}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        results["BERT"] = {"error": str(e)}
    
    # Save results
    output_path = output_dir / "semantic_axis_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to: {output_path}")
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("MODULE 2: DIACHRONIC SEMANTICS")
    print("Time Capsule Paper - ACM C&C 2026")
    print("="*60)
    
    # Setup output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Run analyses
    print("\n" + "="*60)
    print("ANALYSIS C: NEAREST NEIGHBOR EXTRACTION (CLEANED)")
    print("="*60)
    neighbor_results = run_neighbor_analysis(output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS D: SEMANTIC AXIS PROJECTION")
    print("="*60)
    axis_results = run_semantic_axis_analysis(output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("MODULE 2 SUMMARY")
    print("="*60)
    
    print("\nNearest Neighbors (TimeCapsule vs BERT):")
    for word in TARGET_WORDS:
        word_key = word.upper()
        tc = neighbor_results.get("TimeCapsule", {}).get(word_key, [])
        bert = neighbor_results.get("BERT", {}).get(word_key, [])
        print(f"\n   {word_key}:")
        print(f"      TimeCapsule: {', '.join([n['word'] for n in tc[:3]])}")
        print(f"      BERT:        {', '.join([n['word'] for n in bert[:3]])}")
    
    print("\n\nSemantic Axis (TIME on Nature→Factory):")
    for model in ["TimeCapsule", "BERT"]:
        if model in axis_results and "projection" in axis_results[model]:
            print(f"   {model}: {axis_results[model]['projection']:.4f} ({axis_results[model]['interpretation']})")
    
    print(f"\n✅ All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
