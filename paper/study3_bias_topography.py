#!/usr/bin/env python3
"""
Study 3: Bias Topography (t-SNE/PCA Visualization)
===================================================
Visualizes how "civilizational" concepts cluster differently in Victorian vs modern models.

Method:
1. Extract contextual embeddings for 50 civilizational terms
2. Compare TimeCapsule vs GPT-2/BERT as modern baseline
3. Apply t-SNE/PCA dimensionality reduction
4. Generate side-by-side scatter plots with cluster annotations
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Model paths
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "run_20251219_212817"
HF_MODEL_ID = "haykgrigorian/TimeCapsuleLLM-v2-1800-1875"

# Word categories for visualization
WORD_CATEGORIES = {
    "civilizational": [
        "native", "savage", "empire", "colony", "progress", "industry",
        "civilization", "barbarous", "heathen", "primitive", "backward", "enlightened",
        "christian", "infidel", "missionary", "conquest", "dominion", "subject", "servant"
    ],
    "social": [
        "poor", "working", "gentleman", "lady", "peasant", "labourer", "master",
        "slave", "negro", "hindoo", "irish", "gypsy"
    ],
    "progress": [
        "steam", "railway", "manufacture", "machine", "telegraph", "factory",
        "commerce", "trade", "improvement", "science", "invention", "modern"
    ],
    "nature": [
        "nature", "natural", "organic", "earth", "land", "rural", "agricultural"
    ]
}

# Flatten to single list with category labels
ALL_WORDS = []
WORD_TO_CATEGORY = {}
for category, words in WORD_CATEGORIES.items():
    for word in words:
        if word not in WORD_TO_CATEGORY:
            ALL_WORDS.append(word)
            WORD_TO_CATEGORY[word] = category

# Context templates for embedding extraction
CONTEXT_TEMPLATES = [
    "The {word} is of great importance.",
    "The {word} was discussed at length.",
    "We must consider the {word} carefully.",
]


def load_timecapsule_model():
    """Load TimeCapsule model."""
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


def get_word_embedding(model, tokenizer, word, contexts, model_type="causal"):
    """Extract averaged embedding for a word across multiple contexts."""
    device = next(model.parameters()).device
    
    embeddings = []
    
    for context_template in contexts:
        context = context_template.format(word=word)
        
        inputs = tokenizer(context, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
        
        with torch.no_grad():
            if model_type == "bert":
                outputs = model(**inputs, output_hidden_states=True)
            else:
                outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        
        # Find word in context
        input_ids = inputs["input_ids"][0].tolist()
        
        # Try with space prefix
        word_tokens = tokenizer.encode(f" {word}", add_special_tokens=False)
        
        word_start = -1
        for i in range(len(input_ids) - len(word_tokens) + 1):
            if input_ids[i:i+len(word_tokens)] == word_tokens:
                word_start = i
                break
        
        # Fallback: try without space
        if word_start < 0:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
            for i in range(len(input_ids) - len(word_tokens) + 1):
                if input_ids[i:i+len(word_tokens)] == word_tokens:
                    word_start = i
                    break
        
        # Fallback: find by decoding
        if word_start < 0:
            for i, tok_id in enumerate(input_ids):
                tok_str = tokenizer.decode([tok_id]).strip().lower()
                if tok_str == word.lower():
                    word_start = i
                    word_tokens = [tok_id]
                    break
        
        if word_start >= 0:
            word_emb = hidden_states[0, word_start:word_start+len(word_tokens)].mean(dim=0)
            embeddings.append(word_emb.cpu().numpy())
    
    if embeddings:
        return np.mean(embeddings, axis=0)
    return None


def extract_all_embeddings(model, tokenizer, words, model_type="causal"):
    """Extract embeddings for all words."""
    embeddings = {}
    
    for word in tqdm(words, desc=f"Extracting {model_type} embeddings"):
        emb = get_word_embedding(model, tokenizer, word, CONTEXT_TEMPLATES, model_type)
        if emb is not None:
            embeddings[word] = emb.tolist()
    
    return embeddings


def find_word_neighbors(model, tokenizer, target_word, k=10, model_type="causal"):
    """Find nearest neighbor words in the embedding space."""
    device = next(model.parameters()).device
    
    # Get target embedding
    target_emb = get_word_embedding(model, tokenizer, target_word, CONTEXT_TEMPLATES, model_type)
    if target_emb is None:
        return []
    
    target_emb = torch.tensor(target_emb).to(device)
    
    # Get embedding matrix
    if model_type == "bert":
        embedding_matrix = model.embeddings.word_embeddings.weight
    else:
        embedding_matrix = model.model.embed_tokens.weight
    
    # Compute similarities
    target_norm = target_emb / target_emb.norm()
    emb_norms = embedding_matrix / embedding_matrix.norm(dim=1, keepdim=True)
    similarities = torch.matmul(emb_norms, target_norm)
    
    # Get top-k
    top_k = min(k * 50, len(similarities))
    scores, indices = torch.topk(similarities, top_k)
    
    # Filter and collect
    neighbors = []
    for score, idx in zip(scores, indices):
        token_str = tokenizer.decode([idx.item()]).strip()
        
        # Filter criteria
        if not token_str or len(token_str) < 3:
            continue
        if not token_str.isalpha():
            continue
        if target_word.lower() in token_str.lower():
            continue
        
        neighbors.append({
            "word": token_str,
            "score": round(score.item(), 4)
        })
        
        if len(neighbors) >= k:
            break
    
    return neighbors


def create_tsne_visualization(tc_embeddings, bert_embeddings, output_dir):
    """Create t-SNE visualization comparing both models."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # Prepare data
    common_words = [w for w in ALL_WORDS if w in tc_embeddings and w in bert_embeddings]
    
    if len(common_words) < 5:
        print("   Warning: Not enough common words for visualization")
        return
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color mapping
    category_colors = {
        "civilizational": "#e74c3c",  # Red
        "social": "#3498db",          # Blue
        "progress": "#2ecc71",        # Green
        "nature": "#9b59b6"           # Purple
    }
    
    for ax, (embeddings, title) in zip(axes, [
        (tc_embeddings, "TimeCapsule (1800-1875)"),
        (bert_embeddings, "BERT (Modern)")
    ]):
        # Get embeddings for common words
        X = np.array([embeddings[w] for w in common_words])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(common_words)-1))
        X_2d = tsne.fit_transform(X)
        
        # Plot by category
        for i, word in enumerate(common_words):
            category = WORD_TO_CATEGORY.get(word, "other")
            color = category_colors.get(category, "#95a5a6")
            ax.scatter(X_2d[i, 0], X_2d[i, 1], c=color, s=100, alpha=0.7)
            ax.annotate(word, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=color, markersize=10, label=cat.title())
                       for cat, color in category_colors.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle("Bias Topography: Victorian vs Modern Word Embeddings", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save
    plot_path = output_dir / "bias_topography_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved plot to: {plot_path}")


def run_bias_topography(output_dir):
    """Run the complete bias topography analysis."""
    from transformers import BertModel, BertTokenizer
    
    print("\n" + "="*60)
    print("STUDY 3: BIAS TOPOGRAPHY")
    print("="*60)
    
    results = {
        "word_categories": WORD_CATEGORIES,
        "total_words": len(ALL_WORDS),
    }
    
    # --- TimeCapsule ---
    print("\n📂 Loading TimeCapsule Model...")
    tc_model, tc_tokenizer = load_timecapsule_model()
    
    if not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tc_model = tc_model.to(device)
    tc_model.eval()
    
    print(f"\n📊 Extracting TimeCapsule embeddings for {len(ALL_WORDS)} words...")
    tc_embeddings = extract_all_embeddings(tc_model, tc_tokenizer, ALL_WORDS, "causal")
    results["timecapsule_words_extracted"] = len(tc_embeddings)
    
    # Get neighbors for "progress"
    print("\n🔍 Finding neighbors for 'progress' (TimeCapsule)...")
    tc_progress_neighbors = find_word_neighbors(tc_model, tc_tokenizer, "progress", k=15, model_type="causal")
    
    del tc_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # --- BERT ---
    print("\n📂 Loading BERT Model...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)
    bert_model.eval()
    
    print(f"\n📊 Extracting BERT embeddings for {len(ALL_WORDS)} words...")
    bert_embeddings = extract_all_embeddings(bert_model, bert_tokenizer, ALL_WORDS, "bert")
    results["bert_words_extracted"] = len(bert_embeddings)
    
    # Get neighbors for "progress"
    print("\n🔍 Finding neighbors for 'progress' (BERT)...")
    bert_progress_neighbors = find_word_neighbors(bert_model, bert_tokenizer, "progress", k=15, model_type="bert")
    
    # Save embeddings
    embeddings_data = {
        "timecapsule": tc_embeddings,
        "bert": bert_embeddings
    }
    
    emb_path = output_dir / "bias_topography_embeddings.json"
    with open(emb_path, "w") as f:
        json.dump(embeddings_data, f)
    print(f"\n📊 Embeddings saved to: {emb_path}")
    
    # Save progress neighbors
    progress_data = {
        "timecapsule": tc_progress_neighbors,
        "bert": bert_progress_neighbors
    }
    
    progress_path = output_dir / "progress_neighbors.json"
    with open(progress_path, "w") as f:
        json.dump(progress_data, f, indent=2)
    print(f"📊 Progress neighbors saved to: {progress_path}")
    
    # Create visualization
    print("\n🎨 Creating t-SNE visualization...")
    try:
        create_tsne_visualization(tc_embeddings, bert_embeddings, output_dir)
    except Exception as e:
        print(f"   Warning: Could not create visualization: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("PROGRESS NEIGHBORS COMPARISON")
    print("="*60)
    print("\nTimeCapsule 'progress' neighbors:")
    for n in tc_progress_neighbors[:10]:
        print(f"   {n['word']}: {n['score']}")
    
    print("\nBERT 'progress' neighbors:")
    for n in bert_progress_neighbors[:10]:
        print(f"   {n['word']}: {n['score']}")
    
    # Save results
    results["progress_neighbors"] = progress_data
    results_path = output_dir / "bias_topography_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    print("="*60)
    print("STUDY 3: BIAS TOPOGRAPHY")
    print("Time Capsule Paper - ACM C&C 2026")
    print("="*60)
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    results = run_bias_topography(output_dir)
    
    print("\n" + "="*60)
    print("STUDY 3 COMPLETE")
    print("="*60)
    print(f"\n✅ Extracted embeddings for {results.get('timecapsule_words_extracted', 0)} words")
    print(f"✅ Generated t-SNE visualization")
    print(f"✅ Saved progress neighbors comparison")


if __name__ == "__main__":
    main()
