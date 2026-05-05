#!/usr/bin/env python3
"""
Module 1: Quantitative Validation (Engineering Proof)
======================================================
Analysis A: Domain-Specific Perplexity Comparison
Analysis B: Tokenizer Fertility Comparison

Compares TimeCapsule model vs modern baselines on Victorian text.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Local model path fallback
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "run_20251219_212817"
HF_MODEL_ID = "haykgrigorian/TimeCapsuleLLM-v2-1800-1875"

# ============================================================
# ANALYSIS A: PERPLEXITY COMPARISON
# ============================================================

def compute_perplexity(model, tokenizer, texts, batch_size=4, max_length=512):
    """Compute perplexity on a list of texts."""
    device = next(model.parameters()).device
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing PPL"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            
            # Account for padding
            valid_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


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
        return model, tokenizer
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
        return model, tokenizer
    
    raise RuntimeError("Could not load TimeCapsule model from HuggingFace or local checkpoint")


def run_perplexity_analysis(test_texts, output_dir):
    """
    Compare perplexity of TimeCapsule vs Llama-3-8B on Victorian text.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    results = {}
    
    # ---- TimeCapsule Model ----
    print("\n" + "="*60)
    print("Loading TimeCapsule Model")
    print("="*60)
    
    try:
        tc_model, tc_tokenizer = load_timecapsule_model()
        
        if not torch.cuda.is_available():
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            tc_model = tc_model.to(device)
        
        # Ensure pad token
        if tc_tokenizer.pad_token is None:
            tc_tokenizer.pad_token = tc_tokenizer.eos_token
        
        tc_ppl, tc_loss = compute_perplexity(tc_model, tc_tokenizer, test_texts)
        results["TimeCapsule"] = {"perplexity": tc_ppl, "loss": tc_loss}
        print(f"✓ TimeCapsule PPL: {tc_ppl:.2f}")
        
        # Free memory
        del tc_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"✗ Error loading TimeCapsule: {e}")
        results["TimeCapsule"] = {"error": str(e)}
    
    # ---- GPT-2 Baseline (more accessible than Llama-3) ----
    print("\n" + "="*60)
    print("Loading GPT-2 Baseline")
    print("="*60)
    
    try:
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
            
        gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                             "cuda" if torch.cuda.is_available() else "cpu")
        gpt2_model = gpt2_model.to(device)
        
        gpt2_ppl, gpt2_loss = compute_perplexity(gpt2_model, gpt2_tokenizer, test_texts)
        results["GPT-2"] = {"perplexity": gpt2_ppl, "loss": gpt2_loss}
        print(f"✓ GPT-2 PPL: {gpt2_ppl:.2f}")
        
        del gpt2_model
        
    except Exception as e:
        print(f"✗ Error loading GPT-2: {e}")
        results["GPT-2"] = {"error": str(e)}
    
    # ---- Try Mistral-7B if available ----
    print("\n" + "="*60)
    print("Loading Mistral-7B Baseline (if available)")
    print("="*60)
    
    try:
        mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        if mistral_tokenizer.pad_token is None:
            mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
            
        mistral_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            mistral_model = mistral_model.to(device)
        
        mistral_ppl, mistral_loss = compute_perplexity(mistral_model, mistral_tokenizer, test_texts)
        results["Mistral-7B"] = {"perplexity": mistral_ppl, "loss": mistral_loss}
        print(f"✓ Mistral-7B PPL: {mistral_ppl:.2f}")
        
    except Exception as e:
        print(f"✗ Mistral-7B not available: {e}")
        results["Mistral-7B"] = {"error": str(e)}
    
    # Save results
    output_path = output_dir / "perplexity_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to: {output_path}")
    
    return results


# ============================================================
# ANALYSIS B: TOKENIZER FERTILITY
# ============================================================

VICTORIAN_SAMPLE = """
The Parliament did convene on the morrow, wherefore the Members assembled with great 
solemnity. The connexion between commerce and manufacture was discussed at length, 
with particular reference to the condition of the labouring poor. The Hon. Member 
for Liverpool spoke eloquently of the factory system, noting that many operatives 
had not received their wages. "Whereupon," said he, "the grievances of these 
unfortunates must be addressed forthwith." The debate continued thusly until the 
candles were lit, when a motion was made that the House should adjourn. The Speaker 
did thereupon put the question, and it was resolved in the affirmative.

Mr. Dickens, in his recent work, hath described the condition of the workhouse with 
singular vividness. The orphans and destitute persons who dwell therein are shown 
to suffer greatly from want and neglect. The beadle, a personage of considerable 
self-importance, administers the establishment with a severity that might justly 
be termed cruel. It is indeed a melancholy reflection upon our civilization that 
such institutions should be necessary, and that they should be conducted in such 
a manner as to add to, rather than alleviate, the sufferings of the poor.

The railway has wrought such changes in our manner of living as could scarcely have 
been imagined by our forefathers. Where formerly the mail-coach required two days 
to convey passengers from London to Manchester, the locomotive engine now performs 
the journey in a matter of hours. The terminus at Euston-square presents a scene of 
extraordinary activity, with travellers of every description hastening to and fro. 
The porters, in their distinctive livery, attend to the luggage with commendable 
dispatch, whilst the guards inspect the carriages ere departure.
"""

# Alternative dense Victorian sample (Dickens + Parliamentary)
DICKENS_PARLIAMENTARY_SAMPLE = """
It was the best of times, it was the worst of times, it was the age of wisdom, 
it was the age of foolishness, it was the epoch of belief, it was the epoch of 
incredulity, it was the season of Light, it was the season of Darkness, it was 
the spring of hope, it was the winter of despair, we had everything before us, 
we had nothing before us, we were all going direct to Heaven, we were all going 
direct the other way—in short, the period was so far like the present period, 
that some of its noisiest authorities insisted on its being received, for good 
or for evil, in the superlative degree of comparison only.

There were a king with a large jaw and a queen with a plain face, on the throne 
of England; there were a king with a large jaw and a queen with a fair face, on 
the throne of France. In both countries it was clearer than crystal to the lords 
of the State preserves of loaves and fishes, that things in general were settled 
for ever.

The Honourable Member for Westminster rose to address the House upon the subject 
of the Poor Law Amendment Bill. "Mr. Speaker," quoth he, "the condition of the 
labouring classes in our manufacturing districts is such as to excite the gravest 
apprehensions. The workhouses, whereto the destitute are consigned, are administered 
with a rigour that borders upon cruelty. I have received numerous communications 
from my constituents, describing the sufferings endured therein."

Wherefore, he continued, the Government ought to institute an inquiry into the 
management of these establishments. The Honourable Gentleman opposite had spoken 
of the necessity for economy in the public expenditure; but surely, the welfare 
of the poor was a consideration of paramount importance. The connexion between 
pauperism and crime was well established, and it behoved Parliament to address 
this evil ere it assumed yet more alarming proportions.
"""


def compute_tokenizer_fertility(tokenizer, text):
    """
    Compute tokenizer fertility = number of tokens / number of words
    Lower is better (more efficient).
    """
    # Count words (simple whitespace split)
    words = text.split()
    word_count = len(words)
    
    # Count tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_count = len(tokens)
    
    fertility = token_count / word_count
    
    return {
        "word_count": word_count,
        "token_count": token_count,
        "fertility": fertility,
        "efficiency": word_count / token_count  # inverse of fertility
    }


def run_tokenizer_fertility_analysis(output_dir):
    """
    Compare tokenizer efficiency on Victorian text.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    
    # Combine samples for ~1000 words
    sample_text = VICTORIAN_SAMPLE + "\n\n" + DICKENS_PARLIAMENTARY_SAMPLE
    
    results = {}
    
    print("\n" + "="*60)
    print("TOKENIZER FERTILITY ANALYSIS")
    print("="*60)
    print(f"Sample length: {len(sample_text.split())} words\n")
    
    # TimeCapsule tokenizer
    print("Testing TimeCapsule...", end=" ")
    try:
        # Try HF first, then local
        try:
            tc_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        except:
            tc_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(LOCAL_MODEL_PATH))
        
        stats = compute_tokenizer_fertility(tc_tokenizer, sample_text)
        results["TimeCapsule"] = stats
        print(f"✓ Fertility: {stats['fertility']:.3f} ({stats['token_count']} tokens)")
    except Exception as e:
        print(f"✗ Error: {e}")
        results["TimeCapsule"] = {"error": str(e)}
    
    # Other tokenizers
    other_tokenizers = [
        ("GPT-2/OpenAI", "gpt2"),
    ]
    
    for name, model_id in other_tokenizers:
        print(f"Testing {name}...", end=" ")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            stats = compute_tokenizer_fertility(tokenizer, sample_text)
            results[name] = stats
            print(f"✓ Fertility: {stats['fertility']:.3f} ({stats['token_count']} tokens)")
        except Exception as e:
            print(f"✗ Error: {e}")
            results[name] = {"error": str(e)}
    
    # Calculate efficiency comparison
    if "TimeCapsule" in results and "fertility" in results["TimeCapsule"]:
        tc_fertility = results["TimeCapsule"]["fertility"]
        print("\n📊 Efficiency Comparison (vs TimeCapsule):")
        for name, stats in results.items():
            if name != "TimeCapsule" and "fertility" in stats:
                improvement = (stats["fertility"] - tc_fertility) / stats["fertility"] * 100
                print(f"   {name}: {improvement:+.1f}% {'more' if improvement > 0 else 'less'} efficient")
    
    # Save results
    output_path = output_dir / "tokenizer_fertility_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to: {output_path}")
    
    # Save sample text for reference
    sample_path = output_dir / "victorian_sample.txt"
    with open(sample_path, "w") as f:
        f.write(sample_text)
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("MODULE 1: QUANTITATIVE VALIDATION")
    print("Time Capsule Paper - ACM C&C 2026")
    print("="*60)
    
    # Setup output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Load test texts from dataset
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "outputs" / "dataset"
    
    test_texts = []
    if dataset_dir.exists():
        print(f"\n📂 Loading test set from: {dataset_dir}")
        try:
            ds = load_from_disk(str(dataset_dir))
            if "test" in ds:
                # Get raw text samples
                test_data = ds["test"]
                # Limit to reasonable number
                max_samples = min(200, len(test_data))
                
                # We need to decode the texts from token IDs
                from transformers import PreTrainedTokenizerFast
                try:
                    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(LOCAL_MODEL_PATH))
                except:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
                
                for i in range(max_samples):
                    input_ids = test_data[i]["input_ids"]
                    text = tokenizer.decode(input_ids, skip_special_tokens=True)
                    if len(text) > 100:  # Only use substantial texts
                        test_texts.append(text)
                
                print(f"   Loaded {len(test_texts)} test samples")
        except Exception as e:
            print(f"   Error loading dataset: {e}")
    
    # Fallback: use sample Victorian text
    if not test_texts:
        print("\n⚠️  Using sample Victorian text for evaluation")
        test_texts = [VICTORIAN_SAMPLE, DICKENS_PARLIAMENTARY_SAMPLE]
    
    # Run analyses
    print("\n" + "="*60)
    print("ANALYSIS A: DOMAIN-SPECIFIC PERPLEXITY")
    print("="*60)
    ppl_results = run_perplexity_analysis(test_texts[:50], output_dir)  # Limit for speed
    
    print("\n" + "="*60)
    print("ANALYSIS B: TOKENIZER FERTILITY")
    print("="*60)
    fertility_results = run_tokenizer_fertility_analysis(output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("MODULE 1 SUMMARY")
    print("="*60)
    
    print("\nPerplexity Results:")
    for model, data in ppl_results.items():
        if "perplexity" in data:
            print(f"   {model}: {data['perplexity']:.2f}")
        else:
            print(f"   {model}: {data.get('error', 'N/A')}")
    
    print("\nTokenizer Fertility (tokens/word):")
    for name, data in fertility_results.items():
        if "fertility" in data:
            print(f"   {name}: {data['fertility']:.3f}")
    
    print(f"\n✅ All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
