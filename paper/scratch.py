import sys
import json
import torch
import numpy as np
from module2_diachronic import load_timecapsule_model, get_word_embedding_from_model, compute_semantic_axis, project_onto_axis, HISTORICAL_CONTEXTS
from transformers import BertModel, BertTokenizer

def run():
    words = ["time", "labor", "machine"]
    results = {}
    
    POLE_A = "nature"
    POLE_B = "factory"
    context_template = "The {word} is central to our understanding."
    
    print("Loading TimeCapsule...")
    tc_model, tc_tokenizer, tc_type = load_timecapsule_model()
    if not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tc_model = tc_model.to(device)
    tc_model.eval()
    
    axis_tc, origin_tc, _ = compute_semantic_axis(tc_model, tc_tokenizer, POLE_A, POLE_B, context_template, tc_type)
    
    for word in words:
        ctx = HISTORICAL_CONTEXTS.get(word, f"The {word} is important.")
        emb = get_word_embedding_from_model(tc_model, tc_tokenizer, word, ctx, tc_type)
        proj = project_onto_axis(emb, axis_tc, origin_tc)
        results[word] = {"TimeCapsule": float(proj)}
    
    del tc_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("Loading BERT...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)
    bert_model.eval()
    
    axis_bert, origin_bert, _ = compute_semantic_axis(bert_model, bert_tokenizer, POLE_A, POLE_B, context_template, "bert")
    
    for word in words:
        ctx = f"The {word} is important."
        emb = get_word_embedding_from_model(bert_model, bert_tokenizer, word, ctx, "bert")
        proj = project_onto_axis(emb, axis_bert, origin_bert)
        results[word]["BERT"] = float(proj)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run()
