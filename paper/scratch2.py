import sys
import json
import torch
import numpy as np
from module2_diachronic import load_timecapsule_model, get_word_embedding_from_model, compute_semantic_axis, project_onto_axis
from transformers import BertModel, BertTokenizer

def run():
    print("Loading TimeCapsule...")
    tc_model, tc_tokenizer, tc_type = load_timecapsule_model()
    if not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        tc_model = tc_model.to(device)
    tc_model.eval()
    
    print("Loading BERT...")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    device_bert = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device_bert)
    bert_model.eval()

    def get_projection(pole_a, pole_b, target):
        context_template = "The {word} is central to our understanding."
        ctx = f"The {target} is important."
        
        # TC
        axis_tc, origin_tc, _ = compute_semantic_axis(tc_model, tc_tokenizer, pole_a, pole_b, context_template, tc_type)
        emb_tc = get_word_embedding_from_model(tc_model, tc_tokenizer, target, ctx, tc_type)
        proj_tc = project_onto_axis(emb_tc, axis_tc, origin_tc)
        
        # BERT
        axis_bert, origin_bert, _ = compute_semantic_axis(bert_model, bert_tokenizer, pole_a, pole_b, context_template, "bert")
        emb_bert = get_word_embedding_from_model(bert_model, bert_tokenizer, target, ctx, "bert")
        proj_bert = project_onto_axis(emb_bert, axis_bert, origin_bert)
        
        return {"TimeCapsule": float(proj_tc), "BERT": float(proj_bert), "Shift": float(proj_tc - proj_bert)}

    results = {}
    results["VALUE on VIRTUE -> COMMERCE"] = get_projection("virtue", "commerce", "value")
    results["POWER on AUTHORITY -> STEAM"] = get_projection("authority", "steam", "power")

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run()
