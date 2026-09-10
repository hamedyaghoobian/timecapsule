"""
Configuration for the future TimeCapsuleLLM v3 training run.

This config is intentionally aimed at the next stage of preparation:
- raw corpus exists
- tokenizer is trained
- tokenized / packed Arrow shards do not exist yet

Nothing here should be read as "start training now". The intended pipeline is:

raw JSONL.GZ text -> tokenize and pack to fixed 4096-token Arrow shards -> train later

Current expected first training environment:
- Georgia State University
- 3 x NVIDIA L40 GPUs
- 48 GB VRAM each
"""

import torch


# ============================================================
# TOKENIZER / DATA SETTINGS
# ============================================================

VOCAB_SIZE = 32000
CONTEXT_LENGTH = 4096

SPECIAL_TOKENS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "year_token": "<|year|>",
    "title_token": "<|title|>",
}

SPECIAL_TOKEN_IDS = {
    "bos_token_id": 0,
    "eos_token_id": 1,
    "unk_token_id": 2,
    "pad_token_id": 3,
    "year_token_id": 4,
    "title_token_id": 5,
}


# ============================================================
# MODEL ARCHITECTURE (~2B target)
# ============================================================

MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "hidden_size": 2560,
    "intermediate_size": 6912,
    "num_hidden_layers": 26,
    "num_attention_heads": 20,
    "num_key_value_heads": 5,
    "hidden_act": "silu",
    "max_position_embeddings": CONTEXT_LENGTH,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "use_cache": False,
    "bos_token_id": SPECIAL_TOKEN_IDS["bos_token_id"],
    "eos_token_id": SPECIAL_TOKEN_IDS["eos_token_id"],
    "pad_token_id": SPECIAL_TOKEN_IDS["pad_token_id"],
    "unk_token_id": SPECIAL_TOKEN_IDS["unk_token_id"],
    "tie_word_embeddings": False,
    "initializer_range": 0.02,
    "attention_bias": False,
    "mlp_bias": False,
}


# ============================================================
# TRAINING DEFAULTS
# ============================================================

TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 2e-4,
    "warmup_steps": 4000,
    # Placeholder budget only. Real value depends on:
    # - final packed token count
    # - world size
    # - effective batch size
    "max_steps": 100000,
    "bf16": True,
    "tf32": True,
    "optim": "adamw_torch_fused",
    "gradient_checkpointing": True,
    "logging_steps": 10,
    "save_steps": 2000,
    "save_total_limit": 3,
    "report_to": "none",
    "remove_unused_columns": False,
}


# ============================================================
# PREPROCESSING DEFAULTS
# ============================================================

PREPROCESSING_CONFIG = {
    "sequence_length": CONTEXT_LENGTH,
    "add_bos_token": True,
    "add_eos_token": True,
    "documents_per_flush": 256,
    "target_sequences_per_arrow_shard": 8192,
}


# ============================================================
# DEVICE / DTYPE
# ============================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype():
    return torch.bfloat16 if get_device().type == "cuda" else torch.float32
