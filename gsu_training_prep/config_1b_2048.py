"""
Legacy 1.2B-ish / 2048-context config kept for reference.

This preserves the earlier working shape instead of silently overwriting it.
"""

import torch

VOCAB_SIZE = 32000
CONTEXT_LENGTH = 2048

MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "hidden_size": 2048,
    "intermediate_size": 5504,
    "num_hidden_layers": 22,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "max_position_embeddings": CONTEXT_LENGTH,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "use_cache": False,
    "pad_token_id": 3,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "unk_token_id": 2,
}

TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "warmup_steps": 2000,
    "max_steps": 60000,
    "bf16": True,
    "tf32": True,
    "optim": "adamw_torch_fused",
    "gradient_checkpointing": True,
    "logging_steps": 10,
    "save_steps": 1000,
    "save_total_limit": 3,
    "report_to": "none",
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype():
    return torch.bfloat16 if get_device().type == "cuda" else torch.float32
