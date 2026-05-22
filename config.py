"""
Configuration for Historical LLM Training Pipeline (1800–1875)
Optimized for H100 SXM 80GB
"""

from pathlib import Path
import torch

# ============================================================
# PATHS (RunPod — VERIFIED)
# ============================================================
WORKSPACE = Path("/workspace")
DATASET_PATH = Path(
    "/workspace/hf/datasets/postgrammar___london-llm-1800/train"
)

OUTPUT_DIR = WORKSPACE / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# ============================================================
# DATA / TOKEN SETTINGS
# ============================================================
VOCAB_SIZE = 32000
CONTEXT_LENGTH = 2048

SPECIAL_TOKENS = {
    "bos_token_id": 0,
    "eos_token_id": 1,
    "unk_token_id": 2,
    "pad_token_id": 3,
}

# ============================================================
# MODEL ARCHITECTURE (≈1.25B params)
# ============================================================
MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,         # GQA (2:1 ratio)
    "hidden_act": "silu",
    "max_position_embeddings": CONTEXT_LENGTH,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "use_cache": False,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "bos_token_id": SPECIAL_TOKENS["bos_token_id"],
    "eos_token_id": SPECIAL_TOKENS["eos_token_id"],
    "pad_token_id": SPECIAL_TOKENS["pad_token_id"],
    "attn_implementation": "flash_attention_2", # Explicit for H100
}

# ============================================================
# TRAINING SETTINGS (OPTIMIZED FOR H100 SXM 80GB)
# ============================================================
TRAINING_CONFIG = {
    "per_device_train_batch_size": 16,   # Increased from 4; H100 handles this easily
    "gradient_accumulation_steps": 8,    # Global batch size = 128 (approx 262k tokens)
    "learning_rate": 2e-4,              # Standard for 1B model
    "weight_decay": 0.1,
    "warmup_steps": 2000,               # ~3% of training for stability
    "lr_scheduler_type": "cosine",
    "max_steps": 182_000,
    "save_steps": 1000,
    "save_total_limit": 3,
    "logging_steps": 10,                # Frequent logging for early issue detection
    "bf16": True,                       # High precision for H100
    "tf32": True,                       # Enable for faster matrix multiplication
    "gradient_checkpointing": False,     # Disabled: trade VRAM for speed
    "dataloader_num_workers": 8,         # Parallel loading for 120GB dataset
    "dataloader_pin_memory": True,       # Faster host-to-device transfer
}

# ============================================================
# UTILITIES
# ============================================================
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dtype() -> torch.dtype:
    return torch.bfloat16