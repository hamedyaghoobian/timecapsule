"""
Configuration for Historical LLM Training Pipeline (1800-1875)
Target: Apple Silicon M3 Max with MPS + BFloat16
"""
import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "subset_split"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Output subdirectories
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
DATASET_DIR = OUTPUT_DIR / "dataset"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# ============================================================
# REPLICATION SETTINGS
# ============================================================
RANDOM_SEED = 42
DATASET_VERSION = "1.0.0"

# ============================================================
# TOKENIZER SETTINGS
# ============================================================
VOCAB_SIZE = 32000              # 32K vocabulary (efficient for 500M model)
CONTEXT_LENGTH = 2048           # Token context window
MIN_FREQUENCY = 2               # Minimum token frequency for BPE

SPECIAL_TOKENS = [
    "<s>",          # BOS
    "</s>",         # EOS
    "<unk>",        # Unknown
    "<pad>",        # Padding
    "<|year|>",     # Year metadata marker
    "<|title|>",    # Title metadata marker
]

# ============================================================
# DATA CLEANING SETTINGS
# ============================================================
NORMALIZE_LONG_S = True         # ſ → s (for semantic analysis)
REMOVE_CONTROL_CHARS = True     # Strip \x00-\x1f (except newlines)
PRESERVE_LIGATURES = True       # Keep æ, œ, etc.
MIN_TEXT_LENGTH = 100           # Minimum characters per document

# Characters to normalize
CHAR_NORMALIZATION = {
    'ſ': 's',                   # Long-s to s
    'ﬀ': 'ff',                  # ff ligature
    'ﬁ': 'fi',                  # fi ligature
    'ﬂ': 'fl',                  # fl ligature
    'ﬃ': 'ffi',                 # ffi ligature
    'ﬄ': 'ffl',                 # ffl ligature
    '\u00AD': '',               # Soft hyphen removal
}

# ============================================================
# MODEL ARCHITECTURE (500M params)
# ============================================================
MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "hidden_size": 1024,
    "intermediate_size": 2752,
    "num_hidden_layers": 20,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,       # GQA for efficiency
    "max_position_embeddings": CONTEXT_LENGTH,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000,
    "use_cache": False,             # Disable for training
}

# ============================================================
# TRAINING SETTINGS (MPS-optimized)
# ============================================================
TRAINING_CONFIG = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_steps": 1000,
    "max_steps": 50000,
    "save_steps": 2500,
    "save_total_limit": 3,
    "logging_steps": 100,
    "eval_steps": 1000,
    "seed": RANDOM_SEED,
}

# ============================================================
# DEVICE SETTINGS
# ============================================================
import torch

def get_device():
    """Get the best available device for training."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_dtype():
    """Get the best dtype for the current device."""
    device = get_device()
    if device.type == "mps":
        return torch.bfloat16  # Native M3 support
    elif device.type == "cuda":
        return torch.float16
    else:
        return torch.float32

# MPS memory optimization
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
