#!/bin/bash
# RunPod Setup Script for Historical LLM Training
# Run this after starting your pod

set -e

echo "========================================"
echo "Historical LLM - RunPod Setup"
echo "========================================"

# Update system
apt-get update && apt-get install -y htop nvtop

# Create workspace structure
mkdir -p /workspace/data
mkdir -p /workspace/outputs
mkdir -p /workspace/code

# Install Python dependencies
pip install --upgrade pip
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tokenizers accelerate tqdm numpy tensorboard

# Install Flash Attention 2 (for A100/H100)
echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

# Verify installation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except:
    print('Flash Attention: Not installed')
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Upload your data to /workspace/data/"
echo "   - tokenizer/ folder"
echo "   - dataset/ folder"
echo ""
echo "2. Upload training script:"
echo "   scp 05_train_model_cuda.py runpod:/workspace/code/"
echo ""
echo "3. Start training:"
echo "   cd /workspace/code"
echo "   python 05_train_model_cuda.py --max_steps 100000"
echo ""
