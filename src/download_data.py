import os
from pathlib import Path
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

def main():
    print("="*60)
    print("DOWNLOAD DATA FOR RUNPOD TRAINING")
    print("="*60)

    # Output directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. Download Dataset
    print("\n⬇️  Downloading dataset from 'postgrammar/london-llm-1800'...")
    dataset = load_dataset("postgrammar/london-llm-1800")
    
    dataset_path = data_dir / "dataset"
    print(f"💾 Saving dataset to: {dataset_path}")
    dataset.save_to_disk(dataset_path)
    
    # 2. Download Tokenizer
    print("\n⬇️  Downloading tokenizer...")
    tokenizer_path = data_dir / "tokenizer"
    
    # Load from Hub (it's inside the 'tokenizer' folder in the repo)
    # We uploaded it to root/tokenizer. HF Hub path: "postgrammar/london-llm-1800", subfolder="tokenizer"
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained("postgrammar/london-llm-1800", subfolder="tokenizer")
        print(f"💾 Saving tokenizer to: {tokenizer_path}")
        tokenizer.save_pretrained(tokenizer_path)
    except Exception as e:
        print(f"❌ Failed to download tokenizer: {e}")
        print("Trying root...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained("postgrammar/london-llm-1800")
        tokenizer.save_pretrained(tokenizer_path)

    print("\n" + "="*60)
    print("✅ DOWNLOAD COMPLETE")
    print("="*60)
    print("You can now run training with:")
    print("python src/05_train_model_cuda.py --data_dir data --output_dir outputs")

if __name__ == "__main__":
    main()
