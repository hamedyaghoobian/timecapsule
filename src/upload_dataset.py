from huggingface_hub import HfApi
import time

def upload_dataset():
    print("🚀 Starting Robust Dataset Upload...")
    api = HfApi()
    
    try:
        api.upload_large_folder(
            folder_path="outputs_full/dataset",
            repo_id="postgrammar/london-llm-1800",
            repo_type="dataset",
            num_workers=8,  # Parallel uploads
        )
        print("✅ Upload Completed Successfully!")
    except Exception as e:
        print(f"❌ Upload Failed: {e}")

if __name__ == "__main__":
    upload_dataset()
