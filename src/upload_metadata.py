from huggingface_hub import HfApi

def main():
    api = HfApi()
    repo_id = "postgrammar/london-llm-1800"
    
    print(f"🔍 Checking repository: {repo_id}")
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"   Private: {repo_info.private}")
        print(f"   Downloads: {repo_info.downloads}")
        print(f"   Likes: {repo_info.likes}")
    except Exception as e:
        print(f"   Error checking repo info: {e}")

    print("\n🚀 Uploading README.md (Citation)...")
    try:
        api.upload_file(
            path_or_fileobj="dataset_README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card with citation"
        )
        print("✅ README.md uploaded successfully!")
    except Exception as e:
        print(f"❌ Upload Failed: {e}")

if __name__ == "__main__":
    main()
