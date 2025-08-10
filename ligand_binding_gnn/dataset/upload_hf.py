import os
from huggingface_hub import HfApi, HfFolder, upload_file, create_repo


# HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
HF_USERNAME = "your-hf-username"
REPO_NAME = "mini_ligand_binding_dataset"
DATA_FILES = [
    "./data/bindingdb_filtered.csv", 
    # Add your cached embeddings
    # e.g., "./cache/pair_0.npz"
]
PRIVATE = False  # True if you want the dataset private
# -------------------------

def main():
    if HF_TOKEN == "YOUR_HF_TOKEN_HERE":
        raise ValueError("Please set your HF token in HF_TOKEN or as an environment variable.")

    # Save token locally
    HfFolder.save_token(HF_TOKEN)

    # Create repo (ignore if exists)
    api = HfApi()
    full_repo_name = f"{HF_USERNAME}/{REPO_NAME}"
    try:
        create_repo(full_repo_name, repo_type="dataset", private=PRIVATE, exist_ok=True, token=HF_TOKEN)
        print(f"Repo created: https://huggingface.co/datasets/{full_repo_name}")
    except Exception as e:
        print(f"Repo creation warning: {e}")

    # Upload each file
    for file_path in DATA_FILES:
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue

        dest_path = os.path.basename(file_path)
        print(f"Uploading {file_path} → {dest_path}")
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=dest_path,
            repo_id=full_repo_name,
            repo_type="dataset",
            token=HF_TOKEN
        )

    print("✅ Upload complete.")

if __name__ == "__main__":
    main()
