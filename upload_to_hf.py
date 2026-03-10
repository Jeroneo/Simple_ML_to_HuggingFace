"""
Upload trained model artefacts to the Hugging Face Hub.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo


def upload(repo_id: str, token: str):
    api = HfApi()

    print(f"Creating / verifying repo: {repo_id}")
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")

    files = [
        ("model/iris_classifier.pkl", "iris_classifier.pkl"),
        ("model/metrics.json",        "metrics.json"),
        ("model/README.md",           "README.md"),
    ]

    success = 0
    for local, remote in files:
        if not Path(local).exists():
            print(f"  [SKIP] {local} not found")
            continue
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            print(f"  [OK]   {local} -> {remote}")
            success += 1
        except Exception as e:
            print(f"  [FAIL] {local}: {e}")

    print(f"\nUploaded {success}/{len(files)} files to https://huggingface.co/{repo_id}")
    if success == 0:
        sys.exit(1)


if __name__ == "__main__":
    token   = os.environ.get("HF_TOKEN", "")
    repo_id = os.environ.get("HF_REPO_ID", "YOUR_USERNAME/iris-classifier")

    if not token:
        print("Error: HF_TOKEN environment variable is not set.")
        sys.exit(1)

    upload(repo_id, token)
