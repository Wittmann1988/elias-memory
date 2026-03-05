#!/usr/bin/env python3
"""Upload SFT dataset to HuggingFace Hub."""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

DATASET_ID = "erik1988/elias-memory-traces-v1"
DATASET_FILE = Path(__file__).parent.parent / "data" / "sft_dataset.jsonl"


def main():
    api = HfApi(token=os.environ.get("HUGGINGFACE_API_KEY"))

    # Create dataset repo (ignore if exists)
    try:
        create_repo(DATASET_ID, repo_type="dataset", token=api.token)
        print(f"Created dataset repo: {DATASET_ID}")
    except Exception as e:
        print(f"Repo may already exist: {e}")

    # Upload the JSONL file
    api.upload_file(
        path_or_fileobj=str(DATASET_FILE),
        path_in_repo="train.jsonl",
        repo_id=DATASET_ID,
        repo_type="dataset",
    )
    print(f"Uploaded {DATASET_FILE.name} to https://huggingface.co/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
