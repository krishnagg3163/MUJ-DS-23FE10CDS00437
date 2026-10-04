"""ONE-TIME setup: download the base model and the dataset CSVs into this project folder.

Run once:   python download.py
After this, nothing else in the project touches the internet.
"""
from huggingface_hub import hf_hub_download, snapshot_download

from config import BASE_MODEL_DIR, DATA_DIR, HF_DATASET_ID, HF_MODEL_ID, LANGUAGES

# 1. Base model: weights, config and SentencePiece tokenizer files.
print(f"Downloading {HF_MODEL_ID} -> {BASE_MODEL_DIR}")
snapshot_download(
    repo_id=HF_MODEL_ID,
    local_dir=BASE_MODEL_DIR,
    allow_patterns=["*.json", "*.bin", "spiece.*"],
)

# 2. Dataset: one CSV per language.
DATA_DIR.mkdir(parents=True, exist_ok=True)
for lang in LANGUAGES:
    print(f"Downloading {lang}.csv -> {DATA_DIR}")
    hf_hub_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        filename=f"{lang}.csv",
        local_dir=DATA_DIR,
    )

print("Done. Model and data are stored locally.")
