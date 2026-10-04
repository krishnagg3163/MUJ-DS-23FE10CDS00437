"""Shared settings for the headline generator. Every other script imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Hugging Face sources (only used by download.py) ----
HF_MODEL_ID = "ai4bharat/IndicBARTSS"
HF_DATASET_ID = "l3cube-pune/IndicHeadline-ID"

# ---- Local folders: everything lives inside the project after download.py runs once ----
BASE_MODEL_DIR = ROOT / "models" / "IndicBARTSS"        # pretrained weights + tokenizer
DATA_DIR = ROOT / "data" / "raw"                        # original CSVs from the Hub
FINETUNED_DIR = ROOT / "models" / "indicbartss-lora"    # LoRA adapters saved by train.py

# ---- Languages: file name on the Hub -> IndicBART language tag ----
# Hindi and Marathi are both written in Devanagari, which is the script IndicBARTSS uses.
LANGUAGES = {
    "hindi": "<2hi>",
    "marathi": "<2mr>",
}

# ---- Columns in the CSVs ----
# IndicHeadline-ID is a headline *identification* dataset: only `original_title` is the article's real
# headline. `semantic_title`, `lexical_title` and `random_title` are decoys taken from OTHER articles.
INPUT_COL = "full_news"
TARGET_COL = "original_title"

# ---- Tokenization / generation lengths (in tokens) ----
# Measured on the data: articles ~630 tokens (median), headlines ~29 (median) / ~41 (95th pct).
MAX_SOURCE_LENGTH = 600
MAX_TARGET_LENGTH = 48
