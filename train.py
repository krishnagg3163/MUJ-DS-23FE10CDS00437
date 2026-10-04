"""Fine-tune IndicBARTSS with LoRA to write Hindi/Marathi headlines.

Run:          python train.py
Quick test:   python train.py --max_steps 20
Output:       LoRA adapters in models/indicbartss-lora/ (loaded by app.py)
"""
import argparse

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed

from config import BASE_MODEL_DIR, DATA_DIR, FINETUNED_DIR, INPUT_COL, LANGUAGES, ROOT, TARGET_COL
from headline import encode_article, encode_headline, generate_headline, load_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--max_steps", type=int, default=-1, help="stop early (for a quick smoke test)")
cli = parser.parse_args()

SEED = 42
set_seed(SEED)
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- 1. DATA: CSV -> article/headline strings
frames = []
for lang in LANGUAGES:
    df = pd.read_csv(DATA_DIR / f"{lang}.csv", usecols=[INPUT_COL, TARGET_COL]).dropna()
    df = df[(df[INPUT_COL].str.strip() != "") & (df[TARGET_COL].str.strip() != "")]
    df["lang"] = lang
    print(f"{lang}: {len(df)} usable rows")
    frames.append(df)
data = pd.concat(frames, ignore_index=True)

dataset = Dataset.from_pandas(data).train_test_split(test_size=0.02, seed=SEED)
print(dataset)

# ---------------------------------------------------------------- 2. TOKENIZATION: strings -> token IDs
tok = load_tokenizer()


def preprocess(example):
    decoder_input_ids, labels = encode_headline(tok, example[TARGET_COL], example["lang"])
    return {
        "input_ids": encode_article(tok, example[INPUT_COL], example["lang"]),
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }


tokenized = dataset.map(preprocess, remove_columns=dataset["train"].column_names)


def collate(batch):
    """Pad a list of examples into rectangular tensors. Label padding is -100 so the loss ignores it."""
    def pad(seqs, value):
        width = max(len(s) for s in seqs)
        return torch.tensor([s + [value] * (width - len(s)) for s in seqs])

    input_ids = pad([b["input_ids"] for b in batch], tok.pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != tok.pad_token_id).long(),  # 1 = real token, 0 = padding
        "decoder_input_ids": pad([b["decoder_input_ids"] for b in batch], tok.pad_token_id),
        "labels": pad([b["labels"] for b in batch], -100),
    }


# ---------------------------------------------------------------- 3. MODEL: pretrained IndicBARTSS + LoRA
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],  # MBart layer names
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------------- 4. TRAINING
args = Seq2SeqTrainingArguments(
    output_dir=str(ROOT / "checkpoints"),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    max_steps=cli.max_steps,
    learning_rate=3e-4,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # bf16 is stable for BART; fp16 can NaN
    label_names=["labels"],
    remove_unused_columns=False,
    report_to="none",
    seed=SEED,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=collate,
)
trainer.train()
print("Final eval:", trainer.evaluate())

# ---------------------------------------------------------------- 5. SAVE
model.save_pretrained(FINETUNED_DIR)
print("Saved LoRA adapters to:", FINETUNED_DIR)

# ---------------------------------------------------------------- 6. QUICK LOOK at unseen test articles
model.eval()
for row in dataset["test"].select(range(5)):
    print("\nLANG     :", row["lang"])
    print("ARTICLE  :", row[INPUT_COL][:200], "...")
    print("ACTUAL   :", row[TARGET_COL])
    print("GENERATED:", generate_headline(model, tok, row[INPUT_COL], row["lang"]))
