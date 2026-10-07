-KRISHNA AGGARWAL
-23FE10CDS00437
-BTECH CSE(DATA SCIENCE)
-SECTION:F
-Project Title: Hindi & Marathi News Headline Generator
-Github username: krishnagg3163
# 📰 Hindi & Marathi News Headline Generator

Paste a Hindi or Marathi news article and get a short headline back.

The project fine-tunes [**IndicBARTSS**](https://huggingface.co/ai4bharat/IndicBARTSS), a multilingual sequence-to-sequence model from AI4Bharat, with **LoRA** on the [**IndicHeadline-ID**](https://huggingface.co/datasets/l3cube-pune/IndicHeadline-ID) dataset from L3Cube-Pune. A small **Streamlit** app serves the result.

It is meant to be read and learned from. There are five short Python files, no notebooks, and no extra frameworks.

```
"चीन की राजधानी बेजिंग में प्रदूषण की रोकथाम के लिए ..."   ──►   "राजनीति: हवा के हजार दुश्मन"
              news article                                           generated headline
```

---

## ✨ Features

- **Two languages, one model:** Hindi (`<2hi>`) and Marathi (`<2mr>`), both written in Devanagari.
- **Parameter-efficient:** LoRA trains about 4.3M of 248M parameters (1.7%), so training fits on an 8 GB laptop GPU.
- **Fully offline after setup:** the model and data are downloaded once into the project folder, and nothing is fetched again.
- **Small adapter:** the fine-tuned weights are a ~17 MB file that sits on top of the untouched base model.
- **One-page UI:** pick a language, paste an article, click a button.

---

## 🗂️ Project structure

```
Finetune_indicbartss/
├── config.py            # all settings: paths, languages, column names, max lengths
├── download.py          # ONE-TIME: fetch base model + CSVs from Hugging Face
├── headline.py          # shared code: text ↔ token IDs, model loading, generation
├── train.py             # fine-tune with LoRA, evaluate, save adapters, show samples
├── app.py               # Streamlit UI
├── requirements.txt     # pinned Python dependencies (PyTorch is installed separately)
├── .gitignore           # keeps the heavy files below out of git
│
├── models/
│   ├── IndicBARTSS/          # 🚫 git-ignored · pretrained base model (~976 MB, never modified)
│   └── indicbartss-lora/     # ✅ committed   · fine-tuned LoRA adapters (~17 MB)
├── data/raw/                 # 🚫 git-ignored · hindi.csv (~199 MB), marathi.csv (~124 MB)
└── checkpoints/              # 🚫 git-ignored · training snapshots (last 3 kept, for resuming)
```

---

## 🚀 Setup guide

Follow these steps to go from an empty machine to a running app.

### What's in git vs. what gets downloaded

Git holds only the code and the small fine-tuned adapters. The heavy files are listed in `.gitignore`, and `download.py` recreates them on your machine.

| Item | Size | In git? | How you get it |
|---|---|---|---|
| Code, README, `requirements.txt` | < 50 KB | ✅ | `git clone` |
| Fine-tuned adapters `models/indicbartss-lora/` | ~17 MB | ✅ | `git clone` (or retrain) |
| Base model `models/IndicBARTSS/` | ~976 MB | 🚫 | `python download.py` |
| Dataset `data/raw/*.csv` | ~322 MB | 🚫 | `python download.py` |
| Training snapshots `checkpoints/` | ~50 MB each | 🚫 | created by `python train.py` |

### 0. Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda) and Git
- An NVIDIA GPU with **8 GB+ VRAM** and an up-to-date driver for training. The app alone also runs on CPU, just more slowly.
- About **3 GB free disk space** (model, data, checkpoints and the Python environment)

### 1. Get the code

```powershell
git clone <your-repo-url> Finetune_indicbartss
cd Finetune_indicbartss
```

### 2. Create the conda environment

```powershell
conda create -n dlenv python=3.12 -y
conda activate dlenv
```

> **PowerShell says `conda` is not recognized?** Run `conda init powershell` once from the *Anaconda Prompt*, then reopen the terminal. You can also skip activation and call the environment's Python directly: `C:\Users\<you>\miniconda3\envs\dlenv\python.exe train.py`.

### 3. Install PyTorch with CUDA

PyTorch isn't in `requirements.txt` because the right build depends on your GPU driver. Get the install command for your CUDA version from [pytorch.org/get-started](https://pytorch.org/get-started/locally/). For example:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

Check that the GPU is visible. This should print `True` and your GPU's name:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4. Install the remaining dependencies

```powershell
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.14 (CUDA 13.2), transformers 5.18, peft 0.21, datasets 5.0 and streamlit 1.65.

### 5. Download the model and data (once)

```powershell
python download.py
```

This downloads about 1.3 GB:

```
models/IndicBARTSS/   config.json, pytorch_model.bin, spiece.model, tokenizer files
data/raw/             hindi.csv, marathi.csv
```

After this step, nothing in the project touches the internet. `headline.py` sets `HF_HUB_OFFLINE=1`, and every path points to a local folder.

### 6. Train (skip this if the adapters came with your clone)

```powershell
python train.py --max_steps 20   # smoke test: checks the whole pipeline in ~2 minutes
python train.py                  # full run: 3 epochs, ~28k steps
```

The full run takes roughly 3–4 hours on an RTX 4060 Laptop GPU (8 GB). When it finishes, the adapters are saved to `models/indicbartss-lora/` (replacing the old ones), and the script prints five test articles with their real and generated headlines.

### 7. Launch the app

```powershell
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), choose **Hindi** or **Marathi**, paste an article, and click **Generate headline**.

### Optional tweaks

| Tweak | Why |
|---|---|
| `setx HF_HUB_DISABLE_SYMLINKS_WARNING 1` | Hides a harmless Windows warning during `download.py` |
| `setx PYTHONIOENCODING utf-8` | Keeps Hindi and Marathi text readable in the terminal |
| VS Code: *Python: Select Interpreter* → `dlenv` | Makes the editor and its Run button use the right environment |
| Keep the project **outside OneDrive or Dropbox** | Otherwise ~1.3 GB of model and data syncs to the cloud. If it stays, mark `models/` and `data/` as *Free up space* |

---

## 🧠 How it works

### The pipeline

```
data/raw/*.csv
     │   pandas: keep full_news + original_title, drop empty rows, tag language
     ▼
article / headline strings
     │   AlbertTokenizer (SentencePiece)                          headline.py
     ▼
token IDs ── encoder: article </s> <2hi>
          ── decoder: <2hi> headline
          ── labels : headline </s>
     │   collate(): pad to rectangles, attention_mask, label padding → -100
     ▼
PyTorch tensors
     │   IndicBARTSS encoder → decoder  (+ LoRA adapters)          train.py
     ▼
logits → cross-entropy loss → backprop → only LoRA weights update
     │
     ▼   saved: models/indicbartss-lora/
model.generate()  (beam search, starts with <2hi>/<2mr>)           app.py
     │
     ▼
token IDs → strip </s>, <2xx> → decoded headline
```

### IndicBART's input format

IndicBART uses a language tag to know which language it is reading and which one it should write:

| Sequence | Format | Example (Hindi) |
|---|---|---|
| Encoder input | `article </s> <2xx>` | `भारत ने आज मैच जीता </s> <2hi>` |
| Decoder input | `<2xx> headline` | `<2hi> भारत की जीत` |
| Labels | `headline </s>` | `भारत की जीत </s>` |

The labels are the decoder input shifted one position to the left. At every step the model sees the tokens so far and is trained to predict the next one.

### Why `-100`?

Batches have to be rectangular, so shorter headlines get padded. Label positions holding `-100` are skipped by PyTorch's cross-entropy loss, so the model is never trained to predict padding.

### Why LoRA?

LoRA leaves the original weights frozen. It adds small trainable matrices (rank 16) next to the attention layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`) and feed-forward layers (`fc1`, `fc2`), and only those are trained. That keeps memory use low and the saved result small. At inference time `merge_and_unload()` folds the adapters back into the weights, so generation runs as fast as a normal model.

---

## ⚙️ Configuration

Everything you are likely to change is in `config.py` or at the top of `train.py`:

| Setting | Value | Where | Notes |
|---|---|---|---|
| `LANGUAGES` | Hindi, Marathi | `config.py` | Other Devanagari languages can be added the same way |
| `INPUT_COL` / `TARGET_COL` | `full_news` → `original_title` | `config.py` | See the dataset note below |
| `MAX_SOURCE_LENGTH` | 600 tokens | `config.py` | Articles: median ~630 tokens |
| `MAX_TARGET_LENGTH` | 48 tokens | `config.py` | Headlines: 95th percentile ~41 tokens |
| LoRA rank / alpha / dropout | 16 / 32 / 0.1 | `train.py` | |
| Batch size / epochs / LR | 4 / 3 / 3e-4 | `train.py` | |
| Precision | bf16 | `train.py` | fp16 can produce NaN losses on BART-family models |
| Train / test split | 98% / 2% (seed 42) | `train.py` | 37,707 / 770 articles |

---

## ⚠️ Gotchas worth knowing

These caused real bugs during development:

1. **Use `original_title`, not `semantic_title`.** IndicHeadline-ID is a headline *identification* dataset. Only `original_title` belongs to the article. `semantic_title`, `lexical_title` and `random_title` are deliberate decoys taken from other articles. For example, an article about Pope Benedict's death has "Pope Francis will visit India" as its `semantic_title`.
2. **IndicBART ends sequences with `</s>`, not `[SEP]`.** The Albert tokenizer reports `[SEP]` as its end-of-sequence token, which is wrong for this model, so `headline.py` looks up `</s>` explicitly.
3. **`skip_special_tokens=True` does not remove `</s>` or `<2hi>`.** The generation code filters those IDs out by hand before decoding.
4. **Language tags are required.** Without `<2hi>`/`<2mr>` the model does not know which language to produce.
5. **The LoRA layer is `out_proj`, not `o_proj`.** IndicBARTSS is an MBart model, and a wrong name in `target_modules` is skipped silently.
6. **Some Marathi rows have no article.** 1,523 of them, all dropped before training.
7. **IndicBARTSS is Devanagari-only.** Bengali, Tamil and other scripts would need transliteration (for example with `indic-nlp-library`) both before and after the model.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| App says *"No fine-tuned model found"* | Run `python train.py` first |
| App output just repeats the article | You are loading smoke-test adapters. Run the full `python train.py` |
| CUDA out of memory | Lower `per_device_train_batch_size` in `train.py` or `MAX_SOURCE_LENGTH` in `config.py` |
| Training was interrupted | Checkpoints are in `checkpoints/`. Pass `resume_from_checkpoint=True` to `trainer.train()` |
| `conda` not found in PowerShell | Call the env's Python directly: `C:\Users\<you>\miniconda3\envs\dlenv\python.exe train.py` |
| Hindi or Marathi text shows as `???` in the terminal | Set `PYTHONIOENCODING=utf-8` before running |

---

## 🔭 Ideas for next steps

- Add a ROUGE evaluation script. The default `rouge_score` tokenizer drops non-Latin characters, so it needs a whitespace tokenizer for Devanagari.
- Add more languages with transliteration (Bengali, Gujarati, Punjabi, …).
- Expose beam size and length controls in the Streamlit sidebar.
- Compare LoRA against full fine-tuning. At 244M parameters, full fine-tuning also fits on 8 GB with bf16.

---

## 🙏 Credits

- **Model:** [IndicBARTSS](https://huggingface.co/ai4bharat/IndicBARTSS) by AI4Bharat, from Dabre et al., *IndicBART: A Pre-trained Model for Indic Natural Language Generation* (2022).
- **Dataset:** [IndicHeadline-ID](https://huggingface.co/datasets/l3cube-pune/IndicHeadline-ID) by L3Cube-Pune (CC BY 4.0).
- **Libraries:** 🤗 Transformers, PEFT, Datasets, PyTorch and Streamlit.
