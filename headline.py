"""Shared helpers used by both train.py and app.py.

This is where text becomes token IDs, and where generated token IDs become text again.

IndicBART formats (from the model card):
    encoder input :  <article tokens> </s> <2hi>
    decoder input :  <2hi> <headline tokens>
    labels        :  <headline tokens> </s>
The language tag (<2hi> Hindi, <2mr> Marathi) tells the model which language it is reading/writing.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # everything is local after download.py; never hit the Hub

import torch
from peft import PeftModel
from transformers import AlbertTokenizer, AutoModelForSeq2SeqLM

from config import BASE_MODEL_DIR, FINETUNED_DIR, LANGUAGES, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer():
    # Settings from the IndicBART model card: keep case and accents (matras) exactly as written.
    return AlbertTokenizer.from_pretrained(BASE_MODEL_DIR, do_lower_case=False, keep_accents=True)


def eos_id(tok):
    # Note: tok.eos_token is "[SEP]" (an Albert default), but IndicBART actually ends sequences with "</s>".
    return tok.convert_tokens_to_ids("</s>")


def lang_id(tok, lang):
    return tok.convert_tokens_to_ids(LANGUAGES[lang])


def encode_article(tok, text, lang):
    """article string -> encoder input_ids:  article </s> <2xx>"""
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=MAX_SOURCE_LENGTH - 2).input_ids
    return ids + [eos_id(tok), lang_id(tok, lang)]


def encode_headline(tok, text, lang):
    """headline string -> (decoder_input_ids, labels).  Labels are the decoder inputs shifted left by one."""
    ids = tok(text, add_special_tokens=False, truncation=True, max_length=MAX_TARGET_LENGTH - 1).input_ids
    decoder_input_ids = [lang_id(tok, lang)] + ids
    labels = ids + [eos_id(tok)]
    return decoder_input_ids, labels


def load_model(finetuned=True):
    """Load the base model from disk and (optionally) apply the trained LoRA adapters on top."""
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_DIR)
    if finetuned:
        model = PeftModel.from_pretrained(model, FINETUNED_DIR)
        model = model.merge_and_unload()  # fold adapters into the weights -> plain model, faster inference
    return model.to(DEVICE).eval()


@torch.no_grad()
def generate_headline(model, tok, text, lang, num_beams=4):
    input_ids = torch.tensor([encode_article(tok, text, lang)], device=model.device)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        decoder_start_token_id=lang_id(tok, lang),  # start writing in the chosen language
        bos_token_id=tok.convert_tokens_to_ids("<s>"),
        eos_token_id=eos_id(tok),
        pad_token_id=tok.pad_token_id,
        max_length=MAX_TARGET_LENGTH + 1,
        num_beams=num_beams,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )[0].tolist()

    # skip_special_tokens does not remove IndicBART's </s>, <s> and <2xx> tags, so drop them by ID.
    drop = {tok.pad_token_id, eos_id(tok), tok.convert_tokens_to_ids("<s>")} | {lang_id(tok, l) for l in LANGUAGES}
    return tok.decode([i for i in output if i not in drop], skip_special_tokens=True).strip()
