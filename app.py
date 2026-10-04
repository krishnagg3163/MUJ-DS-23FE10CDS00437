"""Streamlit UI: paste a news article, get a headline.

Run:  streamlit run app.py
"""
import streamlit as st

from config import FINETUNED_DIR, LANGUAGES
from headline import generate_headline, load_model, load_tokenizer

st.set_page_config(page_title="Headline Generator", page_icon="📰")
st.title("📰 News Headline Generator")
st.caption("IndicBARTSS fine-tuned with LoRA · Hindi & Marathi")

if not FINETUNED_DIR.exists():
    st.error("No fine-tuned model found. Run `python train.py` first.")
    st.stop()


@st.cache_resource  # load once per server start, not on every click
def get_model():
    return load_tokenizer(), load_model()


tok, model = get_model()

lang = st.selectbox("Language", list(LANGUAGES), format_func=str.title)
article = st.text_area("News article", height=300, placeholder="Paste the full article here...")

if st.button("Generate headline", type="primary", disabled=not article.strip()):
    with st.spinner("Generating..."):
        headline = generate_headline(model, tok, article, lang)
    st.subheader(headline)
