import streamlit as st
from ngram_model import NGramModel
import nltk
from nltk.corpus import gutenberg

import streamlit as st
import nltk

# Download required NLTK data safely
@st.cache_resource
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'gutenberg']
    
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

download_nltk_data()

@st.cache_resource
def load_model(n_order):
    """Load model for specific n-gram order."""
    model = NGramModel(n=n_order)
    # Use first 3 Gutenberg texts for training
    fileids = gutenberg.fileids()[:3]
    texts = [gutenberg.raw(fileid) for fileid in fileids]
    full_text = ' '.join(texts)
    model.train(full_text)
    return model

st.set_page_config(page_title="N-Gram Text Generator", layout="wide")
st.title("🧠 N-Gram Text Generator")
st.markdown("Trained on classic literature. Generates coherent text sequences.")

# Sidebar controls
st.sidebar.header("Settings")
seed = st.sidebar.text_input("Seed phrase:", "to be or")
length = st.sidebar.slider("Max words:", 1, 100, 25)  # FIXED: Starts at 1
ngram_order = st.sidebar.selectbox("N-gram order:", ["Unigram","Bigram", "Trigram", "4-gram"], index=1)

# Main content area
col1, col2 = st.columns([2, 3])

with col1:
    st.info(f"**N-gram:** {ngram_order}\n**Seed:** {seed or 'random'}\n**Length:** {length}")
    if st.button("🎲 Generate Text", type="primary", use_container_width=True):
        if ngram_order == "Unigram":
            ngram_order = 1
        elif ngram_order == "Bigram":
            ngram_order = 2
        elif ngram_order == "Trigram":
            ngram_order = 3
        elif ngram_order == "4-gram":
            ngram_order = 4
        model = load_model(ngram_order)
        # model = load_model(int(ngram_order) + 1)
        generated = model.generate(seed.lower().split() if seed else None, length)
        st.session_state.generated = generated

with col2:
    if 'generated' in st.session_state:
        st.success("**Generated:**")
        st.markdown(f"```{st.session_state.generated}```")
        if st.button("🔄 New Generation"):
            del st.session_state.generated
