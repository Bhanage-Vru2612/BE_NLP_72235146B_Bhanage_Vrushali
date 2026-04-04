# N-Gram Text Generation Web App

Trigram model for text generation, deployed with Streamlit.

## Quick Start
1. `pip install -r requirements.txt`
2. `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('gutenberg')"`
3. `streamlit run app.py`

## Files
- `app.py`: UI and model loading
- `ngram_model.py`: Core logic
- Customize training by editing `app.py` corpus selection.

Built with Python 3.10+ [web:1].
