import os
import re
import math
import json
import pickle
import nltk
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# --- TensorFlow / Keras ---
import tensorflow as tf
# NEW (required for TF 2.20 + Keras 3)
from keras.models import load_model
from keras.utils import pad_sequences


# --- Dash UI ---
import dash
from dash import dcc, html, Input, Output, State

# --- NLTK assets ---
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

load_dotenv()

# ====================================================================================
# Environment & artifact paths
# ====================================================================================
ENV = os.environ.get("environment", "development").lower()
MODEL_PATH     = Path(os.environ.get("LSTM_MODEL_PATH", "/mnt/data/lstm_fake_news_model.h5"))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", "/mnt/data/tokenizer.pkl"))

NUM_WORDS  = int(os.environ.get("NUM_WORDS", 50000))
MAX_LEN    = int(os.environ.get("MAX_LEN",   256))
OOV_TOKEN  = os.environ.get("OOV_TOKEN", "<UNK>")
PADDING    = os.environ.get("PADDING", "post")
TRUNCATING = os.environ.get("TRUNCATING", "post")

LABEL_MAP = {0: "REAL", 1: "FAKE"}

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH.resolve()}")
if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Tokenizer not found at: {TOKENIZER_PATH.resolve()}")

# ====================================================================================
# Load artifacts
# ====================================================================================
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
MODEL = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    TOKENIZER = pickle.load(f)

# ====================================================================================
# NLTK setup
# ====================================================================================
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    _ = nltk.word_tokenize("test")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ====================================================================================
# Preprocessing (exactly like training)
# ====================================================================================
URL_RE   = re.compile(r'https?://\S+|www\.\S+|\S+\.(com|org|net|edu|gov|io|co|uk)\S*|bit\.ly/\S+|t\.co/\S+')
HTML_RE  = re.compile(r'<.*?>')
NONALPH  = re.compile(r'[^a-z\s]+')
WS_RE    = re.compile(r'\s+')

def preprocess_text_lowercase_url(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    text = URL_RE.sub("", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _keep_alpha_only(text: str) -> str:
    text = NONALPH.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text

def preprocess_and_lemmatize(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return " ".join(tokens)
    return ""

def apply_preprocessing(text: str) -> str:
    text = preprocess_text_lowercase_url(text)
    text = _keep_alpha_only(text)
    text = preprocess_and_lemmatize(text)
    return text

# ====================================================================================
# Inference helper
# ====================================================================================
def _predict_lstm_on_combined(title: str, body: str):
    title = (title or "").strip()
    body  = (body or "").strip()
    if not title and not body:
        return None, [None, None]

    title_clean = apply_preprocessing(title)
    body_clean  = apply_preprocessing(body)
    combined    = (title_clean + " " + body_clean).strip()
    if not combined:
        return None, [None, None]

    seq = TOKENIZER.texts_to_sequences([combined])
    x   = pad_sequences(seq, maxlen=MAX_LEN, padding=PADDING, truncating=TRUNCATING)
    pred = MODEL.predict(x, verbose=0)
    if pred.ndim == 2 and pred.shape[1] == 1:
        p_fake = float(pred[0, 0])
        label  = int(p_fake >= 0.5)
        return label, [1.0 - p_fake, p_fake]
    return 1, [0.5, 0.5]

# ====================================================================================
# Dash app with Tier-1 UI styling
# ====================================================================================
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    style={
        "maxWidth": "760px",
        "margin": "40px auto",
        "padding": "24px",
        "border": "1px solid #eee",
        "borderRadius": "12px",
        "boxShadow": "0 6px 18px rgba(0,0,0,0.06)",
        "fontFamily": "Segoe UI, Roboto, Helvetica, Arial, sans-serif",
        "color": "#222",
        "lineHeight": "1.45",
    },
    children=[
        html.H1(
            "📰 Fake News Detector (LSTM)",
            style={"textAlign": "center", "margin": "0 0 14px 0", "fontSize": "28px", "color": "#333"},
        ),
        html.P(
            "Paste an article (title optional) and click Classify.",
            style={"textAlign": "center", "margin": "0 0 24px 0", "color": "#666"},
        ),
        html.Label("Title (optional)", style={"fontWeight": 600, "display": "block", "marginBottom": "6px"}),
        dcc.Input(
            id="input-title",
            placeholder="Enter article title...",
            type="text",
            debounce=True,
            style={
                "width": "100%",
                "height": "42px",
                "padding": "8px 12px",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "marginBottom": "16px",
                "outline": "none",
            },
        ),
        html.Label("Article Body", style={"fontWeight": 600, "display": "block", "marginBottom": "6px"}),
        dcc.Textarea(
            id="input-body",
            placeholder="Paste article body...",
            style={
                "width": "100%",
                "height": "220px",
                "padding": "12px",
                "border": "1px solid #ddd",
                "borderRadius": "8px",
                "resize": "vertical",
                "outline": "none",
            },
        ),
        html.Div(style={"height": "16px"}),
        html.Button(
            "🔍 Classify",
            id="submit-button",
            n_clicks=0,
            style={
                "backgroundColor": "#0d6efd",
                "color": "white",
                "border": "none",
                "padding": "10px 18px",
                "borderRadius": "8px",
                "cursor": "pointer",
                "fontWeight": 600,
                "letterSpacing": "0.3px",
            },
        ),
        html.Div(id="output-container", style={"marginTop": "18px", "fontSize": "18px", "fontWeight": 600}),
    ],
)

# ====================================================================================
# Callback
# ====================================================================================
@app.callback(
    Output("output-container", "children"),
    Input("submit-button", "n_clicks"),
    State("input-title", "value"),
    State("input-body", "value"),
    prevent_initial_call=True
)
def on_submit(n_clicks, title, body):
    try:
        label, probs = _predict_lstm_on_combined(title, body)
    except Exception as e:
        return f"Error: {e}"

    if label is None:
        return "Please enter article text first."

    sentiment = LABEL_MAP.get(label, str(label))
    if isinstance(probs, (list, tuple)) and len(probs) == 2 and all(p is not None for p in probs):
        probs_fmt = [round(float(p), 4) for p in probs]
        return f"Prediction: {sentiment}  |  Probabilities (REAL, FAKE): {probs_fmt}"
    return f"Prediction: {sentiment}"

# Warm-up
try:
    _ = _predict_lstm_on_combined("warmup title", "warmup body")
except Exception:
    pass

# ====================================================================================
# Entrypoint
# ====================================================================================
if __name__ == "__main__":
    if ENV == "development":
        app.run(debug=True, port=int(os.environ.get("PORT", 8000)))
    else:
        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port, debug=False)