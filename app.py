import os
import re
import math
import joblib
import nltk
from pathlib import Path
from dotenv import load_dotenv

# Dash UI
import dash
from dash import dcc, html, Input, Output, State

# NLTK assets
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

load_dotenv()

# =============================================================================
# Environment & artifact paths
# =============================================================================
ENV = os.environ.get("environment", "development").lower()

# You said you’ll use the two joblib files you saved during training:
MODEL_PATH = Path(os.environ.get("LOGREG_MODEL_PATH", "./models/logreg_tuned_model.joblib"))
VEC_PATH   = Path(os.environ.get("TFIDF_PATH", "./models/tfidf_tuned_vectorizer.joblib"))

LABEL_MAP = {0: "REAL", 1: "FAKE"}

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH.resolve()}")
MODEL = joblib.load(MODEL_PATH)

# Detect if MODEL is a Pipeline with tfidf inside; if not, load vectorizer
IS_PIPELINE = hasattr(MODEL, "predict_proba") and hasattr(MODEL, "get_params")
VEC = None
if not IS_PIPELINE:
    if not VEC_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found at: {VEC_PATH.resolve()}")
    VEC = joblib.load(VEC_PATH)

# =============================================================================
# NLTK setup (download at runtime if missing; avoids baking corpora into slug)
# =============================================================================
def _ensure_nltk():
    """
    Ensures required NLTK resources are available at runtime.
    Heroku slugs don't include nltk_data, so this downloads missing corpora.
    """
    # Stopwords
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    # Tokenizers (both legacy and new split)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    # WordNet + Lemmatization resources
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


_ensure_nltk()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# =============================================================================
# Preprocessing (mirrors your training)
# =============================================================================
URL_RE   = re.compile(r'https?://\S+|www\.\S+|\S+\.(com|org|net|edu|gov|io|co|uk)\S*|bit\.ly/\S+|t\.co/\S+')
HTML_RE  = re.compile(r'<.*?>')
NONALPH  = re.compile(r'[^a-z\s]+')
WS_RE    = re.compile(r'\s+')

def _is_nan(x):
    # Avoid bringing in pandas just for isna
    return x is None or (isinstance(x, float) and math.isnan(x))

def preprocess_text_lowercase_url(text):
    if _is_nan(text):
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
    """
      1) lowercase + URL removal + whitespace clean
      2) remove non-alphabetic, collapse spaces
      3) tokenize, drop stopwords, lemmatize
    """
    text = preprocess_text_lowercase_url(text)
    # If HTML present in some sources, you can optionally:
    # text = HTML_RE.sub(" ", text)
    text = _keep_alpha_only(text)
    text = preprocess_and_lemmatize(text)
    return text

# =============================================================================
# Inference helpers
# =============================================================================
def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def _predict_logreg_on_combined(title: str, body: str):
    title = (title or "").strip()
    body  = (body  or "").strip()
    if not title and not body:
        return None, [None, None]

    title_clean = apply_preprocessing(title)
    body_clean  = apply_preprocessing(body)
    combined    = (title_clean + " " + body_clean).strip()
    if not combined:
        return None, [None, None]

    # Vectorize + predict
    if IS_PIPELINE:
        # MODEL already includes TF-IDF
        proba_supported = hasattr(MODEL, "predict_proba")
        if proba_supported:
            proba = MODEL.predict_proba([combined])[0]
        else:
            # Fallback via decision_function -> pseudo-proba
            margin = float(MODEL.decision_function([combined])[0])
            p1 = _sigmoid(margin)  # probability for class 1
            proba = [1.0 - p1, p1]
    else:
        X = VEC.transform([combined])
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)[0]
        else:
            margin = float(MODEL.decision_function(X)[0])
            p1 = _sigmoid(margin)
            proba = [1.0 - p1, p1]

    # Make sure we order probs as [p_real (0), p_fake (1)] using classes_
    if hasattr(MODEL, "classes_"):
        # MODEL is estimator or pipeline with clf.classes_
        classes = getattr(MODEL, "classes_", None)
    else:
        classes = None

    # If classes_ is available and not [0,1], reorder
    if classes is not None:
        # Build a map from class label -> prob
        prob_map = {int(classes[i]): float(proba[i]) for i in range(len(classes))}
        p_real = prob_map.get(0, None)
        p_fake = prob_map.get(1, None)
        # If somehow missing, fall back to original order
        if p_real is None or p_fake is None:
            p_real, p_fake = float(proba[0]), float(proba[1])
    else:
        p_real, p_fake = float(proba[0]), float(proba[1])

    label = 1 if p_fake >= 0.5 else 0
    return label, [p_real, p_fake]

# =============================================================================
# Dash app (Tier-1 polished UI)
# =============================================================================
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
            "📰 Fake News Detector (Logistic Regression)",
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

@app.callback(
    Output("output-container", "children"),
    Input("submit-button", "n_clicks"),
    State("input-title", "value"),
    State("input-body", "value"),
    prevent_initial_call=True
)
def on_submit(n_clicks, title, body):
    try:
        label, probs = _predict_logreg_on_combined(title, body)
    except Exception as e:
        return f"Error: {e}"

    if label is None:
        return "Please enter article text first."

    sentiment = LABEL_MAP.get(label, str(label))
    if isinstance(probs, (list, tuple)) and len(probs) == 2 and all(p is not None for p in probs):
        probs_fmt = [round(float(p), 4) for p in probs]  # [p_real, p_fake]
        return f"Prediction: {sentiment}  |  Probabilities (REAL, FAKE): {probs_fmt}"
    return f"Prediction: {sentiment}"

# Optional warm-up
try:
    _ = _predict_logreg_on_combined("warmup title", "warmup body")
except Exception:
    pass

if __name__ == "__main__":
    if ENV == "development":
        app.run(debug=True, port=int(os.environ.get("PORT", 8000)))
    else:
        port = int(os.environ.get("PORT", 8080))
        app.run(host="0.0.0.0", port=port, debug=False)
