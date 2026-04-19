"""
model.py — Load the saved model and make predictions
------------------------------------------------------
This module is imported by the Flask app (app.py).
It loads the three serialised artifacts and exposes
a single `predict(text)` function.
"""

import os
import pickle
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Load artifacts once at import time ───────────────────────────────────────
def _load_artifact(name: str):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Please run  python train.py  first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


vectorizer = _load_artifact("tfidf_vectorizer.pkl")
classifier = _load_artifact("svm_classifier.pkl")
label_enc  = _load_artifact("label_encoder.pkl")

print(f"Model loaded. Classes: {list(label_enc.classes_)}")


# ── Public API ────────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    """
    Classify a user message.

    Returns
    -------
    dict with keys:
        intent      – predicted class name  (str)
        confidence  – probability 0.0-1.0   (float)
        probabilities – {class: prob, …}    (dict)
    """
    vec   = vectorizer.transform([text])
    label = classifier.predict(vec)[0]
    proba = classifier.predict_proba(vec)[0]

    intent      = label_enc.inverse_transform([label])[0]
    confidence  = float(np.max(proba))
    probs_dict  = {
        cls: round(float(p), 4)
        for cls, p in zip(label_enc.classes_, proba)
    }

    return {
        "intent":        intent,
        "confidence":    round(confidence, 4),
        "probabilities": probs_dict,
    }
