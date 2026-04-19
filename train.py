"""
train.py — Train the TF-IDF + SVM intent classifier
------------------------------------------------------
Run this script once to train the model and save it:
    python train.py

Outputs (saved in models/ folder):
    - tfidf_vectorizer.pkl   : fitted TF-IDF vectorizer
    - svm_classifier.pkl     : trained SVM classifier
    - label_encoder.pkl      : label encoder for class names
"""

import json
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(BASE_DIR, "intents.json")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(intents_path: str):
    """Parse intents.json and return parallel lists of texts and labels."""
    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent_obj in data["intents"]:
        tag = intent_obj["intent"]
        for example in intent_obj["examples"]:
            texts.append(example.strip())
            labels.append(tag)

    print(f"Loaded {len(texts)} examples across {len(set(labels))} intents.")
    return texts, labels


def train():
    # ── 1. Load data ──────────────────────────────────────────────────────
    texts, labels = load_data(INTENTS_FILE)

    # ── 2. Encode labels ──────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print(f"   Classes: {list(le.classes_)}")

    # ── 3. Train/test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.15, random_state=42, stratify=y
    )

    # ── 4. TF-IDF Vectorization ───────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # apply log normalization
        min_df=2,                # ignore very rare terms
        max_features=5000,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    # ── 5. Train SVM ──────────────────────────────────────────────────────
    print("Training SVM classifier ...")
    clf = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,   # enables predict_proba for confidence scores
        random_state=42,
    )
    clf.fit(X_train_vec, y_train)
    print("Training complete.")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    y_pred = clf.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\n" + classification_report(
        y_test, y_pred, target_names=le.classes_
    ))

    # ── 7. Save artifacts ─────────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "svm_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"\nModels saved to: {MODELS_DIR}")


if __name__ == "__main__":
    train()
