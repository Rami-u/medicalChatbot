"""
chat.py — Flask backend for the AI Medical Symptom Chatbot
------------------------------------------------------------
Run:
    python chat.py

API endpoints:
    POST /api/chat    — classify user message → return intent + response
    GET  /api/intents — return list of all known intents
    GET  /            — serve the frontend (index.html)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time

# ── Import model (loads artifacts on startup) ─────────────────────────────────
import model as ml_model
from responses import get_response

# ── Flask setup ───────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)   # allow cross-origin requests from the frontend dev server


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend HTML file."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Request body (JSON):  { "message": "I have itching and rash" }
    Response (JSON):
        {
          "intent":      "allergy",
          "confidence":  0.9512,
          "response":    "🤧 Your symptoms suggest an allergic reaction …",
          "timestamp":   1713556923.45
        }
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body."}), 400

    user_message = str(data["message"]).strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Classify the intent
    prediction  = ml_model.predict(user_message)
    intent      = prediction["intent"]
    confidence  = prediction["confidence"]
    bot_response = get_response(intent)

    return jsonify({
        "intent":     intent,
        "confidence": confidence,
        "response":   bot_response,
        "timestamp":  time.time(),
    })


@app.route("/api/intents", methods=["GET"])
def list_intents():
    """Return the list of all supported intents."""
    classes = list(ml_model.label_enc.classes_)
    return jsonify({"intents": classes, "count": len(classes)})


@app.route("/api/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok", "model": "TF-IDF + SVM"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Medical Symptom Chatbot \u2026")
    print(f"   Frontend  -> http://127.0.0.1:5000/")
    print(f"   Chat API  -> http://127.0.0.1:5000/api/chat")
    app.run(host="0.0.0.0", port=5000, debug=True)
