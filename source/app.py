from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

from train_model import negation_aware_tokenize

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
model = joblib.load(MODEL_DIR / "sentiment_model.joblib")

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("review"), str):
        return jsonify({"error": "Send a JSON object with a text review."}), 400
    review = payload["review"].strip()
    if len(review) < 3:
        return jsonify({"error": "Please enter a review with at least three characters."}), 400
    if len(review) > 5000:
        return jsonify({"error": "Please keep the review below 5,000 characters."}), 400

    processed = negation_aware_tokenize(review)
    features = vectorizer.transform([processed])
    probabilities = model.predict_proba(features)[0]
    prediction_ids, margins = model.predict_with_margin(features)
    prediction_id = int(prediction_ids[0])
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[prediction_id]
    scores = {labels[i]: round(float(probabilities[i]), 4) for i in range(3)}

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(float(probabilities[prediction_id]), 4),
        "scores": scores,
        "margin": round(float(margins[0]), 4),
        "processed_preview": processed,
        "model": "Negation-Aware Contrastive Centroid Classification",
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
