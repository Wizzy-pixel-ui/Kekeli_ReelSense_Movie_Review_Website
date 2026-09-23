import json
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
nltk.download("stopwords", quiet=True)
RANDOM_STATE = 42

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "cannot",
    "can't", "didn't", "doesn't", "don't", "isn't", "wasn't",
    "weren't", "won't", "wouldn't", "shouldn't", "couldn't"
}
STOPWORDS_TO_KEEP = {"not", "no", "never", "neither", "nor"}
CUSTOM_STOPWORDS = set(stopwords.words("english")) - STOPWORDS_TO_KEEP


def map_label(label):
    label = int(label)
    if label in [0, 1]:
        return "negative"
    if label == 2:
        return "neutral"
    if label in [3, 4]:
        return "positive"
    raise ValueError(f"Unexpected label: {label}")


def negation_aware_tokenize(text):
    raw_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[.!?,;:]", str(text).lower())
    processed = []
    negation_active = False
    negation_count = 0
    for token in raw_tokens:
        if token in {".", "!", "?", ",", ";", ":"}:
            negation_active = False
            negation_count = 0
            continue
        clean = token.replace("'", "")
        if clean in NEGATION_WORDS:
            processed.append(clean)
            negation_active = True
            negation_count = 0
            continue
        if clean in CUSTOM_STOPWORDS:
            continue
        if negation_active and negation_count < 3:
            processed.append("NOT_" + clean)
            negation_count += 1
            if negation_count >= 3:
                negation_active = False
        else:
            processed.append(clean)
    return " ".join(processed)


from model_logic import NegationAwareContrastiveCentroid


def main():
    print("Loading the public SST-5 movie-review corpus...")
    dataset = load_dataset("SetFit/sst5")
    texts, labels = [], []
    for split in ["train", "validation", "test"]:
        for row in dataset[split]:
            text = str(row["text"]).strip()
            if text:
                texts.append(text)
                labels.append(map_label(row["label"]))

    cleaned = {}
    for text, label in zip(texts, labels):
        cleaned.setdefault((text, label), True)
    texts = [key[0] for key in cleaned]
    labels = [key[1] for key in cleaned]
    processed = [negation_aware_tokenize(text) for text in texts]
    pairs = [(text, label) for text, label in zip(processed, labels) if text]
    processed, labels = zip(*pairs)

    label_to_number = {"negative": 0, "neutral": 1, "positive": 2}
    y = np.array([label_to_number[label] for label in labels])
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=60000,
        sublinear_tf=True,
        norm="l2",
    )
    X = vectorizer.fit_transform(processed)
    model = NegationAwareContrastiveCentroid()
    model.fit(X, y)
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump(model, MODEL_DIR / "sentiment_model.joblib", compress=3)
    (MODEL_DIR / "metadata.json").write_text(json.dumps({
        "classes": ["negative", "neutral", "positive"],
        "dataset": "SetFit SST-5 on Hugging Face",
        "algorithm": "Negation-Aware Contrastive Centroid Classification",
        "review_count": len(processed),
        "feature_count": int(X.shape[1]),
        "neutral_margin": model.neutral_margin,
    }, indent=2))
    print(f"Saved {len(processed)} reviews and {X.shape[1]} TF-IDF features.")


if __name__ == "__main__":
    main()
