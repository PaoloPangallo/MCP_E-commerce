from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

# modello leggero e veloce
model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------
# Helper: estrai testi dalle recensioni
# ------------------------------------------------

def _extract_texts(feedbacks: List[Dict]) -> List[str]:

    texts = []

    for f in feedbacks:

        text = (
            f.get("CommentText")
            or f.get("comment")
            or f.get("text")
        )

        if text and len(text.strip()) > 3:
            texts.append(text.strip())

    return texts


# ------------------------------------------------
# Sentiment score tramite embedding similarity
# ------------------------------------------------

positive_anchor = model.encode("excellent seller fast shipping great service")
negative_anchor = model.encode("terrible seller scam fake item bad service")


def compute_sentiment_score(feedbacks: List[Dict]) -> float:

    texts = _extract_texts(feedbacks)

    if not texts:
        return 0.5

    embeddings = model.encode(texts)

    scores = []

    for emb in embeddings:

        sim_pos = np.dot(emb, positive_anchor) / (
            np.linalg.norm(emb) * np.linalg.norm(positive_anchor)
        )

        sim_neg = np.dot(emb, negative_anchor) / (
            np.linalg.norm(emb) * np.linalg.norm(negative_anchor)
        )

        raw = sim_pos - sim_neg

        scores.append(raw)

    mean_score = np.mean(scores)

    # normalizzazione 0..1
    normalized = (mean_score + 1) / 2

    return float(np.clip(normalized, 0, 1))