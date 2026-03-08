import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

# ============================================================
# GPU CONFIG & MODEL (singleton)
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
# Usiamo un modello multilingue per gestire i feedback italiani su eBay
_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
_MODEL.to(device)


# ============================================================
# ANCHORS (normalized)
# ============================================================

_POS_ANCHOR = _MODEL.encode(
    "excellent seller fast shipping great service",
    normalize_embeddings=True
).astype("float32")

_NEG_ANCHOR = _MODEL.encode(
    "terrible seller scam fake item bad service",
    normalize_embeddings=True
).astype("float32")


def _extract_texts(feedbacks: List[Dict], max_texts: int = 25) -> List[str]:
    texts: List[str] = []

    for f in feedbacks:
        text = (
            f.get("CommentText")
            or f.get("comment")
            or f.get("text")
        )
        if text and len(text.strip()) > 3:
            texts.append(text.strip())
            if len(texts) >= max_texts:
                break

    return texts


def compute_sentiment_score(feedbacks: List[Dict], max_texts: int = 25) -> float:
    """
    Returns sentiment in [0,1] using embedding similarity to pos/neg anchors.
    Uses limited number of texts for performance.
    """
    texts = _extract_texts(feedbacks, max_texts=max_texts)
    if not texts:
        return 0.5

    embs = _MODEL.encode(
        texts,
        normalize_embeddings=True
    ).astype("float32")

    # Since embeddings are normalized, cosine = dot
    sim_pos = np.dot(embs, _POS_ANCHOR)
    sim_neg = np.dot(embs, _NEG_ANCHOR)

    raw = sim_pos - sim_neg
    mean_score = float(np.mean(raw))

    # map roughly [-1, +1] → [0,1]
    normalized = (mean_score + 1.0) / 2.0
    return float(np.clip(normalized, 0.0, 1.0))