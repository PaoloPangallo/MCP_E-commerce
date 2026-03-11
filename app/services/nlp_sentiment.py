from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# MODEL (lazy singleton)
# ============================================================

from app.services.model_singleton import get_sentence_transformer as _get_model



# ============================================================
# TEXT EXTRACTION
# ============================================================

def _extract_texts(feedbacks: List[Dict], max_texts: int = 20) -> List[str]:
    texts: List[str] = []

    for f in feedbacks:
        text = (
            f.get("CommentText")
            or f.get("comment")
            or f.get("text")
        )

        if not isinstance(text, str):
            continue

        cleaned = text.strip()
        if len(cleaned) < 4:
            continue

        texts.append(cleaned)

        if len(texts) >= max_texts:
            break

    return texts


def _stable_text_fingerprint(texts: Sequence[str]) -> str:
    joined = "||".join(sorted(t.strip().lower() for t in texts if t.strip()))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


# ============================================================
# EMBEDDING ANCHORS
# ============================================================

@lru_cache(maxsize=1)
def _anchor_vectors():

    model = _get_model()

    positive_examples = [
        "excellent seller fast shipping",
        "great product highly recommended",
        "perfect transaction very happy",
        "ottimo venditore spedizione veloce",
        "perfetto consigliato"
    ]

    negative_examples = [
        "terrible seller never again",
        "fake item scam seller",
        "bad service broken product",
        "venditore pessimo prodotto rotto",
        "truffa non comprate"
    ]

    pos = model.encode(
        positive_examples,
        normalize_embeddings=True
    )

    neg = model.encode(
        negative_examples,
        normalize_embeddings=True
    )

    return pos, neg


# ============================================================
# CACHE
# ============================================================

_SENTIMENT_CACHE: Dict[str, float] = {}


# ============================================================
# FAST HEURISTIC
# ============================================================

_POS_WORDS = {
    "excellent", "great", "perfect", "fast", "recommended", "good",
    "ottimo", "perfetto", "veloce", "consigliato", "positivo"
}

_NEG_WORDS = {
    "terrible", "scam", "fake", "bad", "broken", "slow", "refund",
    "terribile", "truffa", "falso", "rotto", "lento", "negativo"
}


def _heuristic_sentiment(texts: Sequence[str]) -> float:
    score = 0.0
    seen = 0

    for text in texts:
        t = text.lower()
        local = 0.5

        pos_hits = sum(1 for w in _POS_WORDS if w in t)
        neg_hits = sum(1 for w in _NEG_WORDS if w in t)

        if pos_hits > neg_hits:
            local = 0.8
        elif neg_hits > pos_hits:
            local = 0.2

        score += local
        seen += 1

    if seen == 0:
        return 0.5

    return round(score / seen, 4)


# ============================================================
# PUBLIC API
# ============================================================

def compute_sentiment_score(
    feedbacks: List[Dict],
    max_texts: int = 20,
    use_cache: bool = True,
    use_fast_path: bool = True,
) -> float:
    """
    Returns sentiment score in [0,1].

    Optimizations:
    - lazy model loading
    - content-based cache
    - fast heuristic path for very small inputs
    - limited embedding workload
    """
    texts = _extract_texts(feedbacks, max_texts=max_texts)

    if not texts:
        return 0.5

    fingerprint = _stable_text_fingerprint(texts)

    if use_cache and fingerprint in _SENTIMENT_CACHE:
        return _SENTIMENT_CACHE[fingerprint]

    # Fast path for tiny batches
    if use_fast_path and len(texts) <= 5:
        val = _heuristic_sentiment(texts)
        if use_cache:
            _SENTIMENT_CACHE[fingerprint] = val
        return val

    try:
        model = _get_model()
        pos_anchor, neg_anchor = _anchor_vectors()

        embs = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=min(16, len(texts)),
            show_progress_bar=False,
        ).astype("float32")

        sim_pos = np.max(np.dot(embs, pos_anchor.T), axis=1)
        sim_neg = np.max(np.dot(embs, neg_anchor.T), axis=1)

        raw = sim_pos - sim_neg
        mean_score = float(np.mean(raw))

        normalized = float(1 / (1 + np.exp(-3 * mean_score)))

    except Exception as e:
        logger.warning("Embedding sentiment failed, using heuristic: %s", e)
        normalized = _heuristic_sentiment(texts)

    if use_cache:
        _SENTIMENT_CACHE[fingerprint] = normalized

    return round(normalized, 4)