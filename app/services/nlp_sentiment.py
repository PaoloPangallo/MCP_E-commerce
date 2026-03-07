from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ============================================================
# MODEL (lazy singleton)
# ============================================================

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        logger.info("Loading SentenceTransformer model: all-MiniLM-L6-v2")
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


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
def _anchor_vectors() -> tuple[np.ndarray, np.ndarray]:
    model = _get_model()

    pos = model.encode(
        "excellent seller fast shipping great service authentic item",
        normalize_embeddings=True
    ).astype("float32")

    neg = model.encode(
        "terrible seller scam fake item bad service broken product",
        normalize_embeddings=True
    ).astype("float32")

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

        sim_pos = np.dot(embs, pos_anchor)
        sim_neg = np.dot(embs, neg_anchor)

        raw = sim_pos - sim_neg
        mean_score = float(np.mean(raw))

        normalized = float(np.clip((mean_score + 1.0) / 2.0, 0.0, 1.0))

    except Exception as e:
        logger.warning("Embedding sentiment failed, using heuristic: %s", e)
        normalized = _heuristic_sentiment(texts)

    if use_cache:
        _SENTIMENT_CACHE[fingerprint] = normalized

    return round(normalized, 4)