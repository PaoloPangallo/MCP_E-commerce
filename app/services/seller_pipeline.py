from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.services.feedback import get_seller_feedback
from app.services.nlp_sentiment import compute_sentiment_score
from app.services.trust import compute_trust_score

# Simple in-process cache
_SELLER_CACHE: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
_SCORE_CACHE: Dict[str, Dict[str, float]] = {}



def _normalize_seller_name(seller_name: str) -> str:
    return (seller_name or "").strip()



def _get_feedbacks_cached(seller_name: str, limit: int) -> List[Dict[str, Any]]:
    key = (seller_name.lower(), limit)

    if key in _SELLER_CACHE:
        return _SELLER_CACHE[key]

    feedbacks = get_seller_feedback(seller_name, limit=limit) or []
    _SELLER_CACHE[key] = feedbacks
    return feedbacks



def run_seller_pipeline(
    seller_name: str,
    page: int = 1,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Seller analysis service.

    Compatibility notes:
    - keeps the original public keys
    - adds `status` and `error` for consumers that need explicit no-data semantics
    - avoids returning misleading 0.5 trust/sentiment when no feedback exists
    """
    seller_name = _normalize_seller_name(seller_name)

    if not seller_name:
        raise ValueError("seller_name vuoto")

    page = max(1, int(page))
    limit = min(max(int(limit), 1), 50)

    needed = page * limit
    feedbacks = _get_feedbacks_cached(seller_name, limit=needed)

    start = (page - 1) * limit
    end = start + limit
    paginated = feedbacks[start:end]

    if not feedbacks:
        return {
            "seller_name": seller_name,
            "page": page,
            "limit": limit,
            "count": 0,
            "feedbacks": [],
            "trust_score": None,
            "sentiment_score": None,
            "status": "no_data",
            "error": "Nessun feedback disponibile per questo venditore.",
        }

    seller_key = seller_name.lower()

    cached_scores = _SCORE_CACHE.get(seller_key)
    if cached_scores and cached_scores.get("count") == len(feedbacks):
        sentiment_score = cached_scores["sentiment_score"]
        trust_score = cached_scores["trust_score"]
    else:
        sentiment_score = compute_sentiment_score(feedbacks, max_texts=20)
        trust_score = compute_trust_score(
            feedbacks,
            sentiment_score=sentiment_score,
        )

        _SCORE_CACHE[seller_key] = {
            "count": float(len(feedbacks)),
            "sentiment_score": float(sentiment_score),
            "trust_score": float(trust_score),
        }

    return {
        "seller_name": seller_name,
        "page": page,
        "limit": limit,
        "count": len(feedbacks),
        "feedbacks": paginated,
        "trust_score": round(float(trust_score), 3),
        "sentiment_score": round(float(sentiment_score), 3),
        "status": "ok",
        "error": None,
    }
