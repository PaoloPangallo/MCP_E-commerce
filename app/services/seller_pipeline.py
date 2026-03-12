from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from app.services.feedback import get_seller_feedback_async
from app.services.nlp_sentiment import compute_sentiment_score
from app.services.trust import compute_trust_score

_CACHE_TTL = 300.0  # 5 minutes
_SELLER_CACHE: Dict[Tuple[str, int], Tuple[float, List[Dict[str, Any]]]] = {}
_SCORE_CACHE: Dict[str, Tuple[float, Dict[str, float]]] = {}

def _normalize_seller_name(seller_name: str) -> str:
    return (seller_name or "").strip()

async def _get_feedbacks_cached_async(seller_name: str, limit: int) -> List[Dict[str, Any]]:
    key = (seller_name.lower(), limit)
    now = time.time()
    cached = _SELLER_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]
    
    feedbacks = await get_seller_feedback_async(seller_name, limit=limit) or []
    _SELLER_CACHE[key] = (now, feedbacks)
    return feedbacks

async def run_seller_pipeline_async(
    seller_name: str,
    page: int = 1,
    limit: int = 50,
) -> Dict[str, Any]:
    seller_name = _normalize_seller_name(seller_name)
    if not seller_name: raise ValueError("seller_name vuoto")

    needed = page * limit
    feedbacks = await _get_feedbacks_cached_async(seller_name, limit=needed)

    if not feedbacks:
        return {
            "seller_name": seller_name, "page": page, "limit": limit, "count": 0,
            "feedbacks": [], "trust_score": None, "sentiment_score": None,
            "status": "no_data", "error": "Nessun feedback disponibile per questo venditore."
        }

    # Sentiment/Trust are fast and synchronous, we can just call them
    sentiment_score = compute_sentiment_score(feedbacks, max_texts=50)
    trust_score = compute_trust_score(feedbacks, sentiment_score=sentiment_score)

    start = (page - 1) * limit
    end = start + limit
    paginated = feedbacks[start:end]

    return {
        "seller_name": seller_name, "page": page, "limit": limit, "count": len(feedbacks),
        "feedbacks": paginated, "trust_score": round(float(trust_score), 3),
        "sentiment_score": round(float(sentiment_score), 3), "status": "ok"
    }
