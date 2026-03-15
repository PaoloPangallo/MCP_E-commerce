import asyncio
import logging
from typing import Any, Dict

from app.db.redis import redis_client
from app.services.feedback import get_seller_feedback
from app.services.nlp_sentiment import compute_sentiment_score
from app.services.trust import compute_trust_score

logger = logging.getLogger(__name__)

_CACHE_TTL = 300  # 5 minuti


async def run_seller_pipeline(
    seller_name: str,
    page: int = 1,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Seller analysis service with Redis caching and async fetch.
    """
    seller_name = (seller_name or "").strip()
    if not seller_name:
        raise ValueError("seller_name vuoto")

    page = max(1, int(page))
    limit = min(max(int(limit), 1), 100)
    needed = page * limit

    cache_key = f"seller_analysis:{seller_name.lower()}:{needed}"
    cached = redis_client.get_json(cache_key)
    if cached:
        return cached

    # Controlla la signature reale di get_seller_feedback:
    # se serve use_cache, aggiungilo esplicitamente
    feedbacks = await get_seller_feedback(
        seller_name,
        limit=needed,
        # use_cache=True,
    )

    start = (page - 1) * limit
    end = start + limit
    paginated = feedbacks[start:end]

    if not feedbacks:
        res = {
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
        redis_client.set_json(cache_key, res, ttl_seconds=_CACHE_TTL)
        return res

    sentiment_score = await asyncio.to_thread(
        compute_sentiment_score,
        feedbacks,
        max_texts=50,
    )
    trust_score = await asyncio.to_thread(
        compute_trust_score,
        feedbacks,
        sentiment_score=sentiment_score,
    )

    res = {
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

    redis_client.set_json(cache_key, res, ttl_seconds=_CACHE_TTL)
    return res