from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import redis

from app.config.settings import settings

logger = logging.getLogger(__name__)

# ============================================================
# REDIS (SESSION MEMORY)
# ============================================================

redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True
)


def get_session_memory(user_key: str) -> Dict[str, Any]:

    history = redis_client.lrange(f"session:{user_key}:history", 0, -1)
    queries = redis_client.lrange(f"session:{user_key}:queries", 0, -1)
    sellers = redis_client.lrange(f"session:{user_key}:sellers", 0, -1)

    return {
        "chat_history": history,
        "recent_queries": queries,
        "recent_sellers": sellers,
    }


def add_query_to_session(user_key: str, query: str):

    redis_client.lpush(f"session:{user_key}:queries", query)
    redis_client.ltrim(f"session:{user_key}:queries", 0, 20)


def add_chat_message(user_key: str, message: str):

    redis_client.lpush(f"session:{user_key}:history", message)
    redis_client.ltrim(f"session:{user_key}:history", 0, 50)


def clear_session_memory(user_key: str):
    """Clears all session memory (history, queries, sellers) from Redis for a specific user."""
    keys = [
        f"session:{user_key}:history",
        f"session:{user_key}:queries",
        f"session:{user_key}:sellers"
    ]
    for key in keys:
        redis_client.delete(key)

