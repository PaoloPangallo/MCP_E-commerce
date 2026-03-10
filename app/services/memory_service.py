from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import redis
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

# ============================================================
# REDIS (SESSION MEMORY)
# ============================================================

redis_client = redis.Redis(
    host="localhost",
    port=6379,
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


# ============================================================
# POSTGRES (LONG TERM MEMORY)
# ============================================================

def get_user_memory(db: Session, user_key: str) -> Dict[str, Any]:

    rows = db.execute(
        text("""
        SELECT key,value
        FROM user_memory
        WHERE user_id=:uid
        """),
        {"uid": user_key},
    )

    memory = {}

    for r in rows:
        try:
            memory[r.key] = json.loads(r.value)
        except Exception:
            memory[r.key] = r.value

    return memory


def save_user_memory(db: Session, user_key: str, key: str, value: Any):

    db.execute(
        text("""
        INSERT INTO user_memory(user_id,key,value)
        VALUES(:uid,:k,:v)
        """),
        {
            "uid": user_key,
            "k": key,
            "v": json.dumps(value),
        },
    )

    db.commit()


# ============================================================
# FAISS (SEMANTIC MEMORY)
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_DIM = 384

faiss_index = faiss.IndexFlatL2(VECTOR_DIM)

semantic_store: List[Dict[str, Any]] = []


def add_semantic_memory(text_data: str, metadata: Dict[str, Any]):

    vector = embedding_model.encode([text_data])[0]
    vector = np.array([vector]).astype("float32")

    faiss_index.add(vector)

    semantic_store.append(
        {
            "text": text_data,
            "metadata": metadata,
        }
    )


def search_semantic_memory(query: str, k: int = 5):

    if faiss_index.ntotal == 0:
        return []

    q_vector = embedding_model.encode([query])[0]
    q_vector = np.array([q_vector]).astype("float32")

    distances, indices = faiss_index.search(q_vector, k)

    results = []

    for idx in indices[0]:

        if idx < len(semantic_store):

            results.append(semantic_store[idx])

    return results