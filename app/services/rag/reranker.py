from typing import List, Dict
import numpy as np

from app.services.rag.embedding import embed


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_similarity(query: str, title: str):

    try:
        q_vec = embed(query)
        t_vec = embed(title)

        return float(cosine_similarity(q_vec, t_vec))

    except Exception:
        return 0.0


def rerank_products(query: str, items: List[Dict]) -> List[Dict]:

    ranked = []

    for item in items:

        title = item.get("title", "")
        price = item.get("price") or 0
        trust = item.get("trust_score") or 0
        rating = item.get("seller_rating") or 0

        similarity = compute_similarity(query, title)

        # normalizzazione prezzo
        price_penalty = price / 1000 if price else 0

        score = (
            0.45 * trust +
            0.25 * similarity +
            0.20 * (rating / 100) -
            0.10 * price_penalty
        )

        item["_rerank_score"] = round(score, 4)

        ranked.append(item)

    ranked.sort(
        key=lambda x: x.get("_rerank_score", 0),
        reverse=True
    )

    return ranked