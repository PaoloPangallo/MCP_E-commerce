from typing import List, Dict
import numpy as np

from app.services.rag.embedding import embed


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ============================================================
# RERANK PRODUCTS
# ============================================================

def rerank_products(query: str, items: List[Dict]) -> List[Dict]:

    if not items:
        return items

    ranked = []

    # embed query once
    q_vec = embed(query)

    # compute price statistics
    prices = [i.get("price") for i in items if i.get("price")]

    avg_price = np.mean(prices) if prices else 0
    std_price = np.std(prices) if prices else 1

    for item in items:

        title = item.get("title", "")

        try:

            t_vec = embed(title)
            similarity = cosine_similarity(q_vec, t_vec)

        except Exception:

            similarity = 0

        trust = item.get("trust_score") or 0
        rating = (item.get("seller_rating") or 0) / 100
        price = item.get("price") or avg_price

        # detect expensive outliers
        price_z = abs(price - avg_price) / std_price if std_price else 0
        price_penalty = min(price_z / 5, 0.2)

        score = (
            0.50 * similarity +
            0.30 * trust +
            0.15 * rating -
            0.05 * price_penalty
        )

        item["_rerank_score"] = round(score, 4)

        ranked.append(item)

    ranked.sort(
        key=lambda x: x.get("_rerank_score", 0),
        reverse=True
    )

    return ranked