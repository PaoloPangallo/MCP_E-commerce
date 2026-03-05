from typing import List, Dict, Optional
import numpy as np

from app.services.rag.embedding import embed


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# ============================================================
# RERANK PRODUCTS
# ============================================================

def rerank_products(
        query: str,
        items: List[Dict],
        user: Optional[object] = None
) -> List[Dict]:
    if not items:
        return items

    ranked = []

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    q_vec = embed(query)

    # --------------------------------------------------------
    # PRICE STATISTICS
    # --------------------------------------------------------

    prices = [
        i.get("price")
        for i in items
        if isinstance(i.get("price"), (int, float))
    ]

    avg_price = np.mean(prices) if prices else 0
    std_price = np.std(prices) if prices else 1

    # --------------------------------------------------------
    # USER PREFERENCES
    # --------------------------------------------------------

    fav_brands = []

    if user and getattr(user, "favorite_brands", None):
        fav_brands = [
            b.strip().lower()
            for b in user.favorite_brands.split(",")
        ]

    # --------------------------------------------------------
    # RERANK LOOP
    # --------------------------------------------------------

    for item in items:

        title = item.get("title", "") or ""

        # ----------------------------------------
        # EMBEDDING SIMILARITY
        # ----------------------------------------

        try:

            t_vec = embed(title)

            similarity = cosine_similarity(q_vec, t_vec)

        except Exception:

            similarity = 0.0

        # ----------------------------------------
        # TRUST SIGNALS
        # ----------------------------------------

        trust = item.get("trust_score") or 0

        rating = (item.get("seller_rating") or 0) / 100

        price = item.get("price") or avg_price

        # ----------------------------------------
        # PRICE OUTLIER PENALTY
        # ----------------------------------------

        price_z = abs(price - avg_price) / std_price if std_price else 0

        price_penalty = min(price_z / 5, 0.2)

        # ----------------------------------------
        # PERSONALIZATION BONUS
        # ----------------------------------------

        personalization = 0

        if fav_brands:

            title_lower = title.lower()

            if any(b in title_lower for b in fav_brands):
                personalization += 0.10

        # ----------------------------------------
        # FINAL SCORE
        # ----------------------------------------

        score = (
                0.50 * similarity +
                0.30 * trust +
                0.15 * rating -
                0.05 * price_penalty +
                personalization
        )

        item["_rerank_score"] = round(float(score), 4)

        ranked.append(item)

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    ranked.sort(
        key=lambda x: x.get("_rerank_score", 0),
        reverse=True
    )

    return ranked
