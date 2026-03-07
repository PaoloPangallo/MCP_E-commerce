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
        user: Optional[object] = None,
        parsed_query: Optional[Dict] = None
) -> List[Dict]:
    if not items:
        return items

    ranked = []

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------
    q_vec = embed(query)
    
    # --------------------------------------------------------
    # KEYWORD EXTRACTION (per Keyword Resonance)
    # --------------------------------------------------------
    # Se abbiamo il parsed_query, usiamo quello, altrimenti proviamo a estrarre termini "caldi"
    resonance_words = set()
    if parsed_query:
        # Estraiamo brands, constraints, preferences
        for k in ["brands", "preferences", "product"]:
            val = parsed_query.get(k)
            if isinstance(val, list):
                resonance_words.update([str(v).lower() for v in val])
            elif val:
                resonance_words.update(str(val).lower().split())
        
        # Aggiungiamo vincoli di prezzo/colore se presenti
        for c in parsed_query.get("constraints", []):
            if isinstance(c, dict) and "value" in c:
                resonance_words.add(str(c["value"]).lower())
    else:
        # Fallback: estraiamo parole significative dalla query string (escludendo stop words comuni)
        stops = {"e", "o", "con", "di", "per", "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "del", "dei", "degli", "delle", "sto", "cercando", "un", "una"}
        resonance_words = {w.lower() for w in query.split() if w.lower() not in stops and len(w) > 2}

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
        title_lower = title.lower()

        # ----------------------------------------
        # EMBEDDING SIMILARITY
        # ----------------------------------------
        try:
            t_vec = embed(title)
            similarity = cosine_similarity(q_vec, t_vec)
        except Exception:
            similarity = 0.0

        # ----------------------------------------
        # KEYWORD RESONANCE BONUS (Attributi specifici: colore, ram, GB)
        # ----------------------------------------
        resonance_bonus = 0.0
        matches = 0
        for word in resonance_words:
            if word in title_lower:
                matches += 1
                resonance_bonus += 0.08 # Bonus per ogni parola chiave matchata
        
        # Bonus extra per match multipli (molto rilevante per es. "iPhone 13 bianco 128gb")
        if matches >= 2:
            resonance_bonus += 0.10

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
            if any(b in title_lower for b in fav_brands):
                personalization += 0.10

        # ----------------------------------------
        # FINAL SCORE (Bilanciato: 40% Simil, 20% Resonance, 25% Trust, 15% Price/Rating)
        # ----------------------------------------
        score = (
                0.40 * similarity +
                0.20 * resonance_bonus +
                0.25 * trust +
                0.10 * rating -
                0.05 * price_penalty +
                personalization
        )

        item["_rerank_score"] = round(float(score), 4)
        # Aggiungiamo un'annotazione per l'explanation
        if resonance_bonus > 0:
            if not item.get("explanations"): item["explanations"] = []
            item["explanations"].append(f"Highly relevant match for specific attributes from your query.")

        ranked.append(item)

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------
    ranked.sort(
        key=lambda x: x.get("_rerank_score", 0),
        reverse=True
    )

    return ranked
