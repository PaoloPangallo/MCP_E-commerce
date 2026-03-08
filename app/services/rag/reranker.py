import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Optional
import numpy as np

# ============================================================
# GPU CONFIG & CROSS-ENCODER LOAD
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
# Usiamo un Cross-Encoder (come suggerito nell'articolo) per una precisione estrema
_reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)

# ============================================================
# RERANK PRODUCTS (POTENZIATO CON CROSS-ENCODER)
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
    
    # Prepariamo le coppie (query, titolo) per il calcolo batch su GPU
    # Il Cross-Encoder lavora sulle coppie per massimizzare la 'cross-attention'
    pairs = [[query, i.get("title") or ""] for i in items]
    
    try:
        # Calcoliamo tutti gli score di rilevanza con un singolo passaggio in GPU
        cross_scores = _reranker.predict(pairs, batch_size=len(items))
    except Exception:
        # Fallback in caso di errore (es: vram out)
        cross_scores = [0.0] * len(items)


    # --------------------------------------------------------
    # KEYWORD EXTRACTION (per Keyword Resonance)
    # --------------------------------------------------------
    resonance_words = set()
    if parsed_query:
        for k in ["brands", "preferences", "product"]:
            val = parsed_query.get(k)
            if isinstance(val, list):
                resonance_words.update([str(v).lower() for v in val])
            elif val:
                resonance_words.update(str(val).lower().split())
        for c in parsed_query.get("constraints", []):
            if isinstance(c, dict) and "value" in c:
                resonance_words.add(str(c["value"]).lower())
    else:
        stops = {"e", "o", "con", "di", "per", "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "del", "dei", "degli", "delle", "sto", "cercando"}
        resonance_words = {w.lower() for w in query.split() if w.lower() not in stops and len(w) > 2}

    # --------------------------------------------------------
    # PRICE STATISTICS
    # --------------------------------------------------------
    prices = [i.get("price") for i in items if isinstance(i.get("price"), (int, float))]
    avg_price = np.mean(prices) if prices else 0
    std_price = np.std(prices) if prices else 1

    # --------------------------------------------------------
    # USER PREFERENCES
    # --------------------------------------------------------
    fav_brands = []
    if user and getattr(user, "favorite_brands", None):
        fav_brands = [b.strip().lower() for b in user.favorite_brands.split(",")]

    # --------------------------------------------------------
    # RERANK LOOP
    # --------------------------------------------------------
    for idx, item in enumerate(items):
        title = item.get("title", "") or ""
        title_lower = title.lower()

        # RILEVANZA SEMANTICA (da Cross-Encoder in GPU)
        semantic_relevance = float(cross_scores[idx])
        # Normalizzazione logistica semplice per portarlo verso 0-1 se necessario, 
        # ma i cross-encoder spesso funzionano bene anche come score assoluti.
        similarity = 1 / (1 + np.exp(-semantic_relevance))

        # KEYWORD RESONANCE BONUS
        resonance_bonus = 0.0
        matches = 0
        for word in resonance_words:
            if word in title_lower:
                matches += 1
                resonance_bonus += 0.08
        if matches >= 2: resonance_bonus += 0.10

        # TRUST & PRICE SIGNALS
        trust = item.get("trust_score") or 0
        rating = (item.get("seller_rating") or 0) / 100
        price = item.get("price") or avg_price
        price_z = abs(price - avg_price) / std_price if std_price else 0
        price_penalty = min(price_z / 5, 0.2)

        # PERSONALIZATION
        personalization = 0.10 if fav_brands and any(b in title_lower for b in fav_brands) else 0

        # FINAL SCORE
        score = (
                0.50 * similarity +    # Maggiore peso alla precisione del Cross-Encoder
                0.15 * resonance_bonus +
                0.20 * trust +
                0.10 * rating -
                0.05 * price_penalty +
                personalization
        )

        item["_rerank_score"] = round(float(score), 4)
        item["_cross_encoder_raw"] = round(semantic_relevance, 2)
        
        if resonance_bonus > 0:
            if not item.get("explanations"): item["explanations"] = []
            item["explanations"].append("Ottimo match semantico con la tua ricerca.")

        ranked.append(item)

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------
    ranked.sort(
        key=lambda x: x.get("_rerank_score", 0),
        reverse=True
    )

    return ranked
