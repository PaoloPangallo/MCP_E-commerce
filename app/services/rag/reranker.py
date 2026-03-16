from typing import List, Dict, Optional, Tuple
import numpy as np
import re

from app.services.rag.cross_encoder import cross_rerank
from app.services.rag.embedding import embed
from app.services.rag.retriever import retrieve_context


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ============================================================
# SIMPLE LEXICAL MATCH
# ============================================================

def lexical_score(query: str, title: str) -> float:
    q_tokens = set(re.findall(r"\w+", (query or "").lower()))
    t_tokens = set(re.findall(r"\w+", (title or "").lower()))

    if not q_tokens:
        return 0.0

    overlap = q_tokens.intersection(t_tokens)
    return len(overlap) / len(q_tokens)


# ============================================================
# ACCESSORY DETECTION
# ============================================================

ACCESSORY_WORDS = [
    "case",
    "cover",
    "charger",
    "caricatore",
    "cavo",
    "cable",
    "vetro",
    "pellicola",
    "screen protector",
    "glass",
    "custodia",
    "adattatore",
    "adapter",
]

# Non usiamo più POSITIVE_HINTS/NEGATIVE_HINTS ma le label LLM-based ("POSITIVE", "NEGATIVE", "NEUTRAL")


def accessory_penalty(query: str, title: str) -> float:
    q = (query or "").lower()
    t = (title or "").lower()

    # penalizza accessori se l'utente sembra cercare un device
    device_terms = ["iphone", "samsung", "galaxy", "pixel", "smartphone", "telefono"]
    query_has_device_intent = any(term in q for term in device_terms)

    if query_has_device_intent and any(w in t for w in ACCESSORY_WORDS):
        return 0.15

    return 0.0


# ============================================================
# TEXT HELPERS
# ============================================================

def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split()).strip()


def _title_tokens(text: str) -> set:
    return set(re.findall(r"\w+", _normalize_text(text)))


def _token_overlap_ratio(a: str, b: str) -> float:
    ta = _title_tokens(a)
    tb = _title_tokens(b)

    if not ta or not tb:
        return 0.0

    inter = ta.intersection(tb)
    return len(inter) / max(1, min(len(ta), len(tb)))


# ============================================================
# RAG SIGNAL EXTRACTION
# ============================================================

def _retrieve_rag_evidence(query: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Recupera evidenze RAG separate per:
    - product docs
    - seller feedback docs
    """
    product_docs = retrieve_context(
        query=query,
        k=8,
        doc_type="product",
        per_source=12,
    )

    seller_docs = retrieve_context(
        query=query,
        k=8,
        doc_type="seller_feedback",
        per_source=12,
    )

    return product_docs, seller_docs


def _compute_product_rag_signal(item: Dict, product_docs: List[Dict]) -> Tuple[float, List[str], List[Dict]]:
    """
    Segnale RAG legato al match prodotto <-> documenti recuperati.
    """
    title = item.get("title") or ""
    seller = (item.get("seller_name") or item.get("seller_username") or "").strip().lower()

    if not title or not product_docs:
        return 0.0, [], []

    best_signal = 0.0
    reasons: List[str] = []
    matched_docs: List[Dict] = []

    for doc in product_docs:
        doc_text = doc.get("text") or ""
        doc_seller = (doc.get("seller") or "").strip().lower()

        overlap = _token_overlap_ratio(title, doc_text)
        rrf = float(doc.get("_rrf_score") or 0.0)

        seller_match_bonus = 0.08 if seller and doc_seller and seller == doc_seller else 0.0
        signal = min(0.22, 0.14 * overlap + seller_match_bonus + min(rrf * 2.0, 0.04))

        if signal > 0.03:
            matched_docs.append(doc)

        if signal > best_signal:
            best_signal = signal

    if best_signal > 0.10:
        reasons.append("retrieved product context strongly matches this listing")
    elif best_signal > 0.05:
        reasons.append("retrieved product context supports this listing")

    return best_signal, reasons, matched_docs[:3]


def _compute_seller_rag_signal(item: Dict, seller_docs: List[Dict]) -> Tuple[float, float, List[str], List[Dict]]:
    """
    Restituisce:
    - seller_rag_boost
    - seller_sentiment_signal (positivo o negativo)
    - reasons
    - matched feedback docs
    """
    seller = (item.get("seller_name") or item.get("seller_username") or "").strip().lower()

    if not seller or not seller_docs:
        return 0.0, 0.0, [], []

    matched = []
    pos_hits = 0
    neg_hits = 0
    rrf_sum = 0.0

    for doc in seller_docs:
        doc_seller = (doc.get("seller") or "").strip().lower()
        if not doc_seller or doc_seller != seller:
            continue

        text = _normalize_text(doc.get("text") or "")
        if not text:
            continue

        matched.append(doc)
        rrf_sum += float(doc.get("_rrf_score") or 0.0)

        sentiment = doc.get("sentiment_label", "NEUTRAL")
        if sentiment == "POSITIVE":
            pos_hits += 1
        elif sentiment == "NEGATIVE":
            neg_hits += 1

    if not matched:
        return 0.0, 0.0, [], []

    seller_rag_boost = min(0.18, 0.04 * len(matched) + min(rrf_sum, 0.06))

    if pos_hits > neg_hits:
        seller_sentiment_signal = min(0.08, 0.02 * (pos_hits - neg_hits))
    elif neg_hits > pos_hits:
        seller_sentiment_signal = -min(0.08, 0.02 * (neg_hits - pos_hits))
    else:
        seller_sentiment_signal = 0.0

    reasons: List[str] = []
    if seller_rag_boost > 0.08:
        reasons.append("seller feedback evidence is strong")
    elif seller_rag_boost > 0.03:
        reasons.append("seller feedback evidence is available")

    if seller_sentiment_signal > 0:
        reasons.append("recent feedback is mostly positive")
    elif seller_sentiment_signal < 0:
        reasons.append("recent feedback contains negative signals")

    return seller_rag_boost, seller_sentiment_signal, reasons, matched[:3]


# ============================================================
# RERANK PRODUCTS
# ============================================================

def rerank_products(
    query: str,
    items: List[Dict],
    user: Optional[object] = None,
    product_docs: Optional[List[Dict]] = None,
    seller_docs: Optional[List[Dict]] = None,
) -> List[Dict]:

    if not items:
        return items

    ranked = []

    # --------------------------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------------------------

    q_vec = embed(query)

    # --------------------------------------------------------
    # RAG EVIDENCE (use pre-fetched if available)
    # --------------------------------------------------------

    if product_docs is None or seller_docs is None:
        product_docs, seller_docs = _retrieve_rag_evidence(query)

    # --------------------------------------------------------
    # PRICE STATISTICS
    # --------------------------------------------------------

    prices = [
        i.get("price")
        for i in items
        if isinstance(i.get("price"), (int, float))
    ]

    avg_price = float(np.mean(prices)) if prices else 0.0
    std_price = float(np.std(prices)) if prices else 1.0
    if std_price == 0:
        std_price = 1.0

    # --------------------------------------------------------
    # USER PREFERENCES
    # --------------------------------------------------------

    fav_brands = []

    if user and getattr(user, "favorite_brands", None):
        fav_brands = [
            b.strip().lower()
            for b in user.favorite_brands.split(",")
            if b and b.strip()
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
            if "_embedding" in item:
                t_vec = item["_embedding"]
            else:
                t_vec = embed(title)
                item["_embedding"] = t_vec

            similarity = cosine_similarity(q_vec, t_vec)
        except Exception:
            similarity = 0.0

        # ----------------------------------------
        # LEXICAL MATCH
        # ----------------------------------------

        lex_score = lexical_score(query, title)

        # ----------------------------------------
        # TRUST SIGNALS
        # ----------------------------------------

        trust = float(item.get("trust_score") or 0.0)
        rating = float(item.get("seller_rating") or 0.0) / 100.0
        trust_boost = trust ** 1.1

        # ----------------------------------------
        # PRICE NORMALIZATION
        # ----------------------------------------

        price = item.get("price")
        if not isinstance(price, (int, float)):
            price = avg_price

        price_z = abs(float(price) - avg_price) / std_price if std_price else 0.0
        price_penalty = min(price_z / 5.0, 0.2)

        # ----------------------------------------
        # ACCESSORY PENALTY
        # ----------------------------------------

        acc_penalty = accessory_penalty(query, title)

        # ----------------------------------------
        # TITLE QUALITY
        # ----------------------------------------

        length_penalty = 0.05 if len(title.split()) > 20 else 0.0

        # ----------------------------------------
        # PERSONALIZATION BONUS
        # ----------------------------------------

        personalization = 0.0
        if fav_brands and any(b in title_lower for b in fav_brands):
            personalization += 0.10

        # ----------------------------------------
        # RAG SIGNALS
        # ----------------------------------------

        product_rag_boost, product_rag_reasons, matched_product_docs = _compute_product_rag_signal(
            item=item,
            product_docs=product_docs,
        )

        seller_rag_boost, seller_sentiment_signal, seller_rag_reasons, matched_seller_docs = _compute_seller_rag_signal(
            item=item,
            seller_docs=seller_docs,
        )

        # ----------------------------------------
        # COLOR & FIT MATCHING (CRITICAL)
        # ----------------------------------------
        
        color_penalty = 0.0
        fit_boost = 0.0
        
        # Color match logic (simple but effective for Italian/English)
        colors_requested = []
        if "nero" in query.lower() or "neri" in query.lower() or "black" in query.lower():
            colors_requested.append("black")
        if "blu" in query.lower() or "blue" in query.lower():
            colors_requested.append("blue")
        if "bianco" in query.lower() or "white" in query.lower():
            colors_requested.append("white")
            
        # Brand match penalty (CRITICAL)
        brand_penalty = 0.0
        requested_brands = []
        # Fallback list of known brands from query (simple scan)
        from app.services.parser import BRAND_WHITELIST
        for b_low, b_canonical in BRAND_WHITELIST.items():
            if b_low in query.lower():
                requested_brands.append(b_canonical.lower())
        
        if requested_brands:
            # If ANY requested brand is NOT found in the title, apply penalty
            found_any = False
            for rb in requested_brands:
                if rb in title_lower:
                    found_any = True
                    break
            if not found_any:
                brand_penalty = 0.45 # Massive penalty for wrong brand
            
        conflicting_colors = {
            "black": ["blu", "blue", "bianco", "white", "rosso", "red", "chiaro", "light"],
            "blue": ["nero", "black", "bianco", "white", "rosso", "red"],
        }
        
        for cr in colors_requested:
            if cr in conflicting_colors:
                for conflict in conflicting_colors[cr]:
                    if conflict in title_lower:
                        color_penalty += 0.20 # Sharp penalty for wrong color
                        
        # Fit match booster
        if "baggy" in query.lower() or "relaxed" in query.lower() or "larghi" in query.lower():
            if any(w in title_lower for w in ["baggy", "relaxed", "loose", "wide", "ampio", "largo"]):
                fit_boost += 0.15
            elif any(w in title_lower for w in ["slim", "skinny", "tight", "stretto"]):
                color_penalty += 0.15 # Using same accumulator for generic fit penalty
        
        # ----------------------------------------
        # FINAL SCORE
        # ----------------------------------------

        score = (
            0.34 * similarity +
            0.18 * lex_score +
            0.16 * trust_boost +
            0.08 * rating -
            0.05 * price_penalty -
            acc_penalty -
            length_penalty -
            color_penalty -
            brand_penalty +
            fit_boost +
            personalization +
            product_rag_boost +
            seller_rag_boost +
            seller_sentiment_signal
        )

        item["_rerank_score"] = round(float(score), 4)
        item["_rag_product_boost"] = round(float(product_rag_boost), 4)
        item["_rag_seller_boost"] = round(float(seller_rag_boost), 4)
        item["_rag_sentiment_signal"] = round(float(seller_sentiment_signal), 4)

        # feedback/evidence utili per explainability
        rag_feedback = []
        for d in matched_seller_docs[:2]:
            rag_feedback.append(
                {
                    "comment": d.get("text"),
                    "seller": d.get("seller"),
                    "rrf_score": d.get("_rrf_score"),
                    "sources": d.get("_sources") or [],
                }
            )

        item["rag_feedback"] = rag_feedback
        item["rag_product_context"] = [
            {
                "text": d.get("text"),
                "seller": d.get("seller"),
                "rrf_score": d.get("_rrf_score"),
                "sources": d.get("_sources") or [],
            }
            for d in matched_product_docs[:2]
        ]

        explanations = list(item.get("explanations") or [])

        if similarity > 0.55:
            explanations.append("semantic match with the query is strong")
        elif similarity > 0.35:
            explanations.append("semantic match with the query is good")

        if lex_score > 0.5:
            explanations.append("title has strong lexical overlap with the query")

        if trust > 0.85:
            explanations.append("seller has very strong trust score")
        elif trust > 0.70:
            explanations.append("seller shows generally positive feedback")

        if acc_penalty > 0:
            explanations.append("listing may be an accessory rather than the main device")

        explanations.extend(product_rag_reasons)
        explanations.extend(seller_rag_reasons)

        # dedup explanations preserving order
        seen = set()
        clean_explanations = []
        for reason in explanations:
            if not reason:
                continue
            if reason in seen:
                continue
            seen.add(reason)
            clean_explanations.append(reason)

        item["explanations"] = clean_explanations[:6]

        ranked.append(item)

    # --------------------------------------------------------
    # SORT BY FEATURE SCORE
    # --------------------------------------------------------

    ranked.sort(
        key=lambda x: x.get("_rerank_score", 0),
        reverse=True
    )

    # --------------------------------------------------------
    # CROSS-ENCODER ON TOP-K
    # --------------------------------------------------------

    top_k = ranked[:5]
    top_k = cross_rerank(query, top_k)

    # fuse cross score with previous score, not replace it
    for item in top_k:
        cross_score = float(item.get("_cross_score", 0.0))
        base_score = float(item.get("_rerank_score", 0.0))

        # squash cross score into a bounded bonus
        cross_bonus = max(min(cross_score / 20.0, 0.15), -0.05)

        item["_final_score"] = round(base_score + cross_bonus, 4)

    top_k.sort(
        key=lambda x: x.get("_final_score", x.get("_rerank_score", 0)),
        reverse=True
    )

    remainder = ranked[5:]

    for item in remainder:
        item["_final_score"] = item.get("_rerank_score", 0)

    ranked = top_k + remainder

    # --------------------------------------------------------
    # CLEAN NON SERIALIZABLE FIELDS
    # --------------------------------------------------------

    for item in ranked:

        # numpy vectors
        if "_embedding" in item:
            del item["_embedding"]

        # internal signals not needed in API
        if "_similarity" in item:
            del item["_similarity"]

        # keep cross score only for top_k if you want
        # otherwise remove
        # if "_cross_score" in item:
        #     del item["_cross_score"]

    return ranked