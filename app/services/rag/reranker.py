from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import re

from app.services.rag.cross_encoder import cross_rerank
from app.services.rag.embedding import embed
from app.services.rag.retriever import retrieve_context
from app.services.rag.ltr_features import extract_ltr_features, get_feature_names

import os
import json
import logging

logger = logging.getLogger(__name__)
@dataclass
class RerankWeights:
    SIMILARITY: float = 0.34
    LEXICAL: float = 0.18
    TRUST: float = 0.16
    RATING: float = 0.08
    PRICE_PENALTY: float = 0.05
    
    # Thresholds for explanations
    STRONG_MATCH: float = 0.55
    GOOD_MATCH: float = 0.35
    STRONG_LEXICAL: float = 0.5
    VERY_STRONG_TRUST: float = 0.85
    GOOD_TRUST: float = 0.70

WEIGHTS = RerankWeights()

SCORING_WEIGHTS = {
    "lexical_sim": 0.25,
    "semantic_sim": 0.35,
    "trust_score": 0.15,
    "seller_rating": 0.05,
    "log_price": -0.02,
    "has_image": 0.05,
    "is_new": 0.05,
    "has_brand": 0.05,
    "has_model": 0.03,
    "num_specs": 0.02,
    "price_z": -0.10,  # Penalize high relative price
    "rag_product_boost": 0.20,
    "rag_seller_boost": 0.15,
    "rag_sentiment": 0.05,
    "price_match_constraint": 0.30
}

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
    constraints: Optional[List[Dict]] = None,
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
            item["_semantic_sim"] = similarity
        except Exception:
            similarity = 0.0
            item["_semantic_sim"] = 0.0

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

        price_z = (float(price) - avg_price) / std_price if std_price else 0.0
        
        price_penalty = 0.0
        if price_z < -1.5:
            price_penalty = 0.40  # Massive penalty for suspiciously cheap items (likely accessories/boxes)
        elif price_z > 2.0:
            price_penalty = 0.15  # Small penalty for overly expensive items

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
        # FINAL SCORE
        # ----------------------------------------

        # ----------------------------------------
        # LTR FEATURES & MODEL PREDICTION
        # ----------------------------------------

        # Inject RAG signals into item for feature extraction
        item["_rag_product_boost"] = product_rag_boost
        item["_rag_seller_boost"] = seller_rag_boost
        item["_rag_sentiment_signal"] = seller_sentiment_signal

        # ----------------------------------------
        # LTR FEATURES & MODEL PREDICTION
        # ----------------------------------------

        # Inject RAG signals into item for feature extraction
        item["_rag_product_boost"] = product_rag_boost
        item["_rag_seller_boost"] = seller_rag_boost
        item["_rag_sentiment_signal"] = seller_sentiment_signal

        ltr_context = {
            "avg_price": avg_price,
            "std_price": std_price,
            "constraints": constraints
        }
        
        features = extract_ltr_features(query, item, context=ltr_context)

        # ----------------------------------------
        # WEIGHT-BASED SCORING (Replaces LTR)
        # ----------------------------------------
        
        score = 0.0
        for feat_name, feat_val in features.items():
            weight = SCORING_WEIGHTS.get(feat_name, 0.0)
            score += feat_val * weight

        # Apply personalization (not part of base features)
        score += personalization

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

        if similarity > WEIGHTS.STRONG_MATCH:
            explanations.append("semantic match with the query is strong")
        elif similarity > WEIGHTS.GOOD_MATCH:
            explanations.append("semantic match with the query is good")

        if lex_score > WEIGHTS.STRONG_LEXICAL:
            explanations.append("title has strong lexical overlap with the query")

        if trust > WEIGHTS.VERY_STRONG_TRUST:
            explanations.append("seller has very strong trust score")
        elif trust > WEIGHTS.GOOD_TRUST:
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

    top_k = ranked[:10]
    top_k = cross_rerank(query, top_k)

    for item in top_k:
        cross_score = float(item.get("_cross_score", 0.0))
        base_score = float(item.get("_rerank_score", 0.0))

        # Normalize cross-encoder logit into a probability (0-1) via sigmoid
        prob_score = 1.0 / (1.0 + np.exp(-cross_score))

        # Blend: give cross-encoder majority weight (70%) over the heuristical base (30%)
        item["_final_score"] = round((0.70 * prob_score) + (0.30 * base_score), 4)

    top_k.sort(
        key=lambda x: x.get("_final_score", x.get("_rerank_score", 0)),
        reverse=True
    )

    remainder = ranked[10:]

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