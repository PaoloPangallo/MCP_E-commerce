from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session

from app.models.listing import Listing
from app.services.ebay import search_items
from app.services.feedback import get_seller_feedback
from app.services.metrics.ir_metrics import ndcg_at_k, precision_at_k, recall_at_k
from app.services.nlp_sentiment import compute_sentiment_score
from app.services.parser import parse_query_service
from app.services.rag.context_builder import build_context
from app.services.rag.explainer import explain_results
from app.services.rag.retriever import retrieve_context
from app.services.rag.reranker import rerank_products
from app.services.rag.query_expansion import expand_query
from app.services.trust import compute_trust_score
from app.services.user_profiling import update_user_profile

logger = logging.getLogger(__name__)

# ============================================================
# IN-MEMORY CACHES
# ============================================================

_SELLER_FEEDBACK_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_SELLER_TRUST_CACHE: Dict[str, Dict[str, float]] = {}

MAX_RESULTS_FROM_EBAY = 20
MAX_SELLERS_FOR_TRUST = 5
MAX_FEEDBACK_PER_SELLER = 40
FEEDBACK_WORKERS = 6


def _normalize_llm_engine(llm_engine: str) -> str:
    llm_engine = (llm_engine or "").strip().lower()
    if llm_engine in {"gemini", "ollama", "rule_based"}:
        return llm_engine
    return "ollama"


def _dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_ids: Set[str] = set()
    deduped: List[Dict[str, Any]] = []

    for item in items:
        ebay_id = item.get("ebay_id")
        if not ebay_id or ebay_id in seen_ids:
            continue

        seen_ids.add(ebay_id)
        deduped.append(item)

    return deduped


def _build_ebay_query(parsed: Dict[str, Any], fallback_query: str) -> str:
    product = parsed.get("product")
    brands = parsed.get("brands") or []

    if product and brands:
        return f"{product} {' '.join(brands)}"

    if product:
        return str(product)

    return parsed.get("semantic_query") or fallback_query


def _fetch_feedback_cached(seller_name: str, limit: int = MAX_FEEDBACK_PER_SELLER) -> List[Dict[str, Any]]:
    key = seller_name.strip().lower()

    if key in _SELLER_FEEDBACK_CACHE:
        return _SELLER_FEEDBACK_CACHE[key]

    feedbacks = get_seller_feedback(seller_name, limit=limit) or []
    _SELLER_FEEDBACK_CACHE[key] = feedbacks
    return feedbacks


def _compute_seller_trust_cached(seller_name: str) -> Optional[float]:
    seller_key = seller_name.strip().lower()
    feedbacks = _fetch_feedback_cached(seller_name, limit=MAX_FEEDBACK_PER_SELLER)

    if not feedbacks:
        return None

    cached = _SELLER_TRUST_CACHE.get(seller_key)
    if cached and int(cached.get("count", -1)) == len(feedbacks):
        return round(float(cached["trust_score"]), 3)

    sentiment_score = compute_sentiment_score(feedbacks, max_texts=20)
    trust_score = compute_trust_score(feedbacks, sentiment_score=sentiment_score)

    _SELLER_TRUST_CACHE[seller_key] = {
        "count": float(len(feedbacks)),
        "sentiment_score": float(sentiment_score),
        "trust_score": float(trust_score),
    }

    return round(float(trust_score), 3)


def _prefetch_top_sellers_feedback(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute trust only for the most relevant sellers, not for everybody.
    This avoids N expensive feedback lookups on each request.
    """
    sellers: List[str] = []
    seen = set()

    for item in items:
        seller = item.get("seller_name")
        if seller and seller not in seen:
            seen.add(seller)
            sellers.append(seller)

        if len(sellers) >= MAX_SELLERS_FOR_TRUST:
            break

    scores: Dict[str, float] = {}
    if not sellers:
        return scores

    with ThreadPoolExecutor(max_workers=FEEDBACK_WORKERS) as executor:
        futures = {
            executor.submit(_compute_seller_trust_cached, seller): seller
            for seller in sellers
        }

        for future in as_completed(futures):
            seller = futures[future]
            try:
                score = future.result()
                if score is not None:
                    scores[seller] = score
            except Exception:
                logger.warning("Trust computation failed for seller=%s", seller)

    return scores


def _batch_fetch_existing_ebay_ids(db: Session, items: List[Dict[str, Any]]) -> Set[str]:
    ebay_ids = [item.get("ebay_id") for item in items if item.get("ebay_id")]
    if not ebay_ids:
        return set()

    rows = (
        db.query(Listing.ebay_id)
        .filter(Listing.ebay_id.in_(ebay_ids))
        .all()
    )

    return {row[0] for row in rows}


def _prepare_and_persist_items(
    db: Session,
    items: List[Dict[str, Any]],
    seller_trust_map: Dict[str, float],
) -> tuple[List[Dict[str, Any]], int]:
    existing_ids = _batch_fetch_existing_ebay_ids(db, items)
    saved_count = 0
    results_out: List[Dict[str, Any]] = []

    for item in items:
        ebay_id = item.get("ebay_id")
        if not ebay_id:
            continue

        already = ebay_id in existing_ids

        if not already:
            listing = Listing(
                ebay_id=ebay_id,
                title=item.get("title"),
                price=item.get("price"),
                currency=item.get("currency"),
                condition=item.get("condition"),
                seller_name=item.get("seller_name"),
                seller_rating=item.get("seller_rating"),
                url=item.get("url"),
                image_url=item.get("image_url"),
            )
            db.add(listing)
            saved_count += 1

        seller_name = item.get("seller_name")
        trust_score = seller_trust_map.get(seller_name) if seller_name else None

        item_copy = dict(item)
        item_copy["_already_in_db"] = already
        item_copy["trust_score"] = trust_score
        results_out.append(item_copy)

    return results_out, saved_count


def _apply_final_ranking(
    items: List[Dict[str, Any]],
    user: Optional[object] = None,
) -> None:
    for item in items:
        relevance = float(item.get("_rerank_score", 0) or 0)
        trust = float(item.get("trust_score") or 0)
        price = item.get("price") or 0

        price_score = 0.0
        try:
            if price:
                price_score = max(0.0, 1.0 - float(price) / 1000.0)
        except Exception:
            price_score = 0.0

        ranking_score = (
            0.55 * relevance
            + 0.25 * trust
            + 0.10 * price_score
        )

        explanations = []

        if user:
            favorite_brands = getattr(user, "favorite_brands", None)
            if favorite_brands:
                brands_pref = {
                    b.strip().lower()
                    for b in favorite_brands.split(",")
                    if b.strip()
                }

                title = (item.get("title") or "").lower()
                for b in brands_pref:
                    if b in title:
                        ranking_score += 0.10
                        item["brand_match"] = True
                        explanations.append(f"This item matches your preferred brand '{b}'.")
                        break

            price_pref = getattr(user, "price_preference", None)
            if price_pref and price:
                try:
                    pref = float(price_pref)
                    if float(price) <= pref:
                        ranking_score += 0.05
                        item["price_match"] = True
                        explanations.append("This product falls within your typical price range.")
                except Exception:
                    pass

        if trust >= 0.8:
            explanations.append("Seller has very strong feedback and trust score.")
        elif trust >= 0.6:
            explanations.append("Seller shows generally positive feedback.")

        item["ranking_score"] = round(ranking_score, 3)
        if explanations:
            item["explanations"] = explanations


def run_search_pipeline(
    query: str,
    db: Session,
    user: Optional[object] = None,
    llm_engine: str = "gemini",
) -> Dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("Query vuota")

    llm_engine = _normalize_llm_engine(llm_engine)
    t0 = time.time()
    timings: Dict[str, float] = {}

    # ============================================================
    # 1) PARSE QUERY
    # ============================================================

    logger.info("PIPELINE STEP 1 parse_query")

    t = time.time()
    parsed = parse_query_service(
        query,
        use_llm=(llm_engine != "rule_based"),
        include_meta=True,
    )
    timings["parse_query_s"] = round(time.time() - t, 3)

    ebay_query_used = _build_ebay_query(parsed, query)

    # ============================================================
    # 2) USER PROFILE UPDATE (NO INTERNAL COMMIT)
    # ============================================================
    logger.info("PIPELINE STEP 2 ebay_search")
    if user:
        try:
            update_user_profile(user, parsed, db)
        except Exception:
            logger.warning("User profiling update failed")

    # ============================================================
    # 3) EBAY SEARCH
    # ============================================================
    logger.info("PIPELINE STEP 3 rerank")

    t = time.time()
    items = search_items(parsed_query=parsed, limit=MAX_RESULTS_FROM_EBAY) or []
    timings["ebay_search_s"] = round(time.time() - t, 3)

    items = _dedupe_items(items)

    # ============================================================
    # 4) RERANK EARLY
    # ============================================================
    logger.info("PIPELINE STEP 4 seller_trust")

    t = time.time()
    if items:
        try:
            items = rerank_products(query, items, user=user)
        except Exception:
            logger.warning("Rerank failed, keeping original order")
    timings["rerank_s"] = round(time.time() - t, 3)

    # ============================================================
    # 5) TRUST ONLY FOR TOP SELLERS
    # ============================================================

    t = time.time()
    seller_trust_map = _prefetch_top_sellers_feedback(items[:MAX_SELLERS_FOR_TRUST * 2])
    timings["seller_trust_s"] = round(time.time() - t, 3)

    # ============================================================
    # 6) SAVE DB + PREP RESULTS
    # ============================================================

    t = time.time()
    results_out, saved_count = _prepare_and_persist_items(
        db=db,
        items=items,
        seller_trust_map=seller_trust_map,
    )
    timings["db_prepare_s"] = round(time.time() - t, 3)

    # ============================================================
    # 7) FINAL RANKING
    # ============================================================

    _apply_final_ranking(results_out, user=user)

    # ============================================================
    # 8) RAG RETRIEVE ONLY
    # NOTE: removed runtime ingest_seller_feedback
    # ============================================================

    t = time.time()
    try:
        expanded_query = expand_query(query)
        logger.info(f"Query expansion: '{query}' -> '{expanded_query}'")
        rag_docs_after = retrieve_context(expanded_query, k=10)
    except Exception as e:
        logger.warning(f"RAG retrieve failed: {e}")
        rag_docs_after = []
    timings["rag_retrieve_s"] = round(time.time() - t, 3)

    for item in results_out:
        seller_name = item.get("seller_name")
        if not seller_name:
            item["rag_feedback"] = []
            continue

        item["rag_feedback"] = [
            d for d in rag_docs_after
            if d.get("seller") == seller_name
        ][:3]

    rag_context_text = build_context(query, results_out, rag_docs_after)

    # ============================================================
    # 9) OPTIONAL EXPLANATION
    # ============================================================

    t = time.time()
    try:
        analysis = explain_results(query, results_out[:5]) if results_out else None
    except Exception:
        analysis = None
    timings["explain_s"] = round(time.time() - t, 3)

    # ============================================================
    # 10) IR METRICS
    # ============================================================

    binary_relevance = [
        1 if (item.get("ranking_score", 0) >= 0.75) else 0
        for item in results_out
    ]

    metrics = {
        "precision@5": precision_at_k(binary_relevance, 5),
        "precision@10": precision_at_k(binary_relevance, 10),
        "recall@10": recall_at_k(
            binary_relevance,
            total_relevant=sum(binary_relevance),
            k=10
        ),
        "ndcg@10": ndcg_at_k(binary_relevance, 10),
    }

    # ============================================================
    # FINAL COMMIT
    # ============================================================

    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("DB commit failed; transaction rolled back")

    timings["total_s"] = round(time.time() - t0, 3)

    return {
        "parsed_query": parsed,
        "ebay_query_used": ebay_query_used,
        "results_count": len(results_out),
        "saved_new_count": saved_count,
        "analysis": analysis,
        "results": results_out,
        "rag_context": rag_context_text,
        "metrics": metrics,
        "_timings": timings,
    }