import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session

from app.db.redis import redis_client
from app.config.cache import SEARCH_PIPELINE_TTL
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
from app.services.ebay_metadata import get_return_policies

logger = logging.getLogger(__name__)

_CACHE_TTL = SEARCH_PIPELINE_TTL

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
        if ebay_id:
            if ebay_id in seen_ids:
                continue
            seen_ids.add(ebay_id)
        
        deduped.append(item)

    return deduped


def _build_ebay_query(parsed: Dict[str, Any], fallback_query: str) -> str:
    parts: List[str] = []
    
    brands = parsed.get("brands") or []
    product = parsed.get("product")
    
    if brands:
        parts.extend(str(b).strip() for b in brands if str(b).strip())
    if product:
        parts.append(str(product).strip())
        
    # Includiamo constraints testuali (no price/condition)
    constraints = parsed.get("constraints") or []
    for c in constraints:
        ctype = c.get("type")
        val = c.get("value")
        if ctype not in ("price", "condition", "aspect") and val:
            # Gestione negazioni per query eBay
            vals = val if isinstance(val, list) else [val]
            for v in vals:
                v_str = str(v).strip()
                # Se la query originale contiene "no", "senza", "nè" riferiti a questo valore
                # cerchiamo di usare l'operatore NOT di eBay (-)
                neg_patterns = [r"\bno\b", r"\bsenza\b", r"\bn[èe]\b"]
                orig_low = fallback_query.lower()
                is_negated = any(re.search(pf + r"\s+" + re.escape(v_str.lower()), orig_low) for pf in neg_patterns)
                
                if is_negated:
                    parts.append(f"-{v_str}")
                else:
                    parts.append(v_str)
                
    if not parts:
        return fallback_query
        
    # Deduplicazione parole mantenendo ordine
    final_tokens: List[str] = []
    seen_tokens_low = set()
    
    raw_query = " ".join(parts)
    # LOG PER ANALISI PARSER (Richiesta Utente)
    print(f"\n[PARSER ANALYSIS] EBAY QUERY: {raw_query}\n")
    logger.info("EBAY QUERY: %s", raw_query)
    
    for word in raw_query.split():
        w_low = word.lower()
        if w_low not in seen_tokens_low:
            final_tokens.append(word)
            seen_tokens_low.add(w_low)
            
    return " ".join(final_tokens).strip()


async def _fetch_feedback_cached(seller_name: str, limit: int = MAX_FEEDBACK_PER_SELLER) -> List[Dict[str, Any]]:
    seller_key = seller_name.strip().lower()
    cache_key = f"seller_feedback:{seller_key}"
    
    cached = redis_client.get_json(cache_key)
    if cached is not None:
        return cached

    # get_seller_feedback is now async
    feedbacks = await get_seller_feedback(seller_name, limit=limit) or []
    redis_client.set_json(cache_key, feedbacks, ttl_seconds=int(_CACHE_TTL))
    return feedbacks


async def _compute_seller_trust_cached(seller_name: str) -> Optional[float]:
    seller_key = seller_name.strip().lower()
    cache_key = f"seller_trust:{seller_key}"
    
    feedbacks = await _fetch_feedback_cached(seller_name, limit=MAX_FEEDBACK_PER_SELLER)
    if not feedbacks:
        return None

    cached = redis_client.get_json(cache_key)
    if cached is not None:
        if int(cached.get("count", -1)) == len(feedbacks):
            return round(float(cached["trust_score"]), 3)

    # These are CPU bound, but let's keep them in thread for now if complex, 
    # or just run them here if they are fast enough.
    sentiment_score = await asyncio.to_thread(compute_sentiment_score, feedbacks, max_texts=20)
    trust_score = await asyncio.to_thread(compute_trust_score, feedbacks, sentiment_score=sentiment_score)

    result = {
        "count": float(len(feedbacks)),
        "sentiment_score": float(sentiment_score),
        "trust_score": float(trust_score),
    }
    redis_client.set_json(cache_key, result, ttl_seconds=int(_CACHE_TTL))

    return round(float(trust_score), 3)


async def _prefetch_top_sellers_feedback(items: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute trust only for the most relevant sellers using async gather.
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

    # Parallel async execution
    tasks = [_compute_seller_trust_cached(s) for s in sellers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for seller, score in zip(sellers, results):
        if isinstance(score, (float, int)):
            scores[seller] = score
        elif isinstance(score, Exception):
            logger.warning("Trust computation failed for seller=%s: %s", seller, score)

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

    logger.info("_prepare_and_persist_items: received %d items", len(items))

    for item in items:
        ebay_id = item.get("ebay_id")
        if not ebay_id:
            logger.warning("Skipping item without ebay_id: title=%s", item.get("title", "?"))
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
        # Use the highly calibrated _final_score from the RAG reranker and Cross-Encoder
        base_relevance = float(item.get("_final_score", item.get("_rerank_score", 0)) or 0)
        trust = float(item.get("trust_score") or 0)
        price = item.get("price") or 0

        ranking_score = base_relevance

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

    # Sort one final time to respect user preferences (like brand match bonuses) injected above
    items.sort(key=lambda x: x.get("ranking_score", 0), reverse=True)


async def run_search_pipeline(
    query: str,
    db: Session,
    user: Optional[object] = None,
    llm_engine: str = "gemini",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("Query vuota")

    llm_engine = _normalize_llm_engine(llm_engine)
    t0 = time.time()
    timings: Dict[str, float] = {}

    # ============================================================
    # 1) PARSE QUERY
    # ============================================================

    logger.info("PIPELINE STEP 1: parse_query")

    # Recupera il contesto se disponibile (session_id or user_id)
    context_info = ""
    target_id = session_id or (str(getattr(user, "id", "")) if user else None)
    if target_id:
        history = redis_client.get_user_queries(target_id)
        if history:
            # Prendi le ultime 3 per non appesantire troppo il prompt
            context_info = " | ".join(history[:3])
            logger.info("PIPELINE: Found context in history: %s", context_info)

    t = time.time()
    parsed = await parse_query_service(
        query,
        use_llm=(llm_engine != "rule_based"),
        include_meta=True,
        context_info=context_info,
    )
    timings["parse_query_s"] = round(time.time() - t, 3)

    ebay_query_used = _build_ebay_query(parsed, query)

    # ============================================================
    # 2) PARALLEL: EBAY SEARCH + RAG RETRIEVAL (including expansion)
    # ============================================================
    logger.info("PIPELINE: Parallel Search & RAG")
    t = time.time()

    async def _do_ebay_search(parsed_query, limit):
        try:
            results = await search_items(parsed_query, limit=limit)
            return results
        except Exception as e:
            logger.error(f"eBay search failed: {e}")
            return {"itemSummaries": [], "aspectDistributions": []}

    async def _do_rag():
        try:
            expanded = await expand_query(query)
            docs = await asyncio.to_thread(retrieve_context, expanded, k=10)
            return expanded, docs
        except Exception as e:
            logger.warning("RAG retrieve failed: %s", e)
            return query, []

    # Run heavy I/O in parallel
    results = await asyncio.gather(
        _do_ebay_search(parsed, MAX_RESULTS_FROM_EBAY),
        _do_rag()
    )
    items = results[0].get("itemSummaries", []) if isinstance(results[0], dict) else []
    aspect_distributions = results[0].get("aspectDistributions", []) if isinstance(results[0], dict) else []
    logger.info("eBay returned %d itemSummaries", len(items))
    expanded_query, rag_docs = results[1]
    
    timings["parallel_io_s"] = round(time.time() - t, 3)
    items = _dedupe_items(items) if items else []

    # ============================================================
    # 2.5 PROACTIVE METADATA ENRICHMENT
    # ============================================================
    dominant_category = None
    if items:
        # Trova la categoria più frequente tra i risultati
        cats = [it.get("categoryId") for it in items if it.get("categoryId")]
        if cats:
            dominant_category = max(set(cats), key=cats.count)
            logger.info("PIPELINE: Detected dominant category %s", dominant_category)

    # Inizia il fetch delle policy in background
    meta_task = None
    if dominant_category:
        meta_task = asyncio.create_task(get_return_policies(category_id=dominant_category))

    # Separate RAG docs for reranker
    product_docs = [d for d in rag_docs if d.get("type") == "product"]
    seller_docs = [d for d in rag_docs if d.get("type") == "seller_feedback"]

    # ============================================================
    # 3) USER PROFILE UPDATE (Non-blocking enough)
    # ============================================================
    logger.info("PIPELINE STEP 2: user_profile")
    if user:
        try:
            update_user_profile(user, parsed, db)
        except Exception:
            logger.warning("User profiling update failed")

    # ============================================================
    # 4) SELLER TRUST  (deve venire PRIMA del rerank per alimentarlo)
    # ============================================================
    logger.info("PIPELINE STEP 5: seller_trust")
    t = time.time()
    seller_trust_map = await _prefetch_top_sellers_feedback(items[:MAX_SELLERS_FOR_TRUST * 2])
    timings["seller_trust_s"] = round(time.time() - t, 3)

    # Inietta trust_score negli item prima del reranker
    for item in items:
        seller_name = item.get("seller_name")
        item["trust_score"] = seller_trust_map.get(seller_name) if seller_name else None

    # ============================================================
    # 5) RERANK  (ora ha trust_score disponibile per ogni item)
    # ============================================================
    logger.info("PIPELINE STEP 6: rerank")

    t = time.time()
    if items:
        try:
            items = await asyncio.to_thread(
                rerank_products,
                query, items, user=user,
                product_docs=product_docs,
                seller_docs=seller_docs,
            )
        except Exception:
            logger.warning("Rerank failed, keeping original order")
    timings["rerank_s"] = round(time.time() - t, 3)

    # ============================================================
    # 6) DB PERSIST
    # ============================================================
    logger.info("PIPELINE STEP 7: db_persist")
    t = time.time()
    results_out, saved_count = _prepare_and_persist_items(db, items, seller_trust_map)
    timings["db_prepare_s"] = round(time.time() - t, 3)

    # ============================================================
    # 7) FINAL RANKING + CONTEXT
    # ============================================================
    logger.info("PIPELINE STEP 8: final_ranking")

    _apply_final_ranking(results_out, user=user)
    
    # Attach RAG feedback
    logger.info("PIPELINE STEP 9: rag_attach")
    for item in results_out:
        seller_name = item.get("seller_name")
        item["rag_feedback"] = [d for d in rag_docs if d.get("seller") == seller_name][:3] if seller_name else []

    # Recupera i risultati del fetch meta se completato
    meta_policies = None
    if meta_task:
        try:
            meta_policies = await meta_task
        except Exception:
            pass

    rag_context_text = build_context(query, results_out, rag_docs)
    
    if meta_policies and "returnPolicies" in meta_policies:
        # Aggiungi info sulle policy ai primi risultati
        policy = meta_policies["returnPolicies"][0] if meta_policies["returnPolicies"] else {}
        if policy:
          accepted = "accettato" if policy.get("returnsAccepted") else "non accettato"
          period = f"entro {policy.get('returnPeriod', {}).get('value')} {policy.get('returnPeriod', {}).get('unit')}" if policy.get("returnPeriod") else ""
          extra_info = f"\n[INFO EBAY] In questa categoria ({dominant_category}), il reso è generalmente {accepted} {period}."
          rag_context_text += extra_info
          logger.info("PIPELINE: Enriched RAG context with return policies")

    # ============================================================
    # 8) EXPLAIN (if results exist)
    # ============================================================
    logger.info("PIPELINE STEP 10: explain")

    t = time.time()
    analysis = None
    if results_out:
        try:
            analysis = await asyncio.to_thread(explain_results, query, results_out[:5])
        except Exception:
            pass
    timings["explain_s"] = round(time.time() - t, 3)

    # ============================================================
    # 9) IR METRICS
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
        "aspect_distributions": aspect_distributions,
        "rag_context": rag_context_text,
        "metrics": metrics,
        "_timings": timings,
    }