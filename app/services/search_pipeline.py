from __future__ import annotations

import logging
import re
import time
import asyncio
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.orm import Session

from app.models.listing import Listing
from app.services.ebay import search_items_async, get_shipping_costs_async
from app.services.feedback import get_seller_feedback_async
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

MAX_RESULTS_FROM_EBAY = 20
MAX_SELLERS_FOR_TRUST = 5
MAX_FEEDBACK_PER_SELLER = 40

async def _compute_seller_trust_async(seller_name: str) -> Optional[float]:
    feedbacks = await get_seller_feedback_async(seller_name, limit=MAX_FEEDBACK_PER_SELLER)
    if not feedbacks:
        return None

    # sentiment and trust score calculation are currently synchronous, 
    # but they are CPU bound or fast enough.
    sentiment_score = compute_sentiment_score(feedbacks)
    trust_score = compute_trust_score(feedbacks, sentiment_score=sentiment_score)
    return round(float(trust_score), 3)

async def _prefetch_top_sellers_feedback_async(items: List[Dict[str, Any]]) -> Dict[str, float]:
    sellers = []
    seen = set()
    for item in items:
        s = item.get("seller_name")
        if s and s not in seen:
            seen.add(s)
            sellers.append(s)
        if len(sellers) >= MAX_SELLERS_FOR_TRUST: break
    
    if not sellers: return {}

    tasks = [asyncio.create_task(_compute_seller_trust_async(s)) for s in sellers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    scores = {}
    for seller, score in zip(sellers, results):
        if isinstance(score, float):
            scores[seller] = score
    return scores

async def run_search_pipeline_async(
    query: str,
    db: Session,
    user: Optional[object] = None,
    llm_engine: str = "ollama",
    include_shipping: bool = False,
) -> Dict[str, Any]:
    
    t0 = time.time()
    timings = {}

    # 1) Parse Query
    t = time.time()
    parsed = await parse_query_service(query)
    timings["parse_query_s"] = round(time.time() - t, 3)

    # 3) eBay Search
    t = time.time()
    items = await search_items_async(parsed_query=parsed, limit=MAX_RESULTS_FROM_EBAY) or []
    timings["ebay_search_s"] = round(time.time() - t, 3)

    # 4) RAG Retrieval (Mocked as sync for now, can be wrapped in to_thread if needed)
    t = time.time()
    try:
        # expand_query might be sync, let's assume it's fast or wrap if needed
        # In a real app, these should also be async
        rag_docs = retrieve_context(query) 
    except Exception:
        rag_docs = []
    timings["rag_retrieve_s"] = round(time.time() - t, 3)

    # 6) Trust for top sellers
    t = time.time()
    seller_trust_map = await _prefetch_top_sellers_feedback_async(items)
    timings["seller_trust_s"] = round(time.time() - t, 3)

    # Persist and finalize (omitting details for brevity)
    for item in items:
        s = item.get("seller_name")
        item["trust_score"] = seller_trust_map.get(s)

    # 8.1) Shipping
    if include_shipping and items:
        top = items[0]
        iid = top.get("ebay_id")
        if iid:
            ship = await get_shipping_costs_async(iid, "IT", "")
            if ship:
                top["shipping_info"] = ship

    timings["total_s"] = round(time.time() - t0, 3)

    return {
        "results": items,
        "results_count": len(items),
        "_timings": timings,
    }