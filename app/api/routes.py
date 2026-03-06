import logging
import os
import time
from typing import Literal, Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import get_optional_user
from app.db.database import get_db
from app.models.listing import Listing
from app.services.ebay import search_items
from app.services.metrics.ir_metrics import precision_at_k, recall_at_k, ndcg_at_k
from app.services.nlp_sentiment import compute_sentiment_score
from app.services.parser import parse_query_service
from app.services.rag import retrieve_context, build_context
from app.services.rag.explainer import explain_results
from app.services.rag.product_ingest import ingest_products
from app.services.rag.reranker import rerank_products

router = APIRouter()
logger = logging.getLogger(__name__)

print("SEARCH ROUTER FILE:", os.path.abspath(__file__))


# ============================================================
# REQUEST MODEL
# ============================================================

class SearchRequest(BaseModel):
    query: str
    llm_engine: Literal["gemini", "ollama", "rule_based"] = "gemini"


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# PARSE ENDPOINT
# ============================================================

@router.post("/parse")
def parse(request: SearchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    use_llm = request.llm_engine != "rule_based"

    try:
        return parse_query_service(
            request.query,
            use_llm=use_llm,
            include_meta=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore parser: {str(e)}")


# ============================================================
# SEARCH ENDPOINT
# ============================================================

@router.post("/search")
def search(
    request: SearchRequest,
    db: Session = Depends(get_db),
    user=Depends(get_optional_user),
):

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    logger.info("Search query: %s", request.query)

    import traceback

    t0 = time.time()
    timings: Dict[str, float] = {}

    # ============================================================
    # 1) PARSE QUERY
    # ============================================================

    try:

        t = time.time()

        parsed = parse_query_service(
            request.query,
            use_llm=(request.llm_engine != "rule_based"),
            include_meta=True,
        )

        if isinstance(parsed, str):
            parsed = {"semantic_query": parsed}

        timings["parse_query_s"] = round(time.time() - t, 3)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Parser error: {str(e)}")

    ebay_query_used = parsed.get("semantic_query") or request.query

    # ============================================================
    # 2) EBAY SEARCH
    # ============================================================

    try:

        t = time.time()

        items = search_items(
            parsed_query=parsed,
            limit=20
        ) or []

        timings["ebay_search_s"] = round(time.time() - t, 3)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"eBay search error: {str(e)}")

    # ============================================================
    # 3) REMOVE DUPLICATES
    # ============================================================

    seen_ids = set()
    deduped = []

    for item in items:

        ebay_id = item.get("ebay_id")

        if not ebay_id:
            continue

        if ebay_id in seen_ids:
            continue

        seen_ids.add(ebay_id)
        deduped.append(item)

    items = deduped

    # ============================================================
    # 4) SELLER FEEDBACK INGEST
    # ============================================================

    from app.services.feedback import get_seller_feedback
    from app.services.trust import compute_trust_score
    from app.services.nlp_sentiment import compute_sentiment_score
    from app.services.rag.ingest import ingest_seller_feedback

    seller_feedback_cache: Dict[str, Any] = {}
    seller_trust_cache: Dict[str, float] = {}

    try:

        t = time.time()

        sellers = {
            i.get("seller_name")
            for i in items
            if i.get("seller_name")
        }

        for seller_name in sellers:

            try:

                feedbacks = get_seller_feedback(seller_name, limit=50)

                seller_feedback_cache[seller_name] = feedbacks

                ingest_seller_feedback(
                    seller_name=seller_name,
                    feedbacks=feedbacks,
                    max_docs=20
                )

            except Exception:
                continue

        timings["feedback_ingest_s"] = round(time.time() - t, 3)

    except Exception:
        traceback.print_exc()

    # ============================================================
    # 5) SAVE DB + TRUST SCORE
    # ============================================================

    saved_count = 0
    results_out = []

    try:

        t = time.time()

        for item in items:

            ebay_id = item.get("ebay_id")

            if not ebay_id:
                continue

            exists = db.query(Listing).filter_by(ebay_id=ebay_id).first()
            already = exists is not None

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
            trust_score = None

            if seller_name:

                try:

                    if seller_name in seller_trust_cache:

                        trust_score = seller_trust_cache[seller_name]

                    else:

                        feedbacks = seller_feedback_cache.get(seller_name) or []

                        sentiment_score = compute_sentiment_score(feedbacks)

                        trust_score = compute_trust_score(
                            feedbacks,
                            sentiment_score=sentiment_score
                        )

                        seller_trust_cache[seller_name] = trust_score

                except Exception:
                    trust_score = None

            item_copy = dict(item)

            item_copy["_already_in_db"] = already
            item_copy["trust_score"] = trust_score

            results_out.append(item_copy)

        db.commit()

        timings["db_save_s"] = round(time.time() - t, 3)

    except Exception as e:

        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

    # ============================================================
    # 6) RERANK PRODUCTS
    # ============================================================

    try:

        from app.services.rag.reranker import rerank_products

        t = time.time()

        if results_out:
            results_out = rerank_products(request.query, results_out)

        timings["rerank_s"] = round(time.time() - t, 3)

    except Exception:
        pass

    # ============================================================
    # 7) FINAL RANKING SCORE
    # ============================================================

    for item in results_out:

        relevance = item.get("_rerank_score", 0)
        trust = item.get("trust_score") or 0
        price = item.get("price") or 0

        price_score = 0.0

        if price:
            price_score = max(0.0, 1.0 - float(price) / 1000.0)

        ranking_score = (
            0.5 * float(relevance)
            + 0.25 * float(trust)
            + 0.15 * float(price_score)
        )

        item["ranking_score"] = round(ranking_score, 3)

    # ============================================================
    # 8) RAG RETRIEVE
    # ============================================================

    from app.services.rag import retrieve_context, build_context

    try:

        t = time.time()

        rag_docs_after = retrieve_context(request.query, k=10)

        timings["rag_retrieve_s"] = round(time.time() - t, 3)

    except Exception:
        rag_docs_after = []

    for item in results_out:

        seller_name = item.get("seller_name")

        if not seller_name:
            item["rag_feedback"] = []
            continue

        item["rag_feedback"] = [
            d for d in rag_docs_after
            if d.get("seller") == seller_name
        ][:3]

    rag_context_text = build_context(
        request.query,
        results_out,
        rag_docs_after
    )

    # ============================================================
    # 9) EXPLAIN RESULTS
    # ============================================================

    from app.services.rag.explainer import explain_results

    try:

        t = time.time()

        analysis = explain_results(request.query, results_out)

        timings["explain_s"] = round(time.time() - t, 3)

    except Exception:
        analysis = None

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

    timings["total_s"] = round(time.time() - t0, 3)

    # ============================================================
    # RESPONSE
    # ============================================================

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