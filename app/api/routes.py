import logging
import os
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
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
# DB
# ============================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
def search(request: SearchRequest, db: Session = Depends(get_db)):

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    t0 = time.time()
    timings = {}

    # ============================================================
    # 1) PARSE
    # ============================================================

    use_llm = request.llm_engine != "rule_based"

    try:
        t = time.time()

        parsed = parse_query_service(
            request.query,
            use_llm=use_llm,
            include_meta=True,
        )

        timings[f"parse_{request.llm_engine}_s"] = round(time.time() - t, 3)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore parser: {str(e)}")

    # debug output
    ebay_query_used = parsed.get("semantic_query") or request.query

    # ============================================================
    # 1.5) RAG retrieval (pre-query context)
    # ============================================================

    try:
        rag_docs = retrieve_context(request.query, k=10)
    except Exception:
        rag_docs = []

    # ============================================================
    # 2) EBAY SEARCH
    # ============================================================

    try:
        t = time.time()

        # ✅ NEW: ebay.py ora vuole TUTTA la parsed_query
        items = search_items(
            parsed_query=parsed,
            limit=20
        )

        # indicizza prodotti nel RAG
        try:
            ingest_products(items)
        except Exception:
            pass

        timings["ebay_search_s"] = round(time.time() - t, 3)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore eBay search: {str(e)}")

    # ============================================================
    # 2.5) REMOVE DUPLICATES
    # ============================================================

    seen_ids = set()
    filtered_items = []

    for item in items:
        ebay_id = item.get("ebay_id")
        if not ebay_id:
            continue
        if ebay_id in seen_ids:
            continue
        seen_ids.add(ebay_id)
        filtered_items.append(item)

    items = filtered_items

    # ============================================================
    # 3) SAVE DB + TRUST SCORE (+ ingest feedback into RAG)
    # ============================================================

    from app.services.feedback import get_seller_feedback
    from app.services.trust import compute_trust_score
    from app.services.rag.ingest import ingest_seller_feedback  # ✅ NEW

    saved_count = 0
    results_out = []

    seller_feedback_cache = {}
    seller_trust_cache = {}

    try:
        t_db = time.time()

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

            # ---------------- TRUST SCORE ----------------

            seller_name = item.get("seller_name")
            trust_score = None

            # usa il RAG pre-query, filtrato sul seller
            rag_context = [
                d for d in rag_docs
                if d.get("seller") == seller_name
            ][:3]

            if seller_name:
                try:
                    if seller_name in seller_trust_cache:
                        trust_score = seller_trust_cache[seller_name]
                    else:
                        # ---------- feedback fetch (cached) ----------
                        if seller_name not in seller_feedback_cache:
                            feedbacks = get_seller_feedback(
                                seller_name,
                                limit=10
                            )
                            seller_feedback_cache[seller_name] = feedbacks

                            # ✅ ingest feedback in RAG (persistente)
                            try:
                                ingest_seller_feedback(
                                    seller_name=seller_name,
                                    feedbacks=feedbacks,
                                    max_docs=20
                                )
                            except Exception:
                                pass
                        else:
                            feedbacks = seller_feedback_cache[seller_name]

                        sentiment_score = compute_sentiment_score(feedbacks)

                        trust_score = compute_trust_score(
                            feedbacks,
                            sentiment_score=sentiment_score
                        )

                        seller_trust_cache[seller_name] = trust_score

                except Exception:
                    trust_score = None

            # ---------------- OUTPUT ----------------

            item_copy = dict(item)
            item_copy["_already_in_db"] = already
            item_copy["trust_score"] = trust_score
            item_copy["rag_feedback"] = rag_context

            results_out.append(item_copy)

        db.commit()
        timings["db_commit_s"] = round(time.time() - t_db, 3)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Errore DB: {str(e)}")

    timings["total_s"] = round(time.time() - t0, 3)

    # ============================================================
    # 4) RERANK
    # ============================================================

    results_out = rerank_products(request.query, results_out)

    # ============================================================
    # 5) ADD RANKING SCORE
    # ============================================================

    for item in results_out:

        relevance = item.get("score", item.get("_rerank_score", 0)) or 0
        trust = item.get("trust_score") or 0
        price = item.get("price") or 0

        price_score = 0
        if price:
            price_score = max(0, 1 - price / 1000)

        ranking_score = (
            0.6 * relevance +
            0.3 * trust +
            0.1 * price_score
        )

        item["ranking_score"] = round(float(ranking_score), 3)

    # ============================================================
    # 6) EXPLAINABLE RANKING
    # ============================================================

    for item in results_out:

        explanations = []

        if (item.get("trust_score") or 0) > 0.8:
            explanations.append("Trusted seller")

        if (item.get("seller_rating") or 0) > 95:
            explanations.append("Excellent seller rating")

        if item.get("ranking_score", 0) > 0.8:
            explanations.append("High relevance match")

        price = item.get("price")
        if price and price < 200:
            explanations.append("Competitive price")

        item["explanations"] = explanations

    # ============================================================
    # 7) RAG CONTEXT + AI ANALYSIS
    # ============================================================

    try:
        rag_docs_after = retrieve_context(request.query, k=10)
    except Exception:
        rag_docs_after = rag_docs

    rag_context_text = build_context(
        request.query,
        results_out,
        rag_docs_after
    )

    analysis = explain_results(request.query, results_out)

    # ============================================================
    # 8) IR METRICS (proxy labels)
    # ============================================================

    relevance_labels = [
        item.get("ranking_score", 0)
        for item in results_out
    ]

    metrics = {
        "precision@5": precision_at_k(relevance_labels, 5),
        "precision@10": precision_at_k(relevance_labels, 10),
        "recall@10": recall_at_k(
            relevance_labels,
            total_relevant=sum(relevance_labels),
            k=10
        ),
        "ndcg@10": ndcg_at_k(relevance_labels, 10),
    }

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