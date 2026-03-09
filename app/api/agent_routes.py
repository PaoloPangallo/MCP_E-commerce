import logging
import os
import time
import traceback
from typing import Literal, Dict, Any, Tuple, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import get_optional_user
from app.db.database import get_db
from app.models.listing import Listing
from app.services.ebay import search_items
from app.services.metrics.ir_metrics import precision_at_k, recall_at_k, ndcg_at_k
from app.services.parser import parse_query_service
from app.services.user_profiling import update_user_profile
from app.services.rag.product_ingest import ingest_products

# importimosimos old orchestrator
# from app.services.agent_orchestrator import ask_agent_orchestrator

router = APIRouter()
logger = logging.getLogger(__name__)

class MessageDict(BaseModel):
    role: str
    content: str

class SearchRequest(BaseModel):
    query: str
    llm_engine: Literal["gemini", "ollama", "rule_based"] = "ollama"
    history: List[MessageDict] = []
    context: Dict[str, Any] = {}
    reset_context: bool = False
    session_id: Optional[str] = None

@router.get("/health")
def health():
    return {"status": "ok"}

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
# LA VOSTRA LOGICA INOSSIDABILE, ISOLATA IN UNA FUNZIONE!
# ============================================================
def execute_full_ecommerce_search(query: str, db: Session, user: Any, t0: float) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    
    # 1) PARSE QUERY
    t = time.time()
    parsed = parse_query_service(query, use_llm=True, include_meta=True)
    if isinstance(parsed, str):
        parsed = {"semantic_query": parsed}
    timings["parse_query_s"] = round(time.time() - t, 3)

    product = parsed.get("product")
    brands = parsed.get("brands") or []
    if product and brands:
        ebay_query_used = f"{product} {' '.join(brands)}"
    elif product:
        ebay_query_used = product
    else:
        ebay_query_used = parsed.get("semantic_query") or query

    # 1.5) IMPLICIT USER PROFILING
    if user:
        try:
            update_user_profile(user, parsed, db)
        except Exception:
            pass

    # 2) EBAY SEARCH
    t = time.time()
    items = search_items(parsed_query=parsed, limit=20) or []
    timings["ebay_search_s"] = round(time.time() - t, 3)

    # 3) REMOVE DUPLICATES
    seen_ids = set()
    deduped = []
    for item in items:
        ebay_id = item.get("ebay_id")
        if not ebay_id or ebay_id in seen_ids:
            continue
        seen_ids.add(ebay_id)
        deduped.append(item)
    items = deduped

    # 3.5) AUTOMATIC RAG INGEST
    # Ingest search results into RAG to provide context for subsequent messages in the same chat
    try:
        ingest_products(items)
    except Exception as e:
        logger.warning(f"RAG Ingest failed: {e}")

    # 4) SELLER FEEDBACK INGEST
    from app.services.feedback import get_seller_feedback
    from app.services.trust import compute_trust_score
    from app.services.nlp_sentiment import compute_sentiment_score
    from app.services.rag.ingest import ingest_seller_feedback

    seller_feedback_cache: Dict[str, Any] = {}
    seller_trust_cache: Dict[str, float] = {}

    t = time.time()
    sellers = {i.get("seller_name") for i in items if i.get("seller_name")}
    for seller_name in sellers:
        try:
            feedbacks = get_seller_feedback(seller_name, limit=50)
            seller_feedback_cache[seller_name] = feedbacks
            ingest_seller_feedback(seller_name=seller_name, feedbacks=feedbacks, max_docs=20)
        except Exception:
            continue
    timings["feedback_ingest_s"] = round(time.time() - t, 3)

    # 5) SAVE DB + TRUST SCORE
    saved_count = 0
    results_out = []
    t = time.time()
    for item in items:
        ebay_id = item.get("ebay_id")
        if not ebay_id:
            continue
        
        exists = db.query(Listing).filter_by(ebay_id=ebay_id).first()
        already = exists is not None
        if not already:
            listing = Listing(
                ebay_id=ebay_id, title=item.get("title"), price=item.get("price"),
                currency=item.get("currency"), condition=item.get("condition"),
                seller_name=item.get("seller_name"), seller_rating=item.get("seller_rating"),
                url=item.get("url"), image_url=item.get("image_url")
            )
            db.add(listing)
            saved_count += 1

        seller_name = item.get("seller_name")
        trust_score = None
        if seller_name:
            if seller_name in seller_trust_cache:
                trust_score = seller_trust_cache[seller_name]
            else:
                feedbacks = seller_feedback_cache.get(seller_name) or []
                sentiment_score = compute_sentiment_score(feedbacks)
                trust_score = compute_trust_score(feedbacks, sentiment_score=sentiment_score)
                seller_trust_cache[seller_name] = trust_score

        item_copy = dict(item)
        item_copy["_already_in_db"] = already
        item_copy["trust_score"] = trust_score
        results_out.append(item_copy)
    timings["db_prepare_s"] = round(time.time() - t, 3)

    # 6) RERANK PRODUCTS
    from app.services.rag.reranker import rerank_products
    t = time.time()
    if results_out:
        results_out = rerank_products(query, results_out)
    timings["rerank_s"] = round(time.time() - t, 3)

    # 7) FINAL RANKING SCORE (PERSONALIZED + EXPLANATIONS)
    for item in results_out:
        relevance = item.get("_rerank_score", 0)
        trust = item.get("trust_score") or 0
        price = item.get("price") or 0
        price_score = max(0.0, 1.0 - float(price) / 1000.0) if price else 0.0

        ranking_score = 0.5 * float(relevance) + 0.25 * float(trust) + 0.15 * float(price_score)
        explanations = []

        if user:
            if user.favorite_brands:
                user_brands = {b.strip().lower() for b in user.favorite_brands.split(",")}
                title = (item.get("title") or "").lower()
                for b in user_brands:
                    if b in title:
                        ranking_score += 0.10
                        item["brand_match"] = True
                        explanations.append(f"This item matches your preferred brand '{b}'.")
                        break
            if user.price_preference and price:
                try:
                    if price <= float(user.price_preference):
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

    # 8) RAG RETRIEVE
    from app.services.rag import retrieve_context, build_context
    t = time.time()
    try:
        rag_docs_after = retrieve_context(query, k=10)
    except Exception:
        rag_docs_after = []
    timings["rag_retrieve_s"] = round(time.time() - t, 3)

    for item in results_out:
        seller_name = item.get("seller_name")
        if not seller_name:
            item["rag_feedback"] = []
            continue
        item["rag_feedback"] = [d for d in rag_docs_after if d.get("seller") == seller_name][:3]

    rag_context_text = build_context(query, results_out, rag_docs_after)

    # 10) IR METRICS
    binary_relevance = [1 if (item.get("ranking_score", 0) >= 0.75) else 0 for item in results_out]
    metrics = {
        "precision@5": precision_at_k(binary_relevance, 5),
        "precision@10": precision_at_k(binary_relevance, 10),
        "recall@10": recall_at_k(binary_relevance, total_relevant=sum(binary_relevance), k=10),
        "ndcg@10": ndcg_at_k(binary_relevance, 10),
    }

    try:
        db.commit()
    except Exception:
        db.rollback()

    timings["total_s"] = round(time.time() - t0, 3)

    return {
        "parsed_query": parsed,
        "ebay_query_used": ebay_query_used,
        "results_count": len(results_out),
        "saved_new_count": saved_count,
        "results": results_out,
        "rag_context": rag_context_text,
        "metrics": metrics,
        "_timings": timings,
    }


from app.services.mcp_client import ask_mcp_orchestrator

# ============================================================
# SEARCH ENDPOINT A CUI IL FRONTEND REACT RISPONDE (MCP ASYNC)
# ============================================================

@router.post("/search")
async def search(
    request: SearchRequest,
    db: Session = Depends(get_db),
    user=Depends(get_optional_user),
):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    logger.info("MCP Search start: %s", request.query)
    t0 = time.time()
    
    try:
        # Iniziamo la magia: affidiamo il compito a Ollama via Model Context Protocol!
        final_payload = await ask_mcp_orchestrator(
            user_message=request.query,
            history=[h.dict() for h in request.history],
            db_session=db,
            user_obj=user,
            t0=t0,
            context=request.context,
            ecommerce_pipeline_func=execute_full_ecommerce_search,
            reset_context=request.reset_context,
            session_id=request.session_id
        )
        
        if "error" in final_payload:
            logger.error(f"MCP CLIENT ERROR: {final_payload['error']}")
            raise HTTPException(status_code=500, detail=f"MCP Agent Error: {final_payload['error']}")
            
        return final_payload
    except Exception as e:
        logger.error("!!! CRITICAL ERROR IN /search ENDPOINT !!!")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
