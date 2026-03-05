import logging
import re
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
    return parse_query_service(
        request.query,
        llm_engine=request.llm_engine,
        include_meta=True,
    )


# ============================================================
# BUILD EBAY QUERY (robusta + <=100 chars)
# ============================================================

def build_ebay_query(parsed: dict, original_query: str) -> str:
    semantic = (parsed.get("semantic_query") or "").strip()
    product = (parsed.get("product") or "").strip()
    brands = parsed.get("brands", []) or []

    # Preferisci semantic_query (di solito è già "scarpe adidas", "garmin watch", ecc.)
    if semantic:
        q = semantic
    else:
        tokens = []
        if brands:
            tokens.extend([str(b).strip() for b in brands if str(b).strip()])
        if product:
            tokens.append(product)
        q = " ".join(tokens).strip()

    # Fallback finale
    if not q:
        q = original_query.strip()

    # normalizza token (evita simboli strani)
    q = " ".join(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{2,}", q))

    # eBay tronca >100 char → tronchiamo noi
    return q[:100].strip() if q else original_query


# ============================================================
# SEARCH ENDPOINT
# ============================================================

@router.post("/search")
def search(request: SearchRequest, db: Session = Depends(get_db)):

    print("SEARCH STARTED")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    t0 = time.time()
    timings = {}

    # ============================================================
    # 1) PARSE
    # ============================================================

    try:
        t = time.time()

        parsed = parse_query_service(
            request.query,
            llm_engine=request.llm_engine,
            include_meta=True,
        )

        timings[f"parse_{request.llm_engine}_s"] = round(time.time() - t, 3)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore parser: {str(e)}")

    constraints = parsed.get("constraints", []) or []
    preferences = parsed.get("preferences", []) or []

    ebay_query = build_ebay_query(parsed, request.query)

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

        items = search_items(
            query_text=ebay_query,
            constraints=constraints,
            preferences=preferences,
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

        t = time.time()

        for item in items:

            ebay_id = item.get("ebay_id")

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

                            # ✅ NEW: ingest feedback in RAG (persistente)
                            # indicizziamo solo quando li scarichiamo davvero
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

        timings["db_commit_s"] = round(time.time() - t, 3)

    except Exception as e:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Errore DB: {str(e)}"
        )

    timings["total_s"] = round(time.time() - t0, 3)

    # ============================================================
    # 4) RERANK
    # ============================================================

    results_out = rerank_products(request.query, results_out)

    # ============================================================
    # 5) ADD RANKING SCORE
    # ============================================================

    for item in results_out:

        # NB: il tuo reranker potrebbe mettere "score" oppure "_rerank_score"
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

        item["ranking_score"] = round(ranking_score, 3)

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

    # (opzionale ma utile) ricarica RAG docs post-ingest,
    # così build_context/explain_results vedono anche i feedback appena indicizzati
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
    # 8) IR METRICS
    # ============================================================

    relevance_labels = []

    for item in results_out:
        score = item.get("ranking_score", 0)

        # proxy relevance
        relevance = 1 if score > 0.75 else 0

        relevance_labels.append(relevance)

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

    # ============================================================
    # RESPONSE
    # ============================================================

    return {
        "parsed_query": parsed,
        "ebay_query_used": ebay_query,
        "results_count": len(results_out),
        "saved_new_count": saved_count,
        "analysis": analysis,
        "results": results_out,
        "rag_context": rag_context_text,
        "metrics": metrics,
        "_timings": timings,
    }