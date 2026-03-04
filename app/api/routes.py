from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import time
import logging
from typing import Literal, Optional
import re

from app.db.database import SessionLocal
from app.models.listing import Listing
from app.services.ebay import search_items
from app.services.nlp_sentiment import compute_sentiment_score
from app.services.parser import parse_query_service
from app.services.feedback import get_seller_feedback
from app.services.trust import compute_trust_score
from app.services.nlp_sentiment import compute_sentiment_score

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
    # 2) EBAY SEARCH
    # ============================================================
    try:
        t = time.time()

        items = search_items(
            query_text=ebay_query,
            constraints=constraints,
            preferences=preferences,
            limit=5
        )

        timings["ebay_search_s"] = round(time.time() - t, 3)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore eBay search: {str(e)}")

    # ============================================================
    # 3) SAVE DB + TRUST SCORE
    # ============================================================
    from app.services.feedback import get_seller_feedback
    from app.services.trust import compute_trust_score

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

            # -------------------------
            # TRUST SCORE
            # -------------------------

            seller_name = item.get("seller_name")
            trust_score = None

            if seller_name:
                try:
                    feedbacks = get_seller_feedback(seller_name, limit=10)

                    sentiment_score = compute_sentiment_score(feedbacks)

                    trust_score = compute_trust_score(
                        feedbacks,
                        sentiment_score=sentiment_score
                    )

                except Exception:
                    trust_score = None

            # -------------------------
            # OUTPUT ITEM
            # -------------------------

            item_copy = dict(item)
            item_copy["_already_in_db"] = already
            item_copy["trust_score"] = trust_score

            results_out.append(item_copy)

        db.commit()
        timings["db_commit_s"] = round(time.time() - t, 3)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Errore DB: {str(e)}")

    timings["total_s"] = round(time.time() - t0, 3)

    return {
        "parsed_query": parsed,
        "ebay_query_used": ebay_query,
        "results_count": len(results_out),
        "saved_new_count": saved_count,
        "results": results_out,
        "_timings": timings,
    }