from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import time
import logging
from typing import Literal
import os

from app.db.database import SessionLocal
from app.models.listing import Listing
from app.services.ebay import search_items
from app.services.parser import parse_query_service

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

    if request.llm_engine == "rule_based":
        use_llm = False
    else:
        # Override temporaneo del provider via env runtime
        os.environ["LLM_PROVIDER"] = request.llm_engine
        use_llm = True

    return parse_query_service(
        request.query,
        use_llm=use_llm,
        include_meta=True
    )


# ============================================================
# BUILD EBAY QUERY
# ============================================================

import re

def build_ebay_query(parsed: dict, original_query: str) -> str:
    product = (parsed.get("product") or "").strip()
    brands = parsed.get("brands", []) or []
    semantic = (parsed.get("semantic_query") or "").strip()

    tokens = []

    # 1️⃣ Se LLM ha identificato un prodotto → è la cosa migliore
    if product:
        tokens.append(product)

    # 2️⃣ Aggiungi brand solo se non già incluso
    if brands:
        b0 = str(brands[0]).strip()
        if b0 and product and b0.lower() not in product.lower():
            tokens.insert(0, b0)

    # 3️⃣ Fallback su semantic_query pulita
    if not tokens and semantic:
        toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{2,}", semantic)
        tokens = toks

    q = " ".join(tokens).strip()
    return q if q else original_query


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
    # 1️⃣ PARSE
    # ============================================================

    try:
        t = time.time()

        if request.llm_engine == "rule_based":
            use_llm = False
        else:
            os.environ["LLM_PROVIDER"] = request.llm_engine
            use_llm = True

        parsed = parse_query_service(
            request.query,
            use_llm=use_llm,
            include_meta=True
        )

        timings[f"parse_{request.llm_engine}_s"] = round(time.time() - t, 3)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore parser: {str(e)}")

    constraints = parsed.get("constraints", [])
    ebay_query = build_ebay_query(parsed, request.query)

    # ============================================================
    # 2️⃣ EBAY SEARCH
    # ============================================================

    try:
        t = time.time()
        items = search_items(query_text=ebay_query, constraints=constraints, limit=5)
        timings["ebay_search_s"] = round(time.time() - t, 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore eBay search: {str(e)}")

    # ============================================================
    # 3️⃣ SAVE DB
    # ============================================================

    saved_items = []

    try:
        t = time.time()

        for item in items:
            ebay_id = item.get("ebay_id")
            if not ebay_id:
                continue

            exists = db.query(Listing).filter_by(ebay_id=ebay_id).first()
            if exists:
                continue

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
            saved_items.append(item)

        db.commit()
        timings["db_commit_s"] = round(time.time() - t, 3)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Errore DB: {str(e)}")

    timings["total_s"] = round(time.time() - t0, 3)

    return {
        "parsed_query": parsed,
        "ebay_query_used": ebay_query,
        "results_count": len(saved_items),
        "results": saved_items,
        "_timings": timings,
    }