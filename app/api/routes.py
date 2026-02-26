from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import time
import logging

from app.db.database import SessionLocal
from app.models.listing import Listing
from app.services.ebay import search_items
from app.services.parser import parse_query_service

router = APIRouter()
logger = logging.getLogger(__name__)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class SearchRequest(BaseModel):
    query: str


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/parse")
def parse(request: SearchRequest):
    return parse_query_service(request.query, use_llm=True, include_meta=True)


import re

def build_ebay_query(parsed: dict, original_query: str) -> str:
    product = (parsed.get("product") or "").strip()
    brands = parsed.get("brands", []) or []
    semantic = (parsed.get("semantic_query") or "").strip()

    tokens = []

    # 1) se c'è un product esplicito, è la cosa migliore per eBay
    if product:
        tokens.append(product)

    # 2) aggiungi UN brand solo se non è già dentro al product (evita duplicati tipo iPhone+iPhone)
    if brands:
        b0 = str(brands[0]).strip()
        if b0 and (product.lower().find(b0.lower()) == -1):
            tokens.insert(0, b0)  # brand prima del modello

    # 3) fallback: semantic_query ripulita in modo "non hardcoded" (tieni solo parole utili)
    if not tokens and semantic:
        # tieni solo token alfanumerici lunghi almeno 2, e numeri (es. "15")
        toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{2,}", semantic)
        tokens = toks

    q = " ".join(tokens).strip()
    return q if q else original_query


@router.post("/search")
def search(request: SearchRequest, db: Session = Depends(get_db)):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query vuota")

    t0 = time.time()
    timings = {}

    # 1) Parse (NO LLM per non bloccare)
    try:
        t = time.time()
        parsed = parse_query_service(request.query, use_llm=True, include_meta=True)
        timings["parse_rule_based_s"] = round(time.time() - t, 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore parser: {str(e)}")

    constraints = parsed.get("constraints", [])
    ebay_query = build_ebay_query(parsed, request.query)

    # 2) eBay search (con timeout in ebay.py)
    try:
        t = time.time()
        items = search_items(query_text=ebay_query, constraints=constraints, limit=5)
        timings["ebay_search_s"] = round(time.time() - t, 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore eBay search: {str(e)}")

    # 3) Save DB
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