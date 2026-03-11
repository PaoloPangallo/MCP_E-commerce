"""
compare_pipeline.py

Runs multiple search_products queries in parallel and compares the top result
of each, scoring them across price, trust, relevance and condition.
Returns a structured comparison with a recommended winner.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================
# SCORING HELPERS
# ============================================================

def _score_price(price: Optional[float], all_prices: List[float]) -> float:
    """Lower price → higher score, normalised in [0, 1]."""
    if price is None or not all_prices:
        return 0.5
    min_p, max_p = min(all_prices), max(all_prices)
    if max_p == min_p:
        return 1.0
    return round(1.0 - (price - min_p) / (max_p - min_p), 4)


def _score_condition(condition: str) -> float:
    mapping = {
        "nuovo": 1.0, "new": 1.0,
        "ricondizionato": 0.7, "refurbished": 0.7,
        "usato": 0.4, "used": 0.4, "gebraucht": 0.4,
    }
    return mapping.get((condition or "").strip().lower(), 0.5)


def _overall_score(price_score: float, trust: float, relevance: float, condition: float) -> float:
    return round(
        0.30 * price_score +
        0.30 * trust +
        0.25 * relevance +
        0.15 * condition,
        4,
    )


# ============================================================
# SYNC WRAPPER (called inside asyncio.to_thread from MCP tool)
# ============================================================

def _run_single_search(query: str, db: Any, llm_engine: str = "ollama") -> Dict[str, Any]:
    """Synchronous search — runs one query through run_search_pipeline."""
    from app.services.search_pipeline import run_search_pipeline  # avoid circular
    try:
        result = run_search_pipeline(query=query, db=db, llm_engine=llm_engine)
        results = result.get("results") or []
        # Return only the top-1 item plus metadata
        return {
            "query": query,
            "top": results[0] if results else None,
            "results_count": len(results),
            "parsed_query": result.get("parsed_query"),
        }
    except Exception as exc:
        logger.warning("compare_pipeline search failed for query=%s: %s", query, exc)
        return {"query": query, "top": None, "results_count": 0, "error": str(exc)}


# ============================================================
# PUBLIC ASYNC API
# ============================================================

async def run_compare_pipeline(
    queries: List[str],
    db: Any,
    llm_engine: str = "ollama",
    max_queries: int = 4,
) -> Dict[str, Any]:
    """
    Run up to `max_queries` searches in parallel and compare their top results.

    Args:
        queries:    List of product search strings to compare.
        db:         SQLAlchemy session.
        llm_engine: LLM backend for query parsing.

    Returns:
        A dict with:
            - candidates: list of scored candidates
            - winner: the winning candidate name/index
            - winner_reason: human-readable explanation
            - comparison_matrix: raw data for display
    """
    queries = [q.strip() for q in queries if q and q.strip()][:max_queries]

    if len(queries) < 2:
        return {
            "status": "error",
            "error": "Servono almeno 2 query per confrontare i prodotti.",
        }

    # Run all searches in parallel
    tasks = [
        asyncio.to_thread(_run_single_search, q, db, llm_engine)
        for q in queries
    ]
    raw_results: List[Dict[str, Any]] = await asyncio.gather(*tasks, return_exceptions=False)

    # Extract top items
    candidates = []
    for raw in raw_results:
        top = raw.get("top")
        if top is None:
            continue
        candidates.append({
            "query": raw["query"],
            "title": top.get("title", "N/A"),
            "price": top.get("price"),
            "currency": top.get("currency", "EUR"),
            "condition": top.get("condition", "N/A"),
            "seller_name": top.get("seller_name"),
            "seller_rating": top.get("seller_rating"),
            "trust_score": top.get("trust_score"),
            "ranking_score": top.get("ranking_score"),
            "url": top.get("url"),
            "image_url": top.get("image_url"),
        })

    if not candidates:
        return {
            "status": "no_data",
            "error": "Nessun prodotto trovato per nessuna delle query.",
            "queries_tried": queries,
        }

    # Build scoring matrix
    all_prices = [c["price"] for c in candidates if c["price"] is not None]

    scored = []
    for c in candidates:
        price_score = _score_price(c["price"], all_prices)
        trust = float(c["trust_score"] or 0.5)
        relevance = float(c["ranking_score"] or 0.5)
        condition = _score_condition(c["condition"])
        overall = _overall_score(price_score, trust, relevance, condition)

        scored.append({
            **c,
            "_scores": {
                "price":      round(price_score, 3),
                "trust":      round(trust, 3),
                "relevance":  round(relevance, 3),
                "condition":  round(condition, 3),
                "overall":    overall,
            },
            "_overall": overall,
        })

    # Sort by overall score descending
    scored.sort(key=lambda x: x["_overall"], reverse=True)
    winner = scored[0]

    # Build human-readable reason
    scores = winner["_scores"]
    reasons = []
    if scores["price"] >= 0.7:
        p = winner.get("price")
        reasons.append(f"prezzo competitivo ({p} {winner.get('currency','EUR')})")
    if scores["trust"] >= 0.7:
        reasons.append(f"venditore affidabile (trust {round(scores['trust']*100):.0f}%)")
    if scores["relevance"] >= 0.7:
        reasons.append("alta rilevanza per la query")
    if winner.get("condition", "").lower() in {"nuovo", "new"}:
        reasons.append("prodotto nuovo")
    if not reasons:
        reasons.append("miglior punteggio complessivo")

    winner_reason = (
        f"«{winner['title'][:60]}» è il prodotto consigliato: "
        + ", ".join(reasons) + "."
    )

    return {
        "status": "ok",
        "queries_compared": len(queries),
        "candidates_found": len(scored),
        "winner": {
            "title": winner["title"],
            "price": winner["price"],
            "currency": winner["currency"],
            "condition": winner["condition"],
            "seller_name": winner["seller_name"],
            "trust_score": winner["trust_score"],
            "url": winner["url"],
            "image_url": winner["image_url"],
            "overall_score": winner["_overall"],
        },
        "winner_reason": winner_reason,
        "comparison_matrix": [
            {
                "query": c["query"],
                "title": c["title"],
                "price": c["price"],
                "currency": c["currency"],
                "condition": c["condition"],
                "seller_name": c["seller_name"],
                "trust_score": c["trust_score"],
                "url": c["url"],
                "scores": c["_scores"],
            }
            for c in scored
        ],
    }
