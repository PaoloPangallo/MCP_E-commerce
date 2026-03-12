from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.services.search_pipeline import run_search_pipeline_async

logger = logging.getLogger(__name__)

async def _run_single_search_async(query: str, db: Any, llm_engine: str = "ollama") -> Dict[str, Any]:
    try:
        result = await run_search_pipeline_async(query=query, db=db, llm_engine=llm_engine)
        results = result.get("results") or []
        return {
            "query": query,
            "top": results[0] if results else None,
            "results_count": len(results),
        }
    except Exception as exc:
        logger.warning("compare_pipeline search failed for query=%s: %s", query, exc)
        return {"query": query, "top": None, "results_count": 0, "error": str(exc)}

def _score_price(price: Optional[float], all_prices: List[float]) -> float:
    if price is None or not all_prices: return 0.5
    min_p, max_p = min(all_prices), max(all_prices)
    if max_p == min_p: return 1.0
    return round(1.0 - (price - min_p) / (max_p - min_p), 4)

def _score_condition(condition: str) -> float:
    mapping = {"nuovo": 1.0, "new": 1.0, "ricondizionato": 0.7, "refurbished": 0.7, "usato": 0.4, "used": 0.4}
    return mapping.get((condition or "").strip().lower(), 0.5)

async def run_compare_pipeline(
    queries: List[str],
    db: Any,
    llm_engine: str = "ollama",
    max_queries: int = 4,
) -> Dict[str, Any]:
    queries = [q.strip() for q in queries if q and q.strip()][:max_queries]
    if len(queries) < 2:
        return {"status": "error", "error": "Servono almeno 2 query."}

    tasks = [asyncio.create_task(_run_single_search_async(q, db, llm_engine)) for q in queries]
    raw_results = await asyncio.gather(*tasks)

    candidates = []
    for raw in raw_results:
        top = raw.get("top")
        if top:
            candidates.append({
                "query": raw["query"],
                "title": top.get("title", "N/A"),
                "price": top.get("price"),
                "trust_score": top.get("trust_score", 0.5),
                "condition": top.get("condition", "N/A"),
                "url": top.get("url"),
                "image_url": top.get("image_url"),
            })

    if not candidates:
        return {"status": "no_data", "error": "Nessun prodotto trovato."}

    # Scoring logic (simplified but keeping the essence)
    all_prices = [c["price"] for c in candidates if c["price"] is not None]
    for c in candidates:
        ps = _score_price(c["price"], all_prices)
        ts = float(c["trust_score"] or 0.5)
        cs = _score_condition(c["condition"])
        c["overall_score"] = round(0.4 * ps + 0.4 * ts + 0.2 * cs, 3)

    candidates.sort(key=lambda x: x["overall_score"], reverse=True)

    return {
        "status": "ok",
        "winner": candidates[0],
        "comparison_matrix": candidates
    }
