"""
compare_pipeline.py

Runs multiple search_products queries in parallel and compares the top result
of each, scoring them across price, trust, relevance and condition.
Returns a structured comparison with a recommended winner and shipping info.
"""
from __future__ import annotations

import asyncio
import logging
import math
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


def _value_score(price: Optional[float], trust: float, condition: float) -> float:
    """Value-for-money: high trust + good condition at low price."""
    if price is None or price <= 0:
        return 0.0
    quality = 0.6 * trust + 0.4 * condition
    price_factor = 1.0 / (1.0 + math.log1p(price / 100))
    return round(quality * (0.5 + price_factor), 4)


async def _run_single_search(
    query: str,
    db: Any,
    llm_engine: str = "ollama",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs one query through run_search_pipeline."""
    from app.services.search_pipeline import run_search_pipeline  # avoid circular
    try:
        result = await run_search_pipeline(query=query, db=db, llm_engine=llm_engine, session_id=session_id)
        results = result.get("results") or []
        return {
            "query": query,
            "top": results[0] if results else None,
            "results_count": len(results),
            "parsed_query": result.get("parsed_query"),
        }
    except Exception as exc:
        logger.warning("compare_pipeline search failed for query=%s: %s", query, exc)
        return {"query": query, "top": None, "results_count": 0, "error": str(exc)}


async def _fetch_cheapest_shipping(
    item_id: str,
    country_code: str = "IT",
) -> Optional[Dict[str, Any]]:
    """Fetches shipping for a single item; returns a lightweight summary."""
    try:
        from app.services.ebay import get_shipping_costs
        raw = await get_shipping_costs(item_id, country_code, "")
        if not raw:
            return None
        options = raw.get("shipping_options") or []
        if not options:
            return None

        def _cost(opt: Dict) -> float:
            try:
                return float(opt.get("shippingCost", {}).get("value", 9999))
            except Exception:
                return 9999.0

        sorted_opts = sorted(options, key=_cost)
        cheapest = sorted_opts[0]
        cost_val = _cost(cheapest)
        is_free = cost_val == 0.0
        currency = cheapest.get("shippingCost", {}).get("currency", "EUR")

        return {
            "cost": 0.0 if is_free else cost_val,
            "currency": currency,
            "free": is_free,
            "service": cheapest.get("shippingServiceCode", "Standard"),
            "min_delivery": cheapest.get("minEstimatedDeliveryDate"),
            "max_delivery": cheapest.get("maxEstimatedDeliveryDate"),
            "options_count": len(options),
        }
    except Exception as exc:
        logger.debug("_fetch_cheapest_shipping failed for %s: %s", item_id, exc)
        return None


# ============================================================
# PUBLIC ASYNC API
# ============================================================

async def run_compare_pipeline(
    queries: List[str],
    db: Any,
    llm_engine: str = "ollama",
    max_queries: int = 4,
    session_id: Optional[str] = None,
    fetch_shipping: bool = True,
    shipping_country: str = "IT",
) -> Dict[str, Any]:
    """
    Run up to `max_queries` searches in parallel and compare their top results.

    Args:
        queries:          List of product search strings to compare.
        db:               SQLAlchemy session.
        llm_engine:       LLM backend for query parsing.
        fetch_shipping:   If True, also fetch cheapest shipping for each candidate.
        shipping_country: ISO country code for shipping estimates.

    Returns:
        A dict with:
            - candidates: list of scored candidates (with shipping info)
            - winner: the winning candidate
            - winner_reason: human-readable explanation
            - comparison_matrix: full data matrix for display
    """
    queries = [q.strip() for q in queries if q and q.strip()][:max_queries]

    if len(queries) < 2:
        return {
            "status": "error",
            "error": "Servono almeno 2 query per confrontare i prodotti.",
        }

    # ── Run all searches in parallel ────────────────────────────────────
    search_tasks = [
        _run_single_search(q, db, llm_engine, session_id=session_id)
        for q in queries
    ]
    raw_results: List[Dict[str, Any]] = await asyncio.gather(*search_tasks, return_exceptions=False)

    # ── Extract top items ────────────────────────────────────────────────
    candidates = []
    for raw in raw_results:
        top = raw.get("top")
        if top is None:
            continue
        candidates.append({
            "query": raw["query"],
            "ebay_id": top.get("ebay_id"),
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

    # ── Shipping fetch (parallel, best-effort) ───────────────────────────
    shipping_map: Dict[str, Optional[Dict]] = {}
    if fetch_shipping:
        shipping_tasks = []
        shipping_ids = []
        for c in candidates:
            eid = c.get("ebay_id")
            if eid:
                shipping_tasks.append(_fetch_cheapest_shipping(eid, shipping_country))
                shipping_ids.append(eid)

        if shipping_tasks:
            shipping_results = await asyncio.gather(*shipping_tasks, return_exceptions=True)
            for eid, result in zip(shipping_ids, shipping_results):
                shipping_map[eid] = None if isinstance(result, Exception) else result

    # ── Scoring matrix ───────────────────────────────────────────────────
    all_prices = [c["price"] for c in candidates if c["price"] is not None]

    scored = []
    for c in candidates:
        price_score = _score_price(c["price"], all_prices)
        trust = float(c["trust_score"] or 0.5)
        relevance = float(c["ranking_score"] or 0.5)
        condition = _score_condition(c["condition"])
        overall = _overall_score(price_score, trust, relevance, condition)
        value = _value_score(c["price"], trust, condition)

        shipping_info = shipping_map.get(c.get("ebay_id") or "") if fetch_shipping else None

        scored.append({
            **c,
            "shipping": shipping_info,
            "_scores": {
                "price":     round(price_score, 3),
                "trust":     round(trust, 3),
                "relevance": round(relevance, 3),
                "condition": round(condition, 3),
                "overall":   overall,
                "value":     value,
            },
            "_overall": overall,
            "_value": value,
        })

    # Sort by overall score descending
    scored.sort(key=lambda x: x["_overall"], reverse=True)
    winner = scored[0]

    # ── Build human-readable reason ──────────────────────────────────────
    scores = winner["_scores"]
    reasons = []
    if scores["price"] >= 0.7:
        p = winner.get("price")
        reasons.append(f"prezzo competitivo ({p} {winner.get('currency', 'EUR')})")
    if scores["trust"] >= 0.7:
        reasons.append(f"venditore affidabile (trust {round(scores['trust'] * 100):.0f}%)")
    if scores["relevance"] >= 0.7:
        reasons.append("alta rilevanza per la query")
    if winner.get("condition", "").lower() in {"nuovo", "new"}:
        reasons.append("prodotto nuovo")
    w_shipping = winner.get("shipping")
    if w_shipping and w_shipping.get("free"):
        reasons.append("spedizione gratuita inclusa")
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
            "ebay_id": winner.get("ebay_id"),
            "title": winner["title"],
            "price": winner["price"],
            "currency": winner["currency"],
            "condition": winner["condition"],
            "seller_name": winner["seller_name"],
            "trust_score": winner["trust_score"],
            "url": winner["url"],
            "image_url": winner["image_url"],
            "overall_score": winner["_overall"],
            "value_score": winner["_value"],
            "shipping": winner.get("shipping"),
        },
        "winner_reason": winner_reason,
        "comparison_matrix": [
            {
                "query": c["query"],
                "ebay_id": c.get("ebay_id"),
                "title": c["title"],
                "price": c["price"],
                "currency": c["currency"],
                "condition": c["condition"],
                "seller_name": c["seller_name"],
                "trust_score": c["trust_score"],
                "url": c["url"],
                "image_url": c.get("image_url"),
                "shipping": c.get("shipping"),
                "scores": c["_scores"],
            }
            for c in scored
        ],
    }
