from typing import Dict, Any

def _normalize_search_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    results = raw.get("results") or raw.get("items") or []
    top_result = results[0] if results else None

    return {
        "status": "ok" if results else "no_results",
        "query": raw.get("query"),
        "results_count": raw.get("results_count", len(results)),
        "results": results,
        "top_result": top_result,
        "analysis": raw.get("analysis"),
        "metrics": raw.get("metrics"),
        "rag_context": raw.get("rag_context"),
        "raw": raw,
    }


def _normalize_seller_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    feedbacks = raw.get("feedbacks") or raw.get("feedback") or []
    count = raw.get("count", len(feedbacks))

    if raw.get("status"):
        status = raw["status"]
    else:
        status = "ok" if count > 0 else "no_data"

    return {
        "status": status,
        "seller_name": raw.get("seller_name"),
        "count": count,
        "feedbacks": feedbacks,
        "trust_score": raw.get("trust_score"),
        "sentiment_score": raw.get("sentiment_score"),
        "error": raw.get("error"),
        "raw": raw,
    }

def _normalize_item_details_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": raw.get("status", "ok"),
        "item_id": raw.get("item_id"),
        "data": raw.get("data"),
        "error": raw.get("error"),
        "message": raw.get("message"),
        "raw": raw,
    }


def _normalize_similar_items_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": raw.get("status", "ok"),
        "item_id": raw.get("item_id"),
        "results": raw.get("results", []),
        "results_count": raw.get("results_count", 0),
        "error": raw.get("error"),
        "raw": raw,
    }


def _normalize_shipping_costs_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": raw.get("status", "ok"),
        "item_id": raw.get("item_id"),
        "data": raw.get("data"),
        "error": raw.get("error"),
        "raw": raw,
    }
