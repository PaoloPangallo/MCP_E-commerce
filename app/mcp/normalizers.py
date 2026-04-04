import re
from typing import Dict, Any
from app.utils.text import clean_text as _clean_text

# ============================================================
# SOGLIE CONDIVISE
# ============================================================

_TRUST_EXCELLENT = 0.85
_TRUST_RELIABLE  = 0.70

_SENTIMENT_POSITIVE = 0.70
_SENTIMENT_NEGATIVE = 0.40


def _trust_label(p: float) -> str:
    if p >= _TRUST_EXCELLENT * 100:
        return "eccellente"
    if p >= _TRUST_RELIABLE * 100:
        return "affidabile"
    return "scarsa affidabilità"


def _sentiment_label(s: float) -> str:
    if s >= _SENTIMENT_POSITIVE * 100:
        return "positivi"
    if s <= _SENTIMENT_NEGATIVE * 100:
        return "negativi"
    return "misti"


def _normalize_search_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    results = raw.get("results") or raw.get("items") or []
    top_result = results[0] if results else None
    count = raw.get("results_count", len(results))

    if count <= 0:
        summary = "Search completata ma senza risultati utili."
    else:
        top_result = top_result or {}
        title = _clean_text(top_result.get("title"))
        price = top_result.get("price")
        currency = _clean_text(top_result.get("currency") or top_result.get("price_currency"))
        seller = _clean_text(top_result.get("seller_name") or top_result.get("seller_username"))
        trust = top_result.get("trust_score")

        if not title:
            summary = f"Ricerca completata: trovati {count} prodotti pertinenti."
        else:
            summary = f"Ho individuato {count} prodotti. Il miglior match è '{title}'"
            if price is not None:
                summary += f" a {price} {currency or 'EUR'}"
            if seller:
                summary += f" (venduto da {seller})"
            if trust is not None:
                try:
                    p = round(float(trust) * 100)
                    summary += f" — affidabilità {_trust_label(p)} ({p}%)"
                except Exception:
                    pass
            summary += "."

    return {
        "status": "ok" if results else "no_data",
        "query": raw.get("query"),
        "ebay_query_used": raw.get("ebay_query_used"),
        "results_count": count,
        "results": results,
        "top_result": top_result,
        "summary": summary,
        "analysis": raw.get("analysis"),
        "metrics": raw.get("metrics"),
        "rag_context": raw.get("rag_context"),
        "dominant_category_name": raw.get("dominant_category_name"),
    }


def _normalize_seller_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    feedbacks = raw.get("feedbacks") or raw.get("feedback") or []
    count = raw.get("count", len(feedbacks))

    if raw.get("status"):
        status = raw["status"]
    else:
        status = "ok" if count > 0 else "no_data"

    seller_name = _clean_text(raw.get("seller_name")) or "venditore"
    trust_score = raw.get("trust_score")
    sentiment_score = raw.get("sentiment_score")
    error = _clean_text(raw.get("error"))

    if count <= 0 or status != "ok":
        reason = error or "Nessun feedback disponibile per questo venditore."
        summary = f"Analisi seller completata senza dati utili per {seller_name}. Motivo: {reason}"
        status = "no_data" if status == "ok" else status
    else:
        summary = f"Analisi completata per {seller_name} ({count} feedback)."

        # Combina trust e sentiment in un'unica valutazione coerente
        try:
            trust_pct = round(float(trust_score) * 100) if trust_score is not None else None
            sent_pct  = round(float(sentiment_score) * 100) if sentiment_score is not None else None

            if trust_pct is not None and sent_pct is not None:
                trust_lbl = _trust_label(trust_pct)
                sent_lbl  = _sentiment_label(sent_pct)
                summary += (
                    f" Profilo {trust_lbl} (Trust: {trust_pct}%),"
                    f" feedback {sent_lbl} (Sentiment: {sent_pct}%)."
                )
            elif trust_pct is not None:
                trust_lbl = _trust_label(trust_pct)
                summary += f" Profilo {trust_lbl} (Trust: {trust_pct}%)."
            elif sent_pct is not None:
                sent_lbl = _sentiment_label(sent_pct)
                summary += f" Feedback {sent_lbl} (Sentiment: {sent_pct}%)."
        except Exception:
            pass

        summary = summary.strip()

    return {
        "status": status,
        "seller_name": raw.get("seller_name"),
        "count": count,
        "feedbacks": feedbacks,
        "trust_score": trust_score,
        "sentiment_score": sentiment_score,
        "error": raw.get("error"),
        "summary": summary,
    }


def _normalize_item_details_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    status = raw.get("status", "ok")
    if status != "ok":
        summary = f"Dettagli oggetto non trovati o errore: {raw.get('error') or raw.get('message')}"
    else:
        data = raw.get("data", {})
        title = data.get("title", "") if isinstance(data, dict) else ""
        summary = f"Recuperati con successo i dettagli dell'oggetto: {title}"

    return {
        "status": status,
        "item_id": raw.get("item_id"),
        "data": raw.get("data"),
        "error": raw.get("error"),
        "message": raw.get("message"),
        "summary": summary,
    }


def _normalize_similar_items_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    status = raw.get("status", "ok")
    count = raw.get("results_count", 0)
    summary = (
        f"Trovati {count} oggetti simili."
        if status == "ok"
        else f"Errore nel recupero oggetti simili: {raw.get('error')}"
    )
    return {
        "status": status,
        "item_id": raw.get("item_id"),
        "results": raw.get("results", []),
        "results_count": count,
        "error": raw.get("error"),
        "summary": summary,
    }


def _normalize_shipping_costs_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    status = raw.get("status", "ok")
    if status != "ok":
        summary = f"Costi di spedizione non trovati o errore: {raw.get('error')}"
    else:
        item_id = raw.get("item_id", "sconosciuto")
        summary = f"Recuperati con successo i costi di spedizione per l'oggetto {item_id}"

    return {
        "status": status,
        "item_id": raw.get("item_id"),
        "data": raw.get("data"),
        "error": raw.get("error"),
        "summary": summary,
    }


def clean_search_query(query: str) -> str:
    # LLM Native parsing: the agent now provides high-quality queries.
    # We just do basic sanitization without risking information loss.
    q = _clean_text(query).lower()
    return q.strip()


_SELLER_NOISE = {
    "venditore", "seller", "feedback", "recensioni", "reputazione", "trust",
}

def extract_explicit_seller(text: str) -> str | None:
    # Left as a minimal fallback, but LLM extracts seller_name directly now.
    raw_text = _clean_text(text)
    if not raw_text:
        return None

    # Semplice estrazione post "venditore" o "seller" se presente
    match = re.search(r"(?:venditore|seller)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9._-]{2,})", raw_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None




def _normalize_playwright_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    results = raw.get("results") or []
    count = raw.get("results_count", len(results))
    query = raw.get("query", "")

    if count <= 0:
        summary = f"Ricerca Playwright completata per '{query}' ma senza risultati."
        status = "no_data"
    else:
        top = results[0]
        title = _clean_text(top.get("title", ""))
        price_raw = _clean_text(top.get("price_raw", ""))
        summary = f"Trovati {count} prodotti su eBay per '{query}'."
        if title:
            summary += f" Miglior risultato: '{title}'"
            if price_raw:
                summary += f" a {price_raw}"
            summary += "."
        status = "ok"

    return {
        "status": status,
        "query": query,
        "results_count": count,
        "results": results,
        "summary": summary,
    }
