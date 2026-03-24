from typing import Dict, Any
from app.utils.text import clean_text as _clean_text

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
                    summary += f" con affidabilità {round(float(trust) * 100):.0f}%"
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
        if trust_score is not None:
            try:
                p = round(float(trust_score) * 100)
                if p >= 85: summary += f" Profilo eccellente (Trust: {p}%)."
                elif p >= 70: summary += f" Profilo affidabile (Trust: {p}%)."
                else: summary += f" Attenzione: affidabilità limitata ({p}%)."
            except Exception:
                pass

        if sentiment_score is not None:
            try:
                s = round(float(sentiment_score) * 100)
                if s >= 70: summary += " Feedback molto positivi."
                elif s < 40: summary += " Segnali di insoddisfazione nei commenti."
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
    summary = f"Trovati {count} oggetti simili." if status == "ok" else f"Errore nel recupero oggetti simili: {raw.get('error')}"
    
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

import re

def clean_search_query(query: str) -> str:
    q = _clean_text(query).lower()
    if not q:
        return ""

    q = re.sub(r"(venditore|seller)\s+[a-zA-Z0-9._-]+", "", q)
    q = re.sub(
        r"(feedback|recensioni|reputazione|trust)\s+(del|della|di)\s+[a-zA-Z0-9._-]+",
        "",
        q,
    )
    q = re.sub(
        r"(dammi i feedback|analizza il venditore|analizza|controlla il venditore|controlla se vende|verifica se vende|feedback del venditore|controlla|verifica|affidabilità|reputazione|trust|feedback|confronta|compara|mi serve|trovami|cerca|mostrami|fammi|bro|poi|grazie|per favore|qualsiasi)",
        "",
        q,
    )
    q = re.sub(r"\s+", " ", q)
    return q.strip()


_SELLER_NOISE = {
    "venditore",
    "seller",
    "feedback",
    "recensioni",
    "reputazione",
    "trust",
    "affidabile",
    "affidabilita",
    "affidabilità",
    "utente",
    "negozio",
    "store",
}

_SELLER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:venditore|seller)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9._-]{2,})", re.IGNORECASE),
    re.compile(
        r"(?:feedback|recensioni|reputazione|trust|affidabil(?:e|it[aà]))\s+(?:del|della|di)\s+([A-Za-z0-9][A-Za-z0-9._-]{2,})",
        re.IGNORECASE,
    ),
    re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]{2,})\s+(?:è|e)\s+(?:affidabile|sicuro)", re.IGNORECASE),
)

def extract_explicit_seller(text: str) -> str | None:
    raw_text = _clean_text(text)
    if not raw_text:
        return None

    for pattern in _SELLER_PATTERNS:
        match = pattern.search(raw_text)
        if not match:
            continue

        candidate = _clean_text(match.group(1)).strip(" .,:;!?()[]{}\"'")
        if not candidate:
            continue

        if candidate.lower() in _SELLER_NOISE:
            continue

        return candidate

    return None
