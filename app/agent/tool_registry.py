import re
from typing import Any, Dict

from app.utils.text import clean_text as _clean_text
from app.mcp.normalizers import clean_search_query
from app.mcp.normalizers import extract_explicit_seller

_QUERY_PATTERNS: Dict[str, tuple[str, ...]] = {
    "conversation": (
        r"\b(ciao|salve|buongiorno|buonasera|hey)\b",
        r"\b(come va|come stai|tutto bene)\b",
        r"\b(cosa ne pensi|che ne pensi|secondo te|mi spieghi|spiegami)\b",
    ),
    "seller": (
        r"\b(venditore|seller)\b",
        r"\b(feedback|recensioni|reputazione|affidabile|affidabilit[aà]|trust)\b",
    ),
    "search": (
        r"\b(cerca|trova|mostra|prezzo|prodotto|prodotti|compra|vende|vendita)\b",
        r"\b\d+[\.,]?\d*\s*(euro|eur|€)\b",
        r"\b(taglia|misura|numero|colore|modello|marca)\b",
    ),
    "multi": (
        r"\b(e poi|oltre a|assieme a|insieme a|anche|controlla|verifica)\b",
        r"[,;:].+\b(cerca|trova|analizza|feedback|seller|venditore|vende)\b",
    ),
    "comparison": (
        r"\b(compara|compari|comparami|confronta|confrontami|confronto|versus|vs|differenza|differenze)\b",
        r"\b(?:meglio di|peggio di|quale\s+.*meglio|quale\s+.*peggio)\b",
    ),
    "market": (
        r"\b(trend|mercato|trends|analisi|statistica|andamento|storico)\b",
        r"\b(serpapi|google trends|prezzo medio|interesse)\b",
    ),
    "deals": (
        r"\b(offerta|offerte|deals|occasioni|promozioni|sconti|sconto|promozione|occasione|deal)\b",
    ),
}

def analyze_user_query(query: str) -> Dict[str, Any]:
    text = _clean_text(query)
    lowered = text.lower()
    explicit_seller = extract_explicit_seller(text)

    conversation_signal = any(
        re.search(pattern, lowered, re.IGNORECASE)
        for pattern in _QUERY_PATTERNS["conversation"]
    )
    seller_signal = bool(explicit_seller) or any(
        re.search(pattern, lowered, re.IGNORECASE)
        for pattern in _QUERY_PATTERNS["seller"]
    )
    search_signal = any(
        re.search(pattern, lowered, re.IGNORECASE)
        for pattern in _QUERY_PATTERNS["search"]
    )
    multi_signal = any(
        re.search(pattern, lowered, re.IGNORECASE)
        for pattern in _QUERY_PATTERNS["multi"]
    )
    comparison_signal = any(
        re.search(pattern, lowered, re.IGNORECASE)
        for pattern in _QUERY_PATTERNS["comparison"]
    )
    market_signal = any(
        re.search(pattern, lowered, re.IGNORECASE)
        for pattern in _QUERY_PATTERNS["market"]
    )
    deals_signal = any(
        re.search(pattern, lowered, re.IGNORECASE)
        for pattern in _QUERY_PATTERNS["deals"]
    )

    cleaned_search_query = clean_search_query(text)
    has_budget_signal = bool(re.search(r"\b\d+[\.,]?\d*\s*(euro|eur|€)\b", lowered))

    return {
        "text": text,
        "cleaned_search_query": cleaned_search_query,
        "explicit_seller": explicit_seller,
        "conversation_signal": conversation_signal,
        "seller_signal": seller_signal,
        "search_signal": search_signal,
        "comparison_signal": comparison_signal,
        "market_signal": market_signal,
        "deals_signal": deals_signal,
        "multi_signal": multi_signal,
        "has_budget_signal": has_budget_signal,
    }
