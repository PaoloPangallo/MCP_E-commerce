from __future__ import annotations

import re
from typing import Any, Dict, Optional, Protocol

from app.services.seller_pipeline import run_seller_pipeline


class ToolContextLike(Protocol):
    db: Any
    user: Optional[object]
    llm_engine: str


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


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _bounded_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
        return min(max(parsed, min_value), max_value)
    except Exception:
        return default


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


def normalize_seller_arguments(
    action_input: Dict[str, Any],
    fallback_seller: str = "",
) -> Dict[str, Any]:
    seller_name = _clean_text(action_input.get("seller_name") or fallback_seller)
    if not seller_name:
        raise ValueError("analyze_seller richiede seller_name.")

    return {
        "seller_name": seller_name,
        "page": _bounded_int(action_input.get("page", 1), default=1, min_value=1, max_value=999),
        "limit": _bounded_int(action_input.get("limit", 50), default=50, min_value=1, max_value=100),
    }


def execute_seller_tool(action_input: Dict[str, Any], context: ToolContextLike) -> Dict[str, Any]:
    clean = normalize_seller_arguments(action_input)

    payload = run_seller_pipeline(
        seller_name=clean["seller_name"],
        page=clean["page"],
        limit=clean["limit"],
    )

    if not isinstance(payload, dict):
        payload = {"result": payload}

    feedbacks = payload.get("feedbacks") or payload.get("feedback") or []
    count = int(payload.get("count", len(feedbacks)))

    payload.setdefault("seller_name", clean["seller_name"])
    payload.setdefault("page", clean["page"])
    payload.setdefault("limit", clean["limit"])
    payload.setdefault("count", count)

    if count > 0:
        payload.setdefault("status", "ok")
    else:
        payload["status"] = "no_data"
        payload.setdefault("error", "Nessun feedback disponibile per questo venditore.")
        payload["trust_score"] = None
        payload["sentiment_score"] = None

    return payload