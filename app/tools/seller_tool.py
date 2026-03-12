from __future__ import annotations

import re
from typing import Any, Dict, Optional, Protocol

from app.services.seller_pipeline import run_seller_pipeline_async


class ToolContextLike(Protocol):
    db: Any
    user: Optional[object]
    llm_engine: str


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


def extract_explicit_seller(text: str) -> str | None:
    raw_text = _clean_text(text)
    if not raw_text:
        return None

    for pattern in _SELLER_PATTERNS:
        match = pattern.search(raw_text)
        if match:
            return _clean_text(match.group(1)).strip(" .,:;!?()[]{}\"'")
    return None


def normalize_seller_arguments(action_input: Dict[str, Any], fallback_seller: str = "") -> Dict[str, Any]:
    seller_name = _clean_text(action_input.get("seller_name") or fallback_seller)
    return {
        "seller_name": seller_name,
        "page": int(action_input.get("page", 1)),
        "limit": int(action_input.get("limit", 50)),
    }


async def execute_seller_tool(action_input: Dict[str, Any], context: ToolContextLike) -> Dict[str, Any]:
    clean = normalize_seller_arguments(action_input)
    payload = await run_seller_pipeline_async(
        seller_name=clean["seller_name"],
        page=clean["page"],
        limit=clean["limit"],
    )
    return payload