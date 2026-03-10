from __future__ import annotations

import re
from typing import Any, Dict, Optional, Protocol

from app.services.search_pipeline import run_search_pipeline


class ToolContextLike(Protocol):
    db: Any
    user: Optional[object]
    llm_engine: str


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


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
        r"(dammi i feedback|analizza il venditore|analizza|controlla il venditore|controlla se vende|verifica se vende|feedback del venditore)",
        "",
        q,
    )
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def normalize_search_arguments(action_input: Dict[str, Any], fallback_query: str = "") -> Dict[str, Any]:
    query = _clean_text(action_input.get("query") or fallback_query)
    query = clean_search_query(query)
    if not query:
        raise ValueError("search_products richiede una query non vuota.")
    return {"query": query}


def execute_search_tool(action_input: Dict[str, Any], context: ToolContextLike) -> Dict[str, Any]:
    clean = normalize_search_arguments(action_input)

    payload = run_search_pipeline(
        query=clean["query"],
        db=context.db,
        user=getattr(context, "user", None),
        llm_engine=getattr(context, "llm_engine", "ollama"),
    )

    if not isinstance(payload, dict):
        payload = {"result": payload}

    payload.setdefault("query", clean["query"])

    results = payload.get("results") or []
    payload.setdefault("results_count", len(results))
    payload.setdefault("status", "ok" if results else "no_data")

    return payload