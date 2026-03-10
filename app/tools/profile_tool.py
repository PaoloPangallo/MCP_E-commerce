from __future__ import annotations

from typing import Any, Dict

from app.services.parser import parse_query_service


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_profile_arguments(action_input: Dict[str, Any], fallback_query: str = "") -> Dict[str, Any]:
    query = _clean_text(action_input.get("query") or fallback_query)
    if not query:
        raise ValueError("profile_query richiede una query non vuota.")
    return {"query": query}


def execute_profile_tool(action_input: Dict[str, Any]) -> Dict[str, Any]:
    clean = normalize_profile_arguments(action_input)
    parsed = parse_query_service(clean["query"])

    if not isinstance(parsed, dict):
        parsed = {"result": parsed}

    return {
        "status": "ok",
        "query": clean["query"],
        "parsed": parsed,
    }