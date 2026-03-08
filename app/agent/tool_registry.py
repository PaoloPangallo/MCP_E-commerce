from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from app.services.search_pipeline import run_search_pipeline
from app.services.seller_pipeline import run_seller_pipeline


@dataclass(slots=True)
class ToolContext:
    db: Session
    user: Optional[object] = None
    llm_engine: str = "ollama"


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    executor: Callable[[Dict[str, Any], ToolContext], Dict[str, Any]]
    required_fields: tuple[str, ...] = field(default_factory=tuple)


def _to_clean_str(value: Any) -> str:
    return str(value or "").strip()


def _to_bounded_int(
    value: Any,
    default: int,
    min_value: int,
    max_value: int,
) -> int:
    try:
        parsed = int(value)
        return min(max(parsed, min_value), max_value)
    except Exception:
        return default


def _normalize_search_input(action_input: Dict[str, Any]) -> Dict[str, Any]:
    query = _to_clean_str(action_input.get("query"))
    if not query:
        raise ValueError("search_pipeline requires a non-empty 'query'")

    return {
        "query": query,
    }


def _normalize_seller_input(action_input: Dict[str, Any]) -> Dict[str, Any]:
    seller_name = _to_clean_str(action_input.get("seller_name"))
    if not seller_name:
        raise ValueError("seller_pipeline requires a non-empty 'seller_name'")

    return {
        "seller_name": seller_name,
        "page": _to_bounded_int(action_input.get("page", 1), default=1, min_value=1, max_value=999),
        "limit": _to_bounded_int(action_input.get("limit", 10), default=10, min_value=1, max_value=50),
    }


def _run_search_pipeline(action_input: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
    clean = _normalize_search_input(action_input)

    return run_search_pipeline(
        query=clean["query"],
        db=context.db,
        user=context.user,
        llm_engine=context.llm_engine,
    )


def _run_seller_pipeline(action_input: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
    clean = _normalize_seller_input(action_input)

    try:
        result = run_seller_pipeline(
            seller_name=clean["seller_name"],
            page=clean["page"],
            limit=clean["limit"],
        )

        if not isinstance(result, dict):
            result = {"result": result}

        result.setdefault("seller_name", clean["seller_name"])
        result.setdefault("page", clean["page"])
        result.setdefault("limit", clean["limit"])
        result.setdefault("feedbacks", [])
        return result

    except Exception as e:
        return {
            "seller_name": clean["seller_name"],
            "page": clean["page"],
            "limit": clean["limit"],
            "count": 0,
            "trust_score": None,
            "sentiment_score": None,
            "feedbacks": [],
            "error": str(e),
        }


TOOLS: Dict[str, ToolSpec] = {
    "search_pipeline": ToolSpec(
        name="search_pipeline",
        description=(
            "Runs the full product search pipeline. "
            "Use it for product discovery, product comparison, budget-based search, "
            "and whenever you need ranked search results enriched with trust, RAG and explanation."
        ),
        input_schema={
            "query": "string",
        },
        required_fields=("query",),
        executor=_run_search_pipeline,
    ),
    "seller_pipeline": ToolSpec(
        name="seller_pipeline",
        description=(
            "Runs seller feedback analysis. "
            "Use it only when a seller name is explicitly provided or already known from previous search results."
        ),
        input_schema={
            "seller_name": "string",
            "page": "int",
            "limit": "int",
        },
        required_fields=("seller_name",),
        executor=_run_seller_pipeline,
    ),
}


def get_tool_descriptions() -> Dict[str, str]:
    return {name: spec.description for name, spec in TOOLS.items()}


def get_tool_names() -> list[str]:
    return list(TOOLS.keys())


def get_tool_spec(name: str) -> Optional[ToolSpec]:
    return TOOLS.get(name)