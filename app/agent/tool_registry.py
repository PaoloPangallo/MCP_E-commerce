from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from app.agent.schemas import ObservationStatus, ObservationQuality
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
    state_key: str = ""
    max_retries: int = 0

    status_resolver: Optional[Callable[[Dict[str, Any]], ObservationStatus]] = None
    quality_resolver: Optional[Callable[[Dict[str, Any]], ObservationQuality]] = None
    terminal_resolver: Optional[Callable[[Dict[str, Any]], bool]] = None
    result_normalizer: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None
    summarizer: Optional[Callable[[Dict[str, Any]], str]] = None


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

    return {"query": query}


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

    result = run_seller_pipeline(
        seller_name=clean["seller_name"],
        page=clean["page"],
        limit=clean["limit"],
    )

    if not isinstance(result, dict):
        result = {"result": result}

    return result


def _normalize_search_result(result: Dict[str, Any], action_input: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(result or {})
    payload.setdefault("query", _to_clean_str(action_input.get("query")))
    payload.setdefault("results", [])
    payload.setdefault("results_count", len(payload.get("results") or []))
    payload.setdefault("analysis", payload.get("analysis") or "")
    payload.setdefault("metrics", payload.get("metrics") or payload.get("ir_metrics"))
    return payload


def _normalize_seller_result(result: Dict[str, Any], action_input: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(result or {})
    payload.setdefault("seller_name", _to_clean_str(action_input.get("seller_name")))
    payload.setdefault("page", _to_bounded_int(action_input.get("page", 1), default=1, min_value=1, max_value=999))
    payload.setdefault("limit", _to_bounded_int(action_input.get("limit", 10), default=10, min_value=1, max_value=50))
    payload.setdefault("count", 0)
    payload.setdefault("trust_score", None)
    payload.setdefault("sentiment_score", None)
    payload.setdefault("feedbacks", [])
    return payload


def _search_status_resolver(payload: Dict[str, Any]) -> ObservationStatus:
    if payload.get("error"):
        return "error"
    return "ok"


def _search_quality_resolver(payload: Dict[str, Any]) -> ObservationQuality:
    results = payload.get("results") or []
    return "good" if results else "empty"


def _search_terminal_resolver(payload: Dict[str, Any]) -> bool:
    return True


def _seller_status_resolver(payload: Dict[str, Any]) -> ObservationStatus:
    error = str(payload.get("error") or "").strip()
    if error:
        return "no_data"
    return "ok"


def _seller_quality_resolver(payload: Dict[str, Any]) -> ObservationQuality:
    if payload.get("error"):
        return "empty"

    count = payload.get("count")
    try:
        count_value = int(count)
    except Exception:
        count_value = 0

    if count_value <= 0:
        return "empty"
    if count_value < 3:
        return "partial"
    return "good"


def _seller_terminal_resolver(payload: Dict[str, Any]) -> bool:
    return True


def summarize_search(data: Dict[str, Any]) -> str:
    results = data.get("results") or []
    results_count = data.get("results_count", len(results))
    analysis = (data.get("analysis") or "").strip()

    parts = [f"Search completata con {results_count} risultati."]
    top = results[0] if results else None

    if top:
        title = top.get("title")
        price = top.get("price")
        currency = top.get("currency")
        seller = top.get("seller_name") or top.get("seller_username")
        trust = top.get("trust_score")

        if title:
            parts.append(f"Top result: {title}.")
        if price is not None:
            price_text = f"{price} {currency or ''}".strip()
            parts.append(f"Prezzo: {price_text}.")
        if seller:
            parts.append(f"Seller: {seller}.")
        if trust is not None:
            try:
                parts.append(f"Trust: {round(float(trust) * 100)}%.")
            except Exception:
                pass

    if analysis:
        parts.append(analysis[:180])

    return " ".join(parts).strip()


def summarize_seller(data: Dict[str, Any]) -> str:
    seller_name = data.get("seller_name")
    trust_score = data.get("trust_score")
    sentiment_score = data.get("sentiment_score")
    count = data.get("count")
    error = data.get("error")

    if error:
        return (
            f"Analisi seller completata senza dati utili per {seller_name or 'seller richiesto'}. "
            f"Motivo: {error}"
        )

    parts = []

    if seller_name:
        parts.append(f"Analizzato seller {seller_name}.")
    if count is not None:
        parts.append(f"Feedback totali: {count}.")
    if trust_score is not None:
        try:
            parts.append(f"Trust score: {round(float(trust_score) * 100)}%.")
        except Exception:
            pass
    if sentiment_score is not None:
        try:
            parts.append(f"Sentiment score: {round(float(sentiment_score) * 100)}%.")
        except Exception:
            pass

    return " ".join(parts).strip() or "Analisi seller completata."


TOOLS: Dict[str, ToolSpec] = {
    "search_pipeline": ToolSpec(
        name="search_pipeline",
        description=(
            "Use for product discovery, product comparison, budget-based search, "
            "and checking what products are being sold."
        ),
        input_schema={"query": "string"},
        required_fields=("query",),
        executor=_run_search_pipeline,
        state_key="search",
        max_retries=1,
        status_resolver=_search_status_resolver,
        quality_resolver=_search_quality_resolver,
        terminal_resolver=_search_terminal_resolver,
        result_normalizer=_normalize_search_result,
        summarizer=summarize_search,
    ),
    "seller_pipeline": ToolSpec(
        name="seller_pipeline",
        description=(
            "Use for seller trust, feedback analysis, reliability checks, "
            "or when the user explicitly asks about a seller."
        ),
        input_schema={
            "seller_name": "string",
            "page": "int",
            "limit": "int",
        },
        required_fields=("seller_name",),
        executor=_run_seller_pipeline,
        state_key="seller",
        max_retries=1,
        status_resolver=_seller_status_resolver,
        quality_resolver=_seller_quality_resolver,
        terminal_resolver=_seller_terminal_resolver,
        result_normalizer=_normalize_seller_result,
        summarizer=summarize_seller,
    ),
}


def get_tool_descriptions() -> Dict[str, str]:
    return {name: spec.description for name, spec in TOOLS.items()}


def get_tool_names() -> list[str]:
    return list(TOOLS.keys())


def get_tool_spec(name: str) -> Optional[ToolSpec]:
    return TOOLS.get(name)
