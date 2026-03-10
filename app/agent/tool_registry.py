from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session

from app.agent.schemas import LatencyClass, ObservationQuality, ObservationStatus
from app.tools import (
    execute_conversation_tool,
    execute_search_tool,
    execute_seller_tool,
)
from app.tools.search_tool import clean_search_query, normalize_search_arguments
from app.tools.seller_tool import extract_explicit_seller, normalize_seller_arguments

InputNormalizer = Callable[[Dict[str, Any], Any], Dict[str, Any]]

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
        r"\b(cerca|trova|mostra|compara|confronta|prezzo|prodotto|prodotti|compra|vende|vendita)\b",
        r"\b\d+[\.,]?\d*\s*(euro|eur|€)\b",
        r"\b(taglia|misura|numero|colore|modello|marca)\b",
    ),
    "multi": (
        r"\b(e poi|oltre a|assieme a|insieme a|anche|controlla|verifica)\b",
        r"[,;:].+\b(cerca|trova|analizza|feedback|seller|venditore|vende)\b",
    ),
}


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

    tags: tuple[str, ...] = field(default_factory=tuple)
    examples: tuple[str, ...] = field(default_factory=tuple)
    required_fields: tuple[str, ...] = field(default_factory=tuple)
    state_key: str = ""
    max_retries: int = 0

    cost: int = 1
    latency_class: LatencyClass = "medium"
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    produced_entities: tuple[str, ...] = field(default_factory=tuple)
    can_run_in_parallel: bool = True
    use_cache: bool = True

    input_normalizer: Optional[InputNormalizer] = None
    status_resolver: Optional[Callable[[Dict[str, Any]], ObservationStatus]] = None
    quality_resolver: Optional[Callable[[Dict[str, Any]], ObservationQuality]] = None
    terminal_resolver: Optional[Callable[[Dict[str, Any]], bool]] = None
    result_normalizer: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None
    summarizer: Optional[Callable[[Dict[str, Any]], str]] = None


TOOLS: Dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> None:
    TOOLS[spec.name] = spec


def get_tool_spec(name: str) -> Optional[ToolSpec]:
    return TOOLS.get(name)


def get_tool_names() -> list[str]:
    return list(TOOLS.keys())


def find_tools_by_tags(*tags: str, match_all: bool = False) -> list[str]:
    wanted = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
    if not wanted:
        return []

    matched: list[str] = []
    for name, spec in TOOLS.items():
        spec_tags = {tag.lower() for tag in spec.tags}
        if match_all and wanted.issubset(spec_tags):
            matched.append(name)
        elif not match_all and wanted & spec_tags:
            matched.append(name)
    return matched


def find_first_tool_by_tags(*tags: str, match_all: bool = False) -> Optional[str]:
    tools = find_tools_by_tags(*tags, match_all=match_all)
    return tools[0] if tools else None


def get_tool_descriptions() -> Dict[str, str]:
    return {name: spec.description for name, spec in TOOLS.items()}


def get_tool_catalog() -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for name, spec in TOOLS.items():
        catalog[name] = {
            "description": spec.description,
            "input_schema": spec.input_schema,
            "required_fields": list(spec.required_fields),
            "tags": list(spec.tags),
            "examples": list(spec.examples[:2]),
            "state_key": spec.state_key or None,
            "cost": spec.cost,
            "latency_class": spec.latency_class,
            "dependencies": list(spec.dependencies),
            "produced_entities": list(spec.produced_entities),
            "can_run_in_parallel": spec.can_run_in_parallel,
            "use_cache": spec.use_cache,
        }
    return catalog


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


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

    cleaned_search_query = clean_search_query(text)
    has_budget_signal = bool(re.search(r"\b\d+[\.,]?\d*\s*(euro|eur|€)\b", lowered))

    return {
        "text": text,
        "cleaned_search_query": cleaned_search_query,
        "explicit_seller": explicit_seller,
        "conversation_signal": conversation_signal,
        "seller_signal": seller_signal,
        "search_signal": search_signal,
        "multi_signal": multi_signal,
        "has_budget_signal": has_budget_signal,
    }


def _normalize_search_action_input(action_input: Dict[str, Any], memory: Any) -> Dict[str, Any]:
    fallback_query = getattr(memory, "user_query", "")
    return normalize_search_arguments(action_input, fallback_query=fallback_query)


def _normalize_seller_action_input(action_input: Dict[str, Any], memory: Any) -> Dict[str, Any]:
    fallback_seller = (
        _clean_text(action_input.get("seller_name"))
        or _clean_text(getattr(memory, "last_seller_name", ""))
        or _clean_text(extract_explicit_seller(getattr(memory, "user_query", "")))
    )
    return normalize_seller_arguments(action_input, fallback_seller=fallback_seller)


def _normalize_conversation_action_input(action_input: Dict[str, Any], memory: Any) -> Dict[str, Any]:
    query = _clean_text(action_input.get("query") or getattr(memory, "user_query", ""))
    if not query:
        raise ValueError("conversation richiede una query non vuota.")
    return {"query": query}


def _normalize_search_result(result: Dict[str, Any], action_input: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(result or {})
    payload.setdefault("query", _clean_text(action_input.get("query")))
    payload.setdefault("results", payload.get("results") or [])
    payload.setdefault("results_count", len(payload["results"]))
    payload.setdefault("status", "ok" if payload["results_count"] > 0 else "no_data")
    return payload


def _normalize_seller_result(result: Dict[str, Any], action_input: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(result or {})
    feedbacks = payload.get("feedbacks") or payload.get("feedback") or []
    count = int(payload.get("count", len(feedbacks)))

    payload.setdefault("seller_name", _clean_text(action_input.get("seller_name")))
    payload.setdefault("page", int(action_input.get("page", 1) or 1))
    payload.setdefault("limit", int(action_input.get("limit", 10) or 10))
    payload["count"] = count
    payload["feedbacks"] = feedbacks

    if count > 0:
        payload["status"] = "ok"
    else:
        payload["status"] = "no_data"
        payload.setdefault("error", "Nessun feedback disponibile per questo venditore.")
        payload["trust_score"] = None
        payload["sentiment_score"] = None

    return payload


def _normalize_conversation_result(result: Dict[str, Any], action_input: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(result or {})
    payload.setdefault("query", _clean_text(action_input.get("query")))
    payload.setdefault("answer", "")
    payload.setdefault("status", "ok" if payload.get("answer") else "error")
    return payload


def _resolve_search_status(payload: Dict[str, Any]) -> ObservationStatus:
    if payload.get("error"):
        return "error"
    if int(payload.get("results_count", 0)) <= 0:
        return "no_data"
    return "ok"


def _resolve_search_quality(payload: Dict[str, Any]) -> ObservationQuality:
    count = int(payload.get("results_count", 0))
    if count <= 0:
        return "empty"
    if count < 3:
        return "partial"
    return "good"


def _resolve_search_terminal(payload: Dict[str, Any]) -> bool:
    return bool(payload.get("error")) or int(payload.get("results_count", 0)) >= 1


def _resolve_seller_status(payload: Dict[str, Any]) -> ObservationStatus:
    if payload.get("error") and payload.get("status") == "error":
        return "error"
    if payload.get("status") == "no_data" or int(payload.get("count", 0)) <= 0:
        return "no_data"
    return "ok"


def _resolve_seller_quality(payload: Dict[str, Any]) -> ObservationQuality:
    count = int(payload.get("count", 0))
    if count <= 0:
        return "empty"
    if count < 3:
        return "partial"
    return "good"


def _resolve_seller_terminal(payload: Dict[str, Any]) -> bool:
    return payload.get("status") in {"ok", "no_data"} or bool(payload.get("error"))


def _resolve_conversation_status(payload: Dict[str, Any]) -> ObservationStatus:
    if payload.get("answer"):
        return "ok"
    return "error"


def _resolve_conversation_quality(payload: Dict[str, Any]) -> ObservationQuality:
    if payload.get("answer"):
        return "good"
    return "empty"


def _resolve_conversation_terminal(payload: Dict[str, Any]) -> bool:
    return True


def _summarize_search(payload: Dict[str, Any]) -> str:
    count = int(payload.get("results_count", 0))
    results = payload.get("results") or []
    if count <= 0:
        return "Search completata ma senza risultati utili."

    top = results[0] if results else {}
    title = _clean_text(top.get("title"))
    price = top.get("price")
    currency = _clean_text(top.get("currency") or top.get("price_currency"))
    seller = _clean_text(top.get("seller_name") or top.get("seller_username"))
    trust = top.get("trust_score")

    chunks = [f"Search completata con {count} risultati."]
    if title:
        chunks.append(f"Top result: {title}.")
    if price is not None:
        chunks.append(f"Prezzo: {price} {currency or 'EUR'}.")
    if seller:
        chunks.append(f"Seller: {seller}.")
    if trust is not None:
        try:
            chunks.append(f"Trust: {round(float(trust) * 100):.0f}%.")
        except Exception:
            pass
    return " ".join(chunks).strip()


def _summarize_seller(payload: Dict[str, Any]) -> str:
    seller_name = _clean_text(payload.get("seller_name")) or "venditore"
    count = int(payload.get("count", 0))
    trust_score = payload.get("trust_score")
    sentiment_score = payload.get("sentiment_score")
    error = _clean_text(payload.get("error"))

    if count <= 0:
        reason = error or "Nessun feedback disponibile per questo venditore."
        return f"Analisi seller completata senza dati utili per {seller_name}. Motivo: {reason}"

    trust_text = ""
    if trust_score is not None:
        try:
            trust_text = f" Trust score: {round(float(trust_score) * 100):.0f}%."
        except Exception:
            trust_text = ""

    sentiment_text = ""
    if sentiment_score is not None:
        try:
            sentiment_text = f" Sentiment score: {round(float(sentiment_score) * 100):.0f}%."
        except Exception:
            sentiment_text = ""

    return (
        f"Analizzato seller {seller_name}. "
        f"Feedback totali: {count}."
        f"{trust_text}{sentiment_text}"
    ).strip()


def _summarize_conversation(payload: Dict[str, Any]) -> str:
    answer = _clean_text(payload.get("answer"))
    if answer:
        return answer
    return _clean_text(payload.get("error")) or "Nessuna risposta conversazionale disponibile."


def _bootstrap_tools() -> None:
    TOOLS.clear()

    register_tool(
        ToolSpec(
            name="search_products",
            description="Cerca prodotti e-commerce usando la pipeline completa di search, ranking e trust.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            executor=execute_search_tool,
            tags=("search", "product", "catalog", "ebay"),
            examples=(
                "iphone 13 massimo 700 euro",
                "felpe adidas da donna taglia 42 o 44",
            ),
            required_fields=("query",),
            state_key="search",
            max_retries=0,
            cost=2,
            latency_class="high",
            dependencies=(),
            produced_entities=("products", "search_analysis", "metrics", "seller"),
            can_run_in_parallel=False,
            use_cache=True,
            input_normalizer=_normalize_search_action_input,
            status_resolver=_resolve_search_status,
            quality_resolver=_resolve_search_quality,
            terminal_resolver=_resolve_search_terminal,
            result_normalizer=_normalize_search_result,
            summarizer=_summarize_search,
        )
    )

    register_tool(
        ToolSpec(
            name="analyze_seller",
            description="Analizza un venditore e-commerce usando feedback, trust score e sentiment.",
            input_schema={
                "type": "object",
                "properties": {
                    "seller_name": {"type": "string"},
                    "page": {"type": "integer", "default": 1},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["seller_name"],
            },
            executor=execute_seller_tool,
            tags=("seller", "trust", "feedback", "ebay"),
            examples=(
                "feedback di mario_store",
                "analizza il venditore dyson-official",
            ),
            required_fields=("seller_name",),
            state_key="seller",
            max_retries=0,
            cost=1,
            latency_class="medium",
            dependencies=(),
            produced_entities=("seller", "trust_score", "sentiment_score", "feedback"),
            can_run_in_parallel=True,
            use_cache=True,
            input_normalizer=_normalize_seller_action_input,
            status_resolver=_resolve_seller_status,
            quality_resolver=_resolve_seller_quality,
            terminal_resolver=_resolve_seller_terminal,
            result_normalizer=_normalize_seller_result,
            summarizer=_summarize_seller,
        )
    )

    register_tool(
        ToolSpec(
            name="conversation",
            description="Risponde a messaggi conversazionali generici quando non serve usare tool eBay.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            executor=execute_conversation_tool,
            tags=("conversation", "chat", "general"),
            examples=(
                "ciao come stai",
                "spiegami come funziona ebay",
            ),
            required_fields=("query",),
            state_key="conversation",
            max_retries=0,
            cost=1,
            latency_class="medium",
            dependencies=(),
            produced_entities=("answer",),
            can_run_in_parallel=False,
            use_cache=False,
            input_normalizer=_normalize_conversation_action_input,
            status_resolver=_resolve_conversation_status,
            quality_resolver=_resolve_conversation_quality,
            terminal_resolver=_resolve_conversation_terminal,
            result_normalizer=_normalize_conversation_result,
            summarizer=_summarize_conversation,
        )
    )


_bootstrap_tools()