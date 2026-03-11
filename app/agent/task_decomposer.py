from __future__ import annotations

import logging
import re
from typing import Dict, List

from app.agent.tool_registry import (
    analyze_user_query,
    find_first_tool_by_tags,
    get_tool_spec,
)

logger = logging.getLogger(__name__)

MULTI_STEP_CONNECTORS = (
    " e ",
    " poi ",
    " inoltre ",
    " anche ",
    " controlla ",
    " verifica ",
    " insieme ",
    " assieme ",
)


def should_decompose_query(query: str) -> bool:
    profile = analyze_user_query(query)
    if not profile["text"]:
        return False

    if not profile["search_signal"]:
        return False

    if not profile["seller_signal"]:
        return False

    if profile["multi_signal"] or profile["comparison_signal"]:
        return True

    lowered = query.lower()
    return any(connector in lowered for connector in MULTI_STEP_CONNECTORS)


def _select_capability_tools() -> Dict[str, str | None]:
    return {
        "seller": find_first_tool_by_tags("seller", "trust", "feedback"),
        "search": find_first_tool_by_tags("search", "product", "catalog"),
        "compare": find_first_tool_by_tags("compare", "product"),
    }


def _normalize_task(tool_name: str, action_input: Dict, query: str, explicit_seller: str | None) -> Dict | None:
    spec = get_tool_spec(tool_name)
    if spec is None:
        return None

    memory_like = type(
        "MemoryLike",
        (),
        {"user_query": query, "last_seller_name": explicit_seller},
    )()

    payload = dict(action_input or {})
    try:
        if spec.input_normalizer:
            payload = spec.input_normalizer(payload, memory_like)
    except Exception:
        return None

    for field_name in spec.required_fields:
        value = payload.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None

    return {"tool": tool_name, "input": payload}


def _build_deterministic_tasks(query: str) -> List[Dict]:
    profile = analyze_user_query(query)
    tools = _select_capability_tools()

    seller_tool = tools["seller"]
    search_tool = tools["search"]
    compare_tool = tools["compare"]
    explicit_seller = profile["explicit_seller"]

    tasks: List[Dict] = []

    # Priorità: se è un confronto, usiamo il tool dedicato
    if profile["comparison_signal"] and compare_tool:
        normalized = _normalize_task(
            compare_tool,
            {"queries": query},
            query=query,
            explicit_seller=explicit_seller,
        )
        if normalized:
            return [normalized]

    # Nei casi ibridi espliciti vogliamo prima la ricerca, poi il controllo seller.
    if profile["search_signal"] and search_tool:
        # Se abbiamo già un compare_tool pianificato, non aggiungiamo search generico
        normalized = _normalize_task(
            search_tool,
            {"query": profile["cleaned_search_query"] or query},
            query=query,
            explicit_seller=explicit_seller,
        )
        if normalized:
            tasks.append(normalized)

    if profile["seller_signal"] and explicit_seller and seller_tool:
        normalized = _normalize_task(
            seller_tool,
            {"seller_name": explicit_seller, "page": 1, "limit": 10},
            query=query,
            explicit_seller=explicit_seller,
        )
        if normalized:
            tasks.append(normalized)

    cleaned: List[Dict] = []
    for task in tasks:
        if cleaned and cleaned[-1] == task:
            continue
        cleaned.append(task)

    return cleaned


def decompose_query(query: str, llm_engine: str) -> List[Dict]:
    """
    Decomposer deterministico e volutamente conservativo.
    In Fase 1 serve solo per casi multi-step molto espliciti.
    """
    if not should_decompose_query(query):
        return []

    try:
        return _build_deterministic_tasks(query)
    except Exception as exc:
        logger.warning("Task decomposition failed: %s", exc)
        return []