
from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace
from typing import Dict, List

from app.agent.tool_registry import (
    find_first_tool_by_tags,
    get_tool_catalog,
    get_tool_spec,
)
from app.services.parser import call_gemini, call_ollama, extract_first_json_object

logger = logging.getLogger(__name__)

DECOMPOSER_PROMPT = """
You are an MCP-style e-commerce task planner.

Decompose ONLY multi-step user requests into an ordered executable tool plan.
Use ONLY tools from the provided catalog.
Do not invent tool names or parameters.

Rules:
- Produce tasks only when multiple actionable needs exist.
- Typical multi-step cases:
  1) seller feedback + product search
  2) seller analysis + check what the seller sells
  3) conversational wrapper + concrete retrieval request
- Keep task order meaningful.
- Prefer seller analysis before product search when both are explicitly requested.
- Return only executable tool tasks.
- If the request is effectively single-step, return an empty task list.
- Keep the output strictly as JSON.

Return ONLY JSON:
{
  "tasks":[
    {"tool":"tool_name","input":{}}
  ]
}
""".strip()


_MULTI_INTENT_PATTERNS = [
    r"\b(e poi|and then|insieme a|oltre a|assieme a)\b",
    r"[?,;:].+\b(cerca|trova|analizza|feedback|vende|seller|venditore|recensioni)\b",
]

_SELLER_HINTS = [
    "seller",
    "venditore",
    "feedback",
    "affidabile",
    "trust",
    "recensioni",
]

_SEARCH_HINTS = [
    "cerca",
    "trova",
    "mostra",
    "vende",
    "selling",
    "prodotto",
    "prodotti",
    "prezzo",
    "compra",
]

_CONVERSATION_HINTS = [
    "ciao",
    "come stai",
    "come va",
    "tutto bene",
    "salve",
]


def _contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def _extract_explicit_seller(text: str) -> str | None:
    pattern = r"(?:venditore|seller)\s+([A-Za-z0-9._-]{3,})"
    match = re.search(pattern, text or "", re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _clean_search_query(query: str) -> str:
    q = (query or "").strip().lower()
    q = re.sub(r"(venditore|seller)\s+[a-zA-Z0-9._-]+", "", q)
    q = re.sub(
        r"(dammi i feedback|analizza il venditore|analizza|feedback del venditore|controlla se vende|verifica se vende)",
        "",
        q,
    )
    return re.sub(r"\s+", " ", q).strip()


def should_decompose_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False

    has_seller = _contains_any(q, _SELLER_HINTS)
    has_search = _contains_any(q, _SEARCH_HINTS)
    has_conversation_wrapper = _contains_any(q, _CONVERSATION_HINTS)

    if has_seller and has_search:
        return True

    if has_conversation_wrapper and (has_seller or has_search):
        return True

    if any(re.search(pattern, q, re.IGNORECASE) for pattern in _MULTI_INTENT_PATTERNS):
        return has_seller or has_search

    return False


def _select_capability_tools() -> Dict[str, str | None]:
    return {
        "seller": find_first_tool_by_tags("seller", "trust", "feedback"),
        "search": find_first_tool_by_tags("search", "product", "catalog"),
    }


def _normalize_task(tool_name: str, action_input: Dict, query: str, explicit_seller: str | None) -> Dict | None:
    spec = get_tool_spec(tool_name)
    if spec is None:
        return None

    payload = dict(action_input or {})
    memory_like = SimpleNamespace(user_query=query, last_seller_name=explicit_seller)

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


def _fallback_multi_tasks(query: str) -> List[Dict]:
    q = (query or "").strip()
    q_lower = q.lower()

    tools = _select_capability_tools()
    seller_tool = tools["seller"]
    search_tool = tools["search"]
    seller_name = _extract_explicit_seller(q)

    tasks: List[Dict] = []

    has_seller = _contains_any(q_lower, _SELLER_HINTS)
    has_search = _contains_any(q_lower, _SEARCH_HINTS)

    if has_seller and seller_name and seller_tool:
        normalized = _normalize_task(
            seller_tool,
            {"seller_name": seller_name, "page": 1, "limit": 10},
            query=q,
            explicit_seller=seller_name,
        )
        if normalized:
            tasks.append(normalized)

    if has_search and search_tool:
        normalized = _normalize_task(
            search_tool,
            {"query": _clean_search_query(q)},
            query=q,
            explicit_seller=seller_name,
        )
        if normalized:
            tasks.append(normalized)

    return tasks


def decompose_query(query: str, llm_engine: str) -> List[Dict]:
    if not should_decompose_query(query):
        return []

    tool_catalog = get_tool_catalog()

    if llm_engine == "rule_based":
        return _fallback_multi_tasks(query)

    prompt = (
        f"{DECOMPOSER_PROMPT}\n\n"
        f"Available tools:{json.dumps(tool_catalog, ensure_ascii=False, separators=(',', ':'))}\n"
        f"User query:{query}"
    )

    explicit_seller = _extract_explicit_seller(query)

    try:
        if llm_engine == "gemini":
            raw = call_gemini(prompt)
        elif llm_engine == "ollama":
            raw = call_ollama(prompt)
        else:
            return _fallback_multi_tasks(query)

        json_text = extract_first_json_object(raw)
        if not json_text:
            logger.info("Task decomposition returned no JSON, using fallback.")
            return _fallback_multi_tasks(query)

        data = json.loads(json_text)
        tasks = data.get("tasks")

        if not isinstance(tasks, list):
            logger.info("Task decomposition returned invalid task list, using fallback.")
            return _fallback_multi_tasks(query)

        cleaned: List[Dict] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue

            tool = str(task.get("tool") or "").strip()
            action_input = task.get("input") or {}

            if tool not in tool_catalog or not isinstance(action_input, dict):
                continue

            normalized = _normalize_task(tool, action_input, query=query, explicit_seller=explicit_seller)
            if normalized:
                cleaned.append(normalized)

        return cleaned or _fallback_multi_tasks(query)

    except Exception as exc:
        logger.warning("Task decomposition failed: %s", exc)
        return _fallback_multi_tasks(query)
