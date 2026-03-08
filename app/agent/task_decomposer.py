from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

from app.services.parser import call_gemini, call_ollama, extract_first_json_object

logger = logging.getLogger(__name__)

DECOMPOSER_PROMPT = """
You are an e-commerce task planner.

Decompose ONLY multi-step user requests into an ordered tool plan.

Available tools:
- search_pipeline
  input: {"query":"string"}
- seller_pipeline
  input: {"seller_name":"string","page":1,"limit":10}

Rules:
- Produce tasks only when multiple actionable needs exist.
- Typical multi-step cases:
  1) seller feedback + product search
  2) seller analysis + check what seller sells
  3) conversational wrapper + concrete retrieval request
- Keep task order meaningful.
- Do not invent seller names.
- Return only executable tool tasks.
- If the request is effectively single-step, return an empty task list.

Return ONLY JSON:
{
  "tasks":[
    {"tool":"seller_pipeline","input":{"seller_name":"example","page":1,"limit":10}},
    {"tool":"search_pipeline","input":{"query":"example"}}
  ]
}
""".strip()


_MULTI_INTENT_PATTERNS = [
    r"\b(e|ed|and|poi|anche|oltre a|insieme a)\b",
    r"[?,;:].+\b(cerca|trova|analizza|feedback|vende|seller|venditore)\b",
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
]


def _contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def should_decompose_query(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False

    has_seller = _contains_any(q, _SELLER_HINTS)
    has_search = _contains_any(q, _SEARCH_HINTS)

    if has_seller and has_search:
        return True

    if any(re.search(pattern, q, re.IGNORECASE) for pattern in _MULTI_INTENT_PATTERNS):
        return True

    return False


def _fallback_multi_tasks(query: str) -> List[Dict]:
    q = (query or "").strip()
    q_lower = q.lower()

    seller_name = _extract_explicit_seller(q)
    tasks: List[Dict] = []

    has_seller = _contains_any(q_lower, _SELLER_HINTS)
    has_search = _contains_any(q_lower, _SEARCH_HINTS)

    if has_seller and seller_name:
        tasks.append(
            {
                "tool": "seller_pipeline",
                "input": {
                    "seller_name": seller_name,
                    "page": 1,
                    "limit": 10,
                },
            }
        )

    if has_search:
        tasks.append(
            {
                "tool": "search_pipeline",
                "input": {
                    "query": _clean_search_query(q),
                },
            }
        )

    return tasks


def _clean_search_query(query: str) -> str:
    q = (query or "").strip().lower()
    q = re.sub(r"(venditore|seller)\s+[a-zA-Z0-9._-]+", "", q)
    q = re.sub(r"(dammi i feedback|analizza il venditore|analizza|feedback del venditore)", "", q)
    return re.sub(r"\s+", " ", q).strip()


def _extract_explicit_seller(text: str) -> str | None:
    pattern = r"(?:venditore|seller)\s+([A-Za-z0-9._-]{3,})"
    match = re.search(pattern, text or "", re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def decompose_query(query: str, llm_engine: str) -> List[Dict]:
    if not should_decompose_query(query):
        return []

    if llm_engine == "rule_based":
        return _fallback_multi_tasks(query)

    prompt = f"{DECOMPOSER_PROMPT}\n\nUser query:{query}"

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
            if tool not in {"search_pipeline", "seller_pipeline"}:
                continue
            if not isinstance(action_input, dict):
                continue
            cleaned.append({"tool": tool, "input": action_input})

        return cleaned or _fallback_multi_tasks(query)

    except Exception as exc:
        logger.warning("Task decomposition failed: %s", exc)
        return _fallback_multi_tasks(query)
