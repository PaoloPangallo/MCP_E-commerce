from __future__ import annotations

import json
from typing import Any, Dict


PLANNER_SYSTEM_PROMPT = """
You are ebayGPT, a small ReAct e-commerce agent.

Available tools:

1) search_pipeline
   Input: {"query":"string"}
   Use for product search, ranking, comparison, budget filtering and convenience analysis.

2) seller_pipeline
   Input: {"seller_name":"string","page":1,"limit":10}
   Use only when a seller name is explicitly known from the user query or from previous results.

Return ONLY valid minified JSON with this schema:
{"thought":"short","action":"search_pipeline|seller_pipeline|finish","action_input":{},"final_answer":null}

Rules:
- Keep "thought" under 12 words.
- Prefer search_pipeline first for product-related requests.
- Use seller_pipeline only if seller_name is already known.
- Never invent seller names.
- Do not repeat a tool if the scratchpad already contains enough information.
- If enough information is available, return action="finish".
- No markdown.
- JSON only.
""".strip()


FINAL_ANSWER_SYSTEM_PROMPT = """
You are ebayGPT.

Write the final answer in Italian.
Use only the provided data.
Do not invent prices, sellers, trust scores, metrics or results.
Be concise, natural and clear.
If no useful result exists, say it clearly.
No markdown.
No bullet points.
""".strip()


def _compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _compact_scratchpad_for_prompt(scratchpad: Dict[str, Any]) -> Dict[str, Any]:
    """
    Riduce il rumore del prompt mantenendo solo i campi più utili.
    """
    return {
        "steps_done": scratchpad.get("steps_done"),
        "has_search_payload": scratchpad.get("has_search_payload"),
        "has_seller_payload": scratchpad.get("has_seller_payload"),
        "has_search_results": scratchpad.get("has_search_results"),
        "last_seller_name": scratchpad.get("last_seller_name"),
        "top_results": scratchpad.get("top_results") or [],
        "seller_summary": scratchpad.get("seller_summary"),
        "recent_errors": scratchpad.get("recent_errors") or [],
    }


def _compact_final_data_for_prompt(final_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evita di passare payload enormi al final LLM.
    """
    search = final_data.get("search") or {}
    seller = final_data.get("seller") or {}

    compact_search = {
        "results_count": search.get("results_count"),
        "analysis": search.get("analysis"),
        "metrics": search.get("metrics"),
        "top_results": (search.get("results") or [])[:3],
    } if search else None

    compact_seller = {
        "seller_name": seller.get("seller_name"),
        "count": seller.get("count"),
        "trust_score": seller.get("trust_score"),
        "sentiment_score": seller.get("sentiment_score"),
    } if seller else None

    return {
        "search": compact_search,
        "seller": compact_seller,
        "top_result": final_data.get("top_result"),
        "last_seller_name": final_data.get("last_seller_name"),
        "search_analysis": final_data.get("search_analysis"),
        "metrics": final_data.get("metrics"),
        "errors": final_data.get("errors") or [],
    }


def build_planner_prompt(
    user_query: str,
    scratchpad: Dict[str, Any],
    step_index: int,
    max_steps: int,
    tool_descriptions: Dict[str, str],
) -> str:
    compact_scratchpad = _compact_scratchpad_for_prompt(scratchpad)

    # tool_descriptions resta supportato, ma lo compattiamo
    compact_tools = {
        name: desc
        for name, desc in (tool_descriptions or {}).items()
        if name in {"search_pipeline", "seller_pipeline"}
    }

    return (
        f"{PLANNER_SYSTEM_PROMPT}\n\n"
        f"Step:{step_index}/{max_steps}\n"
        f"Tools:{_compact_json(compact_tools)}\n"
        f"User query:{user_query}\n"
        f"Scratchpad:{_compact_json(compact_scratchpad)}\n"
        f"Return JSON only."
    )


def build_final_answer_prompt(
    user_query: str,
    scratchpad: Dict[str, Any],
    final_data: Dict[str, Any],
) -> str:
    compact_scratchpad = _compact_scratchpad_for_prompt(scratchpad)
    compact_final_data = _compact_final_data_for_prompt(final_data)

    return (
        f"{FINAL_ANSWER_SYSTEM_PROMPT}\n\n"
        f"User query:{user_query}\n"
        f"Scratchpad:{_compact_json(compact_scratchpad)}\n"
        f"Final data:{_compact_json(compact_final_data)}\n"
        f"Write the final answer now."
    )