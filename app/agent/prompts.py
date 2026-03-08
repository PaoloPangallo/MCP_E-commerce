from __future__ import annotations

import json
from typing import Any, Dict


PLANNER_SYSTEM_PROMPT = """
You are ebayGPT, an agentic e-commerce assistant.

Decide only the NEXT BEST ACTION.
Be conservative, short, and deterministic.

Available intents:
- conversation
- seller_analysis
- product_search
- hybrid

Rules:
- If pending_tasks is not empty, prioritize completing those tasks before finishing.
- Use seller_pipeline only for seller trust, feedback, reliability, or an explicitly known seller.
- Use search_pipeline for product discovery, checking what a seller sells, and product comparison.
- Do not repeat a tool if the scratchpad already has terminal information for that need.
- Never invent seller names.
- Never finish while a required part of the request is still unanswered.
- Return ONLY valid minified JSON.

Few-shot examples:

User: "ciao come va?"
Output:
{"thought":"Richiesta conversazionale.","intent":"conversation","action":"finish","action_input":{},"final_answer":null}

User: "analizza il venditore mario_store"
Output:
{"thought":"Serve analisi seller.","intent":"seller_analysis","action":"seller_pipeline","action_input":{"seller_name":"mario_store","page":1,"limit":10},"final_answer":null}

User: "cerca una maglia inter"
Output:
{"thought":"Serve ricerca prodotto.","intent":"product_search","action":"search_pipeline","action_input":{"query":"maglia inter"},"final_answer":null}

User: "dammi i feedback del venditore mario_store e controlla se vende carte pokemon"
Output:
{"thought":"Richiesta ibrida, prima seller.","intent":"hybrid","action":"seller_pipeline","action_input":{"seller_name":"mario_store","page":1,"limit":10},"final_answer":null}

Schema:
{
  "thought":"short reasoning",
  "intent":"conversation|seller_analysis|product_search|hybrid",
  "action":"search_pipeline|seller_pipeline|finish",
  "action_input":{},
  "final_answer":null
}
""".strip()


FINAL_ANSWER_SYSTEM_PROMPT = """
You are ebayGPT.

Write the final answer in Italian.
Use only the provided data.
Do not invent prices, sellers, trust scores, metrics or results.
If the request was conversational, answer naturally and clearly.
If no useful result exists, say it clearly.
If both seller data and product data are available, combine them in one coherent answer.
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
    return {
        "steps_done": scratchpad.get("steps_done"),
        "intent": scratchpad.get("intent"),
        "pending_tasks": scratchpad.get("pending_tasks") or [],
        "search_status": scratchpad.get("search_status"),
        "seller_status": scratchpad.get("seller_status"),
        "has_search_payload": scratchpad.get("has_search_payload"),
        "has_seller_payload": scratchpad.get("has_seller_payload"),
        "has_search_results": scratchpad.get("has_search_results"),
        "last_seller_name": scratchpad.get("last_seller_name"),
        "top_results": scratchpad.get("top_results") or [],
        "seller_summary": scratchpad.get("seller_summary"),
        "search_analysis": scratchpad.get("search_analysis"),
        "tool_calls": scratchpad.get("tool_calls") or {},
        "recent_errors": scratchpad.get("recent_errors") or [],
    }


def _compact_final_data_for_prompt(final_data: Dict[str, Any]) -> Dict[str, Any]:
    search = final_data.get("search") or {}
    seller = final_data.get("seller") or {}

    compact_search = {
        "results_count": search.get("results_count"),
        "analysis": search.get("analysis"),
        "metrics": search.get("metrics") or search.get("ir_metrics"),
        "top_results": (search.get("results") or [])[:3],
    } if search else None

    compact_seller = {
        "seller_name": seller.get("seller_name"),
        "count": seller.get("count"),
        "trust_score": seller.get("trust_score"),
        "sentiment_score": seller.get("sentiment_score"),
        "error": seller.get("error"),
    } if seller else None

    return {
        "intent": final_data.get("intent"),
        "search": compact_search,
        "seller": compact_seller,
        "top_result": final_data.get("top_result"),
        "last_seller_name": final_data.get("last_seller_name"),
        "search_analysis": final_data.get("search_analysis"),
        "metrics": final_data.get("metrics"),
        "errors": final_data.get("errors") or [],
        "pending_tasks": final_data.get("pending_tasks") or [],
    }


def build_planner_prompt(
    user_query: str,
    scratchpad: Dict[str, Any],
    step_index: int,
    max_steps: int,
    tool_descriptions: Dict[str, str],
) -> str:
    compact_scratchpad = _compact_scratchpad_for_prompt(scratchpad)

    return (
        f"{PLANNER_SYSTEM_PROMPT}\n\n"
        f"Step:{step_index}/{max_steps}\n"
        f"Tools:{_compact_json(tool_descriptions or {})}\n"
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
