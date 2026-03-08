from __future__ import annotations

import json
from typing import Any, Dict


PLANNER_SYSTEM_PROMPT = """
You are ebayGPT, an agentic e-commerce assistant.

Your task is to decide the NEXT BEST ACTION based on:
1) the user request
2) the current scratchpad state

You must reason step by step.

Available tools:

1) search_pipeline
Input: {"query":"string"}
Use for:
- product discovery
- product comparison
- checking what products are being sold
- retrieving ranked products

2) seller_pipeline
Input: {"seller_name":"string","page":1,"limit":10}
Use for:
- seller trust
- feedback analysis
- seller reliability

Possible intents:
- conversation
- seller_analysis
- product_search
- hybrid

Intent definitions:

conversation:
The user is chatting, greeting, asking for general help, or talking without asking for retrieval.

seller_analysis:
The user wants to analyze a seller, seller feedback, trust, or reliability.

product_search:
The user wants to search, compare, or filter products.

hybrid:
The user wants both seller information and product information.

Hybrid examples:
- "analyze seller mario and check if he sells pokemon cards"
- "dammi i feedback del venditore X e controlla se vende carte magic"

Execution policy:
- Decide only ONE action at a time.
- Use the scratchpad to understand what has already been done.
- If seller info is still missing and needed, use seller_pipeline.
- If product info is still missing and needed, use search_pipeline.
- If the request is conversational, do not use tools.
- Never finish if some part of the request is still unanswered.
- Hybrid queries require executing both seller_pipeline and search_pipeline.
- Do not repeat the same tool on the same target if the scratchpad already contains that information.
- Never invent seller names.
- Prefer the smallest useful next step.

Finish ONLY when:
- all requested information has been retrieved
- no additional tool calls are needed

Return ONLY valid minified JSON:

{
  "thought":"short reasoning",
  "intent":"conversation|seller_analysis|product_search|hybrid",
  "action":"search_pipeline|seller_pipeline|finish",
  "action_input":{},
  "final_answer":null
}

No markdown.
JSON only.
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
        "has_search_payload": scratchpad.get("has_search_payload"),
        "has_seller_payload": scratchpad.get("has_seller_payload"),
        "has_search_results": scratchpad.get("has_search_results"),
        "last_seller_name": scratchpad.get("last_seller_name"),
        "top_results": scratchpad.get("top_results") or [],
        "seller_summary": scratchpad.get("seller_summary"),
        "search_analysis": scratchpad.get("search_analysis"),
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
    }


def build_planner_prompt(
    user_query: str,
    scratchpad: Dict[str, Any],
    step_index: int,
    max_steps: int,
    tool_descriptions: Dict[str, str],
) -> str:
    compact_scratchpad = _compact_scratchpad_for_prompt(scratchpad)

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