from __future__ import annotations

import json
from typing import Any, Dict, List

# Approx 4 characters per token as a rough safety heuristic
ROUGH_CHARS_PER_TOKEN = 4
MAX_LLM_PROMPT_TOKENS = 6000


def _estimate_tokens(text: str) -> int:
    return len(str(text)) // ROUGH_CHARS_PER_TOKEN


def _truncate_scratchpad(scratchpad: List[Dict[str, Any]], context_budget_chars: int) -> str:
    """Formats the scratchpad as JSON but aggressively truncates older items if it exceeds the budget."""
    if not scratchpad:
        return "Nessuna azione precedente."

    formatted_items = []
    # Add items starting from most recent backwards
    for item in reversed(scratchpad):
        item_str = json.dumps(item, ensure_ascii=False, indent=2)
        formatted_items.insert(0, item_str)

    full_str = json.dumps([json.loads(i) for i in formatted_items], ensure_ascii=False, indent=2)

    # If it fits the budget, return it all
    if len(full_str) <= context_budget_chars:
        return full_str

    # Otherwise, start dropping the oldest items
    while len(formatted_items) > 1 and len(json.dumps([json.loads(i) for i in formatted_items])) > context_budget_chars:
        formatted_items.pop(0)  # Remove oldest

    leftover = json.dumps([json.loads(i) for i in formatted_items], ensure_ascii=False, indent=2)
    return f"[... {len(scratchpad) - len(formatted_items)} older items omitted ...]\n" + leftover


PLANNER_SYSTEM_PROMPT = """
You are ebayGPT, an e-commerce agent that plans the NEXT BEST ACTION.

You receive:
- the user query
- the current scratchpad/state
- the catalog of available tools

Your job:
- decide only the next best action
- use only tools from the catalog
- prefer deterministic behavior
- use LLM reasoning only when the next step is genuinely ambiguous
- do not invent tool names or parameters
- if the request is already satisfied, finish

Important policy:
- `search_products` is for product discovery and shopping queries
- `analyze_seller` is for seller reliability, feedback, trust and reputation
- `get_item_details` is ONLY to fetch specific technical details or lengthy descriptions of an already identified `item_id`
- `get_shipping_costs` is ONLY to compute exact shipping costs for a specific CAP/country of an `item_id`
- `conversation` is for purely conversational requests with no e-commerce tool need
- for hybrid queries, prefer the unmet need first
- do not repeat a tool call when its state is already terminal and useful
- keep ALL free text and thoughts ONLY in Italian
- return ONLY valid minified JSON

Schema:
{
  "thought":"Strategia attuale e perché questo step è utile (in ITALIANO)",
  "intent":"conversation|seller_analysis|product_search|hybrid|comparison|item_details|shipping",
  "action":"tool_name|finish",
  "action_input":{},
  "final_answer":null
}
""".strip()


FINAL_ANSWER_SYSTEM_PROMPT = """
You are ebayGPT.

Write the final answer in Italian.
Use only the provided data.
Do not invent prices, sellers, trust scores, metrics, or results.
Integrate available tool outputs into one coherent answer.
If the structured data is enough, be concise and direct.
CRITICAL: For simple greetings (e.g., "ciao", "hey") or purely conversational turns, respond directly and briefly.
If there is no useful result, say it clearly.
No markdown.
No bullet points.
""".strip()


def _compact_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def _compact_tool_catalog_for_prompt(tool_catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    compact: Dict[str, Dict[str, Any]] = {}
    for name, spec in (tool_catalog or {}).items():
        if not isinstance(spec, dict):
            continue

        compact[name] = {
            "description": spec.get("description"),
            "input_schema": spec.get("input_schema"),
            "required_fields": spec.get("required_fields") or [],
            "tags": spec.get("tags") or [],
            "examples": (spec.get("examples") or [])[:2],
            "state_key": spec.get("state_key"),
            "cost": spec.get("cost"),
            "latency_class": spec.get("latency_class"),
            "dependencies": spec.get("dependencies") or [],
            "produced_entities": spec.get("produced_entities") or [],
        }
    return compact


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
        "metrics": scratchpad.get("metrics"),
        "tool_calls": scratchpad.get("tool_calls") or {},
        "llm_calls": scratchpad.get("llm_calls") or {},
        "tool_states": scratchpad.get("tool_states") or {},
        "recent_observations": scratchpad.get("recent_observations") or [],
        "recent_errors": scratchpad.get("recent_errors") or [],
        "session_memory": scratchpad.get("session_memory") or {},
        "long_term_memory": scratchpad.get("long_term_memory") or {},
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
        "compare": final_data.get("compare"),
        "top_result": final_data.get("top_result"),
        "last_seller_name": final_data.get("last_seller_name"),
        "search_analysis": final_data.get("search_analysis"),
        "metrics": final_data.get("metrics"),
        "errors": final_data.get("errors") or [],
        "pending_tasks": final_data.get("pending_tasks") or [],
        "tool_states": final_data.get("tool_states") or {},
        "recent_observations": final_data.get("recent_observations") or [],
        "session_memory": final_data.get("session_memory") or {},
        "long_term_memory": final_data.get("long_term_memory") or {},
    }


def build_planner_prompt(
    user_query: str,
    scratchpad: List[Dict[str, Any]], # Changed type hint to List
    step_index: int,
    max_steps: int,
    tool_catalog: Dict[str, Dict[str, Any]],
    custom_instructions: Optional[str] = None
) -> str:
    compact_tool_catalog = _compact_tool_catalog_for_prompt(tool_catalog)
    tools_json = json.dumps(compact_tool_catalog, ensure_ascii=False, indent=2)

    system_prompt = PLANNER_SYSTEM_PROMPT
    if custom_instructions:
        system_prompt += f"\n\nUSER CUSTOM INSTRUCTIONS:\n{custom_instructions}"

    # Estimate token usage to safeguard context window
    current_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(tools_json) + _estimate_tokens(user_query)
    budget_chars = max(1000, (MAX_LLM_PROMPT_TOKENS - current_tokens) * ROUGH_CHARS_PER_TOKEN)

    scratchpad_str = _truncate_scratchpad(scratchpad, budget_chars)

    return (
        f"{system_prompt}\n\n"
        f"Step:{step_index}/{max_steps}\n"
        f"Available tools:{tools_json}\n"
        f"User query:{user_query}\n"
        f"Scratchpad:{scratchpad_str}\n"
        f"Return JSON only."
    )


def build_final_answer_prompt(
    user_query: str,
    scratchpad: List[Dict[str, Any]],
    final_data: Dict[str, Any],
    custom_instructions: Optional[str] = None
) -> str:
    compact_final_data = _compact_final_data_for_prompt(final_data)
    context_info = _compact_json(compact_final_data)

    system_prompt = FINAL_ANSWER_SYSTEM_PROMPT
    if custom_instructions:
        system_prompt += f"\n\nUSER CUSTOM INSTRUCTIONS (Apply these specifically to your tone/style/content):\n{custom_instructions}"
        
    pref_str = ""
    # Make sure we don't blow up context with massive scratchpad
    base_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_query) + _estimate_tokens(context_info)
    budget_chars = max(1000, (MAX_LLM_PROMPT_TOKENS - base_tokens) * ROUGH_CHARS_PER_TOKEN)
    
    scratch_str = _truncate_scratchpad(scratchpad, budget_chars)

    return f"""{system_prompt}

CONVERSATION CONTEXT:
{context_info}

USER STATUS/PREFERENCES:
{pref_str}

OBSERVATIONS HISTORY:
{scratch_str}

USER REQUEST:
{user_query}

Write the final answer now.
"""