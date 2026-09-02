"""
Unit tests for ReactPlanner deterministic routing.

Tests the heuristic _infer_intent() and _ordered_tools_for_intent() methods
without hitting the LLM or any external services. A lightweight MockMemory
stub is used instead of the full RequestState dataclass.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.agent.planner import ReactPlanner


def _log_routing_result(description: str, **values: object) -> None:
    """Print a compact routing trace when pytest is run with ``-s``."""
    details = " | ".join(f"{key}={value!r}" for key, value in values.items())
    print(f"[agent-routing] {description} | {details}")


# ---------------------------------------------------------------------------
# Minimal memory stub — only the attributes used by _infer_intent /
# _ordered_tools_for_intent
# ---------------------------------------------------------------------------

class MockMemory:
    def __init__(self, query: str, mcp_mode: str = "standard", has_explicit_id: bool = False):
        self.user_query = query
        self.mcp_mode = mcp_mode
        self.last_seller_name = None
        self.detected_intent = None
        # Embed an eBay-style ID in the query when needed
        if has_explicit_id:
            self.user_query = query + " 123456789012"


# ---------------------------------------------------------------------------
# _infer_intent tests
# ---------------------------------------------------------------------------

INTENT_CASES = [
    # (query, expected_intent, description)
    ("cerca iPhone 13 ricondizionato", "product_search", "default product search"),
    ("laptop gaming sotto i 1000 euro", "product_search", "product with price — default"),
    ("feedback del venditore pegaso_it", "seller_analysis", "keyword: feedback"),
    ("affidabilità del venditore top_shop", "seller_analysis", "keyword: affidabilità"),
    ("recensioni venditore mediaworld", "seller_analysis", "keyword: recensioni"),
    ("confronta iPhone 14 e Samsung S23", "comparison", "keyword: confronta"),
    ("compara questi due prodotti", "comparison", "keyword: compara"),
    ("qual è la differenza tra i due telefoni", "comparison", "keyword: differenza"),
    ("quanto costa spedire questo articolo", "shipping", "keyword: costa spedire"),
    ("voglio sapere la spedizione", "shipping", "keyword: spedizione"),
    ("contatta il negozio per me", "contact_seller", "keyword: contatta"),
    ("scrivi un messaggio al seller", "contact_seller", "keyword: scrivi"),
    ("andamento prezzi smartphone", "market_trends", "keyword: andamento + prezz"),
    ("statistiche mercato tech", "market_trends", "keyword: statistiche"),
    ("ciao", "conversation", "greeting: ciao"),
    ("hey", "conversation", "greeting: hey"),
    ("hello", "conversation", "greeting: hello"),
    ("salve", "conversation", "greeting: salve"),
    ("", "conversation", "empty query"),
]


@pytest.mark.parametrize("query,expected_intent,description", INTENT_CASES)
def test_infer_intent(query: str, expected_intent: str, description: str):
    planner = ReactPlanner(llm_engine="rule_based")
    memory = MockMemory(query)
    result = planner._infer_intent(memory)
    _log_routing_result(
        description,
        query=query,
        expected_intent=expected_intent,
        detected_intent=result,
    )
    assert result == expected_intent, (
        f"[{description}] query='{query}' → got '{result}', expected '{expected_intent}'"
    )


# ---------------------------------------------------------------------------
# _ordered_tools_for_intent tests — standard MCP mode
# ---------------------------------------------------------------------------

TOOL_ORDER_CASES = [
    # (intent, expected_first_tool, description)
    ("product_search", "search_products", "product search uses search_products"),
    ("seller_analysis", "analyze_seller", "seller analysis uses analyze_seller"),
    ("comparison", "compare_products", "comparison uses compare_products"),
    ("market_trends", "market_trends", "market trends tool"),
    ("deals", "get_ebay_deals", "deals uses get_ebay_deals"),
    ("contact_seller", "contact_seller", "contact seller (no playwright catalog)"),
    ("hybrid", "search_products", "hybrid starts with search_products"),
    ("wishlist", "search_products", "wishlist without explicit ID starts with search"),
    ("shipping", "search_products", "shipping without explicit ID starts with search"),
]


@pytest.mark.parametrize("intent,expected_first_tool,description", TOOL_ORDER_CASES)
def test_ordered_tools_for_intent_standard(intent: str, expected_first_tool: str, description: str):
    planner = ReactPlanner(llm_engine="rule_based")
    memory = MockMemory("test query", mcp_mode="standard")
    tools = planner._ordered_tools_for_intent(intent, memory)
    _log_routing_result(
        description,
        intent=intent,
        expected_first_tool=expected_first_tool,
        routed_tools=tools,
    )
    assert len(tools) > 0, f"[{description}] No tools returned for intent '{intent}'"
    assert tools[0] == expected_first_tool, (
        f"[{description}] intent='{intent}' → first tool='{tools[0]}', expected '{expected_first_tool}'"
    )


def test_item_details_with_explicit_id():
    """When query contains an eBay item ID, item_details should use get_item_details directly."""
    planner = ReactPlanner(llm_engine="rule_based")
    memory = MockMemory("dettagli articolo", has_explicit_id=True)
    tools = planner._ordered_tools_for_intent("item_details", memory)
    _log_routing_result("item details with explicit ID", routed_tools=tools)
    assert tools == ["get_item_details"], f"Expected ['get_item_details'], got {tools}"


def test_item_details_without_explicit_id():
    """Without an item ID, item_details should start with search to find the item first."""
    planner = ReactPlanner(llm_engine="rule_based")
    memory = MockMemory("voglio dettagli su questo prodotto")
    tools = planner._ordered_tools_for_intent("item_details", memory)
    _log_routing_result("item details without explicit ID", routed_tools=tools)
    assert tools[0] == "search_products"
    assert "get_item_details" in tools


def test_wishlist_with_explicit_id():
    """With an explicit eBay ID, wishlist skips the search step."""
    planner = ReactPlanner(llm_engine="rule_based")
    memory = MockMemory("aggiungi alla wishlist", has_explicit_id=True)
    tools = planner._ordered_tools_for_intent("wishlist", memory)
    _log_routing_result("wishlist with explicit ID", routed_tools=tools)
    assert tools == ["manage_wishlist"]


def test_playwright_mode_uses_browser_tools():
    """In playwright_browser mode, product_search routes to browser tools."""
    planner = ReactPlanner(llm_engine="rule_based")
    memory = MockMemory("cerca iPhone", mcp_mode="playwright_browser")
    tools = planner._ordered_tools_for_intent("product_search", memory)
    _log_routing_result("playwright product search", routed_tools=tools)
    assert tools[0] == "browser_navigate"


def test_tool_budget_exceeded_skips_tool():
    """_exceeds_tool_budget returns True when tool call count >= max_calls_per_tool."""
    planner = ReactPlanner(llm_engine="rule_based")
    planner.max_calls_per_tool = 2

    class MemoryWithCounts(MockMemory):
        def tool_call_count(self, tool_name: str) -> int:
            return 2  # already at budget

    memory = MemoryWithCounts("cerca iPhone")
    result = planner._exceeds_tool_budget(memory, "search_products")
    _log_routing_result("tool budget exceeded", tool="search_products", exceeded=result)
    assert result is True


def test_tool_budget_not_exceeded():
    """_exceeds_tool_budget returns False when tool has not been called yet."""
    planner = ReactPlanner(llm_engine="rule_based")

    class MemoryWithCounts(MockMemory):
        def tool_call_count(self, tool_name: str) -> int:
            return 0

    memory = MemoryWithCounts("cerca iPhone")
    result = planner._exceeds_tool_budget(memory, "search_products")
    _log_routing_result("tool budget available", tool="search_products", exceeded=result)
    assert result is False


# ---------------------------------------------------------------------------
# Intent completeness: every VALID_INTENT has a tool mapping
# ---------------------------------------------------------------------------

def test_all_valid_intents_have_tool_mapping():
    """Every intent defined in VALID_INTENTS (except conversation) should return
    at least one tool from _ordered_tools_for_intent."""
    from app.agent.planner import VALID_INTENTS
    planner = ReactPlanner(llm_engine="rule_based")
    memory = MockMemory("test query")

    skip = {"conversation", "playwright_search"}  # conversation has no tool; playwright_search needs playwright mode
    for intent in VALID_INTENTS - skip:
        tools = planner._ordered_tools_for_intent(intent, memory)
        _log_routing_result("valid intent tool mapping", intent=intent, routed_tools=tools)
        assert len(tools) > 0, f"Intent '{intent}' returned no tools"
