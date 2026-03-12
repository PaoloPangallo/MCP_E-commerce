from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Coroutine

from sqlalchemy.orm import Session

from app.agent.schemas import LatencyClass, ObservationQuality, ObservationStatus
from app.tools import (
    execute_compare_tool,
    execute_conversation_tool,
    execute_search_tool,
    execute_seller_tool,
    execute_item_details_tool,
    execute_shipping_costs_tool,
)
from app.tools.search_tool import clean_search_query, normalize_search_arguments
from app.tools.seller_tool import extract_explicit_seller, normalize_seller_arguments
from app.tools.item_details_tool import normalize_item_details_arguments
from app.tools.shipping_costs_tool import normalize_shipping_costs_arguments

InputNormalizer = Callable[[Dict[str, Any], Any], Dict[str, Any]]

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
    executor: Callable[[Dict[str, Any], ToolContext], Coroutine[Any, Any, Dict[str, Any]]]

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

def get_tool_catalog() -> Dict[str, Dict[str, Any]]:
    catalog = {}
    for name, spec in TOOLS.items():
        catalog[name] = {
            "description": spec.description,
            "input_schema": spec.input_schema,
            "tags": list(spec.tags),
        }
    return catalog

def analyze_user_query(query: str) -> Dict[str, Any]:
    # Placeholder for query analysis logic
    return {"text": query}

def _bootstrap_tools() -> None:
    TOOLS.clear()
    
    register_tool(ToolSpec(
        name="search_products",
        description="Search products on eBay.",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        executor=execute_search_tool,
        tags=("search", "ebay"),
    ))

    register_tool(ToolSpec(
        name="analyze_seller",
        description="Analyze seller feedback.",
        input_schema={"type": "object", "properties": {"seller_name": {"type": "string"}}, "required": ["seller_name"]},
        executor=execute_seller_tool,
        tags=("seller", "ebay"),
    ))

    register_tool(ToolSpec(
        name="conversation",
        description="General chat.",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        executor=execute_conversation_tool,
        tags=("conversation",),
    ))

    register_tool(ToolSpec(
        name="compare_products",
        description="Compare products.",
        input_schema={"type": "object", "properties": {"queries": {"type": "string"}}, "required": ["queries"]},
        executor=execute_compare_tool,
        tags=("compare",),
    ))

    register_tool(ToolSpec(
        name="get_item_details",
        description="Get item details.",
        input_schema={"type": "object", "properties": {"item_id": {"type": "string"}}, "required": ["item_id"]},
        executor=execute_item_details_tool,
        tags=("details",),
    ))

    register_tool(ToolSpec(
        name="get_shipping_costs",
        description="Get shipping costs.",
        input_schema={"type": "object", "properties": {"item_id": {"type": "string"}}, "required": ["item_id"]},
        executor=execute_shipping_costs_tool,
        tags=("shipping",),
    ))

_bootstrap_tools()
