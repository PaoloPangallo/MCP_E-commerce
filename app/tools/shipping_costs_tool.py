from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from app.services.ebay import get_shipping_costs_async


class ToolContextLike(Protocol):
    db: Any
    user: Optional[object]
    llm_engine: str


def normalize_shipping_costs_arguments(action_input: Dict[str, Any], memory: Any) -> Dict[str, Any]:
    return {
        "item_id": action_input.get("item_id"),
        "country_code": action_input.get("country_code", "IT"),
        "zip_code": action_input.get("zip_code", ""),
    }


async def execute_shipping_costs_tool(action_input: Dict[str, Any], context: ToolContextLike) -> Dict[str, Any]:
    item_id = action_input.get("item_id")
    country = action_input.get("country_code", "IT")
    zip_code = action_input.get("zip_code", "")

    if not item_id:
        return {"status": "error", "message": "item_id missing"}

    data = await get_shipping_costs_async(item_id, country, zip_code)
    if not data:
        return {"status": "error", "item_id": item_id}

    return {"status": "ok", "data": data, "item_id": item_id}
