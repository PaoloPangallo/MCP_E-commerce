from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from app.services.ebay import get_item_details_async


class ToolContextLike(Protocol):
    db: Any
    user: Optional[object]
    llm_engine: str


def normalize_item_details_arguments(action_input: Dict[str, Any], memory: Any) -> Dict[str, Any]:
    item_id = action_input.get("item_id")
    if not item_id:
        # try to get from memory if available
        pass
    return {"item_id": item_id}


async def execute_item_details_tool(action_input: Dict[str, Any], context: ToolContextLike) -> Dict[str, Any]:
    item_id = action_input.get("item_id")
    if not item_id:
        return {"status": "error", "message": "item_id missing"}

    data = await get_item_details_async(item_id)
    if not data:
        return {"status": "not_found", "item_id": item_id}

    return {"status": "ok", "data": data}
