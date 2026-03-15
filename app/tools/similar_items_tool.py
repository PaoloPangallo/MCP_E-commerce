import logging
from typing import Any, Dict, List

from app.services.ebay import get_similar_items

logger = logging.getLogger(__name__)

async def execute_similar_items_tool(arguments: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Executor for the similar_items tool.
    Arguments:
        item_id (str): The eBay item ID to find similar items for.
    """
    item_id = arguments.get("item_id")
    if not item_id:
        return {
            "status": "error",
            "message": "item_id is required"
        }

    try:
        logger.info("Executing similar_items_tool for item_id: %s", item_id)
        items = await get_similar_items(item_id)
        
        return {
            "status": "ok",
            "item_id": item_id,
            "results_count": len(items),
            "results": items
        }
    except Exception as e:
        logger.exception("Error in execute_similar_items_tool")
        return {
            "status": "error",
            "message": str(e)
        }
