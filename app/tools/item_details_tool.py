from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from app.agent.tool_registry import ToolContext

from app.services.ebay import get_item_details

logger = logging.getLogger(__name__)


from app.utils.text import clean_text as _clean_text
# Using shared _clean_text from app.utils.text


def normalize_item_details_arguments(action_input: Dict[str, Any], memory: Any) -> Dict[str, Any]:
    item_id = _clean_text(action_input.get("item_id"))
    if not item_id and getattr(memory, "search_payload", None):
        results = memory.search_payload.get("results")
        if results and len(results) > 0:
            item_id = results[0].get("ebay_id", "")
            
    if not item_id:
        raise ValueError("item_details_tool richiede un item_id valido.")
    return {"item_id": item_id}


def execute_item_details_tool(action_input: Dict[str, Any], context: "ToolContext") -> Dict[str, Any]:
    """
    Esegue il tool per scaricare i dettagli completi di un oggetto.
    """
    clean = normalize_item_details_arguments(action_input, getattr(context, "memory", None))
    item_id = clean["item_id"]

    try:
        data = get_item_details(item_id)
    except Exception as exc:
        logger.exception("Errore in execute_item_details_tool per %s: %s", item_id, exc)
        return {"item_id": item_id, "status": "error", "error": str(exc)}

    if not data:
        return {
            "item_id": item_id,
            "status": "not_found",
            "message": "Nessun dettaglio trovato o oggetto non esistente.",
        }

    return {
        "item_id": item_id,
        "status": "ok",
        "data": data,
    }
