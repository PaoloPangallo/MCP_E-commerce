from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from app.agent.tool_registry import ToolContext

from app.services.ebay import get_item_details

logger = logging.getLogger(__name__)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_item_details_arguments(action_input: Dict[str, Any]) -> Dict[str, Any]:
    item_id = _clean_text(action_input.get("item_id"))
    if not item_id:
        raise ValueError("item_details_tool richiede un item_id valido.")
    return {"item_id": item_id}


async def execute_item_details_tool(action_input: Dict[str, Any], context: "ToolContext") -> Dict[str, Any]:
    """
    Esegue il tool per scaricare i dettagli completi di un oggetto.
    A differenza degli altri tool, questo esegue una request I/O-bound asincrona/bloccante
    quindi avvolgiamo la chiamata sincrona in un thread executor se necessario, ma dato 
    che services/ebay.py è sincrono con requests.Session, lo eseguiamo in un offtrack async.
    """
    clean = normalize_item_details_arguments(action_input)
    item_id = clean["item_id"]

    try:
        import asyncio
        data = await asyncio.to_thread(get_item_details, item_id)
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
