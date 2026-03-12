import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from app.agent.schemas import ObservationStatus
if TYPE_CHECKING:
    from app.agent.tool_registry import ToolContext
from app.services.ebay import get_shipping_costs

logger = logging.getLogger(__name__)


def normalize_shipping_costs_arguments(args: Dict[str, Any], memory: Any) -> Dict[str, Any]:
    """
    Normalizes inputs for the get_shipping_costs tool.
    Extracts item_id either explicitly or from the user query's implicit context.
    Default country to IT if not specified.
    """
    item_id = str(args.get("item_id", "")).strip()

    if not item_id and getattr(memory, "search_payload", None):
        results = memory.search_payload.get("results")
        if results and len(results) > 0:
            item_id = results[0].get("ebay_id", "")

    country_code = str(args.get("country_code", "IT")).strip().upper()
    zip_code = str(args.get("zip_code", "")).strip()

    return {
        "item_id": item_id,
        "country_code": country_code,
        "zip_code": zip_code,
    }


def execute_shipping_costs_tool(action_input: Dict[str, Any], context: "ToolContext") -> Dict[str, Any]:
    """
    Esegue la ricerca dei costi di spedizione per un oggetto eBay e una posizione specifica.
    """
    item_id = action_input.get("item_id")
    country_code = action_input.get("country_code", "IT")
    zip_code = action_input.get("zip_code", "")

    if not item_id:
        return {
            "status": "error",
            "error": "Manca l'ID dell'oggetto per calcolare la spedizione.",
            "data": None,
        }

    try:
        # get_shipping_costs() is synchronous (and slow), but executor uses asyncio.to_thread
        data = get_shipping_costs(item_id, country_code, zip_code)
        
        if not data:
            return {
                "status": "error",
                "error": f"Nessun dato di spedizione trovato per l'oggetto {item_id}.",
                "data": None,
            }

        return {
            "status": "ok",
            "data": data,
        }
    except Exception as exc:
        logger.exception("Error in execute_shipping_costs_tool: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "data": None,
        }
