import logging
from typing import Dict, Any, Annotated
from pydantic import Field

from app.mcp.core import mcp, _db_context, _build_context, _tool_error
from app.mcp.normalizers import (
    _normalize_item_details_output,
    _normalize_similar_items_output,
    _normalize_shipping_costs_output,
)
from app.services.ebay import (
    get_item_details as _svc_get_item_details,
    get_similar_items as _svc_get_similar_items,
    get_shipping_costs as _svc_get_shipping_costs,
)

logger = logging.getLogger(__name__)

@mcp.tool(
    name="get_item_details",
    description="Recupera i dettagli estesi, descrizione e specifiche tecniche di un prodotto conoscendo il suo ID eBay (item_id).",
)
async def get_item_details(
    item_id: Annotated[str, Field(description="ID univoco dell'oggetto eBay (es. '112345678901')")],
    user_query: Annotated[str, Field(description="Query originale dell'utente, usata per filtrare i prodotti simili")] = "",
    session_id: Annotated[str, Field(description="ID di sessione utente")] = ""
) -> Dict[str, Any]:
    try:
        with _db_context() as db:
            context = _build_context(db=db, session_id=session_id)
            logger.info("MCP TOOL get_item_details START | item_id=%s", item_id)
            result_data = await _svc_get_item_details(item_id, user_query=user_query)
            result = {
                "item_id": item_id,
                "status": "ok" if result_data else "not_found",
                "data": result_data,
                "message": "" if result_data else "Nessun dettaglio trovato o oggetto non esistente."
            }
            normalized = _normalize_item_details_output(result)
            normalized["_backend"] = "mcp"
            logger.info("MCP TOOL get_item_details END")
            return normalized
    except Exception as exc:
        logger.exception("MCP get_item_details failed")
        return _tool_error(item_id=item_id, error=str(exc))


@mcp.tool(
    name="get_shipping_costs",
    description="Recupera i costi e le opzioni di spedizione precisi per un determinato oggetto verso un CAP e Paese specifici.",
)
async def get_shipping_costs(
    item_id: Annotated[str, Field(description="ID univoco dell'oggetto eBay")],
    country_code: Annotated[str, Field(description="Codice ISO nazione destinazione (default: IT)")] = "IT",
    zip_code: Annotated[str, Field(description="CAP destinazione per calcolo accurato")] = "",
    session_id: Annotated[str, Field(description="ID di sessione utente")] = ""
) -> Dict[str, Any]:
    try:
        with _db_context() as db:
            context = _build_context(db=db, session_id=session_id)
            logger.info("MCP TOOL get_shipping_costs START | item_id=%s", item_id)
            
            data = await _svc_get_shipping_costs(item_id, country_code, zip_code)
            result = {
                "status": "ok" if data else "error",
                "error": "" if data else f"Nessun dato di spedizione trovato per {item_id}",
                "data": data,
                "item_id": item_id
            }
            normalized = _normalize_shipping_costs_output(result)
            normalized["_backend"] = "mcp"
            logger.info("MCP TOOL get_shipping_costs END")
            return normalized
    except Exception as exc:
        logger.exception("MCP get_shipping_costs failed")
        return _tool_error(item_id=item_id, error=str(exc))


@mcp.tool(
    name="get_similar_items",
    description="Recupera prodotti simili o correlati per un oggetto a partire dal suo ID (item_id)."
)
async def get_similar_items(
    item_id: Annotated[str, Field(description="ID univoco dell'oggetto eBay")],
    session_id: Annotated[str, Field(description="ID di sessione utente")] = ""
) -> Dict[str, Any]:
    try:
        with _db_context() as db:
            context = _build_context(db=db, session_id=session_id)
            logger.info("MCP TOOL get_similar_items START | item_id=%s", item_id)
            items = await _svc_get_similar_items(item_id)
            result = {
                "status": "ok",
                "item_id": item_id,
                "results_count": len(items) if items else 0,
                "results": items or []
            }
            normalized = _normalize_similar_items_output(result)
            normalized["_backend"] = "mcp"
            logger.info("MCP TOOL get_similar_items END")
            return normalized
    except Exception as exc:
        logger.exception("MCP get_similar_items failed")
        return _tool_error(item_id=item_id, error=str(exc))
