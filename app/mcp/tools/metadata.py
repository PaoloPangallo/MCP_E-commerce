import logging
from typing import Dict, Any, Annotated
from pydantic import Field

from app.mcp.core import mcp, _db_context, _build_context, _tool_error
from app.services.ebay_metadata import get_marketplace_metadata as service_get_marketplace_metadata

logger = logging.getLogger(__name__)

@mcp.tool(
    name="get_marketplace_metadata",
    description="Recupera i metadata tecnici e le policy eBay (condizioni usato, regole restituzione). NON usare per cercare prodotti o offerte."
)
async def get_marketplace_metadata(
    policy_type: Annotated[str, Field(description="Tipo query metadata ('item_conditions', 'return_policies', 'listing_structure')")] = "item_conditions",
    marketplace_id: Annotated[str, Field(description="ID del marketplace (es. 'EBAY_IT')")] = "",
    category_id: Annotated[str, Field(description="ID della categoria per restrizioni puntuali")] = "",
    session_id: Annotated[str, Field(description="ID di sessione utente")] = ""
) -> Dict[str, Any]:
    try:
        with _db_context() as db:
            context = _build_context(db=db, session_id=session_id)
            logger.info("MCP TOOL get_marketplace_metadata START | policy_type=%s", policy_type)
            raw_result = await service_get_marketplace_metadata(
                policy_type=policy_type,
                marketplace_id=marketplace_id,
                category_id=category_id or None
            )

            has_error = "error" in raw_result and not any(
                k for k in raw_result if k not in ("error", "detail", "policy_type", "marketplace_id", "category_id")
            )

            message = ""
            if not has_error:
                for list_key in ("itemConditionPolicies", "returnPolicies", "listingStructurePolicies"):
                    if list_key in raw_result and isinstance(raw_result[list_key], list):
                        total = len(raw_result[list_key])
                        if total > 10:
                            raw_result[list_key] = raw_result[list_key][:10]
                            message = f"Warning: data truncated (showing 10 of {total} items). Specifying a category_id is recommended for full results."

            result = {
                "status": "error" if has_error else "ok",
                "message": message,
                "policy_type": policy_type,
                "marketplace_id": raw_result.get("marketplace_id", ""),
                "category_id": category_id,
                "results": raw_result,
                "results_count": len(raw_result) if not has_error else 0,
            }
            result["_backend"] = "mcp"
            
            status = result.get("status")
            if status != "ok":
                result["summary"] = f"Errore nel recupero metadata: {result.get('error', 'sconosciuto')}"
            else:
                pt = result.get("policy_type", "metadata")
                result["summary"] = f"Recuperati con successo i metadata eBay ({pt})."
                
            logger.info("MCP TOOL get_marketplace_metadata END")
            return result
    except Exception as exc:
        logger.exception("MCP get_marketplace_metadata failed")
        return _tool_error(policy_type=policy_type, error=str(exc))
