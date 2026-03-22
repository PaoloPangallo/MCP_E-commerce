import logging
from typing import Dict, Any, Annotated
from pydantic import Field

from app.mcp.core import mcp, _db_context, _build_context, _tool_error
from app.tools import execute_metadata_tool

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
            result = await execute_metadata_tool(
                {"policy_type": policy_type, "marketplace_id": marketplace_id, "category_id": category_id or None},
                context
            )
            result["_backend"] = "mcp"
            logger.info("MCP TOOL get_marketplace_metadata END")
            return result
    except Exception as exc:
        logger.exception("MCP get_marketplace_metadata failed")
        return _tool_error(policy_type=policy_type, error=str(exc))
