import logging
from typing import Dict, Any, Annotated
from pydantic import Field

from app.mcp.core import mcp, _db_context, _build_context, _tool_error
from app.mcp.normalizers import _normalize_seller_output
from app.services.seller_pipeline import run_seller_pipeline

logger = logging.getLogger(__name__)

@mcp.tool(
    name="analyze_seller",
    description="Analizza la reputazione di un venditore e-commerce estraendo feedback pertinenti, trust score e sentiment ratio.",
)
async def analyze_seller(
    seller_name: Annotated[str, Field(description="Nome o username del venditore su eBay (es. 'shopping_ita')")],
    page: Annotated[int, Field(description="Numero di pagina per la paginazione dei feedback")] = 1,
    limit: Annotated[int, Field(description="Numero di feedback da analizzare (default 100, max 300) per sentiment e trust")] = 100,
    session_id: Annotated[str, Field(description="ID di sessione")] = ""
) -> Dict[str, Any]:
    try:
        with _db_context() as db:
            raw = await run_seller_pipeline(seller_name=seller_name, page=page, limit=limit)
            normalized = _normalize_seller_output(raw)
            normalized["_backend"] = "mcp"
            return normalized
    except Exception as exc:
        logger.exception("MCP analyze_seller failed")
        return _tool_error(seller_name=seller_name, error=str(exc))
