import logging
from typing import Dict, Any, Annotated, Optional
from pydantic import Field

from app.mcp.core import mcp
from app.services.serp.ebay_deals import fetch_ebay_deals

logger = logging.getLogger(__name__)

@mcp.tool(
    name="get_ebay_deals",
    description=(
        "Recupera le migliori offerte del giorno (Daily Deals) e offerte a tempo limitato (Limited Time Deals) da eBay. "
        "È il tool principale da usare quando l'utente chiede 'offerte', 'deals', 'occasioni' o 'promozioni'. "
        "Supporta il filtraggio per categoria (category_id) o per parola chiave (query). "
        "Se l'utente specifica una categoria con un ID (es. 9355), DEVI passare tale valore a category_id."
    ),
)
async def get_ebay_deals(
    query: Annotated[str, Field(description="Parola chiave per filtrare le offerte (es. 'iphone', 'lego')")] = "",
    category_id: Annotated[str, Field(description="ID categoria eBay (es. '9355' per cellulari)")] = "",
    ebay_domain: Annotated[str, Field(description="Dominio eBay da utilizzare (default: ebay.it)")] = "ebay.it",
    session_id: Annotated[str, Field(description="ID di sessione utente")] = ""
) -> Dict[str, Any]:
    """
    Recupera le offerte (deals) da eBay tramite SerpApi.
    """
    try:
        logger.info(f"MCP TOOL get_ebay_deals START | query='{query}', category='{category_id}', session='{session_id}'")
        
        # Gestiamo le stringhe vuote come None per il servizio sottostante
        q = query if query.strip() else None
        cat = category_id if category_id.strip() else None

        result = await fetch_ebay_deals(
            query=q,
            category_id=cat,
            ebay_domain=ebay_domain
        )
        
        if not result:
            return {
                "status": "error",
                "message": "Nessuna offerta trovata per i criteri specificati o errore SerpApi.",
                "data": None,
                "summary": "Nessuna offerta trovata per i criteri specificati."
            }

        return {
            "status": "ok",
            "title": result.title,
            "subtitle": result.subtitle,
            "deals": [item.dict() for item in result.items],
            "summary": f"Offerte speciali recuperate: {result.title} ({len(result.items)} oggetti).",
            "_backend": "mcp"
        }

    except Exception as e:
        logger.error(f"MCP TOOL get_ebay_deals ERROR: {e}")
        return {
            "status": "error",
            "message": str(e),
            "data": None,
            "summary": f"Errore nel recupero delle offerte: {str(e)}"
        }
