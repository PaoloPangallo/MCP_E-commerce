import logging
from typing import Dict, Any, Annotated, Optional
from pydantic import Field

from app.mcp.core import mcp
from app.services.serp.ebay_deals import fetch_ebay_deals

logger = logging.getLogger(__name__)

@mcp.tool(
    name="get_ebay_deals",
    description=(
        "Recupera le migliori offerte del giorno (Daily Deals) e offerte a tempo limitato (Limited Time Deals) da eBay usando SerpApi. "
        "È il tool principale da usare quando l'utente chiede 'offerte', 'deals', 'occasioni' o 'promozioni'. "
        "Permette di trovare sconti attivi su categorie specifiche o tramite parole chiave. "
        "Restituisce prezzi scontati, percentuali di sconto, prezzi originali e link diretti."
    ),
)
async def get_ebay_deals(
    query: Annotated[Optional[str], Field(description="Parola chiave per filtrare le offerte (es. 'iphone', 'lego')")] = None,
    category_id: Annotated[Optional[str], Field(description="ID categoria eBay (es. '9355' per cellulari)")] = None,
    ebay_domain: Annotated[str, Field(description="Dominio eBay da utilizzare (default: ebay.it)")] = "ebay.it"
) -> Dict[str, Any]:
    """
    Recupera le offerte (deals) da eBay tramite SerpApi.
    """
    try:
        logger.info(f"MCP TOOL get_ebay_deals START | query={query}, category={category_id}")
        
        result = await fetch_ebay_deals(
            query=query,
            category_id=category_id,
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
            "summary": f"Offerte speciali recuperate: {result.title} ({len(result.items)} oggetti)."
        }

    except Exception as e:
        logger.error(f"MCP TOOL get_ebay_deals ERROR: {e}")
        return {
            "status": "error",
            "message": str(e),
            "data": None,
            "summary": f"Errore nel recupero delle offerte: {str(e)}"
        }
