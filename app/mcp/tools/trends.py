import logging
from typing import Dict, Any, Annotated
from pydantic import Field

from app.mcp.core import mcp, _tool_error
import os
from dotenv import load_dotenv
from app.services.serp.market_trends import get_market_trends_analysis

load_dotenv()
SERP_API_KEY = os.getenv("SERP_API_KEY")

logger = logging.getLogger(__name__)

@mcp.tool(
    name="market_trends",
    description="Analizza i trend di mercato (prezzi e interesse) per un prodotto tramite servizi esterni (es. SerpApi e Google Trends).",
)
async def market_trends(
    query: Annotated[str, Field(description="Nome del prodotto da cercare (es. 'playstation 5')")],
) -> Dict[str, Any]:
    try:
        logger.info("MCP TOOL market_trends START | query=%s", query)
        
        if not SERP_API_KEY:
            raise ValueError("SERP_API_KEY mancante dal file .env")

        import re
        query_cleaned = re.sub(r'^(trend di mercato|analisi prezzi|mostrami i trend).*?:', '', query.lower()).strip()
        query_cleaned = re.sub(r'^[:\s\-]+', '', query_cleaned).strip()

        raw = await get_market_trends_analysis(query_cleaned, SERP_API_KEY)
        
        # Generiamo il summary
        shop = raw.get("shopping_data", {})
        trends = raw.get("trends_data", {})
        if shop.get("status") != "ok" and trends.get("status") != "ok":
            summary = "Nessun dato di mercato disponibile."
        else:
            avg = shop.get("average_price")
            direction = trends.get("trend_direction")
            summary = f"Trend di mercato analizzati: prezzo medio online = {avg}€, interesse = {direction}."
            
        raw["summary"] = summary
        raw["_backend"] = "mcp"
        logger.info("MCP TOOL market_trends END")
        return raw

    except Exception as exc:
        logger.exception("MCP market_trends failed")
        return _tool_error(query=query, error=str(exc))
