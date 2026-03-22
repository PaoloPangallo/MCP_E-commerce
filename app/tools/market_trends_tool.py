from typing import Any, Dict, Optional
import os
import logging
import re
from dotenv import load_dotenv
from app.services.serp.market_trends import get_market_trends_analysis

load_dotenv()

logger = logging.getLogger(__name__)

SERP_API_KEY = os.getenv("SERP_API_KEY")

def normalize_market_trends_arguments(action_input: Dict[str, Any]) -> Dict[str, Any]:
    query = action_input.get("query")
    if not query:
        raise ValueError("market_trends richiede una query (nome del prodotto).")

    prefixes_to_strip = [
        "trend di mercato, statistiche e andamento prezzi medi online per:",
        "trend di mercato e analisi prezzi per:",
        "analisi dei trend di mercato per:",
        "mostrami i trend di mercato per:",
        "mostrami i trend di mercato e i prezzi medi online per:",
        "analisi mercato",
        "trend per",
        "prezzi per",
        "per:"
    ]

    clean_q = query.lower()
    for prefix in prefixes_to_strip:
        if clean_q.startswith(prefix):
            query = query[len(prefix):].strip()
            clean_q = query.lower()

    query = re.sub(r'^[:\s\-]+', '', query).strip()
    return {"query": query}

async def execute_market_trends_tool(action_input: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    Analizza i trend di mercato (prezzi e interesse) per un prodotto tramite il servizio Serp centralizzato.
    """
    try:
        clean = normalize_market_trends_arguments(action_input)
        query = clean["query"]

        logger.info(f"Market Trends Analysis (Service-based) for: {query}")

        if not SERP_API_KEY:
            logger.warning("SERP_API_KEY non configurata nel .env")
            return {
                "status": "error",
                "error": "SERP_API_KEY mancante. Configurala nel file .env."
            }

        # Delega l'intera logica complessa al servizio dedicato
        result = await get_market_trends_analysis(query, SERP_API_KEY)
        
        return result

    except Exception as e:
        logger.error(f"Error in execute_market_trends_tool: {e}")
        return {
            "status": "error",
            "error": str(e)
        }