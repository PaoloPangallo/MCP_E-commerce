"""
playwright_tool.py
MCP Tool: ebay_scrape
Usa Playwright per cercare prodotti su eBay.it navigando il sito reale
(scraping browser-based, indipendente dalle API eBay).
"""

import logging
from typing import Any, Annotated, Dict
from pydantic import Field

from app.mcp.core import mcp, _tool_error
from app.mcp.normalizers import _normalize_playwright_output
from app.services.ebay_playwright import scrape_ebay_search

logger = logging.getLogger(__name__)


@mcp.tool(
    name="ebay_scrape",
    description=(
        "Cerca prodotti su eBay.it navigando direttamente il sito con un browser reale (Playwright). "
        "Utile per cercare prodotti senza usare le API eBay, raccogliendo dati live dalla pagina pubblica. "
        "Restituisce titolo, prezzo, URL, immagine, condizione, venditore e spedizione per ogni annuncio. "
        "Se 'visible' è true, il browser viene mostrato (headed mode)."
    ),
)
async def ebay_scrape(
    query: Annotated[
        str,
        Field(description="Query di ricerca per eBay (es. 'iphone 13 128gb', 'scarpe nike air max 42')"),
    ],
    max_results: Annotated[
        int,
        Field(description="Numero massimo di risultati da restituire (default 10, max 24)", ge=1, le=24),
    ] = 10,
    visible: Annotated[
        bool,
        Field(description="Se true, mostra la finestra del browser (utile per debug o demo)"),
    ] = False,
    session_id: Annotated[
        str,
        Field(description="ID di sessione utente (fornito dal sistema)"),
    ] = "",
) -> Dict[str, Any]:
    try:
        logger.info("MCP TOOL ebay_scrape START | query=%s | max=%d | visible=%s", query, max_results, visible)

        results = await scrape_ebay_search(
            query=query, 
            max_results=max_results,
            headless=not visible
        )

        raw = {
            "query": query,
            "results": results,
            "results_count": len(results),
        }
        normalized = _normalize_playwright_output(raw)
        normalized["_backend"] = "mcp"

        logger.info("MCP TOOL ebay_scrape END | count=%d", len(results))
        return normalized

    except Exception as exc:
        logger.exception("MCP ebay_scrape failed")
        return _tool_error(query=query, error=str(exc))
