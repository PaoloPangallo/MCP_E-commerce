"""
MCP Server — Playwright Browser World
Espone solo i tool che richiedono un browser reale (Chromium visibile).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Annotated

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from app.mcp.normalizers import _normalize_playwright_output
from app.services.ebay_playwright import scrape_ebay_search

logger = logging.getLogger(__name__)

# ── Seconda istanza MCP: mondo Playwright ────────────────────────────────────
mcp_playwright = FastMCP("mcp-playwright-world")

@dataclass
class _PlaywrightDeps:
    db_factory: Optional[Callable[[], Any]] = None

_PLAYWRIGHT_DEPS = _PlaywrightDeps()


def configure_playwright_mcp(
    db_factory: Optional[Callable[[], Any]] = None,
) -> None:
    _PLAYWRIGHT_DEPS.db_factory = db_factory
    logger.info("Playwright MCP configured | db_factory=%s", bool(db_factory))


@mcp_playwright.tool(
    name="search_products",
    description=(
        "Cerca prodotti su eBay.it navigando il sito con un browser Chromium reale e VISIBILE. "
        "Il browser si apre sullo schermo così l'utente può vedere la navigazione in tempo reale. "
        "Restituisce titolo, prezzo, URL, immagine, condizione, venditore e spedizione per ogni annuncio."
    ),
)
async def pw_search_products(
    query: Annotated[
        str,
        Field(description="Query di ricerca per eBay (es. 'iphone 13 128gb')"),
    ],
    max_results: Annotated[
        int,
        Field(description="Numero massimo di risultati (default 10, max 24)", ge=1, le=24),
    ] = 10,
) -> Dict[str, Any]:
    try:
        logger.info("PW search_products START | query=%s | max=%d", query, max_results)
        results = await scrape_ebay_search(
            query=query,
            max_results=max_results,
            headless=False,  # SEMPRE visibile nel mondo Playwright
        )
        raw = {
            "query": query,
            "results": results,
            "results_count": len(results),
        }
        normalized = _normalize_playwright_output(raw)
        normalized["_backend"] = "playwright_browser"
        logger.info("PW search_products END | count=%d", len(results))
        return normalized
    except Exception as exc:
        logger.exception("PW search_products failed")
        return {"status": "error", "query": query, "error": str(exc)}
