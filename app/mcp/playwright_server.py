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
    # Chiamata esplicita alla sincronizzazione
    sync_standard_tools()


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
        
@mcp_playwright.tool(name="list_playwright_entities", description="DEBUG: Elenca tool, prompt e risorse nel mondo Playwright.")
async def list_playwright_entities() -> Dict[str, Any]:
    return {
        "tools": list(mcp_playwright._tool_manager._tools.keys()),
        "prompts": list(mcp_playwright._prompt_manager._prompts.keys()),
        "resources": list(mcp_playwright._resource_manager._resources.keys()),
    }


# ── Sincronizzazione dal mondo Standard ─────────────────────────────────────
def sync_standard_tools():
    """Copia tool, prompt e risorse dall'istanza standard a quella playwright usando l'API ufficiale."""
    try:
        from app.mcp.server import mcp as standard_mcp
        
        # 1. Sync Tools
        st_tm = getattr(standard_mcp, "_tool_manager", None)
        pw_tm = getattr(mcp_playwright, "_tool_manager", None)
        t_count = 0
        if st_tm and pw_tm:
            for name, tool_obj in st_tm._tools.items():
                if name not in pw_tm._tools:
                    mcp_playwright.add_tool(tool_obj.func, name=name, description=tool_obj.description)
                    t_count += 1
        
        # 2. Sync Prompts
        st_pm = getattr(standard_mcp, "_prompt_manager", None)
        pw_pm = getattr(mcp_playwright, "_prompt_manager", None)
        p_count = 0
        if st_pm and pw_pm:
            # st_pm._prompts è un dict {name: Prompt}
            for name, prompt_obj in st_pm._prompts.items():
                if name not in pw_pm._prompts:
                    try:
                        mcp_playwright.add_prompt(prompt_obj.func, name=name, description=prompt_obj.description)
                        p_count += 1
                    except Exception as pe:
                        logger.warning("Failed to sync prompt %s: %s", name, pe)
                
        # 3. Sync Resources
        st_rm = getattr(standard_mcp, "_resource_manager", None)
        pw_rm = getattr(mcp_playwright, "_resource_manager", None)
        r_count = 0
        if st_rm and pw_rm:
            # Sincronizzazione risorse statiche
            for uri, res_obj in st_rm._resources.items():
                if uri not in pw_rm._resources:
                    try:
                        mcp_playwright.add_resource(res_obj.func, uri=uri, description=res_obj.description)
                        r_count += 1
                    except Exception as re:
                        logger.warning("Failed to sync resource %s: %s", uri, re)
            
            # Sincronizzazione template (es: profile://{session_id})
            # st_rm._templates è un dict {uri_template: ResourceTemplate}
            for uri_template, tmpl_obj in st_rm._templates.items():
                if uri_template not in pw_rm._templates:
                    try:
                        mcp_playwright.add_resource(
                            tmpl_obj.func, 
                            uri=uri_template, 
                            name=tmpl_obj.name, 
                            description=tmpl_obj.description
                        )
                        r_count += 1
                        logger.info("Synced resource template: %s", uri_template)
                    except Exception as te:
                        logger.warning("Failed to sync template %s: %s", uri_template, te)

        logger.info(
            "Playwright MCP: synced [tools=%d, prompts=%d, resources=%d] from standard world using official API", 
            t_count, p_count, r_count
        )
    except Exception as e:
        logger.error("Failed to sync standard world entities to playwright world: %s", e)
