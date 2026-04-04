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


# Rimosso pw_search_products: costringiamo l'agente a usare i tool primitivi di navigazione.

        
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
        
        st_tm = getattr(standard_mcp, "_tool_manager", None)
        pw_tm = getattr(mcp_playwright, "_tool_manager", None)
        t_count = 0
        
        # Strumenti macro da NON esporre nel mondo iterativo Playwright
        excluded_tools = {
            "search_products", "ebay_scrape", "compare_products", 
            "get_item_details", "get_similar_items", "get_ebay_deals"
        }
        
        if st_tm and pw_tm:
            for name, tool_obj in st_tm._tools.items():
                if name not in pw_tm._tools and name not in excluded_tools:
                    try:
                        func = getattr(tool_obj, "fn", getattr(tool_obj, "func", getattr(tool_obj, "function", None)))
                        if func:
                            mcp_playwright.add_tool(func, name=name, description=tool_obj.description)
                            t_count += 1
                        else:
                            logger.warning("Could not find callable for tool %s", name)
                    except Exception as e:
                        logger.warning("Error adding tool %s: %s", name, e)

        
        # 2. Sync Prompts
        st_pm = getattr(standard_mcp, "_prompt_manager", None)
        pw_pm = getattr(mcp_playwright, "_prompt_manager", None)
        p_count = 0
        if st_pm and pw_pm:
            # st_pm._prompts è un dict {name: Prompt}
            for name, prompt_obj in st_pm._prompts.items():
                if name not in pw_pm._prompts:
                    try:
                        func = getattr(prompt_obj, "fn", getattr(prompt_obj, "func", getattr(prompt_obj, "function", None)))
                        if func:
                            mcp_playwright.add_prompt(func, name=name, description=prompt_obj.description)
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
                        func = getattr(res_obj, "fn", getattr(res_obj, "func", getattr(res_obj, "function", None)))
                        if func:
                            mcp_playwright.add_resource(func, uri=uri, description=res_obj.description)
                            r_count += 1

                    except Exception as re:
                        logger.warning("Failed to sync resource %s: %s", uri, re)
            
            # Sincronizzazione template (es: profile://{session_id})
            # st_rm._templates è un dict {uri_template: ResourceTemplate}
            for uri_template, tmpl_obj in st_rm._templates.items():
                if uri_template not in pw_rm._templates:
                    try:
                        func = getattr(tmpl_obj, "fn", getattr(tmpl_obj, "func", getattr(tmpl_obj, "function", None)))
                        if func:
                            mcp_playwright.add_resource(
                                func, 
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
