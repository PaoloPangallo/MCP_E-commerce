"""
browser_tools.py
MCP Tools per l'agente browser Playwright.
Permettono una navigazione interattiva e persistente.
"""

import logging
from typing import Any, Annotated, Dict
from pydantic import Field

from app.mcp.core import mcp, _tool_error
from app.services.browser_manager import BrowserManager

logger = logging.getLogger(__name__)

# Istanza globale del manager
browser_mgr = BrowserManager.get_instance()

@mcp.tool(
    name="browser_navigate",
    description=(
        "Naviga verso una URL specifica usando un browser reale e persistente. "
        "Il browser rimarrà aperto per sessioni successive. "
        "Restituisce URL corrente, titolo e uno screenshot (base64)."
    ),
)
async def browser_navigate(
    url: Annotated[str, Field(description="URL da visitare (es. https://www.ebay.it)")],
    timeout: Annotated[int, Field(description="Timeout in secondi per l'inattività (default 60)")] = 60,
) -> Dict[str, Any]:
    try:
        logger.info("MCP: browser_navigate -> %s", url)
        browser_mgr.set_timeout(timeout)
        result = browser_mgr.navigate(url)
        return result
    except Exception as exc:
        logger.exception("MCP: browser_navigate failed")
        return _tool_error(url=url, error=str(exc))

@mcp.tool(
    name="browser_click",
    description=(
        "Clicca su un elemento della pagina usando un selettore CSS o XPath. "
        "Dopo il click, restituisce lo stato aggiornato della pagina."
    ),
)
async def browser_click(
    selector: Annotated[str, Field(description="Selettore CSS o XPath dell'elemento da cliccare")],
) -> Dict[str, Any]:
    try:
        logger.info("MCP: browser_click -> %s", selector)
        result = browser_mgr.click(selector)
        return result
    except Exception as exc:
        logger.exception("MCP: browser_click failed")
        return _tool_error(selector=selector, error=str(exc))

@mcp.tool(
    name="browser_type",
    description=(
        "Inserisce del testo in un campo di input (es. una barra di ricerca). "
        "Può simulare la pressione del tasto Invio dopo l'inserimento."
    ),
)
async def browser_type(
    selector: Annotated[str, Field(description="Selettore CSS/XPath dell'input field")],
    text: Annotated[str, Field(description="Testo da scrivere")],
    press_enter: Annotated[bool, Field(description="Se true, preme Invio dopo aver scritto")] = False,
) -> Dict[str, Any]:
    try:
        logger.info("MCP: browser_type -> %s | text=%s", selector, text)
        result = browser_mgr.type(selector, text, press_enter=press_enter)
        return result
    except Exception as exc:
        logger.exception("MCP: browser_type failed")
        return _tool_error(selector=selector, error=str(exc))

@mcp.tool(
    name="browser_get_view",
    description="Ottiene lo stato corrente del browser (URL, Titolo e Screenshot) senza eseguire azioni.",
)
async def browser_get_view() -> Dict[str, Any]:
    try:
        return browser_mgr.get_view()
    except Exception as exc:
        return _tool_error(error=str(exc))

@mcp.tool(
    name="browser_close",
    description="Chiude immediatamente la sessione del browser e rilascia le risorse.",
)
async def browser_close() -> Dict[str, Any]:
    try:
        return browser_mgr.close()
    except Exception as exc:
        return _tool_error(error=str(exc))
