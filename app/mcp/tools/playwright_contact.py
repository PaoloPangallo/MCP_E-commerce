"""
playwright_contact.py
MCP Tool: contact_seller_playwright
Apre Chromium visibile, naviga su una pagina prodotto eBay e tenta di
contattare il venditore compilando il form di messaggistica.

NOTA: eBay richiede login per inviare messaggi. Se l'utente non è loggato,
ritorna status="login_required" con istruzioni.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Annotated, Dict, Optional

from pydantic import Field

from app.mcp.playwright_server import mcp_playwright
from app.services.ebay_playwright import _PLAYWRIGHT_EXECUTOR

logger = logging.getLogger(__name__)


def _build_contact_result(
    product_url: str,
    success: bool,
    status: str,
    detail: str,
    message_sent: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": "ok" if success else "error",
        "success": success,
        "product_url": product_url,
        "contact_status": status,
        "detail": detail,
        "message_sent": message_sent,
    }


async def _async_contact_seller(
    product_url: str,
    message: str,
    timeout_ms: int,
) -> Dict[str, Any]:
    """Logica Playwright pura — chiamare solo da dentro un ProactorEventLoop."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright non installato. Esegui: pip install playwright && playwright install chromium"
        ) from exc

    logger.info("PW contact_seller START | url=%s", product_url)
    _launch_args = ["--no-sandbox", "--disable-dev-shm-usage"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=_launch_args)
        context = await browser.new_context(
            locale="it-IT",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        try:
            await page.goto(product_url, timeout=timeout_ms, wait_until="load")
        except Exception as exc:
            logger.warning("PW contact_seller: goto timeout | %s", exc)

        current_url = page.url
        if "signin" in current_url or "login" in current_url:
            await browser.close()
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="login_required",
                detail=(
                    "eBay richiede il login per contattare i venditori. "
                    "Effettua il login su eBay.it nel browser che si è aperto, "
                    "poi riprova l'operazione."
                ),
            )

        contact_selectors = [
            "a[href*='contactseller']",
            "a[href*='contact_seller']",
            "a:has-text('Contatta il venditore')",
            "a:has-text('Contact seller')",
            "button:has-text('Contatta il venditore')",
            "[data-testid*='contact']",
        ]

        contact_link = None
        for selector in contact_selectors:
            try:
                el = await page.query_selector(selector)
                if el:
                    contact_link = el
                    break
            except Exception:
                continue

        if not contact_link:
            await browser.close()
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="contact_button_not_found",
                detail=(
                    "Non ho trovato il pulsante 'Contatta il venditore' su questa pagina. "
                    "Potrebbe essere che il venditore non accetti messaggi, "
                    "o che la pagina abbia una struttura diversa."
                ),
            )

        await contact_link.click()

        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass

        current_url = page.url
        if "signin" in current_url or "login" in current_url:
            await browser.close()
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="login_required",
                detail=(
                    "eBay richiede il login per inviare messaggi. "
                    "Effettua il login su eBay.it e riprova."
                ),
            )

        message_selectors = [
            "textarea[name='body']",
            "textarea[id*='message']",
            "textarea[placeholder*='messaggio']",
            "textarea[placeholder*='message']",
            "textarea",
        ]

        textarea = None
        for sel in message_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    textarea = el
                    break
            except Exception:
                continue

        if not textarea:
            await browser.close()
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="message_form_not_found",
                detail=(
                    "La pagina di contatto è aperta nel browser. "
                    "Non ho trovato il campo di testo in modo automatico. "
                    "Puoi compilare e inviare il messaggio manualmente nel browser."
                ),
            )

        await textarea.fill(message)

        submit_selectors = [
            "button[type='submit']",
            "input[type='submit']",
            "button:has-text('Invia')",
            "button:has-text('Send')",
            "button:has-text('Invia messaggio')",
        ]

        submit_btn = None
        for sel in submit_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    submit_btn = el
                    break
            except Exception:
                continue

        if not submit_btn:
            await browser.close()
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="submit_button_not_found",
                detail=(
                    "Ho compilato il messaggio nel form. "
                    "Non ho trovato il bottone di invio automaticamente — "
                    "clicca tu stesso 'Invia' nel browser aperto."
                ),
                message_sent=message,
            )

        await submit_btn.click()

        try:
            await page.wait_for_load_state("networkidle", timeout=8_000)
        except Exception:
            pass

        await browser.close()
        logger.info("PW contact_seller: message sent | url=%s", product_url)
        return _build_contact_result(
            product_url=product_url,
            success=True,
            status="message_sent",
            detail="Messaggio inviato al venditore con successo tramite browser.",
            message_sent=message,
        )


def _run_contact_in_proactor_loop(
    product_url: str,
    message: str,
    timeout_ms: int,
) -> Dict[str, Any]:
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _async_contact_seller(product_url, message, timeout_ms)
        )
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@mcp_playwright.tool(
    name="contact_seller_playwright",
    description=(
        "Contatta un venditore eBay inviando un messaggio direttamente dalla pagina del prodotto, "
        "usando un browser Chromium reale e visibile. "
        "Naviga automaticamente alla pagina del prodotto, trova il pulsante 'Contatta il venditore', "
        "compila il form e invia il messaggio. "
        "NOTA: richiede che l'utente sia loggato su eBay nel browser che si apre. "
        "Se non loggato, il tool restituirà istruzioni per effettuare il login."
    ),
)
async def contact_seller_playwright(
    product_url: Annotated[
        str,
        Field(description="URL completo della pagina prodotto eBay (es. https://www.ebay.it/itm/123456789012)"),
    ],
    message: Annotated[
        str,
        Field(description="Testo del messaggio da inviare al venditore"),
    ],
) -> Dict[str, Any]:
    try:
        logger.info("MCP contact_seller_playwright START | url=%s", product_url)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _PLAYWRIGHT_EXECUTOR,
            _run_contact_in_proactor_loop,
            product_url,
            message,
            30_000,
        )
        logger.info(
            "MCP contact_seller_playwright END | status=%s",
            result.get("contact_status"),
        )
        return result
    except Exception as exc:
        logger.exception("MCP contact_seller_playwright failed")
        return _build_contact_result(
            product_url=product_url,
            success=False,
            status="error",
            detail=str(exc),
        )
