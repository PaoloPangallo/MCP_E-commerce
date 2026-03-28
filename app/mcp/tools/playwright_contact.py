"""
playwright_contact.py
MCP Tool: contact_seller_playwright
Apre Chromium visibile, naviga su una pagina prodotto eBay e tenta di
contattare il venditore compilando il form di messaggistica.

NOTA: eBay richiede login per inviare messaggi. Se l'utente non è loggato,
il browser rimane APERTO così l'utente può fare login e riprovare.
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
            # IMPORTANT: leave browser open so user can log in manually
            logger.info("PW contact_seller: login required — browser left open for manual login")
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="login_required",
                detail=(
                    "eBay richiede il login per contattare i venditori. "
                    "Il browser è rimasto aperto: effettua il login su eBay.it, "
                    "poi naviga al prodotto e clicca 'Contatta il venditore'."
                ),
            )

        contact_selectors = [
            "a[href*='contactseller']",
            "a[href*='contact_seller']",
            "a[href*='vi/contact']",
            "a:has-text('Contatta il venditore')",
            "a:has-text('Contact seller')",
            "button:has-text('Contatta il venditore')",
            "button:has-text('Contact seller')",
            "[data-testid*='contact']",
            "[data-testid='contact-seller']",
            ".vi-VR-cvipFreeShipping a",
        ]

        contact_link = None
        for selector in contact_selectors:
            try:
                el = await page.query_selector(selector)
                if el:
                    contact_link = el
                    logger.info("PW contact_seller: found contact button via selector=%s", selector)
                    break
            except Exception:
                continue

        if not contact_link:
            # Leave browser open — user can navigate manually
            logger.info("PW contact_seller: contact button not found — browser left open")
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="contact_button_not_found",
                detail=(
                    "Non ho trovato il pulsante 'Contatta il venditore' su questa pagina. "
                    "Il browser è rimasto aperto: cerca il pulsante manualmente e clicca tu."
                ),
            )

        await contact_link.click()

        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except Exception:
            pass

        current_url = page.url
        if "signin" in current_url or "login" in current_url:
            # Leave browser open for manual login
            logger.info("PW contact_seller: redirect to login after click — browser left open")
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="login_required",
                detail=(
                    "eBay richiede il login per inviare messaggi. "
                    "Il browser è rimasto aperto: effettua il login e riprova."
                ),
            )

        message_selectors = [
            "textarea[name='body']",
            "textarea[id*='message']",
            "textarea[placeholder*='messaggio']",
            "textarea[placeholder*='message']",
            "#message-to-seller-textarea",
            ".msg-form__contenteditable",
            "textarea",
        ]

        textarea = None
        for sel in message_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    textarea = el
                    logger.info("PW contact_seller: found textarea via selector=%s", sel)
                    break
            except Exception:
                continue

        if not textarea:
            # Leave browser open — page is on contact form, user can type manually
            logger.info("PW contact_seller: textarea not found — browser left open on contact page")
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="message_form_not_found",
                detail=(
                    "La pagina di contatto è aperta nel browser. "
                    "Non ho trovato il campo di testo automaticamente — "
                    "scrivi e invia il messaggio tu direttamente nel browser aperto."
                ),
            )

        await textarea.fill(message)

        submit_selectors = [
            "button[type='submit']",
            "input[type='submit']",
            "button:has-text('Invia')",
            "button:has-text('Send')",
            "button:has-text('Invia messaggio')",
            "button:has-text('Send message')",
            "[data-testid='send-message']",
        ]

        submit_btn = None
        for sel in submit_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    submit_btn = el
                    logger.info("PW contact_seller: found submit btn via selector=%s", sel)
                    break
            except Exception:
                continue

        if not submit_btn:
            # Message is filled, leave browser open for manual submit
            logger.info("PW contact_seller: submit button not found — browser left open with message filled")
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="submit_button_not_found",
                detail=(
                    "Ho compilato il messaggio nel form. "
                    "Il browser è rimasto aperto: clicca tu 'Invia' per inviare."
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
        "Se non loggato, il browser rimane aperto così l'utente può fare login manualmente."
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
        loop = asyncio.get_running_loop()
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
