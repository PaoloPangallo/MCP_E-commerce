"""
playwright_contact.py
MCP Tool: contact_seller_playwright

Usa il Chrome già installato e già loggato dall'utente:
- Se Chrome è aperto con --remote-debugging-port=9222 → ci connette via CDP
- Altrimenti → lancia Chrome con il profilo reale dell'utente (richiede Chrome chiuso)

Flusso autonomo:
1. Apre/connette a Chrome con sessione eBay attiva
2. Naviga su IntermediatedFAQ del venditore
3. Clicca "Non riguarda un oggetto"
4. Compila il messaggio
5. Clicca Invia
6. Chiude la tab solo in caso di successo; lascia tutto aperto in caso di errore.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
from typing import Any, Annotated, Dict, Optional

from pydantic import Field

from app.mcp.playwright_server import mcp_playwright
from app.services.ebay_playwright import _PLAYWRIGHT_EXECUTOR

logger = logging.getLogger(__name__)

# Profilo Chrome reale dell'utente — contiene tutti i cookie incluso eBay/Google.
# Richiede che Chrome sia CHIUSO quando Playwright lo usa.
if sys.platform == "win32":
    _CHROME_USER_DATA = os.path.expandvars(
        r"%LOCALAPPDATA%\Google\Chrome\User Data"
    )
else:
    _CHROME_USER_DATA = os.path.join(os.path.expanduser("~"), ".config", "google-chrome")

# Porta CDP su cui Chrome espone il remote debugging.
# Per abilitarla permanentemente: aggiungere --remote-debugging-port=9222
# nelle proprietà di avvio di Chrome (o in chrome://flags per test).
_CDP_PORT = int(os.getenv("CHROME_CDP_PORT", "9222"))


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


def _chrome_cdp_available(port: int = _CDP_PORT) -> bool:
    """Ritorna True se Chrome sta ascoltando sulla porta CDP."""
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except OSError:
        return False


async def _async_contact_seller(
    product_url: str,
    message: str,
    timeout_ms: int,
) -> Dict[str, Any]:
    """Flusso autonomo — usa Chrome reale già loggato.

    Strategia di connessione (in ordine di priorità):
    1. CDP: se Chrome gira con --remote-debugging-port=9222, ci connette direttamente
    2. Profilo reale: lancia Chrome con il profilo utente (Chrome deve essere chiuso)
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright non installato. Esegui: pip install playwright && playwright install chromium"
        ) from exc

    logger.info("PW contact_seller START | url=%s", product_url)
    _launch_args = ["--no-sandbox", "--disable-dev-shm-usage"]

    # ── Selectors ────────────────────────────────────────────────────────────

    not_about_item_selectors = [
        "a:has-text('Non riguarda un oggetto')",
        "button:has-text('Non riguarda un oggetto')",
        "a:has-text('Not about an item')",
        "button:has-text('Not about an item')",
        "a:has-text('Not about a specific item')",
        "a[href*='notAboutItem']",
        "a[href*='not_about_item']",
        "[data-testid*='not-about-item']",
        "[data-testid*='notAboutItem']",
    ]

    message_field_selectors = [
        "textarea[name='body']",
        "textarea[id*='message']",
        "textarea[id*='msg']",
        "textarea[placeholder*='messaggio']",
        "textarea[placeholder*='message']",
        "textarea[placeholder*='Messaggio']",
        "#msg",
        "textarea",
    ]

    submit_selectors = [
        "button:has-text('Invia')",
        "button:has-text('Send')",
        "input[type='submit']",
        "button[type='submit']",
        "[data-testid*='send']",
        "[data-testid*='submit']",
    ]

    contact_page_selectors = [
        "a[href*='contactseller']",
        "a[href*='contact_seller']",
        "a[href*='vi/contact']",
        "a[href*='MsgContact']",
        "a[href*='askSellerQuestion']",
        "a:has-text('Contatta il venditore')",
        "a:has-text('Contact seller')",
        "button:has-text('Contatta il venditore')",
        "button:has-text('Contact seller')",
        "[data-testid='vi-VR-CONTACT-SELLER']",
        "[data-testid='contact-seller']",
    ]

    # ── Connessione a Chrome ──────────────────────────────────────────────────

    pw = await async_playwright().start()
    context = None
    browser_cdp = None  # solo se usiamo CDP
    page = None
    success = False
    using_cdp = False

    try:
        if _chrome_cdp_available(_CDP_PORT):
            # Chrome è già aperto con remote debugging → connessione CDP
            logger.info("PW contact_seller: connecting to running Chrome via CDP port %s", _CDP_PORT)
            browser_cdp = await pw.chromium.connect_over_cdp(f"http://localhost:{_CDP_PORT}")
            # Usa il contesto esistente (sessione già loggata)
            context = browser_cdp.contexts[0] if browser_cdp.contexts else await browser_cdp.new_context(locale="it-IT")
            using_cdp = True
            logger.info("PW contact_seller: CDP connected | contexts=%d", len(browser_cdp.contexts))
        else:
            # Chrome non ha CDP aperta → launch con profilo reale
            logger.info("PW contact_seller: launching Chrome with real profile at %s", _CHROME_USER_DATA)
            context = await pw.chromium.launch_persistent_context(
                _CHROME_USER_DATA,
                channel="chrome",
                headless=False,
                args=_launch_args,
                locale="it-IT",
            )

        page = await context.new_page()

        # ── Step 1: naviga ───────────────────────────────────────────────────
        try:
            await page.goto(product_url, timeout=timeout_ms, wait_until="domcontentloaded")
        except Exception as exc:
            logger.warning("PW contact_seller: goto timeout/error | %s", exc)

        # ── Step 2: login wall? aspetta max 2 minuti ─────────────────────────
        current_url = page.url
        if "signin" in current_url or "login" in current_url:
            logger.info("PW contact_seller: login required — waiting up to 120s")
            try:
                await page.wait_for_function(
                    "() => !window.location.href.includes('signin') && "
                    "!window.location.href.includes('login')",
                    timeout=120_000,
                )
                current_url = page.url
                logger.info("PW contact_seller: login complete | url=%s", current_url)
            except Exception:
                logger.warning("PW contact_seller: login timeout")
                return _build_contact_result(
                    product_url=product_url,
                    success=False,
                    status="login_required",
                    detail=(
                        "eBay richiede il login. Effettua il login nel browser aperto "
                        "e poi riprova."
                    ),
                )

        # ── Step 3: IntermediatedFAQ → clicca "Non riguarda un oggetto" ──────
        current_url = page.url
        if "IntermediatedFAQ" in current_url or "intermediatedfaq" in current_url.lower():
            logger.info("PW contact_seller: on IntermediatedFAQ — clicking 'Non riguarda un oggetto'")
            clicked = False
            for sel in not_about_item_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        await el.click()
                        try:
                            await page.wait_for_load_state("networkidle", timeout=10_000)
                        except Exception:
                            pass
                        clicked = True
                        logger.info("PW contact_seller: clicked '%s' | now at %s", sel, page.url)
                        break
                except Exception:
                    continue

            if not clicked:
                try:
                    elements = await page.evaluate(
                        "() => Array.from(document.querySelectorAll('a,button'))"
                        ".map(el => ({tag: el.tagName, text: el.innerText.trim().slice(0,80), href: el.href || ''}))"
                        ".filter(x => x.text).slice(0, 30)"
                    )
                    logger.warning("PW contact_seller: button not found. Elements: %s", elements)
                except Exception:
                    pass
                return _build_contact_result(
                    product_url=product_url,
                    success=False,
                    status="contact_button_not_found",
                    detail=(
                        "Non riesco a trovare il pulsante 'Non riguarda un oggetto'. "
                        "Il browser è aperto: clicca tu il pulsante."
                    ),
                )

        # ── Step 4: pagina prodotto → "Contatta il venditore" ────────────────
        elif "ebay.it/itm/" in current_url or "ebay.com/itm/" in current_url:
            for sel in contact_page_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        await el.click()
                        try:
                            await page.wait_for_load_state("networkidle", timeout=8_000)
                        except Exception:
                            pass
                        logger.info("PW contact_seller: clicked contact selector=%s | now at %s", sel, page.url)
                        break
                except Exception:
                    continue

        # ── Step 5: compila messaggio ─────────────────────────────────────────
        logger.info("PW contact_seller: filling message | url=%s", page.url)
        filled = False
        for sel in message_field_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    await el.click()
                    await el.fill(message)
                    filled = True
                    logger.info("PW contact_seller: message filled via selector=%s", sel)
                    break
            except Exception:
                continue

        if not filled:
            logger.warning("PW contact_seller: message field not found at %s", page.url)
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="message_form_not_found",
                detail=(
                    "Sono arrivato alla pagina ma non trovo il campo messaggio. "
                    "Il browser è aperto: scrivi tu il messaggio e clicca Invia."
                ),
            )

        # ── Step 6: clicca Invia ──────────────────────────────────────────────
        submitted = False
        for sel in submit_selectors:
            try:
                el = await page.query_selector(sel)
                if el:
                    await el.click()
                    try:
                        await page.wait_for_load_state("networkidle", timeout=10_000)
                    except Exception:
                        pass
                    submitted = True
                    logger.info("PW contact_seller: message submitted | now at %s", page.url)
                    break
            except Exception:
                continue

        if submitted:
            success = True
            await asyncio.sleep(2)
            return _build_contact_result(
                product_url=product_url,
                success=True,
                status="message_sent",
                detail="Messaggio inviato con successo al venditore.",
                message_sent=message,
            )
        else:
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="submit_button_not_found",
                detail=(
                    "Ho compilato il messaggio ma non trovo il bottone 'Invia'. "
                    "Il browser è aperto: clicca tu 'Invia'."
                ),
            )

    except Exception as exc:
        logger.exception("PW contact_seller: unexpected error | %s", exc)
        return _build_contact_result(
            product_url=product_url,
            success=False,
            status="error",
            detail=str(exc),
        )

    finally:
        if success:
            # Chiudi solo la tab aperta, non tutto il browser
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass
            if not using_cdp and context:
                try:
                    await context.close()
                except Exception:
                    pass
        else:
            logger.info("PW contact_seller: browser left open for manual interaction")
        # Ferma sempre il processo Playwright (solo se non siamo in CDP)
        if not using_cdp:
            try:
                await pw.stop()
            except Exception:
                pass


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
        "Contatta un venditore eBay in modo autonomo usando Chrome (già loggato dall'utente). "
        "Se Chrome gira con --remote-debugging-port=9222, si connette direttamente alla sessione attiva. "
        "Altrimenti lancia Chrome con il profilo reale. "
        "Naviga su IntermediatedFAQ, clicca 'Non riguarda un oggetto', compila il messaggio e lo invia."
    ),
)
async def contact_seller_playwright(
    product_url: Annotated[
        str,
        Field(description="URL contatto eBay (es. https://www.ebay.it/cnt/IntermediatedFAQ?seller_name=X)"),
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
        logger.info("MCP contact_seller_playwright END | status=%s", result.get("contact_status"))
        return result
    except Exception as exc:
        logger.exception("MCP contact_seller_playwright failed")
        return _build_contact_result(
            product_url=product_url,
            success=False,
            status="error",
            detail=str(exc),
        )
