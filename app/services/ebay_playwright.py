"""
ebay_playwright.py
Servizio di browser scraping per eBay.it usando Playwright.
Non dipende dall'API eBay: naviga la pagina pubblica con un browser headless reale.

NOTA WINDOWS: Playwright richiede ProactorEventLoop per lanciare sottoprocessi.
Uvicorn usa SelectorEventLoop su Windows, quindi eseguiamo Playwright in un
thread dedicato con il proprio ProactorEventLoop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Costanti / URL
# ─────────────────────────────────────────────

EBAY_SEARCH_URL = "https://www.ebay.it/sch/i.html?_nkw={query}&_ipg=48&LH_BIN=1"

_PRODUCT_CONTAINER = "li:has(.su-card-container)"
_LINK_SEL      = "a.s-card__link"
_TITLE_SEL     = ".s-card__title span, .s-card__title"
_PRICE_SEL     = ".s-card__price .s-price-format, .s-card__price"
_IMAGE_SEL     = ".su-image img, .s-card img"
_CONDITION_SEL = ".s-card__specifics-list"
_SHIPPING_SEL  = ".s-card__delivery"
_SELLER_SEL    = ".s-card__seller-info"
_LOCATION_SEL  = ".s-card__item-location"

# Thread pool dedicato: Playwright gira nel suo thread con ProactorEventLoop
_PLAYWRIGHT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="playwright"
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _parse_price(raw: str) -> Optional[float]:
    """Estrae il primo numero decimale da una stringa prezzo."""
    try:
        cleaned = raw.replace(".", "").replace(",", ".")
        match = re.search(r"[\d]+(?:\.\d+)?", cleaned)
        return float(match.group()) if match else None
    except Exception:
        return None


# ─────────────────────────────────────────────
# Core async scraping (deve girare in ProactorEventLoop)
# ─────────────────────────────────────────────

async def _async_scrape(
    query: str,
    max_results: int,
    headless: bool,
    timeout_ms: int,
) -> List[Dict[str, Any]]:
    """Logica Playwright pura — chiamare solo da dentro un ProactorEventLoop."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright non è installato. Esegui: pip install playwright && playwright install chromium"
        ) from exc

    url = EBAY_SEARCH_URL.format(query=query.replace(" ", "+"))
    logger.info("Playwright scraper | url=%s | max_results=%d", url, max_results)

    results: List[Dict[str, Any]] = []
    _launch_args = ["--no-sandbox", "--disable-dev-shm-usage"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless, args=_launch_args)
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
            await page.goto(url, timeout=timeout_ms, wait_until="load")
        except Exception as exc:
            logger.warning("Playwright: goto load timeout, proceeding anyway | %s", exc)

        try:
            await page.wait_for_selector(".su-card-container", timeout=10_000)
        except Exception as exc:
            logger.warning("Playwright: nessuna card prodotto trovata | %s", exc)
            await browser.close()
            return results

        items = await page.query_selector_all(_PRODUCT_CONTAINER)
        logger.info("Playwright: trovati %d elementi prodotto", len(items))

        for item in items:
            if len(results) >= max_results:
                break

            try:
                link_el = await item.query_selector(_LINK_SEL)
                if not link_el:
                    link_el = await item.query_selector("a[href*='/itm/']")
                url_raw = await link_el.get_attribute("href") if link_el else ""
                if not url_raw or "/itm/" not in url_raw:
                    continue
                item_url = url_raw.split("?")[0]

                title_el = await item.query_selector(_TITLE_SEL)
                title = (await title_el.inner_text()).strip() if title_el else ""
                if not title:
                    title = (await link_el.inner_text()).strip()[:100] if link_el else ""
                if not title or "shop on ebay" in title.lower():
                    continue

                price_el = await item.query_selector(_PRICE_SEL)
                price_raw = (await price_el.inner_text()).strip() if price_el else ""
                price = _parse_price(price_raw)

                img_el = await item.query_selector(_IMAGE_SEL)
                image_url = ""
                if img_el:
                    image_url = await img_el.get_attribute("src") or ""
                    if not image_url or "gif" in image_url:
                        image_url = await img_el.get_attribute("data-src") or ""

                cond_el = await item.query_selector(_CONDITION_SEL)
                condition = (await cond_el.inner_text()).strip() if cond_el else ""

                seller_el = await item.query_selector(_SELLER_SEL)
                seller = (await seller_el.inner_text()).strip() if seller_el else ""

                loc_el = await item.query_selector(_LOCATION_SEL)
                location = (await loc_el.inner_text()).strip() if loc_el else ""

                ship_el = await item.query_selector(_SHIPPING_SEL)
                shipping = (await ship_el.inner_text()).strip() if ship_el else ""

                results.append({
                    "title": title,
                    "price": price,
                    "price_raw": price_raw,
                    "currency": "EUR",
                    "url": item_url,
                    "image_url": image_url,
                    "condition": condition,
                    "seller": seller,
                    "location": location,
                    "shipping": shipping,
                    "_source": "playwright",
                })

            except Exception as exc:
                logger.debug("Playwright: errore parsing item | %s", exc)
                continue

        await browser.close()

    logger.info("Playwright scraper completato | risultati=%d", len(results))
    return results


# ─────────────────────────────────────────────
# Thread worker: ProactorEventLoop isolato
# ─────────────────────────────────────────────

def _run_in_proactor_loop(
    query: str,
    max_results: int,
    headless: bool,
    timeout_ms: int,
) -> List[Dict[str, Any]]:
    """
    Crea un ProactorEventLoop nel thread corrente ed esegue lo scraping.
    Necessario su Windows perché uvicorn gira con SelectorEventLoop che non
    supporta asyncio.create_subprocess_exec() (usato da Playwright).
    """
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_async_scrape(query, max_results, headless, timeout_ms))
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

async def scrape_ebay_search(
    query: str,
    max_results: int = 10,
    headless: bool = True,
    timeout_ms: int = 25_000,
) -> List[Dict[str, Any]]:
    """
    Naviga su eBay.it e ritorna una lista di prodotti trovati.

    Esegue Playwright in un thread dedicato con ProactorEventLoop per
    compatibilità con Windows + uvicorn (che usa SelectorEventLoop).

    Args:
        query:       Testo da cercare su eBay.
        max_results: Numero massimo di risultati (default 10).
        headless:    Se True, browser invisibile (default True).
        timeout_ms:  Timeout in ms per il caricamento della pagina.

    Returns:
        Lista di dict con: title, price, price_raw, url,
        image_url, condition, seller, location, shipping.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _PLAYWRIGHT_EXECUTOR,
        _run_in_proactor_loop,
        query,
        max_results,
        headless,
        timeout_ms,
    )
