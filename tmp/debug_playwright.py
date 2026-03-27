"""
debug_playwright.py — script diagnostico temporaneo.
Naviga eBay.it, cattura un dump HTML minimo e stampa i selettori trovati.
Esegui: python tmp/debug_playwright.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from playwright.async_api import async_playwright

    url = "https://www.ebay.it/sch/i.html?_nkw=iphone+13&_ipg=24"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="it-IT",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        print(f"Navigating to: {url}")
        await page.goto(url, timeout=30_000, wait_until="networkidle")

        # Prova selettori diversi
        for sel in ["li.s-item", ".srp-results .s-item", "div[data-view]", ".srp-river-results li"]:
            els = await page.query_selector_all(sel)
            print(f"  Selector '{sel}': {len(els)} elements")

        # Salva una piccola porzione dell'HTML per analisi
        body_html = await page.content()
        snippet = body_html[:3000]
        print("\n--- HTML snippet (first 3000 chars) ---")
        print(snippet)
        print("--- END HTML snippet ---")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
