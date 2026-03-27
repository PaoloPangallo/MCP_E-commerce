"""
debug_playwright2.py — script diagnostico temporaneo per analizzare struttura item.
Esegui: python tmp/debug_playwright2.py
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from playwright.async_api import async_playwright

    url = "https://www.ebay.it/sch/i.html?_nkw=iphone+13&_ipg=24&LH_BIN=1"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(locale="it-IT", user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36"
        ))
        page = await context.new_page()
        await page.goto(url, timeout=30_000, wait_until="networkidle")

        items = await page.query_selector_all(".srp-river-results li")
        print(f"Total li items: {len(items)}")

        # Analizza i primi 3 item non-vuoti
        count = 0
        for item in items:
            if count >= 3:
                break
            html = await item.inner_html()
            if len(html) < 100:
                continue  # salta spacer/vuoti
            print(f"\n--- Item {count+1} HTML (first 1500 chars) ---")
            print(html[:1500])
            count += 1

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
