import asyncio
import logging
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO)

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating to ebay.it...")
        await page.goto("https://www.ebay.it")
        await page.wait_for_load_state("domcontentloaded")
        
        # Click cookie banner to be safe
        try:
            await page.click("#gdpr-banner-accept", timeout=3000)
        except Exception:
            pass
            
        print("Typing iphone 9 batteria...")
        await page.fill("#gh-ac", "iphone 9 batteria")
        await page.press("#gh-ac", "Enter")
        
        print("Waiting 5s for results...")
        await asyncio.sleep(5)
        
        js_extractor = '''() => {
            let el = document.getElementById('mainContent') || document.getElementById('srp-river-results') || document.querySelector('main');
            if (el) {
                return el.innerText.substring(0, 2000);
            }
            return document.body.innerText.substring(0, 1500);
        }'''
        
        text = await page.evaluate(js_extractor)
        print("EXTRACTED TEXT:")
        print("---")
        print(text)
        print("---")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
