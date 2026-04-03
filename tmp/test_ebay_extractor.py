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
            
        print("Typing maglia roberto baggio...")
        await page.fill("#gh-ac", "maglia roberto baggio")
        await page.press("#gh-ac", "Enter")
        
        print("Waiting 5s for results...")
        await asyncio.sleep(5)
        
        js_extractor = '''() => {
            let items = [...document.querySelectorAll('h1, .s-item__title, .s-item__price')];
            if (items.length > 0) {
                return items.map(el => el.innerText).join('\\n').substring(0, 2000);
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
