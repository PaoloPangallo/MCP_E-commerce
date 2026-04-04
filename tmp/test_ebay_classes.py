import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.ebay.it")
        await page.wait_for_load_state("domcontentloaded")
        
        try:
            await page.click("#gdpr-banner-accept", timeout=3000)
        except Exception:
            pass
            
        await page.fill("#gh-ac", "maglia roberto baggio")
        await page.press("#gh-ac", "Enter")
        await asyncio.sleep(5)
        
        js_extractor = '''() => {
            let el = document.getElementById('mainContent') || document.getElementById('srp-river-results') || document.querySelector('main');
            return el ? el.innerText.substring(0, 1500) : document.body.innerText.substring(0, 1500);
        }'''
        
        text = await page.evaluate(js_extractor)
        print("EXTRACTED:")
        print(text)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
