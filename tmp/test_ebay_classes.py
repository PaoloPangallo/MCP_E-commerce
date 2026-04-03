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
            // Find all anchor tags that might be product links
            let links = [...document.querySelectorAll('a')];
            let texts = [];
            for (let i = 0; i < links.length; i++) {
                if (links[i].innerText && links[i].innerText.toLowerCase().includes('baggio')) {
                    texts.push(links[i].className + " -> " + links[i].innerText.replace(/\\n/g, ' '));
                }
            }
            return texts.join('\\n').substring(0, 2000);
        }'''
        
        text = await page.evaluate(js_extractor)
        print("EXTRACTED:")
        print(text)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
