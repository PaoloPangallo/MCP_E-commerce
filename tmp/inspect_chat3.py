import asyncio, json
from playwright.async_api import async_playwright

async def inspect():
    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = await context.new_page()
        url = "https://www.ebay.it/contact/sendmsg?recipient=papang_76&message_type_id=14"
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        except Exception as e:
            pass
        await asyncio.sleep(5)
        
        buttons = await page.evaluate('''() => {
            let res = [];
            for (let el of document.querySelectorAll("button, input[type='submit'], [role='button']")) {
                res.push({
                    tag: el.tagName,
                    id: el.id || null,
                    cls: (el.className||"").slice(0,80),
                    aria: el.getAttribute("aria-label"),
                    title: el.getAttribute("title"),
                    testid: el.getAttribute("data-testid"),
                    text: (el.innerText||"").trim().slice(0,40)
                });
            }
            return res;
        }''')
        
        inputs = await page.evaluate('''() => {
            let res = [];
            for (let el of document.querySelectorAll("textarea, input[type='text'], div[contenteditable]")) {
                res.push({
                    tag: el.tagName,
                    id: el.id || null,
                    cls: (el.className||"").slice(0,80),
                    ph: el.getAttribute("placeholder"),
                    aria: el.getAttribute("aria-label")
                });
            }
            return res;
        }''')
        
        result = {"buttons": buttons, "inputs": inputs}
        with open("tmp/inspect_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        await page.close()

asyncio.run(inspect())
print("done")
