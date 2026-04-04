import asyncio
from playwright.async_api import async_playwright

async def inspect():
    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = await context.new_page()
        url = "https://www.ebay.it/contact/sendmsg?recipient=papang_76&message_type_id=14"
        print(f"NAV: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        except Exception as e:
            print(f"goto err: {e}")
        await asyncio.sleep(5)
        
        # Dump all buttons with identifiers
        buttons = await page.evaluate('''() => {
            let res = [];
            for (let el of document.querySelectorAll("button, input[type='submit'], [role='button']")) {
                res.push(
                    "tag=" + el.tagName +
                    " id=" + (el.id||"") +
                    " class=" + (el.className||"").slice(0,60) +
                    " aria=" + (el.getAttribute("aria-label")||"") +
                    " title=" + (el.getAttribute("title")||"") +
                    " testid=" + (el.getAttribute("data-testid")||"") +
                    " text=" + (el.innerText||"").trim().slice(0,40)
                );
            }
            return res;
        }''')
        
        print(f"BUTTONS ({len(buttons)}):")
        for b in buttons:
            print(" >", b)
        
        # Dump textareas
        inputs = await page.evaluate('''() => {
            let res = [];
            for (let el of document.querySelectorAll("textarea, input[type='text'], div[contenteditable]")) {
                res.push(
                    "tag=" + el.tagName +
                    " id=" + (el.id||"") +
                    " class=" + (el.className||"").slice(0,60) +
                    " ph=" + (el.getAttribute("placeholder")||"") +
                    " aria=" + (el.getAttribute("aria-label")||"")
                );
            }
            return res;
        }''')
        print(f"INPUTS ({len(inputs)}):")
        for i in inputs:
            print(" >", i)
        
        await page.close()
        print("DONE")

asyncio.run(inspect())
