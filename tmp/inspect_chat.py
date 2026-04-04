import asyncio
from playwright.async_api import async_playwright

async def inspect():
    async with async_playwright() as pw:
        try:
            print("Connecting to CDP on 9222...")
            browser = await pw.chromium.connect_over_cdp("http://localhost:9222")
            context = browser.contexts[0]
            page = await context.new_page()
            url = "https://www.ebay.it/contact/sendmsg?recipient=papang_76&message_type_id=14"
            print(f"Navigating to {url}...")
            await page.goto(url, wait_until="networkidle", timeout=15000)
            await asyncio.sleep(4)
            print("Scanning DOM for buttons...")
            buttons = await page.evaluate('''() => {
                let els = document.querySelectorAll('button');
                let res = [];
                for (let el of els) {
                    let h = el.innerHTML.toLowerCase();
                    let t = el.innerText.toLowerCase();
                    let testid = el.getAttribute('data-testid');
                    // We only want the ones likely to be sending
                    if (h.includes('svg') && !h.includes('search') || t.includes('invia') || testid) {
                        res.push({
                            id: el.id,
                            class: el.className,
                            title: el.getAttribute('title'),
                            aria: el.getAttribute('aria-label'),
                            testid: testid
                        });
                    }
                }
                return res;
            }''')
            
            print(f"Found {len(buttons)} candidates:")
            import pprint
            pprint.pprint(buttons)
            await page.close()
            print("Done.")
        except Exception as e:
            import traceback
            traceback.print_exc()

asyncio.run(inspect())
