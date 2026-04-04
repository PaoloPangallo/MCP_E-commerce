"""
Test del nuovo flusso: cerca item_id del venditore via API, poi invia messaggio con contesto.
"""
import asyncio, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONPATH", ".")

SELLER = "danpang-22"
MESSAGE = "[TEST AUTOMATICO] Ciao! Sono interessato ai vostri prodotti. Grazie!"
CDP_PORT = 9222

async def run():
    # 1. Setup app context
    from app.config.settings import settings
    from app.services.ebay import init_http_client, _get_oauth_token, _perform_search_request, get_client
    await init_http_client()

    # 2. Cerca item_id del venditore
    print(f"[1] Cerco item del venditore {SELLER} via eBay API...")
    client = get_client()
    token = await _get_oauth_token()
    filter_string = f"sellers:{{{SELLER}}}"
    data = await _perform_search_request(
        client=client, token=token, query="a",
        filter_string=filter_string, limit=1, offset=0
    )
    items = data.get("itemSummaries") or []
    if not items:
        print(f"  [FAIL] Nessun item trovato per {SELLER}!")
        return
    
    raw_id = items[0].get("itemId", "")
    numeric_id = raw_id.split("|")[1] if "|" in raw_id else raw_id
    item_title = items[0].get("title", "N/A")
    print(f"  [ok] Trovato item: {numeric_id} — {item_title}")

    # 3. Costruisci URL completo
    url = f"https://www.ebay.it/contact/sendmsg?recipient={SELLER}&item_id={numeric_id}&message_type_id=1"
    print(f"[2] URL di contatto: {url}")

    # 4. Playwright: naviga e invia
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        print(f"[3] Connessione CDP...")
        browser = await pw.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}")
        context = browser.contexts[0]
        page = await context.new_page()
        
        print(f"[4] Navigazione...")
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        await page.screenshot(path="tmp/shot_new_01_after_nav.png")
        print(f"    URL: {page.url}")

        print("[5] Aspetto textarea...")
        try:
            await page.wait_for_selector("#imageupload__sendmessage--textbox", state="visible", timeout=12000)
            print("  [ok] Textarea trovata!")
        except Exception as e:
            print(f"  [FAIL] {e}")
            await page.screenshot(path="tmp/shot_new_FAIL.png")
            await page.close()
            return

        print("[6] Inserisco messaggio...")
        ta = await page.query_selector("#imageupload__sendmessage--textbox")
        await ta.click(force=True, timeout=2000)
        await ta.fill(MESSAGE)
        await asyncio.sleep(0.5)
        await page.screenshot(path="tmp/shot_new_02_after_fill.png")
        print("  >> shot_new_02_after_fill.png")

        print("[7] Cerco bottone Invia...")
        submit = await page.query_selector("#imageupload__send--button")
        if not submit:
            print("  [FAIL] Bottone non trovato!")
            await page.close()
            return
        print("  [ok] Bottone trovato: #imageupload__send--button")

        print("[8] Invio tra 3 secondi... (Ctrl+C per annullare)")
        await asyncio.sleep(3)
        await page.evaluate("(el) => el.click()", submit)
        await asyncio.sleep(3)

        await page.screenshot(path="tmp/shot_new_03_after_submit.png")
        print(f"  >> shot_new_03_after_submit.png")
        print(f"[9] URL finale: {page.url}")
        print("[10] DONE!")
        await page.close()

asyncio.run(run())
