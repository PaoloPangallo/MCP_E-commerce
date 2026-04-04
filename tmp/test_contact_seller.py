"""
Test manuale del flusso contact_seller con screenshot ad ogni step.
"""
import asyncio, sys
from playwright.async_api import async_playwright

SELLER = "danpang-22"
MESSAGE = "[TEST AUTOMATICO] Ciao! Sono interessato ai tuoi prodotti. Grazie!"
CDP_PORT = 9222

async def run():
    async with async_playwright() as pw:
        print(f"[1] Connessione CDP su porta {CDP_PORT}...")
        try:
            browser = await pw.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}")
        except Exception as e:
            print(f"ERRORE CDP: {e}")
            sys.exit(1)

        context = browser.contexts[0]
        page = await context.new_page()
        print(f"[2] Nuova tab aperta.")

        url = f"https://www.ebay.it/contact/sendmsg?recipient={SELLER}&message_type_id=14"
        print(f"[3] Navigazione a: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        except Exception as e:
            print(f"  [warn] goto: {e}")

        print(f"[4] URL corrente: {page.url}")
        await page.screenshot(path="tmp/shot_01_after_nav.png")
        print("  >> screenshot: tmp/shot_01_after_nav.png")

        print("[5] Attendo textarea #imageupload__sendmessage--textbox ...")
        try:
            await page.wait_for_selector(
                "#imageupload__sendmessage--textbox",
                state="visible",
                timeout=10000,
            )
            print("  [ok] Textarea trovata!")
        except Exception as e:
            print(f"  [FAIL] Textarea non trovata: {e}")
            await page.screenshot(path="tmp/shot_FAIL_no_textarea.png")
            print("  >> screenshot: tmp/shot_FAIL_no_textarea.png")
            inputs = await page.evaluate(
                "() => Array.from(document.querySelectorAll('textarea,input,div[contenteditable]'))"
                ".map(e => e.id + '|' + (e.getAttribute('placeholder')||''))"
            )
            print("  Inputs:", inputs)
            await page.close()
            return

        print(f"[6] Inserisco messaggio...")
        textarea = await page.query_selector("#imageupload__sendmessage--textbox")
        await textarea.click(force=True, timeout=2000)
        await textarea.fill(MESSAGE)
        await asyncio.sleep(0.5)
        val = await textarea.input_value()
        print(f"  Testo nel campo: '{val}'")
        await page.screenshot(path="tmp/shot_02_after_fill.png")
        print("  >> screenshot: tmp/shot_02_after_fill.png")

        print("[7] Cerco bottone Invia...")
        submit = None
        for sel in [
            "#imageupload__send--button",
            "[data-testid='message-send-button']",
            "button[aria-label='Invia il messaggio']",
            "button.imageupload__sendbutton",
        ]:
            el = await page.query_selector(sel)
            if el:
                submit = el
                print(f"  [ok] Bottone trovato: {sel}")
                break

        if not submit:
            print("  [FAIL] Bottone Invia non trovato!")
            await page.screenshot(path="tmp/shot_FAIL_no_submit.png")
            print("  >> screenshot: tmp/shot_FAIL_no_submit.png")
            await page.close()
            return

        print("[8] Invio in 3 secondi... (Ctrl+C per annullare)")
        await asyncio.sleep(3)
        await page.evaluate("(el) => el.click()", submit)
        await asyncio.sleep(3)

        await page.screenshot(path="tmp/shot_03_after_submit.png")
        print(f"  >> screenshot: tmp/shot_03_after_submit.png")
        print(f"[9] URL dopo invio: {page.url}")
        print("[10] DONE")

        await asyncio.sleep(2)
        await page.close()

asyncio.run(run())
