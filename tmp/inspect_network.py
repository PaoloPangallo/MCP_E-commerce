"""
Ispezione del form eBay sendmsg: cerca token CSRF, field nascosti, network requests.
"""
import asyncio, json, sys, os
sys.path.insert(0, ".")
os.environ.setdefault("PYTHONPATH", ".")

SELLER = "danpang-22"

async def run():
    from app.config.settings import settings
    from app.services.ebay import init_http_client, _get_oauth_token, _perform_search_request, get_client
    await init_http_client()

    client = get_client()
    token = await _get_oauth_token()
    data = await _perform_search_request(
        client=client, token=token, query="a",
        filter_string=f"sellers:{{{SELLER}}}", limit=1, offset=0
    )
    items = data.get("itemSummaries") or []
    if not items:
        print("Nessun item trovato!")
        return
    raw_id = items[0].get("itemId", "")
    numeric_id = raw_id.split("|")[1] if "|" in raw_id else raw_id
    print(f"item_id: {numeric_id}")

    url = f"https://www.ebay.it/contact/sendmsg?recipient={SELLER}&item_id={numeric_id}&message_type_id=1"
    print(f"URL: {url}")

    from playwright.async_api import async_playwright
    
    # Intercetta le richieste di rete per vedere cosa viene inviato
    network_log = []

    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        page = await context.new_page()

        # Monitor delle richieste
        async def on_request(request):
            if "sendmsg" in request.url or "contact" in request.url or "message" in request.url.lower():
                if request.method in ("POST", "PUT"):
                    try:
                        body = request.post_data
                        network_log.append({
                            "url": request.url,
                            "method": request.method,
                            "headers": dict(request.headers),
                            "body": body
                        })
                    except Exception:
                        pass

        async def on_response(response):
            if "sendmsg" in response.url or ("contact" in response.url and response.status != 200):
                network_log.append({
                    "response_url": response.url,
                    "status": response.status,
                })

        page.on("request", on_request)
        page.on("response", on_response)

        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(4)
        await page.screenshot(path="tmp/shot_inspect_form.png")

        # Cerca hidden inputs e token
        hidden = await page.evaluate("""() => {
            const res = {};
            // Hidden inputs
            for (let el of document.querySelectorAll('input[type=hidden]')) {
                res['hidden_' + (el.name || el.id)] = el.value;
            }
            // Meta tags con CSRF
            for (let el of document.querySelectorAll('meta[name*=csrf], meta[name*=token]')) {
                res['meta_' + el.name] = el.content;
            }
            // Dataset del form
            const form = document.querySelector('form');
            if (form) {
                res['form_action'] = form.action;
                res['form_method'] = form.method;
            }
            return res;
        }""")
        print("Hidden fields:", json.dumps(hidden, indent=2))

        # Ora invia il messaggio e cattura le richieste
        ta = await page.query_selector("#imageupload__sendmessage--textbox")
        await ta.click(force=True)
        await ta.fill("[TEST ISPEZIONE] ciao")
        await asyncio.sleep(0.5)

        submit = await page.query_selector("#imageupload__send--button")
        await page.evaluate("(el) => el.click()", submit)
        await asyncio.sleep(3)
        await page.screenshot(path="tmp/shot_inspect_after_send.png")

        print(f"URL dopo invio: {page.url}")
        print(f"\nNetwork requests intercettate ({len(network_log)}):")
        for req in network_log:
            print(json.dumps(req, indent=2, default=str))

        await page.close()

asyncio.run(run())
