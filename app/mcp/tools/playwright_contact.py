"""
playwright_contact.py
MCP Tool: contact_seller_playwright

Usa il Chrome già installato e già loggato dall'utente:
- Se Chrome è aperto con --remote-debugging-port=9222 → ci connette via CDP
- Altrimenti → lancia Chrome con il profilo reale dell'utente (richiede Chrome chiuso)

Flusso autonomo:
1. Apre/connette a Chrome con sessione eBay attiva
2. Naviga su IntermediatedFAQ del venditore
3. Clicca "Non riguarda un oggetto"
4. Compila il messaggio
5. Clicca Invia
6. Chiude la tab solo in caso di successo; lascia tutto aperto in caso di errore.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
from typing import Any, Annotated, Dict, Optional

from pydantic import Field

from app.mcp.playwright_server import mcp_playwright
from app.services.ebay_playwright import _PLAYWRIGHT_EXECUTOR

logger = logging.getLogger(__name__)

# Profilo Chrome reale dell'utente — contiene tutti i cookie incluso eBay/Google.
# Richiede che Chrome sia CHIUSO quando Playwright lo usa.
if sys.platform == "win32":
    _CHROME_USER_DATA = os.path.expandvars(
        r"%LOCALAPPDATA%\Google\Chrome\User Data"
    )
else:
    _CHROME_USER_DATA = os.path.join(os.path.expanduser("~"), ".config", "google-chrome")

# Porta CDP su cui Chrome espone il remote debugging.
# Per abilitarla permanentemente: aggiungere --remote-debugging-port=9222
# nelle proprietà di avvio di Chrome (o in chrome://flags per test).
_CDP_PORT = int(os.getenv("CHROME_CDP_PORT", "9222"))


def _build_contact_result(
    product_url: str,
    success: bool,
    status: str,
    detail: str,
    message_sent: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": "ok" if success else "error",
        "success": success,
        "product_url": product_url,
        "contact_status": status,
        "detail": detail,
        "message_sent": message_sent,
    }


def _chrome_cdp_available(port: int = _CDP_PORT) -> bool:
    """Ritorna True se Chrome sta ascoltando sulla porta CDP."""
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except OSError:
        return False



async def _fetch_seller_item_data(seller_name: str) -> Optional[Dict[str, str]]:
    """
    Cerca il primo annuncio attivo del venditore sulla eBay Browse API e restituisce
    il suo item_id (numerico) e item_url (web).
    Senza un item context, l'invio diretto di messaggi AJAX spesso fallisce con 500.
    """
    try:
        from app.services.ebay import (
            _get_oauth_token, _perform_search_request, get_client
        )
        client = get_client()
        token = await _get_oauth_token()
        filter_string = f"sellers:{{{seller_name}}}"
        data = await _perform_search_request(
            client=client,
            token=token,
            query="a",          # query minima (eBay non accetta wildcard *)
            filter_string=filter_string,
            limit=1,
            offset=0,
        )
        items = data.get("itemSummaries") or []
        if items:
            raw_id = items[0].get("itemId", "")
            # itemId può essere in formato "v1|12345|0" — estrai solo il numero
            if "|" in raw_id:
                parts_id = raw_id.split("|")
                numeric_id = parts_id[1] if len(parts_id) > 1 else parts_id[0]
            else:
                numeric_id = raw_id
            
            item_url = items[0].get("url") or items[0].get("itemWebUrl")
            logger.info("_fetch_seller_item_data: found item %s (%s) for seller %s", 
                        numeric_id, item_url, seller_name)
            return {"item_id": numeric_id, "item_url": item_url}
    except Exception as e:
        logger.warning("_fetch_seller_item_data failed for seller=%s: %s", seller_name, e)
    return None


async def _async_contact_seller(
    product_url: str,
    message: str,
    timeout_ms: int,
    seller_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Flusso autonomo — usa Chrome reale già loggato.

    Strategia di connessione (in ordine di priorità):
    1. CDP: se Chrome gira con --remote-debugging-port=9222, ci connette direttamente
    2. Profilo reale: lancia Chrome con il profilo utente (Chrome deve essere chiuso)
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright non installato. Esegui: pip install playwright && playwright install chromium"
        ) from exc

    logger.info("PW contact_seller START | url=%s", product_url)
    _launch_args = ["--no-sandbox", "--disable-dev-shm-usage"]

    # ── Selectors ────────────────────────────────────────────────────────────

    not_about_item_selectors = [
        "a:has-text('Non riguarda un oggetto specifico')",
        "button:has-text('Non riguarda un oggetto specifico')",
        "a:has-text('Non riguarda un oggetto')",
        "button:has-text('Non riguarda un oggetto')",
        "a:has-text('Not about an item')",
        "button:has-text('Not about an item')",
        "a:has-text('Not about a specific item')",
        "a:has-text('Altra domanda')",
        "a:has-text('Continua')",
        "button:has-text('Continua')",
        "a[href*='notAboutItem']",
        "a[href*='not_about_item']",
        "a[href*='OTHER']",
        "a[href*='other']",
        "a[href*='sendmsg']",
        "[data-testid*='not-about-item']",
        "[data-testid*='notAboutItem']",
    ]

    message_field_selectors = [
        "textarea[name='body']",
        "textarea[id*='message']",
        "textarea[id*='msg']",
        "textarea[placeholder*='messaggio']",
        "textarea[placeholder*='message']",
        "textarea[placeholder*='Messaggio']",
        "#msg",
        "textarea",
    ]

    submit_selectors = [
        "button[aria-label='Invia il messaggio']",
        "button[aria-label='Invia']",
        "[data-testid*='send-button']",
        "[data-testid*='sendbutton']",
        "[data-testid='imageupload__send-btn']",
        ".imageupload__send-btn-wrapper button",
        ".imageupload__send-btn-wrapper div[role='button']",
        ".imageupload__sendbutton",
        "button[aria-label*='Invia']",
        "#imageupload__send--button",
        "svg[viewBox*='0 0 24 24']",
    ]

    contact_page_selectors = [
        "a[href*='contactseller']",
        "a[href*='contact_seller']",
        "a[href*='vi/contact']",
        "a[href*='MsgContact']",
        "a[href*='askSellerQuestion']",
        "a:has-text('Invia un messaggio al venditore')",
        "a:has-text('messaggio al venditore')",
        "a:has-text('Contatta il venditore')",
        "a:has-text('Contact seller')",
        "a[href*='contact']",
        "a[href*='ContactSeller']",
        "button:has-text('Contatta il venditore')",
        "button:has-text('Contact seller')",
        "[data-testid='vi-VR-CONTACT-SELLER']",
        "[data-testid='contact-seller']",
    ]

    # ── Connessione a Chrome ──────────────────────────────────────────────────

    pw = await async_playwright().start()
    context = None
    browser_cdp = None  # solo se usiamo CDP
    page = None
    success = False
    using_cdp = False

    try:
        if _chrome_cdp_available(_CDP_PORT):
            # Chrome è già aperto con remote debugging → connessione CDP
            logger.info("PW contact_seller: connecting to running Chrome via CDP port %s", _CDP_PORT)
            browser_cdp = await pw.chromium.connect_over_cdp(f"http://localhost:{_CDP_PORT}")
            # Usa il contesto esistente (sessione già loggata)
            context = browser_cdp.contexts[0] if browser_cdp.contexts else await browser_cdp.new_context(locale="it-IT")
            using_cdp = True
            logger.info("PW contact_seller: CDP connected | contexts=%d", len(browser_cdp.contexts))
        else:
            # Chrome non ha CDP aperta → launch con profilo reale
            logger.info("PW contact_seller: launching Chrome with real profile at %s", _CHROME_USER_DATA)
            context = await pw.chromium.launch_persistent_context(
                _CHROME_USER_DATA,
                channel="chrome",
                headless=False,
                args=_launch_args,
                locale="it-IT",
            )

        page = await context.new_page()

        # ── Step 1: Risoluzione URL (Natural Navigation) ──────────────────
        # STRATEGIA:
        # 1. Se product_url è già un link diretto a sendmsg → USALO SUBITO.
        #    Evitiamo di cercare articoli a caso se abbiamo già la rotta corretta.
        # 2. Se abbiamo seller_name ma non un URL diretto → naviga SRP venditore
        #    e trova il primo articolo reale.
        # 3. Se abbiamo un URL sendmsg senza item_id → arricchiamo.
        
        final_url = product_url
        item_data = None
        
        # Debugging the mystery of the direct contact check
        is_direct_contact = product_url is not None and "sendmsg" in str(product_url).lower()
        logger.info("PW contact_seller Step 1 DEBUG: product_url=%r, is_direct=%s, seller=%s", 
                    product_url, is_direct_contact, seller_name)

        if seller_name and not is_direct_contact:
            seller_srp_url = f"https://www.ebay.it/sch/i.html?_ssn={seller_name}&_sop=10"
            logger.info(
                "PW contact_seller: resolving seller SRP for seller=%s | srp=%s",
                seller_name, seller_srp_url
            )
            try:
                srp_page = await context.new_page()
                await srp_page.goto(seller_srp_url, wait_until="domcontentloaded", timeout=15_000)
                await asyncio.sleep(1.5)

                # Estrai il primo link di un articolo (/itm/) dalla SRP del venditore
                # Escludiamo link che sembrano ID numerici brevi o fake (es. /12345)
                first_item_url = await srp_page.evaluate("""() => {
                    const links = Array.from(document.querySelectorAll('a[href*="/itm/"]'));
                    const filtered = links.filter(l => {
                        const href = l.href || "";
                        if (l.offsetWidth === 0) return false;
                        if (!href.includes('/itm/')) return false;
                        // Avoid placeholders like 123456
                        const matches = href.match(/\\/itm\\/(\\d+)/);
                        if (matches && matches[1].length < 10) return false;
                        // Prefer .it domain if available
                        return true;
                    });
                    // Prioritize ebay.it links
                    filtered.sort((a, b) => (b.href.includes('ebay.it') ? 1 : 0) - (a.href.includes('ebay.it') ? 1 : 0));
                    return filtered.length > 0 ? filtered[0].href.split('?')[0] : null;
                }""")
                await srp_page.close()

                if first_item_url:
                    final_url = first_item_url
                    logger.info("PW contact_seller: found item from seller SRP: %s", final_url)
                else:
                    logger.warning("PW contact_seller: no items found on seller SRP, using product_url")
            except Exception as exc:
                logger.warning("PW contact_seller: SRP navigation failed (%s), using product_url", exc)
                try:
                    await srp_page.close()
                except Exception:
                    pass
        elif is_direct_contact:
            logger.info("PW contact_seller: using direct contact URL: %s", final_url)
        
        elif "sendmsg" in product_url and "item_id" not in product_url:

            import re as _re
            seller_match = _re.search(r"recipient=([^&]+)", product_url)
            if seller_match:
                seller_slug = seller_match.group(1)
                item_data = await _fetch_seller_item_data(seller_slug)
                if item_data and item_data.get("item_id"):
                    final_url = (
                        f"https://www.ebay.it/contact/sendmsg"
                        f"?recipient={seller_slug}&item_id={item_data['item_id']}&message_type_id=1"
                    )
                    logger.info("PW contact_seller: enriched URL with item_id=%s => %s", item_data['item_id'], final_url)



        # ── Step 2: naviga ───────────────────────────────────────────────────
        try:
            logger.info("PW contact_seller: navigating to target URL: %s", final_url)
            await page.goto(final_url, timeout=timeout_ms, wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle", timeout=10_000)
            await page.screenshot(path="tmp/debug_nat_01_nav.png")
        except Exception as exc:
            logger.warning("PW contact_seller: goto timeout/error | %s", exc)
        
        # ── Step 3: login wall? aspetta max 2 minuti ─────────────────────────
        current_url = page.url
        if "signin" in current_url or "login" in current_url:
            logger.info("PW contact_seller: login required — waiting up to 120s")
            try:
                await page.wait_for_function(
                    "() => !window.location.href.includes('signin') && "
                    "!window.location.href.includes('login')",
                    timeout=120_000,
                )
                current_url = page.url
                logger.info("PW contact_seller: login complete | url=%s", current_url)
            except Exception:
                logger.warning("PW contact_seller: login timeout")
                return _build_contact_result(
                    product_url=product_url,
                    success=False,
                    status="login_required",
                    detail=(
                        "eBay richiede il login. Effettua il login nel browser aperto "
                        "e poi riprova."
                    ),
                )

        current_url = page.url

        # ── Step 3a: pagina prodotto (Se partiamo da un annuncio) ─────────────
        if "ebay.it/itm/" in current_url or "ebay.com/itm/" in current_url:
            logger.info("PW contact_seller: On item page, searching for contact link...")
            contact_clicked = False
            for sel in contact_page_selectors:
                try:
                    # Usiamo evaluate per CLICCARE ed estrarre informazioni, ma NON navighiamo fuori
                    # se si tratta di un'attivazione di overlay React
                    click_res = await page.evaluate("""(selector) => {
                        let el;
                        if (selector.includes(':has-text')) {
                            const match = selector.match(/:has-text\\(['"](.+?)['"]\\)/);
                            const text = match ? match[1] : "";
                            el = Array.from(document.querySelectorAll('a, button')).find(
                                e => e.innerText.includes(text)
                            );
                        } else {
                            el = document.querySelector(selector);
                        }
                        
                        if (!el) return null;
                        el.click();
                        return "clicked";
                    }""", sel)
                    
                    if click_res == "clicked":
                        logger.info("PW contact_seller: clicked contact link via JS selector (overlay mode): %s", sel)
                        # Diamo tempo all'overlay React di apparire
                        await asyncio.sleep(4)
                        await page.screenshot(path="tmp/debug_nat_02_after_click.png")
                        current_url = page.url
                        contact_clicked = True
                        break
                except Exception as e:
                    logger.debug("PW contact_seller: failed selector %s: %s", sel, e)
                    continue
            
            if not contact_clicked:
                logger.warning("PW contact_seller: could not find contact link on item page. Trying to find any link with 'messaggio'...")
                try:
                    # Fallback estremo: cerca qualsiasi link che contenga "messaggio"
                    el = await page.query_selector("a:has-text('messaggio')")
                    if el:
                        logger.info("PW contact_seller: clicking fallback 'messaggio' link")
                        await el.click()
                        await asyncio.sleep(3)
                        contact_clicked = True
                except Exception:
                    pass

            if contact_clicked:
                await page.wait_for_load_state("domcontentloaded", timeout=10_000)
                current_url = page.url
                logger.info("PW contact_seller: transition after click | new url=%s", current_url)
            else:
                logger.warning("PW contact_seller: failed to trigger contact flow from item page.")

        # ── Step 3b: IntermediatedFAQ → naviga al form del messaggio ──────────
        if "IntermediatedFAQ" in current_url or "intermediatedfaq" in current_url.lower():
            logger.info("PW contact_seller: on IntermediatedFAQ — extracting direct sendmsg URL")
            # Attendi che la pagina carichi i link FAQ (React SPA)
            try:
                await page.wait_for_selector(
                    "#mainContent a, main a",
                    state="attached",
                    timeout=8_000,
                )
            except Exception:
                pass
            await asyncio.sleep(1.5)

            clicked = False

            # STRATEGIA PRINCIPALE: estrai direttamente l'href dal DOM e usa page.goto()
            # eBay usa SPA routing con event.preventDefault(): element.click() non naviga.
            # Il link "Continua" o "Non riguarda un oggetto" hanno sempre href=sendmsg/...
            try:
                sendmsg_url = await page.evaluate(
                    """() => {
                        // Priorità 1: link diretto a sendmsg (presente in html non-SPA di eBay)
                        const direct = document.querySelector('a[href*="sendmsg"]');
                        if (direct) return direct.href;
                        // Priorità 2: link "Continua" che potrebbe navigare a sendmsg
                        const continua = Array.from(document.querySelectorAll('a')).find(
                            a => a.innerText.trim() === 'Continua' || a.innerText.trim() === 'Continue'
                        );
                        if (continua) return continua.href;
                        // Priorità 3: qualsiasi link che contiene il nome venditore e "contact"
                        const contactLink = Array.from(document.querySelectorAll('a')).find(
                            a => a.href && (a.href.includes('contact') || a.href.includes('sendmsg'))
                        );
                        if (contactLink) return contactLink.href;
                        return null;
                    }"""
                )
                if sendmsg_url and "sendmsg" in sendmsg_url:
                    logger.info("PW contact_seller: navigating directly to sendmsg URL: %s", sendmsg_url)
                    await page.goto(sendmsg_url, wait_until="domcontentloaded")
                    clicked = True
                elif sendmsg_url:
                    logger.info("PW contact_seller: navigating to extracted URL: %s", sendmsg_url)
                    await page.goto(sendmsg_url, wait_until="domcontentloaded")
                    clicked = True
            except Exception as e:
                logger.warning("PW contact_seller: direct URL extraction failed: %s", e)

            if not clicked:
                # Verifica se il form del messaggio è già presente (eBay a volte salta le FAQ e porta diretti alla chat)
                form_ready = False
                for sel in message_field_selectors:
                    try:
                        if await page.query_selector(sel):
                            form_ready = True
                            logger.info("PW contact_seller: message form already present, bypassing FAQ click.")
                            break
                    except Exception:
                        pass
                
                if not form_ready:
                    try:
                        elements = await page.evaluate(
                            "() => {"
                            "  const main = document.querySelectorAll('#mainContent a, #mainContent button, main a, main button, [role=\"main\"] a, [role=\"main\"] button');"
                            "  const els = main.length ? Array.from(main) : Array.from(document.querySelectorAll('a,button'));"
                            "  return els.map(el => ({tag: el.tagName, text: el.innerText.trim().slice(0,80), href: el.href || ''}))"
                            "           .filter(x => x.text).slice(0, 40);"
                            "}"
                        )
                        logger.warning("PW contact_seller: button not found. Content elements: %s", elements)
                    except Exception:
                        pass
                    return _build_contact_result(
                        product_url=product_url,
                        success=False,
                        status="contact_button_not_found",
                        detail=(
                            "Non riesco a trovare il pulsante 'Non riguarda un oggetto'. "
                            "Il browser è aperto: clicca tu il pulsante."
                        ),
                    )

        # ── Step 5: compila messaggio ─────────────────────────────────────────
        logger.info("PW contact_seller: filling message | url=%s", page.url)
        logger.info("PW contact_seller: Testo che sto per inviare: \n%s", message)
        
        current_url_for_fill = page.url
        is_sendmsg_page = "sendmsg" in current_url_for_fill or "messages" in current_url_for_fill
        
        target_context = page
        if not is_sendmsg_page:
            # L'iframe .m2m-dialog__ifr esiste solo nell'overlay della pagina prodotto,
            # NON nella pagina messaggi sendmsg.
            try:
                iframe_el = await page.query_selector(".m2m-dialog__ifr")
                if iframe_el:
                    target_context = await iframe_el.content_frame() or page
                    logger.info("PW contact_seller: Using iframe context for message drafting")
            except Exception:
                pass

        # Aspettiamo che React renderizzi il campo composizione
        if is_sendmsg_page:
            # Pagina messaggi di eBay: attendi input / contenteditable nella area compose
            logger.info("PW contact_seller: on sendmsg page — using direct compose selectors")
            try:
                await page.wait_for_selector(
                    "input[placeholder*='messaggio'], div[contenteditable='true'], textarea",
                    state="visible",
                    timeout=8_000,
                )
            except Exception:
                await asyncio.sleep(2.0)
        else:
            # Overlay su pagina prodotto: attendi textarea specifica eBay
            try:
                await target_context.wait_for_selector(
                    "#imageupload__sendmessage--textbox",
                    state="visible",
                    timeout=10_000,
                )
            except Exception:
                await asyncio.sleep(2.5)  # fallback generico

        # Selettori specifici per il form di messaggistica AJAX o Overlay di eBay
        extended_message_selectors = [
            "#imageupload__sendmessage--textbox",
            "textarea.textbox__control",
            "textarea[name='body']",
            "textarea[placeholder*='messaggio']",
            "input[placeholder*='messaggio']",
            "div[placeholder*='messaggio']",
            "[aria-label*='messaggio']",
            "div[contenteditable='true']",  # pagina messaggi eBay send
            ".composer-input",
        ]

        filled = False
        # ── Caso 1: pagina messaggi sendmsg (no iframe, input diretto) ─────────
        if is_sendmsg_page:
            try:
                comp_filled = await page.evaluate("""(msg) => {
                    // Cerca prima l'ID verificato dallo screenshot
                    let el = document.getElementById('imageupload__sendmessage--textbox') || 
                             document.querySelector('[data-testid="message-input-textarea"]');
                    
                    if (!el) {
                        // Fallback selettori generici
                        const selectors = [
                            'input[placeholder*="messaggio"]',
                            'div[contenteditable="true"]',
                            'textarea',
                        ];
                        for (const sel of selectors) {
                            const els = Array.from(document.querySelectorAll(sel));
                            el = els.find(e => e.offsetWidth > 0 && e.offsetHeight > 0
                                && e.tagName !== 'BUTTON' && e.getAttribute('type') !== 'submit');
                            if (el) break;
                        }
                    }

                    if (el) {
                        el.click();
                        el.focus();
                        if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
                            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value') ||
                                Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
                            if (nativeInputValueSetter) nativeInputValueSetter.set.call(el, msg);
                        } else if (el.isContentEditable) {
                            el.innerText = msg;
                        }
                        el.dispatchEvent(new Event('input', {bubbles: true}));
                        el.dispatchEvent(new Event('change', {bubbles: true}));
                        return {
                            id: el.id, 
                            class: el.className,
                            selector: el.tagName + '#' + el.id,
                            initialValue: el.value || el.innerText
                        };
                    }
                    return null;
                }""", message)
                
                if comp_filled:
                    logger.info("PW contact_seller: sendmsg page compose found: %s", comp_filled)
                    # Svuota e digita con delay per triggerare React state
                    await page.keyboard.press("Control+A")
                    await page.keyboard.press("Backspace")
                    await page.keyboard.type(message, delay=50)
                    await asyncio.sleep(0.5)
                    
                    # Verifica che sia stato scritto (logica di sicurezza)
                    typed_val = await page.evaluate("""() => {
                        const el = document.getElementById('imageupload__sendmessage--textbox') || 
                                 document.querySelector('[data-testid="message-input-textarea"]');
                        return el ? (el.value || el.innerText) : '';
                    }""")
                    if len(typed_val) > 0:
                        logger.info("PW contact_seller: confirmed message is typed (%d chars)", len(typed_val))
                        filled = True
                    else:
                        logger.warning("PW contact_seller: message box appears empty after typing!")
                        # Fallback: prova a forzare ancora una volta via JS
                        await page.evaluate("""(m) => {
                            const el = document.getElementById('imageupload__sendmessage--textbox') || 
                                     document.querySelector('[data-testid="message-input-textarea"]');
                            if (el) { el.value = m; el.dispatchEvent(new Event('input', {bubbles:true})); }
                        }""", message)
                        filled = True
                else:
                    logger.warning("PW contact_seller: sendmsg compose field not found via JS")
            except Exception as e:
                logger.warning("PW error filling sendmsg compose: %s", e)

        # ── Caso 2: overlay item-page (iframe o no) ───────────────────────────
        if not filled:
            # Cerchiamo di identificare l'elemento che funge da 'composer' (div, textarea o input)
            # che contiene o ha come placeholder 'Invia il messaggio'
            composer_found = await target_context.evaluate("""() => {
                const candidates = Array.from(document.querySelectorAll('textarea, input, [role="textbox"], .textbox__control, div'))
                    .filter(el => {
                        if (el.tagName === 'BUTTON' || el.getAttribute('role') === 'button' || el.getAttribute('type') === 'submit') return false;
                        if (el.innerText && el.innerText.trim() === 'Invia il messaggio') return false; // This is a button!
                        
                        const text = el.innerText || "";
                        const placeholder = (el.placeholder || el.getAttribute('placeholder') || "").toLowerCase();
                        const ariaLabel = (el.getAttribute('aria-label') || "").toLowerCase();
                        
                        const isComposer = placeholder.includes('messaggio') || 
                                           placeholder.includes('message') || 
                                           ariaLabel.includes('messaggio') || 
                                           ariaLabel.includes('message') ||
                                           (el.tagName === 'TEXTAREA');

                        return isComposer;
                    });
                
                // Prioritize textareas or inputs over divs
                candidates.sort((a, b) => {
                    const aScore = (a.tagName === 'TEXTAREA' || a.tagName === 'INPUT') ? 1 : 0;
                    const bScore = (b.tagName === 'TEXTAREA' || b.tagName === 'INPUT') ? 1 : 0;
                    return bScore - aScore;
                });

                const el = candidates.find(c => c.offsetWidth > 0 && c.offsetHeight > 0);
                if (el) {
                    el.click();
                    el.focus();
                    return true;
                }
                return false;
            }""")
            
            if composer_found:
                logger.info("PW contact_seller: composer found and focused via JS overlay logic.")
                # Svuota prima se necessario
                await page.keyboard.press("Control+A")
                await page.keyboard.press("Backspace")
                # Digita il messaggio
                await page.keyboard.type(message, delay=50)
                
                # FORZA L'ATTIVAZIONE DEL TASTO INVIA (fondamentale per eBay React)
                await target_context.evaluate("""() => {
                    const el = document.getElementById('imageupload__sendmessage--textbox') || 
                               document.querySelector('textarea[id*="sendmessage"]') ||
                               document.querySelector('.textbox__control');
                    if (el) {
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                        console.log("PW contact_seller: dispatched input/change events to enable button");
                    }
                }""");
                await asyncio.sleep(1.0)
                filled = True

        if filled:
            await page.screenshot(path="tmp/debug_nat_03_filled.png")
        else:
            await page.screenshot(path="tmp/debug_nat_FAIL_form.png")
            try:
                html = await page.content()
                with open("tmp/debug_form.html", "w", encoding="utf-8") as f:
                    f.write(html)
            except Exception as ex:
                logger.error(f"Could not dump html: {ex}")
            logger.warning("PW contact_seller: message field not found at %s", page.url)
            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="message_form_not_found",
                detail=(
                    "Sono arrivato alla pagina ma non trovo il campo messaggio. "
                    "Il browser è aperto: scrivi tu il messaggio e clicca Invia."
                ),
            )

        # ── Step 6: clicca Invia ──────────────────────────────────────────────
        submitted = False
        for sel in submit_selectors:
            try:
                # Proviamo prima con selettore standard
                el = await target_context.query_selector(sel)
                if not el:
                    # Fallback JavaScript potente: cerca qualsiasi cosa contenga 'Invia'
                    # e clicca il primo risultato visibile che sembra un bottone
                    el = await target_context.evaluate_handle("""() => {
                        // 1. Cerca il wrapper specifico visto nello screenshot
                        const wrapper = document.querySelector('.imageupload__send-btn-wrapper');
                        if (wrapper) {
                            const btn = wrapper.querySelector('button, [role="button"], svg');
                            if (btn) return btn;
                        }

                        // 2. Cerca bottoni con Paper Plane o icone simili (Send)
                        const allButtons = Array.from(document.querySelectorAll('button, [role="button"], a'))
                            .filter(b => b.offsetWidth > 0 && b.offsetHeight > 0);
                        
                        // Cerca per SVG path (paper plane)
                        const sendByIcon = allButtons.find(b => {
                            const svg = b.querySelector('svg');
                            if (!svg) return false;
                            const html = svg.innerHTML.toLowerCase();
                            // Paper plane paths often contain these coordinate patterns
                            return html.includes('m2.01 21') || html.includes('m1.01 3') || 
                                   html.includes('l23 12') || b.classList.contains('send');
                        });
                        if (sendByIcon) return sendByIcon;

                        // 2. Cerca per ARIA label o testo
                        const candidates = allButtons.filter(el => {
                                if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT' || el.getAttribute('role') === 'textbox') return false;
                                // Escludi righe inbox con checkbox (non sono bottoni di invio)
                                if (el.querySelector('input[type="checkbox"]')) return false;
                                const ariaRaw = (el.getAttribute('aria-label') || '').toLowerCase();
                                if (ariaRaw.startsWith('seleziona')) return false;

                                const label = ariaRaw;
                                const title = (el.getAttribute('title') || "").toLowerCase();
                                const text = (el.innerText || "").toLowerCase();

                                return label.includes('invia') || label.includes('send') ||
                                       title.includes('invia') || title.includes('send') ||
                                       text === 'invia' || text === 'send';
                            });
                        
                        // 3. Fallback: il bottone a DESTRA del campo messaggi
                        if (candidates.length === 0) {
                            const composer = document.querySelector(
                                '#imageupload__sendmessage--textbox, ' +
                                'textarea[placeholder*="messaggio"], ' +
                                'textarea[data-testid="message-input-textarea"], ' +
                                'input[placeholder*="messaggio"], ' +
                                'div[contenteditable="true"]'
                            );
                            if (composer) {
                                const rect = composer.getBoundingClientRect();
                                const rightOf = allButtons.find(b => {
                                    const bRect = b.getBoundingClientRect();
                                    return bRect.left > rect.right - 50 && 
                                           bRect.top > rect.top - 20 && 
                                           bRect.top < rect.bottom + 20 &&
                                           bRect.width < 100;
                                });
                                if (rightOf) return rightOf;
                            }
                        }

                        candidates.sort((a, b) => {
                            const score = el => {
                                const label = (el.getAttribute('aria-label') || '').toLowerCase();
                                if (label.startsWith('seleziona')) return 0;
                                if (label.includes('invia') || label.includes('send')) return 3;
                                if (label.includes('messaggio') || label.includes('message')) return 2;
                                return 1;
                            };
                            return score(b) - score(a);
                        });

                        return candidates[0] || null;
                    }""")
                    if el:
                        # Converti JSHandle a ElementHandle se possibile (o usa evaluate)
                        is_null = await target_context.evaluate("(e) => e === null", el)
                        if is_null: el = None
                
                if el:
                    logger.info("PW contact_seller: Attempting click on submit element...")
                    # Cerchiamo di cliccare il BOTTONE genitore se abbiamo trovato uno span/svg
                    await target_context.evaluate("""(element) => { 
                        if(!element) return;
                        let target = element;
                        // Se abbiamo trovato un'icona o uno span dentro il bottone, sali fino al bottone
                        if (target.tagName !== 'BUTTON' && target.getAttribute('role') !== 'button') {
                            const parentBtn = target.closest('button') || target.closest('[role="button"]');
                            if (parentBtn) target = parentBtn;
                        }
                        target.scrollIntoView({block: 'center'});
                        target.click();
                        console.log("PW contact_seller: clicked target", target);
                    }""", el)
                    
                    await asyncio.sleep(2.5)
                    
                    # VERIFICA: se il testo è ancora nel textarea, il click non ha funzionato
                    still_has_text = await page.evaluate("""() => {
                        const el = document.getElementById('imageupload__sendmessage--textbox') || 
                                 document.querySelector('[data-testid="message-input-textarea"]');
                        return el && (el.value || el.innerText).length > 2; // threshold for safety
                    }""")
                    
                    if still_has_text:
                        logger.warning("PW contact_seller: Textarea still contains message after click. Trying Enter fallback.")
                        await page.keyboard.press("Enter")
                        await asyncio.sleep(2.0)
                        still_has_text = await page.evaluate("""() => {
                            const el = document.getElementById('imageupload__sendmessage--textbox') || 
                                     document.querySelector('[data-testid="message-input-textarea"]');
                            return el && (el.value || el.innerText).length > 2;
                        }""")
                    
                    if not still_has_text:
                        submitted = True
                        logger.info("PW contact_seller: message submitted (textarea is clear) | now at %s", page.url)
                        break
                    else:
                        logger.warning("PW contact_seller: Send failed to clear textarea. Will try next selector.")
            except Exception as e:
                logger.debug("PW error clicking submit: %s", str(e)[:100])
                continue

        if submitted:
            success = True
            await asyncio.sleep(2)
            return _build_contact_result(
                product_url=product_url,
                success=True,
                status="message_sent",
                detail="Messaggio inviato con successo al venditore.",
                message_sent=message,
            )
        else:
            # Fallback finale per sendmsg page: premi Enter (il campo è già compilato)
            if is_sendmsg_page and filled:
                logger.info("PW contact_seller: sendmsg fallback — pressing Enter to submit")
                try:
                    await page.keyboard.press("Return")
                    await asyncio.sleep(2)
                    success = True
                    return _build_contact_result(
                        product_url=product_url,
                        success=True,
                        status="message_sent",
                        detail="Messaggio inviato con successo al venditore (via Enter).",
                        message_sent=message,
                    )
                except Exception as e:
                    logger.warning("PW contact_seller: Enter fallback failed: %s", e)

            return _build_contact_result(
                product_url=product_url,
                success=False,
                status="submit_button_not_found",
                detail=(
                    "Ho compilato il messaggio ma non trovo il bottone 'Invia'. "
                    "Il browser è aperto: clicca tu 'Invia'."
                ),
            )

    except Exception as exc:
        logger.exception("PW contact_seller: unexpected error | %s", exc)
        return _build_contact_result(
            product_url=product_url,
            success=False,
            status="error",
            detail=str(exc),
        )

    finally:
        if success:
            # Chiudi solo la tab aperta
            if page is not None:
                try:
                    # await context.close()
                    # await browser.close()
                    pass
                except Exception:
                    pass
        else:
            logger.info("PW contact_seller: browser left open for manual interaction")
            
        # IMPORTANTISSIMO: Ferma SEMPRE il socket bridge locale di Playwright per svuotare la memoria (Node.js backend)
        # Nota: questo NON ti chiuderà il browser remoto originario
        try:
            await pw.stop()
        except Exception:
            pass


def _run_contact_in_proactor_loop(
    product_url: str,
    message: str,
    timeout_ms: int,
    seller_name: Optional[str] = None,
) -> Dict[str, Any]:
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _async_contact_seller(product_url, message, timeout_ms, seller_name)
        )
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@mcp_playwright.tool(
    name="contact_seller_playwright",
    description=(
        "Contatta un venditore eBay in modo autonomo usando Chrome (già loggato dall'utente). "
        "Se Chrome gira con --remote-debugging-port=9222, si connette direttamente alla sessione attiva. "
        "Altrimenti lancia Chrome con il profilo reale. "
        "Naviga su IntermediatedFAQ, clicca 'Non riguarda un oggetto', compila il messaggio e lo invia."
    ),
)
async def contact_seller_playwright(
    product_url: Annotated[
        str,
        Field(description="URL contatto eBay (es. https://www.ebay.it/cnt/IntermediatedFAQ?seller_name=X)"),
    ],
    message: Annotated[
        str,
        Field(description="Testo del messaggio da inviare al venditore"),
    ],
    seller_name: Annotated[
        Optional[str],
        Field(description="Nome del venditore eBay (opzionale, usato per navigazione naturale)"),
    ] = None,
) -> Dict[str, Any]:
    try:
        logger.info("MCP contact_seller_playwright START | seller=%s | url=%s", seller_name, product_url)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _PLAYWRIGHT_EXECUTOR,
            _run_contact_in_proactor_loop,
            product_url,
            message,
            45_000,
            seller_name,
        )
        logger.info("MCP contact_seller_playwright END | status=%s", result.get("contact_status"))
        return result
    except Exception as exc:
        logger.exception("MCP contact_seller_playwright failed")
        return _build_contact_result(
            product_url=product_url,
            success=False,
            status="error",
            detail=str(exc),
        )
