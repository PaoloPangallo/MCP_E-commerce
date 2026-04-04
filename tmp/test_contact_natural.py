"""
Test della Navigazione Naturale (Natural Navigation).
Il test cerca un prodotto del venditore, naviga alla pagina prodotto (/itm/), 
clicca su 'Contatta il venditore' e invia il messaggio.
"""
import asyncio, sys, os

sys.path.insert(0, ".")
os.environ.setdefault("PYTHONPATH", ".")

# Usiamo un venditore reale per il test
SELLER = "danpang-22"
MESSAGE = "Buongiorno, l'oggetto è ancora disponibile? Grazie."

async def run():
    print(f"--- TEST NATURAL NAVIGATION per {SELLER} ---")
    
    # 1. Init eBay services (per cercare l'item)
    from app.services.ebay import init_http_client
    await init_http_client()
    
    # 2. Importa il tool aggiornato
    from app.mcp.tools.playwright_contact import _async_contact_seller
    
    # Eseguiamo il flusso passano il seller_name
    # Questo forzerà la ricerca dell'item e il 'Natural Navigation'
    print("[1] Avvio _async_contact_seller con Natural Navigation...")
    
    # Nota: usiamo l'URL generic as fallback, ma il tool cercherà un item URL.
    generic_url = f"https://www.ebay.it/contact/sendmsg?recipient={SELLER}"
    
    result = await _async_contact_seller(
        product_url=generic_url,
        message=MESSAGE,
        timeout_ms=45000,
        seller_name=SELLER
    )
    
    print(f"\n[RISULTATO]: {result.get('status')}")
    print(f"Dettaglio: {result.get('detail')}")
    print(f"Success: {result.get('success')}")

if __name__ == "__main__":
    asyncio.run(run())
