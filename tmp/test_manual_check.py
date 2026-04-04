
import asyncio
import sys
import os

# Aggiungi il percorso della cartella radice al sys.path
sys.path.append(os.getcwd())

from app.mcp.tools.playwright_contact import _async_contact_seller

async def main():
    seller_name = "danpang-22"
    message = "Test finale: questo messaggio è scritto dal bot. Clicca tu 'Invia' per favore!"
    
    print(f"--- TEST MANUALE: CONTATTO {seller_name} ---")
    print("Il bot aprirà l'overlay e scriverè il messaggio.")
    print("Il browser RESTERÀ APERTO per permetterti di cliccare 'Invia'.")
    
    # Eseguiamo il contatto con i parametri di test
    # Nota: Abbiamo commentato browser.close() in playwright_contact.py
    
    success_data = await _async_contact_seller(
        product_url=None, # Lasciamo che il bot lo trovi da solo tramite seller_name (Natural Navigation)
        message=message,
        timeout_ms=120000,
        seller_name=seller_name
    )
    
    print(f"\nEsito Script (success flag): {success_data.get('success')}")
    print(f"Messaggio: {success_data.get('message')}")
    
    print("\n--- ATTENZIONE ---")
    print("Il browser dovrebbe essere ora visibile con la chat aperta.")
    print("PROVA A CLICCARE 'INVIA' TU MANUALMENTE E DIMMI COSA SUCCEDE.")
    print("Premi INVIO qui in console per terminare lo script quando hai finito.")
    input()

if __name__ == "__main__":
    asyncio.run(main())
