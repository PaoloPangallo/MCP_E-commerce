import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.mcp.tools.playwright_tool import ebay_scrape

async def main():
    print("🚀 Test avvio ebay_scrape (Playwright)...")
    print("Browser dovrebbe aprirsi in MODALITÀ VISIBILE di default.")
    
    # Eseguiamo il tool direttamente. 
    # Notare che 'visible' ora ha default True in playwright_tool.py
    try:
        result = await ebay_scrape(
            query="iphone 13",
            max_results=3,
            session_id="test_session"
        )
        print("\n✅ Risultati ottenuti:")
        for res in result.get("results", []):
            print(f"- {res.get('title')} | Prezzo: {res.get('price_raw')}")
            
    except Exception as e:
        print(f"\n❌ Errore durante l'esecuzione: {e}")

if __name__ == "__main__":
    asyncio.run(main())
