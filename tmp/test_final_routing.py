import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.ebay_agent import EbayAgent
from app.mcp.client import get_mcp_client

async def main():
    print("🚀 Test finale di routing e avvio Playwright...")
    # Questo test simula il passaggio dal Planner all'esecuzione del tool MCP
    
    agent = EbayAgent()
    query = "Cerca su eBay con Playwright (MODALITÀ VISIBILE): msrcello adcani"
    
    # Questo invocerà internamente il planner, che grazie al fix userà il tool ebay_scrape
    # Chromium dovrebbe aprirsi davanti a te.
    print(f"👉 Inviando query: {query}")
    try:
        # Nota: serve che il backend sia attivo (lo abbiamo già lanciato in background) 
        # perché l'EbayAgent usa il client MCP per parlare con le proprie tools.
        responses = []
        async for event in agent.run(query):
            # Stampiamo solo gli eventi di tool_start per conferma
            if event.get("type") == "tool_start":
                print(f"🛠️  TOOL AVVIATO: {event.get('tool')}")
            responses.append(event)
            
        print("\n✅ Test completato. Dovresti aver visto il browser eBay aprirsi.")
    except Exception as e:
        print(f"\n❌ Errore: {e}")

if __name__ == "__main__":
    asyncio.run(main())
