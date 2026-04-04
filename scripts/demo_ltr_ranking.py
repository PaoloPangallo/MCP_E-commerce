import asyncio
import os
import json
import pandas as pd
from app.services.ebay import search_items, init_http_client, close_http_client
from app.services.ltr import rerank_items

# Test Personas
PERSONAS = {
    "Balanced": None, # Defaults: 0.5 for all
    "Bargain Hunter": {"price_sens": 0.9, "qual_pref": 0.2, "brand_loyalty": 0.3, "cond_aff": 0.8},
    "Luxury Buyer": {"price_sens": 0.1, "qual_pref": 0.9, "brand_loyalty": 0.9, "cond_aff": 1.0}
}

async def run_demo(query: str, limit: int = 30):
    print(f"\n" + "="*80)
    print(f" DEMO QUERY: '{query}' (Limit: {limit})")
    print("="*80)
    
    await init_http_client()
    try:
        parsed_query = {"original_query": query}
        # 1. Fetch RAW items from eBay
        # (Note: we bypass the integrated reranker in search_items to show manual control here)
        result = await search_items(parsed_query, limit=limit)
        raw_items = result.get("itemSummaries", [])
        
        print(f"Loaded {len(raw_items)} raw items from eBay.")
        
        for p_name, traits in PERSONAS.items():
            print(f"\n--- Reranking for: {p_name} ---")
            ranked = await rerank_items(query, list(raw_items), traits)
            
            # Print Top 5
            for idx, item in enumerate(ranked[:8]):
                score = item.get("_ltr_score", 0.0)
                cond = item.get("condition", "N/A")
                price = item.get("price", 0.0)
                title = item.get("title", "")[:45]
                print(f" [{idx+1}] {title:<45} | €{price:>7.2f} | Cond: {cond:<10} | LTR: {score:+.3f}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await close_http_client()

async def main():
    # Setup PYTHONPATH
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    # Run both queries
    await run_demo("iphone 13", limit=30)
    await run_demo("iphone 13 ricondizionato", limit=10)

if __name__ == "__main__":
    asyncio.run(main())
