import asyncio
import os
import json
from app.services.ebay import search_items, init_http_client, close_http_client

async def test_integration():
    print("--- Testing Live eBay Search + L2R Reranking ---")
    await init_http_client()
    
    query_str = "smartphone android"
    print(f"Searching for: '{query_str}'...")
    
    try:
        parsed_query = {"original_query": query_str}
        # We don't pass specific persona traits yet, so it should use 'Balanced' defaults
        result = await search_items(parsed_query, limit=10)
        items = result.get("itemSummaries", [])
        
        print(f"Found {len(items)} items.")
        
        for idx, item in enumerate(items):
            score = item.get("_ltr_score")
            print(f" [{idx+1}] {item['title'][:40]}... | Price: {item['price']} | LTR Score: {score}")
            
        if any(i.get("_ltr_score") is not None for i in items):
            print("\n✅ SUCCESS: L2R Reranking is active (scores detected).")
        else:
            print("\n❌ FAILURE: L2R Reranking scores not found in results.")
            
    except Exception as e:
        print(f"Error during integration test: {e}")
    finally:
        await close_http_client()

if __name__ == "__main__":
    # Ensure working directory is project root
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    asyncio.run(test_integration())
