import asyncio
import logging
import json
from dotenv import load_dotenv

load_dotenv()

from app.services.ebay import search_items, init_http_client, close_http_client
from app.services.parser import parse_query_service

async def test_search():
    logging.basicConfig(level=logging.INFO)
    await init_http_client()
    
    queries = ["iphone 15", "iphone15", "ciao, possiamo cercare un iphone15?"]
    
    for query in queries:
        print(f"\n{'='*20}\nTESTING QUERY: {query}\n{'='*20}")
        
        # 1. Parse
        parsed = await parse_query_service(query)
        print(f"Parsed: {json.dumps(parsed, indent=2)}")
        
        # 2. Search
        results = await search_items(parsed, limit=5)
        
        print(f"\nResults Count: {len(results.get('itemSummaries', []))}")
        
        if results.get('itemSummaries'):
            for i, item in enumerate(results['itemSummaries']):
                print(f"  {i+1}. {item.get('title')} - {item.get('price')} {item.get('currency')}")
        else:
            print("  !!! NO RESULTS !!!")
            
    await close_http_client()

if __name__ == "__main__":
    asyncio.run(test_search())
