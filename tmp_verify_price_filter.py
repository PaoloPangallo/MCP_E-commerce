
import asyncio
import os
import json
import logging
from app.services.search_pipeline import run_search_pipeline
from app.db.database import SessionLocal

logging.basicConfig(level=logging.INFO)

async def test_price_filter():
    db = SessionLocal()
    try:
        # Search for 'iPhone' - generic
        query = "iPhone"
        print(f"Testing query: {query}")
        
        results = await run_search_pipeline(query=query, db=db)
        
        items = results.get("items", [])
        print(f"Found {len(items)} items in 'items' key.")
        
        if items:
            for item in items[:3]:
                print(f"Item: {item.get('title')} | Price: {item.get('price')} {item.get('currency')}")
        else:
            print(f"Full response: {results}")
        if failures:
            print(f"\nFAILED: Found {len(failures)} items above 300€ despite filter.")
            for f in failures:
                print(f" - {f.get('title')} ({f.get('price')}€)")
        else:
            print("\nSUCCESS: All items are below or equal to 300€.")
            
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_price_filter())
