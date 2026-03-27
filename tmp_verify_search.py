
import asyncio
import os
import json
import logging
from app.services.search_pipeline import run_search_pipeline
from app.db.database import SessionLocal

logging.basicConfig(level=logging.INFO)

async def test_search():
    db = SessionLocal()
    try:
        # Search for 'iPhone'
        query = "iPhone sotto 500 euro"
        print(f"Testing query: {query}")
        
        results_payload = await run_search_pipeline(query=query, db=db)
        
        items = results_payload.get("results", [])
        print(f"FOUND ITEMS COUNT: {len(items)}")
        
        for item in items[:5]:
            print(f" - {item.get('title')} | {item.get('price')} {item.get('currency')}")
            
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_search())
