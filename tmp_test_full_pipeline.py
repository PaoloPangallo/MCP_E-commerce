import asyncio
import logging
import json
from dotenv import load_dotenv

load_dotenv()

from app.services.search_pipeline import run_search_pipeline
from app.services.ebay import init_http_client, close_http_client
from app.db.database import SessionLocal

async def test_full_pipeline():
    logging.basicConfig(level=logging.INFO)
    await init_http_client()
    
    db = SessionLocal()
    query = "iphone 15"
    
    print(f"\n--- TESTING FULL PIPELINE: {query} ---")
    
    try:
        results = await run_search_pipeline(
            query=query,
            db=db,
            llm_engine="ollama_cloud"
        )
        
        print(f"\nResults Count: {results.get('results_count')}")
        print(f"Ebay Query Used: {results.get('ebay_query_used')}")
        
        if results.get('results'):
            for i, item in enumerate(results['results'][:5]):
                print(f"  {i+1}. {item.get('title')} - {item.get('price')} {item.get('currency')} (Score: {item.get('ranking_score')})")
        else:
            print("  !!! NO RESULTS !!!")
            
        print("\n--- TIMINGS ---")
        print(json.dumps(results.get('_timings', {}), indent=2))
        
    finally:
        db.close()
        await close_http_client()

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
