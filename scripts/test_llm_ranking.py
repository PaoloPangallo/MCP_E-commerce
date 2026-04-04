import asyncio
import logging
import sys
import os

# Set up paths
sys.path.append(os.getcwd())

from app.services.ltr import rerank_items

# Mock items with rich context
MOCK_ITEMS = [
    {
        "title": "Apple iPhone 13 128GB Blue - Nuovo",
        "price": 699.0,
        "category_name": "Cellulari e Smartphone",
        "condition": "Nuovo",
        "seller_name": "TechStore_Official",
        "seller_rating": 99.8,
        "trust_score": 0.95,
        "rag_feedback": [{"text": "Spedizione velocissima, prodotto perfetto."}]
    },
    {
        "title": "Custodia per iPhone 13 in Silicone Trasparente",
        "price": 19.90,
        "category_name": "Accessori per cellulari",
        "condition": "Nuovo",
        "seller_name": "AccessoryPoint",
        "seller_rating": 95.0,
        "trust_score": 0.70
    },
    {
        "title": "Apple iPhone 13 128GB - Usato",
        "price": 680.0,
        "category_name": "Cellulari e Smartphone",
        "condition": "Usato",
        "seller_name": "Private_Seller_99",
        "seller_rating": 80.0,
        "trust_score": 0.40,
        "rag_feedback": [{"text": "Il venditore ha risposto dopo 3 giorni."}]
    }
]

async def test_ltr():
    logging.basicConfig(level=logging.INFO)
    query = "iPhone 13"
    
    print(f"\n--- Testing COMPLEX LTR with query: '{query}' ---")
    try:
        results = await rerank_items(query, MOCK_ITEMS)
    except Exception:
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nResults returned: {len(results)}")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['title']} | Score: {res['_ltr_score']}")
        print(f"   Motivation: {res.get('_llm_motivation', 'N/A')}")
        
    # Check if the cover was filtered out
    titles = [r['title'] for r in results]
    if any("Custodia" in t for t in titles):
        print("\n❌ FAIL: Accessory was NOT filtered out.")
    else:
        print("\n✅ SUCCESS: Accessory was filtered out correctly.")

if __name__ == "__main__":
    asyncio.run(test_ltr())
