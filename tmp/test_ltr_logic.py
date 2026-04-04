import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.services.rag.reranker import rerank_products

async def test_ltr_logic():
    query = "dyson sotto i 300 euro"
    print(f"Testing LTR Logic for: {query}")
    
    constraints = [{"type": "price", "operator": "less_than", "value": 300}]
    
    items = [
        {
            "ebay_id": "1",
            "title": "Dyson V15 Detect - Nuovo",
            "price": 650.0,
            "condition": "Nuovo",
            "seller_name": "ProShop",
            "seller_rating": 99.5
        },
        {
            "ebay_id": "2",
            "title": "Dyson V8 Absolute - Usato Ottimo",
            "price": 240.0,
            "condition": "Usato",
            "seller_name": "CheapStore",
            "seller_rating": 95.0
        },
        {
             "ebay_id": "3",
            "title": "Filtro per Dyson V15",
            "price": 25.0,
            "condition": "Nuovo",
            "seller_name": "AccessoryPoint",
            "seller_rating": 98.0
        }
    ]
    
    # Run reranker directly
    ranked = rerank_products(
        query, 
        items, 
        constraints=constraints
    )
    
    print("\nResults:")
    for i, item in enumerate(ranked):
        price = item.get("price")
        title = item.get("title")
        score = item.get("_rerank_score")
        final = item.get("_final_score")
        print(f"{i+1}. [{price} EUR] Score={score} Final={final} | {title}")

if __name__ == "__main__":
    asyncio.run(test_ltr_logic())
