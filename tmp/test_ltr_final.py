import asyncio
import sys
import os
import json

# Add project root to sys.path
sys.path.append(os.getcwd())

from app.services.rag.reranker import rerank_products

async def test_ltr():
    query = "iphone 15 pro max"
    
    # Mock some items from eBay-like search
    items = [
        {
            "title": "Apple iPhone 15 Pro Max - 256GB - Natural Titanium (Unlocked)",
            "price": 1050.0,
            "trust_score": 0.95,
            "seller_rating": 99.5,
            "condition": "nuovo",
            "image_url": "http://img.com/1.jpg",
            "ner_attributes": {"brand": "Apple", "model": "iPhone 15 Pro Max"}
        },
        {
            "title": "Custodia iPhone 15 Pro Max MagSafe Silicone",
            "price": 45.0,
            "trust_score": 0.88,
            "seller_rating": 98.0,
            "condition": "nuovo",
            "image_url": "http://img.com/2.jpg",
            "ner_attributes": {"brand": "Apple", "model": "iPhone 15 Pro Max Case"}
        },
        {
            "title": "Scatola Vuota iPhone 15 Pro Max box only",
            "price": 15.0,
            "trust_score": 0.70,
            "seller_rating": 95.0,
            "condition": "usato",
            "image_url": "http://img.com/3.jpg",
            "ner_attributes": {}
        }
    ]
    
    # Run reranking (it will fetch semantic sim and RAG in background usually, 
    # but here we test the LTR scoring specifically)
    ranked = rerank_products(query, items)
    
    print(f"\n--- LTR RERANKING RESULTS FOR: '{query}' ---")
    for i, item in enumerate(ranked):
        print(f"{i+1}. Score: {item.get('_rerank_score')} | Title: {item.get('title')[:60]}...")
        # Check if RAG or other flags were touched
        # print(f"   Signals: Sim={item.get('_semantic_sim', 0)}, RagProd={item.get('_rag_product_boost', 0)}")

if __name__ == "__main__":
    asyncio.run(test_ltr())
