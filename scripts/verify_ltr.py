import asyncio
import os
import sys
import logging

# Aggiungi la root del progetto al path per gli import
sys.path.append(os.getcwd())

# Configura il logging per vedere i messaggi del reranker
logging.basicConfig(level=logging.INFO)

from app.services.rag.reranker import rerank_products
from app.services.rag.ltr_features import extract_ltr_features

async def verify_ltr():
    query = "iphone 15 pro"
    items = [
        {"title": "Apple iPhone 15 Pro 256GB Titanio Naturale", "price": 1100, "trust_score": 0.95, "condition": "Nuovo"},
        {"title": "Custodia MagSafe per iPhone 15 Pro", "price": 50, "trust_score": 0.9, "condition": "Nuovo"},
        {"title": "Samsung Galaxy S24 Ultra", "price": 1000, "trust_score": 0.8, "condition": "Nuovo"}
    ]
    
    print("--- Testing Reranker with LTR Model ---")
    
    # Check features manually
    for item in items:
        feats = extract_ltr_features(query, item)
        print(f"\nFeatures for {item['title']}:")
        print(feats)
    
    results = rerank_products(query, items)
    
    print("\nRanking finale:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['title']} (Score: {res.get('_rerank_score', 'N/A')})")

if __name__ == "__main__":
    asyncio.run(verify_ltr())
