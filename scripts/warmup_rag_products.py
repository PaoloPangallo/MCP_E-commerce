# scripts/warmup_rag_products.py

import os
import sys
import logging

# Aggiunge la root del progetto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ebay import search_items
from app.services.rag.product_ingest import ingest_products
from app.services.rag.category_classifier import category_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-Warmup")

def run():
    # Recuperiamo alcune categorie dal RAG per sapere cosa cercare
    categories = category_store.documents[:10] # Le prime 10
    if not categories:
        logger.warning("No categories found in RAG. Run populate_ebay_taxonomy.py first.")
        return

    for cat in categories:
        logger.info(f"Warming up RAG with products for category: {cat.get('name')} (ID: {cat.get('category_id')})")
        
        # Effettuiamo una ricerca generica per la categoria per "riempire" il RAG
        # Cerchiamo il nome della categoria come query base
        query = {"product": cat.get("name"), "semantic_query": cat.get("name")}
        
        try:
            items = search_items(query, limit=20, category_id=cat.get("category_id"))
            if items:
                logger.info(f"Ingesting {len(items)} products into RAG...")
                ingest_products(items)
            else:
                logger.info("No items found for this category warmup.")
        except Exception as e:
            logger.error(f"Error during warmup for {cat.get('name')}: {e}")

    logger.info("Warmup complete! Your RAG is now pre-populated with some real products.")

if __name__ == "__main__":
    run()
