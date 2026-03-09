# scripts/manage_rag.py

import os
import sys
import argparse

# Aggiunge la root del progetto al path per importare app.*
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag.category_classifier import ingest_categories, category_store, category_bm25
from app.services.rag.vector_store import VectorStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-Manager")

def main():
    parser = argparse.ArgumentParser(description="Gestione RAG per MCP E-commerce")
    parser.add_argument("--action", choices=["reload-categories", "clear-all", "status"], required=True)
    
    args = parser.parse_args()
    
    if args.action == "reload-categories":
        logger.info("Ricaricamento categorie eBay...")
        # Pulizia namespace categorie
        if os.path.exists("rag_index_ebay_categories.faiss"):
            os.remove("rag_index_ebay_categories.faiss")
        if os.path.exists("rag_metadata_ebay_categories.pkl"):
            os.remove("rag_metadata_ebay_categories.pkl")
            
        ingest_categories()
        logger.info("Done.")

    elif args.action == "clear-all":
        logger.info("ATTENZIONE: Eliminazione di tutti gli indici RAG...")
        files_to_remove = [
            "rag_index_default.faiss", "rag_metadata_default.pkl",
            "rag_index_ebay_categories.faiss", "rag_metadata_ebay_categories.pkl",
            "rag_index.faiss", "rag_metadata.pkl" # Vecchi file pre-refactoring
        ]
        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)
                logger.info(f"Rimosso: {f}")
        logger.info("Sistema RAG resettato.")

    elif args.action == "status":
        default_store = VectorStore("default")
        logger.info(f"Collezione 'default': {len(default_store.documents)} documenti.")
        logger.info(f"Collezione 'ebay_categories': {len(category_store.documents)} categorie caricate.")

if __name__ == "__main__":
    main()
