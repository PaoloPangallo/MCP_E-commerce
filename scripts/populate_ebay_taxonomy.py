# scripts/populate_ebay_taxonomy.py

import os
import sys
import logging
import json

# Aggiunge la root del progetto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ebay import _get_oauth_token, TAXONOMY_URL, EBAY_IT_TREE_ID, EBAY_MARKETPLACE_ID, get_category_aspects
from app.services.rag.category_classifier import category_store, category_bm25
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Taxonomy-Ingestor")

def fetch_category_tree():
    """Recupera l'intero albero delle categorie eBay per l'Italia."""
    token = _get_oauth_token()
    url = f"{TAXONOMY_URL}/category_tree/{EBAY_IT_TREE_ID}"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE_ID,
    }
    
    logger.info(f"Fetching full category tree for Marketplace {EBAY_MARKETPLACE_ID}...")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Failed to fetch tree: {response.text}")
        return None
    return response.json()

def process_node(node, path=""):
    """Funzione ricorsiva per estrarre le categorie foglia (leaf)."""
    category = node.get("category", {})
    cat_id = category.get("categoryId")
    cat_name = category.get("categoryName")
    
    current_path = f"{path} > {cat_name}" if path else cat_name
    
    children = node.get("childCategoryTreeNodes", [])
    
    # Se non ha figli, è una categoria foglia (dove si vendono i prodotti)
    if not children:
        return [{
            "category_id": cat_id,
            "name": cat_name,
            "full_path": current_path
        }]
    
    leaves = []
    for child in children:
        leaves.extend(process_node(child, current_path))
    return leaves

def run():
    tree = fetch_category_tree()
    if not tree:
        return
    
    root_node = tree.get("rootCategoryNode", {})
    logger.info("Extracting leaf categories...")
    all_leaves = process_node(root_node)
    
    logger.info(f"Found {len(all_leaves)} leaf categories. Ingesting popular ones for RAG...")
    
    # Per motivi di performance e quote, iniziamo con una selezione o filtriamo quelle più rilevanti
    # In una versione reale, potresti volerle indicizzare tutte (richiede tempo)
    # Per questa demo, prendiamo le prime 500 o quelle con parole chiave 'moda', 'elettronica', etc.
    target_keywords = ["Abbigliamento", "Informatica", "Telefonia", "Orologi", "Scarpe", "Console", "Foto"]
    
    selected = [l for l in all_leaves if any(k in l['full_path'] for k in target_keywords)]
    logger.info(f"Filtered to {len(selected)} relevant categories.")

    texts = []
    metas = []
    
    # Limite per la demo per non bloccare il sistema ore
    BATCH_LIMIT = 100 
    
    for i, cat in enumerate(selected[:BATCH_LIMIT]):
        logger.info(f"[{i+1}/{BATCH_LIMIT}] Fetching aspects for: {cat['name']}...")
        aspects = get_category_aspects(cat['category_id'])
        
        text = f"Category: {cat['name']}. Path: {cat['full_path']}. Required Aspects: {', '.join(aspects)}"
        texts.append(text)
        metas.append({
            "type": "ebay_category",
            "category_id": cat["category_id"],
            "full_path": cat["full_path"],
            "name": cat["name"],
            "required_aspects": aspects
        })

    if texts:
        logger.info(f"Persisting {len(texts)} categories to RAG namespaces...")
        category_store.add_documents(texts, metas)
        category_bm25.add_documents(texts, metas)
        logger.info("Ingestion complete!")

if __name__ == "__main__":
    run()
