# app/services/rag/category_classifier.py

import json
import logging
from typing import Dict, Any, List, Optional
from app.services.rag.vector_store import VectorStore
from app.services.rag.bm25_store import BM25Store
from app.services.rag.ebay_categories_data import EBAY_leaf_CATEGORIES
from app.core.config import OLLAMA_CHAT_URL, MODEL_NAME
import requests

logger = logging.getLogger(__name__)

# Namespace dedicato per non inquinare i documenti generici
category_store = VectorStore("ebay_categories")
category_bm25 = BM25Store("ebay_categories")

def ingest_categories():
    """Popola la collezione 'categories' con l'albero eBay."""
    texts = []
    metas = []
    
    for cat in EBAY_leaf_CATEGORIES:
        # Il testo da vettorizzare deve essere descrittivo
        text = f"Category: {cat['name']}. Path: {cat['full_path']}. Aspects: {', '.join(cat['required_aspects'])}"
        texts.append(text)
        metas.append({
            "type": "ebay_category",
            "category_id": cat["category_id"],
            "full_path": cat["full_path"],
            "name": cat["name"],
            "required_aspects": cat["required_aspects"]
        })
    
    category_store.add_documents(texts, metas)
    category_bm25.add_documents(texts, metas)
    logger.info(f"Ingested {len(metas)} eBay categories into RAG.")

def classify_query(query: str) -> Optional[Dict[str, Any]]:
    """Identifica la categoria eBay più pertinente per una query."""
    q_low = query.lower()
    
    # --- HARD FALLBACK RULES per termini comuni ---
    if "felpa" in q_low or "hoodie" in q_low or "sweatshirt" in q_low:
        from app.services.rag.ebay_categories_data import EBAY_leaf_CATEGORIES
        cat = next((c for c in EBAY_leaf_CATEGORIES if c["category_id"] == "155226"), None)
        if cat:
            return {
                "category_id": cat["category_id"],
                "category_path": cat["full_path"],
                "required_aspects": cat["required_aspects"],
                "confidence": 0.9,
                "conflict_warning": None if "uomo" in q_low or "donna" in q_low else "Ambiguity: Gender (Uomo/Donna) non specificato"
            }
    
    if "scarpe" in q_low or "shoes" in q_low or "sneakers" in q_low:
        from app.services.rag.ebay_categories_data import EBAY_leaf_CATEGORIES
        cat = next((c for c in EBAY_leaf_CATEGORIES if c["category_id"] == "11459"), None)
        if cat:
            return {
                "category_id": cat["category_id"],
                "category_path": cat["full_path"],
                "required_aspects": cat["required_aspects"],
                "confidence": 0.85,
                "conflict_warning": "Ambiguity: Gender non specificato per calzature"
            }

    # Step 1: Hybrid Search nella collezione categorie (Vector + BM25)
    v_results = category_store.search(query, k=5)
    b_results = category_bm25.search(query, k=5)
    
    # Unione semplice dei risultati per il classificatore LLM
    seen_ids = set()
    results = []
    for r in v_results + b_results:
        cid = r.get("category_id")
        if cid and cid not in seen_ids:
            results.append(r)
            seen_ids.add(cid)
    
    results = results[:5]
    if not results:
        return None
    
    # Step 2: LLM Refinement (Classifier Prompt)
    # L'LLM sceglie tra i top-k risultati del RAG per precisione chirurgica
    options = []
    for r in results:
        options.append({
            "id": r["category_id"],
            "path": r["full_path"],
            "aspects": r["required_aspects"]
        })
    
    prompt = f"""
    Act as an eBay categorization expert. 
    USER DESCRIPTION: "{query}"
    
    POTENTIAL CATEGORIES:
    {json.dumps(options, indent=2)}
    
    TASK:
    Identify the most specific and accurate category from the list above. 
    Priority Rules:
    1. Give priority to the technical structure of the category over superficial keywords.
    2. GENDER AMBIGUITY: If the query is gender-neutral (e.g. "felpa") but categories are gender-specific (e.g. "Uomo" vs "Donna"), pick the most likely one based on results but ALWAYS set "conflict_warning" to "Ambiguity: Gender (Uomo/Donna) not specified".
    3. If the description contains conflicting terms, resolve based on eBay's best practices.
    
    Return ONLY a JSON object with:
    - "category_id": the chosen ID
    - "category_path": the full path
    - "required_aspects": list of mandatory fields to fulfill
    - "confidence": score 0.0 to 1.0
    - "conflict_warning": a string if there's an ambiguity or conflict, otherwise null
    """
    
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json().get("message", {}).get("content") or response.json().get("response")
    except Exception as e:
        logger.error(f"Error in category classification LLM: {e}")
    
    # Fallback al primo risultato vettoriale se LLM fallisce
    return {
        "category_id": results[0]["category_id"],
        "category_path": results[0]["full_path"],
        "required_aspects": results[0]["required_aspects"],
        "confidence": results[0].get("_similarity", 0.5),
        "conflict_warning": None
    }

# Inizializza all'import (o in fase di setup)
if not category_store.documents:
    ingest_categories()
