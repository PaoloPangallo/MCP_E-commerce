import asyncio
import os
import sys
import json
import logging
from typing import List, Dict, Any

# Aggiungi la root del progetto al path per gli import
sys.path.append(os.getcwd())

from app.services.ebay import search_items, init_http_client, close_http_client
from app.services.rag.ltr_trainer import generate_training_data, train_ltr_model
from app.services.trust import compute_trust_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERIOUS_QUERIES = [
    "iphone 15 pro",
    "sneakers jordan 1",
    "lego star wars millenium falcon",
    "macbook air m2",
    "vintage casio watch",
    "gaming laptop rtx 4080",
    "coffee machine nespresso",
    "wireless earbuds anc",
    "ps5 console",
    "bose quietcomfort 45",
    "dyson v15 detect",
    "nintendo switch oled",
    "canon eos r6",
    "rolex submariner",
    "kindle paperwhite",
    "mechanical keyboard rgb",
    "electric scooter xiaomi",
    "air fryer ninja",
    "board games strategy",
    "vinyl player audio technica"
]

async def collect_real_data():
    """
    Raccoglie risultati reali da eBay per una lista di query serie.
    """
    await init_http_client()
    
    training_data_input = {}
    
    for query in SERIOUS_QUERIES:
        logger.info(f"Searching eBay for: {query}")
        try:
            # Mock di un oggetto parsed_query per la funzione search_items
            parsed_query = {
                "original_query": query,
                "product": query,
                "brands": [],
                "constraints": [],
                "preferences": []
            }
            
            search_results = await search_items(parsed_query, limit=12)
            items = search_results.get("itemSummaries", [])
            
            if not items:
                logger.warning(f"No results for {query}")
                continue
            
            # Arricchimento base (Trust Score)
            for item in items:
                # Use seller_rating as a proxy for trust if available
                rating = item.get("seller_rating") or 0.5
                item["trust_score"] = float(rating) / 100.0 if float(rating) > 1 else float(rating)
            
            training_data_input[query] = items
            logger.info(f"Collected {len(items)} items for '{query}'")
            
        except Exception as e:
            logger.error(f"Error collecting data for '{query}': {e}")
            
    await close_http_client()
    return training_data_input

async def main():
    logger.info("--- START SERIOUS LTR TRAINING FLOW ---")
    
    # 1. Raccogli dati reali
    real_data = await collect_real_data()
    
    if not real_data:
        logger.error("No data collected. Exiting.")
        return
        
    train_path = "tmp/ltr_serious_dataset.jsonl"
    model_path = "app/services/rag/ltr_model.pkl"
    
    # 2. Genera training set con Ollama (questo prenderà del tempo)
    logger.info("--- Step 2: Labeling with Ollama ---")
    await generate_training_data(real_data, train_path)
    
    # 3. Addestra il modello
    logger.info("--- Step 3: Training Model ---")
    await train_ltr_model(train_path, model_path)
    
    logger.info("--- SERIOUS LTR FLOW COMPLETED ---")

if __name__ == "__main__":
    asyncio.run(main())
