import asyncio
import os
import sys
import json
import logging
from typing import List, Dict, Any

# Aggiungi la root del progetto al path per gli import
sys.path.append(os.getcwd())

from app.services.parser import parse_query_service
from app.services.ebay import search_items, init_http_client, close_http_client
from app.services.rag.ltr_trainer import generate_training_data, train_ltr_model
from app.services.trust import compute_trust_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERIOUS_QUERIES = [
    # TECH & ELECTRONICS
    "iphone 15 pro", "iphone 13 usato", "macbook pro m3 14 pollici", "samsung galaxy s24 ultra",
    "gaming laptop rtx 4090", "ipad air 5 256gb", "nintendo switch oled con giochi", 
    "ps5 console nuova", "bose quietcomfort ultra", "sony wh-1000xm5", "meccanica tastiera rgb",
    "monitor gaming 4k 144hz", "kindle paperwhite 16gb", "dyson v15 detect", "dyson sotto i 300 euro",
    "folletto vk200 usato", "mac mini m2", "gopro hero 12", "fotocamera mirrorless canon",
    # HOME & KITCHEN
    "macchina caffe delonghi", "ninja air fryer dual zone", "kitchenaid artisan rossa",
    "irobot roomba j7", "scrivania ufficio legno", "lampada design cartell",
    # FASHION & LUXURY
    "sneakers nike dunk low", "scarpe jordan 1 retro", "rolex submariner", "omega speedmaster",
    "casio ga-2100", "occhiali da sole rayban aviator", "borsa gucci marmont",
    # HOBBIES & COLLECTIBLES
    "lego star wars millennium falcon", "lego tecnic auto", "carte pokemon charizard",
    "vinile pink floyd dark side", "chitarra elettrica fender stratocaster", "ps4 usata con 2 controller",
    # SEARCH WITH CONSTRAINTS (PRICE/FILTER)
    "iphone tra 500 e 600 euro", "smartphone economico sotto i 150 euro", "aspirapolvere potente sopra i 400 euro",
    "giacca invernale uomo north face", "lego sotto i 50 euro", "computer fisso sopra i 2000 euro"
]

async def collect_real_data():
    """
    Raccoglie risultati reali da eBay per una lista di query serie.
    """
    await init_http_client()
    dataset_config = {}
    
    for query in SERIOUS_QUERIES:
        logger.info(f"Processing query: {query}")
        try:
            # Usa il parser REALE invece di un mock
            parsed = await parse_query_service(query, use_llm=True)
            search_results = await search_items(parsed, limit=12)
            items = search_results.get("itemSummaries", [])
            
            if not items:
                logger.warning(f"No results for {query}")
                continue
            
            # Arricchimento base (Trust Score)
            for item in items:
                rating = item.get("seller_rating") or 98.0
                item["trust_score"] = float(rating) / 100.0 if float(rating) > 1 else float(rating)
            
            dataset_config[query] = {
                "items": items,
                "constraints": parsed.get("constraints") or []
            }
            logger.info(f"Collected {len(items)} items and constraints for '{query}'")
            
        except Exception as e:
            logger.error(f"Error collecting data for '{query}': {e}")
            
    await close_http_client()
    return dataset_config

async def main():
    logger.info("--- START LONG SERIOUS LTR TRAINING FLOW ---")
    
    # 1. Raccogli dati reali
    real_data_config = await collect_real_data()
    
    if not real_data_config:
        logger.error("No data collected. Exiting.")
        return
        
    train_path = "tmp/ltr_serious_dataset.jsonl"
    model_path = "app/services/rag/ltr_model.pkl"
    
    # 2. Genera training set con Ollama Cloud
    logger.info("--- Step 2: Labeling with Ollama Cloud ---")
    await generate_training_data(real_data_config, train_path)
    
    # 3. Addestra il modello
    logger.info("--- Step 3: Training Model ---")
    await train_ltr_model(train_path, model_path)
    
    logger.info("--- SERIOUS LTR FLOW COMPLETED ---")

if __name__ == "__main__":
    asyncio.run(main())
