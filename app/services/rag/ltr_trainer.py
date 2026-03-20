import asyncio
import json
import logging
import os
from typing import List, Dict, Any

from app.services.parser import call_ollama
from app.services.rag.ltr_features import extract_ltr_features, get_feature_names
from app.services.rag.embedding import embed
import numpy as np

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """
Sei un esperto di e-commerce e Information Retrieval. 
Il tuo compito è valutare la rilevanza dei prodotti rispetto a una query dell'utente.
Usa una scala da 0 a 4:
4: Risultato perfetto (stesso modello, marca, caratteristiche principali).
3: Risultato molto buono (stessa categoria, marca corretta, modello simile).
2: Risultato mediocre (stessa categoria ma marca/modello diverso, o accessori correlati).
1: Risultato poco utile (stessa categoria molto ampia, o prodotto vagamente correlato).
0: Risultato irrilevante (completamente fuori tema).

Rispondi solo con un JSON array di numeri (uno per ogni prodotto nell'ordine fornito).
Esempio: [4, 2, 0, 3]
"""

async def label_results_with_ollama(query: str, items: List[Dict[str, Any]]) -> List[int]:
    """
    Uses Ollama to label a list of search results for a given query.
    """
    if not items:
        return []

    titles_text = "\n".join([f"{i+1}. {item.get('title')}" for i, item in enumerate(items)])
    prompt = f"Query: {query}\n\nProdotti:\n{titles_text}\n\nValuta la rilevanza (0-4) per ogni prodotto."

    try:
        response = await call_ollama(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        if response:
            # Clean up potential markdown or thinking blocks if Ollama is chatty
            clean_res = response.strip()
            if "```json" in clean_res:
                clean_res = clean_res.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_res:
                clean_res = clean_res.split("```")[1].split("```")[0].strip()
            
            labels = json.loads(clean_res)
            if isinstance(labels, list) and len(labels) == len(items):
                return [int(l) for l in labels]
    except Exception as e:
        logger.error("Error labeling with Ollama: %s", e)
    
    return [0] * len(items)

async def generate_training_data(queries_with_items: Dict[str, List[Dict[str, Any]]], output_path: str):
    """
    Generates a training dataset by labeling items with Ollama.
    """
    dataset = []
    
    for query, items in queries_with_items.items():
        logger.info("Labeling query: %s", query)
        labels = await label_results_with_ollama(query, items)
        
        # Pre-compute query embedding
        query_emb = embed(query)
        
        # Compute price statistics for the group
        prices = [float(item.get("price", 0)) for item in items if item.get("price")]
        avg_price = sum(prices) / len(prices) if prices else 0
        std_price = np.std(prices) if len(prices) > 1 else 0
        
        ltr_context = {
            "avg_price": avg_price,
            "std_price": std_price
        }

        for item, label in zip(items, labels):
            # Compute item embedding and semantic similarity
            item_title = item.get("title", "")
            item_emb = embed(item_title)
            semantic_sim = float(np.dot(query_emb, item_emb))
            
            # Inject signals into item for feature extraction
            item["_semantic_sim"] = semantic_sim
            
            features = extract_ltr_features(query, item, context=ltr_context)
            dataset.append({
                "features": features,
                "label": label,
                "query": query,
                "title": item_title
            })
            
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    logger.info("Generated %d training samples at %s", len(dataset), output_path)

import pickle
from sklearn.ensemble import HistGradientBoostingRegressor

async def train_ltr_model(training_data_path: str, model_output_path: str):
    """
    Trains a HistGradientBoostingRegressor on the labeled dataset.
    """
    if not os.path.exists(training_data_path):
        logger.error("Training data not found at %s", training_data_path)
        return

    X = []
    y = []
    
    feature_names = get_feature_names()
    
    with open(training_data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            features = entry["features"]
            # Ensure features are in the correct order
            X.append([features.get(name, 0.0) for name in feature_names])
            y.append(float(entry["label"]))
            
    if not X:
        logger.warning("No data to train on.")
        return
        
    logger.info("Training model with %d samples...", len(X))
    model = HistGradientBoostingRegressor(max_iter=100, max_depth=5)
    model.fit(X, y)
    
    # Save the model and the feature names for inference
    payload = {
        "model": model,
        "feature_names": feature_names
    }
    
    with open(model_output_path, "wb") as f:
        pickle.dump(payload, f)
        
    logger.info("Model saved to %s", model_output_path)

if __name__ == "__main__":
    # Example usage (can be run as a script)
    async def main():
        sample_data = {
            "iphone 13 pro": [
                {"title": "Apple iPhone 13 Pro 128GB Sierra Blue", "price": 800, "trust_score": 0.9},
                {"title": "Cover per iPhone 13 Pro Trasparente", "price": 15, "trust_score": 0.8},
                {"title": "Samsung Galaxy S21 Ultra", "price": 700, "trust_score": 0.7}
            ]
        }
        train_path = "tmp/ltr_train_data.jsonl"
        model_path = "app/services/rag/ltr_model.pkl"
        
        await generate_training_data(sample_data, train_path)
        await train_ltr_model(train_path, model_path)
    
    # asyncio.run(main()) # Uncomment to run
