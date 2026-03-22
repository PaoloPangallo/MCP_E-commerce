import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional

from app.llm.client import call_ollama_cloud
from app.services.rag.ltr_features import extract_ltr_features, get_feature_names
from app.services.rag.embedding import embed
import numpy as np

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """
Sei un esperto di e-commerce e Information Retrieval (IR). 
Il tuo compito è valutare la rilevanza dei prodotti rispetto a una query che include specifici vincoli (prezzo, marca, condizione).

Usa una scala da 0 a 4:
4: Risultato perfetto. Il prodotto è ESATTAMENTE quello cercato e rispetta TUTTI i vincoli (es. è nel range di prezzo indicato).
3: Risultato molto buono. Stessa categoria e marca, ma magari colore/storage diverso o prezzo leggermente fuori range.
2: Risultato mediocre. Stessa categoria ma modello/marca diverso, oppure ACCESSORI correlati (es. cover per il telefono cercato).
1: Risultato poco utile. Prodotto molto lontano dai vincoli (es. €800 quando il budget è €500) o categoria vagamente correlata.
0: Risultato irrilevante o RUMORE. Prodotto di una categoria completamente diversa o spam.

IMPORTANTE: Se un prodotto viola palesemente un vincolo di prezzo o marca indicato dall'utente, NON può avere un voto superiore a 1.
Rispondi solo con un JSON array di numeri (uno per ogni prodotto nell'ordine fornito).
Esempio: [4, 0, 1, 2]
"""

async def label_results_with_ollama(query: str, items: List[Dict[str, Any]], constraints: Optional[List[Dict[str, Any]]] = None) -> List[int]:
    """
    Uses Ollama Cloud to label a list of search results for a given query.
    """
    if not items:
        return []

    # Include price context in the prompt for the judge
    titles_text = "\n".join([
        f"{i+1}. {item.get('title')} | Prezzo: €{item.get('price', 'N/A')}" 
        for i, item in enumerate(items)
    ])
    
    constraints_str = json.dumps(constraints or [], indent=2)
    prompt = f"Query: {query}\nVincoli estratti:\n{constraints_str}\n\nProdotti:\n{titles_text}\n\nValuta la rilevanza (0-4) per ogni prodotto rispettando i vincoli."

    try:
        response = await call_ollama_cloud(prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
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

async def generate_training_data(dataset_config: Dict[str, Dict[str, Any]], output_path: str):
    """
    Generates a training dataset by labeling items with Ollama.
    dataset_config format: { "query_string": { "items": [...], "constraints": [...] } }
    """
    dataset = []
    
    for query, config in dataset_config.items():
        items = config.get("items", [])
        constraints = config.get("constraints", [])
        
        logger.info("Labeling query: %s", query)
        labels = await label_results_with_ollama(query, items, constraints=constraints)
        
        # Pre-compute query embedding
        query_emb = embed(query)
        
        # Compute price statistics for the group
        prices = [float(item.get("price", 0)) for item in items if item.get("price")]
        avg_price = sum(prices) / len(prices) if prices else 0
        std_price = np.std(prices) if len(prices) > 1 else 0
        
        ltr_context = {
            "avg_price": avg_price,
            "std_price": std_price,
            "constraints": constraints
        }

        # Proper Cosine Similarity instead of unstable dot product
        def cosine_sim(a, b):
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denom) if denom > 0 else 0.0

        for item, label in zip(items, labels):
            item_title = item.get("title", "")
            item_emb = embed(item_title)
            semantic_sim = cosine_sim(query_emb, item_emb)
            
            # Inject signals
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
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score

async def train_ltr_model(training_data_path: str, model_output_path: str):
    """
    Trains an XGBRanker (LambdaMART) with monotonicity constraints on the labeled dataset.
    """
    if not os.path.exists(training_data_path):
        logger.error("Training data not found at %s", training_data_path)
        return

    dataset_entries = []
    feature_names = get_feature_names()
    
    with open(training_data_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            dataset_entries.append(entry)
            
    if not dataset_entries:
        logger.warning("No data to train on.")
        return
        
    # Sort data by query string (qid requirement logic for XGBoost)
    dataset_entries.sort(key=lambda x: x["query"])
    
    X = []
    y = []
    queries = []
    for entry in dataset_entries:
        features = entry["features"]
        X.append([features.get(name, 0.0) for name in feature_names])
        y.append(float(entry["label"]))
        queries.append(entry["query"])
        
    # Support for Group-based split and training
    from collections import Counter
    unique_queries = sorted(set(queries))
    
    if len(unique_queries) < 5:
        logger.warning("Troppo poche query (%d) per una validazione affidabile. Training procedendo senza split.", len(unique_queries))
        split_available = False
    else:
        split_available = True

    # Convert queries to integer QIDs (for GroupShuffleSplit)
    query_to_id = {q: i for i, q in enumerate(unique_queries)}
    qids = np.array([query_to_id[q] for q in queries])

    X = np.array(X)
    y = np.array(y)

    # Monotonic Constraints: (+1 ascending, -1 descending, 0 none)
    constraint_dict = {
        "lexical_sim": 1, "semantic_sim": 1,
        "trust_score": 1, "seller_rating": 1,
        "log_price": -1, "price_z": -1,
        "has_image": 1, "is_new": 1,
        "has_brand": 1, "has_model": 1, "num_specs": 1,
        "rag_product_boost": 1, "rag_seller_boost": 1, "rag_sentiment": 1,
        "price_match_constraint": 1
    }
    monotone_constraints = tuple([constraint_dict.get(fn, 0) for fn in feature_names])

    if split_available:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=qids))
        
        # Calculate group counts for training/validation
        train_groups = [count for _, count in sorted(Counter(qids[train_idx]).items())]
        val_groups = [count for _, count in sorted(Counter(qids[val_idx]).items())]

        val_model = xgb.XGBRanker(
            objective='rank:ndcg',
            monotone_constraints=monotone_constraints,
            max_depth=5,
            n_estimators=100,
            learning_rate=0.05,
            tree_method='hist'
        )
        
        # Train with early stopping
        val_model.fit(
            X[train_idx], y[train_idx], 
            group=train_groups,
            eval_set=[(X[val_idx], y[val_idx])],
            eval_group=[val_groups],
            verbose=False
        )
        
        preds = val_model.predict(X[val_idx])
        
        # Calculate NDCG@10 per query in validation set
        val_queries_ids = set(qids[val_idx])
        ndcgs = []
        for vq in val_queries_ids:
            mask = (qids[val_idx] == vq)
            true_relevance = np.asarray([y[val_idx][mask]])
            predicted_scores = np.asarray([preds[mask]])
            if true_relevance.shape[1] > 1 and np.max(true_relevance) > 0:
                ndcgs.append(ndcg_score(true_relevance, predicted_scores, k=10))
        
        if ndcgs:
            logger.info("Out-of-Fold Validation NDCG@10: %.4f", np.mean(ndcgs))

    logger.info("Training FINAL model on all %d samples...", len(X))
    final_groups = [count for _, count in sorted(Counter(qids).items())]
    
    model = xgb.XGBRanker(
        objective='rank:ndcg',
        monotone_constraints=monotone_constraints,
        max_depth=5,
        n_estimators=100,
        learning_rate=0.05,
        tree_method='hist'
    )
    model.fit(X, y, group=final_groups)
    
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
