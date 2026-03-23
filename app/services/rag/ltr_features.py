from typing import List, Dict, Any, Optional
import math
import re

def compute_lexical_similarity(query: str, title: str) -> float:
    """Simple Jaccard similarity between query and title tokens."""
    q_tokens = set(re.findall(r'\w+', query.lower()))
    t_tokens = set(re.findall(r'\w+', title.lower()))
    if not q_tokens: return 0.0
    intersection = q_tokens.intersection(t_tokens)
    return len(intersection) / len(q_tokens)

def extract_ltr_features(query: str, item: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Extracts a numerical feature vector for a (Query, Item) pair.
    These features will be used by the LTR model.
    """
    features = {}
    context = context or {}

    # 1. Relevance Features
    features["lexical_sim"] = compute_lexical_similarity(query, item.get("title", ""))
    features["semantic_sim"] = float(item.get("_semantic_sim", 0.0))
    
    # 2. Trust Features
    features["trust_score"] = float(item.get("trust_score") or 0.5)
    features["seller_rating"] = float(item.get("seller_rating") or 0.0) / 100.0
    
    # 3. Price Features
    price = float(item.get("price") or 0.0)
    features["log_price"] = math.log10(price + 1) if price > 0 else 0.0
    
    # Price Z-score (relative to the result set)
    avg_p = context.get("avg_price", 0.0)
    std_p = context.get("std_price", 1.0)
    features["price_z"] = (price - avg_p) / std_p if std_p > 0 else 0.0
    
    # 4. Meta Features
    features["has_image"] = 1.0 if item.get("image_url") else 0.0
    features["is_new"] = 1.0 if "nuov" in str(item.get("condition", "")).lower() else 0.0
    
    # 5. NER Features
    ner = item.get("ner_attributes", {})
    features["has_brand"] = 1.0 if ner.get("brand") else 0.0
    features["has_model"] = 1.0 if ner.get("model") else 0.0
    features["num_specs"] = float(len(ner.get("specs") or {}))

    # 6. RAG signals (pre-computed in search/rerank)
    features["rag_product_boost"] = float(item.get("_rag_product_boost", 0.0))
    features["rag_seller_boost"] = float(item.get("_rag_seller_boost", 0.0))
    features["rag_sentiment"] = float(item.get("_rag_sentiment_signal", 0.0))

    # 7. Constraint Matching
    constraints = context.get("constraints") or []
    price_match = 1.0 # Default if no constraints
    for c in constraints:
        if c.get("type") == "price":
            op = c.get("operator")
            val = c.get("value")
            if op == "between" and isinstance(val, list) and len(val) == 2:
                if price < val[0] or price > val[1]:
                    price_match = 0.0
            elif op == "<=" and val is not None:
                if price > float(val):
                    price_match = 0.0
            elif op == ">=" and val is not None:
                if price < float(val):
                    price_match = 0.0
    features["price_match_constraint"] = price_match
    
    # 8. Accessory Detection (High negative weight if undesired)
    features["accessory_score"] = float(item.get("_accessory_penalty", 0.0))

    return features

def get_feature_names() -> List[str]:
    return [
        "lexical_sim", "semantic_sim", "trust_score", "seller_rating",
        "log_price", "price_z", "has_image", "is_new", "has_brand",
        "has_model", "num_specs", "rag_product_boost", "rag_seller_boost", 
        "rag_sentiment", "price_match_constraint", "accessory_score"
    ]
