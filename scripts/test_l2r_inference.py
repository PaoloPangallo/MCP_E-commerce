import pandas as pd
import numpy as np
import lightgbm as lgb
import os

MODEL_PATH = 'models/l2r_model.txt'

FEATURES = [
    'price_sens', 'qual_pref', 'brand_loyalty', 'cond_aff',
    'price_norm', 'price_median_ratio', 'brand_match', 'cond_norm',
    'seller_rating_norm', 'seller_feedback_norm',
    'product_rating_norm', 'product_review_norm', 'description_len_norm',
    'free_ship'
]

def test_inference():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    # Load model
    bst = lgb.Booster(model_file=MODEL_PATH)

    # 1. Create a mock result set of 5 items
    # Features: [price_norm, price_median_ratio, rating, feedback, ...]
    items = [
        {"id": "CheapItem", "price_norm": 0.1, "price_median": 0.5, "rating": 0.5},
        {"id": "MidItem",   "price_norm": 0.5, "price_median": 1.0, "rating": 0.8},
        {"id": "LuxuryItem","price_norm": 0.9, "price_median": 1.8, "rating": 0.95},
    ]

    # 2. Test for BARGAIN HUNTER (Sens: 0.9, Qual: 0.1)
    print("--- Reranking for BARGAIN HUNTER (Price Sens: 0.9) ---")
    rows_bh = []
    for i in items:
        # Create full feature vector
        feat_vec = [
            0.9, 0.1, 0.5, 0.5, # User traits
            i["price_norm"], i["price_median"], 0.5, 1.0, # item basics
            0.8, 0.8, i["rating"], 0.5, 0.5, 1.0 # ratings/ship
        ]
        rows_bh.append(feat_vec)
    
    scores_bh = bst.predict(np.array(rows_bh))
    for i, s in zip(items, scores_bh):
        print(f" Item: {i['id']:<10} | Score: {s:.4f}")

    # 3. Test for LUXURY BUYER (Sens: 0.1, Qual: 0.9)
    print("\n--- Reranking for LUXURY BUYER (Qual Pref: 0.9) ---")
    rows_lb = []
    for i in items:
        feat_vec = [
            0.1, 0.9, 0.5, 0.5, # User traits
            i["price_norm"], i["price_median"], 0.5, 1.0,
            0.8, 0.8, i["rating"], 0.5, 0.5, 1.0
        ]
        rows_lb.append(feat_vec)
    
    scores_lb = bst.predict(np.array(rows_lb))
    for i, s in zip(items, scores_lb):
        print(f" Item: {i['id']:<10} | Score: {s:.4f}")

if __name__ == "__main__":
    test_inference()
