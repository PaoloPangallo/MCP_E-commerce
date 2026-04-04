import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
import joblib
import os

# Configuration
DATA_PATH = 'scripts/training_set_v2.csv'
MODEL_PATH = 'models/l2r_model.txt'
LOG_DIR = 'models/logs'
os.makedirs('models', exist_ok=True)

# Feature list (must match simulator output)
FEATURES = [
    'price_sens', 'qual_pref', 'brand_loyalty', 'cond_aff',
    'price_norm', 'price_median_ratio', 'brand_match', 'cond_norm',
    'seller_rating_norm', 'seller_feedback_norm',
    'product_rating_norm', 'product_review_norm', 'description_len_norm',
    'free_ship'
]

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Run the simulator first.")
        return

    print(f"--- Loading Dataset: {DATA_PATH} ---")
    df = pd.read_csv(DATA_PATH)
    
    # Pre-processing: ensure numeric types
    for col in FEATURES + ['relevance']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Grouping is based on (query + persona_id)
    df['query_group'] = df['query'] + "_" + df['persona_id']
    
    # Sort by group to ensure consistent blocks for LightGBM
    df = df.sort_values('query_group')
    
    X = df[FEATURES]
    y = df['relevance']
    groups = df.groupby('query_group').size().values

    # Train/Test Split (by group to avoid persona-leaking)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['query_group']))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Recalculate groups for train/test subsets
    groups_train = df.iloc[train_idx].groupby('query_group').size().values
    groups_test  = df.iloc[test_idx].groupby('query_group').size().values

    print(f"Training on {len(groups_train)} groups ({len(X_train)} rows)")
    print(f"Testing on {len(groups_test)} groups ({len(X_test)} rows)")

    # Model definition
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        label_gain=np.arange(21), # Relevance is 0-20
        random_state=42,
        importance_type='gain',
        verbose=-1
    )

    # Fit
    ranker.fit(
        X_train, y_train,
        group=groups_train,
        eval_set=[(X_test, y_test)],
        eval_group=[groups_test],
        eval_at=[5, 10]
    )

    # Save model
    ranker.booster_.save_model(MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")

    # Feature Importance
    importance = pd.DataFrame({
        'feature': FEATURES,
        'gain': ranker.feature_importances_
    }).sort_values('gain', ascending=False)
    
    print("\n--- Feature Importance (Gain) ---")
    print(importance)

if __name__ == "__main__":
    train()
