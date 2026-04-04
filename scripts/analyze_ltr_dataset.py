import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

# Set paths
DATA_FILE = 'scripts/training_set_v2.csv'
MATRIX_FILE = 'scripts/raw_matrix_v2.csv'
OUTPUT_DIR = 'scripts/analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"--- Dataset Overview ---")
    print(f"Total Rows: {len(df)}")
    print(f"Unique Queries: {df['query'].nunique()}")
    print(f"Unique Personas: {df['persona_id'].nunique()}")

    # 1. Relevance Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['relevance'], bins=20, kde=True)
    plt.title('Distribution of Relevance Scores')
    plt.savefig(f"{OUTPUT_DIR}/relevance_dist.png")
    
    # 2. Persona Differences
    avg_rel = df.groupby('persona_id')['relevance'].mean().sort_values()
    print("\n--- Avg Relevance per Persona ---")
    print(avg_rel)

    # 3. Correlation between User Features and Ranking
    print("\n--- Feature Correlation (Global) ---")
    # We look at how user traits interact with item features to drive relevance
    # Higher price_sens should correlate negatively with price_norm
    corr_matrix = df[['price_sens', 'qual_pref', 'price_norm', 'product_rating_norm', 'relevance']].corr()
    print(corr_matrix)

    # 4. Persona-Specific Analysis
    print("\n--- Price Sensitivity Calibration ---")
    for pid in df['persona_id'].unique():
        sub = df[df['persona_id'] == pid]
        p_sens = sub['price_sens'].iloc[0]
        # Correlation between price and relevance for THIS persona
        corr, _ = spearmanr(sub['price_norm'], sub['relevance'])
        print(f" - {pid:<18} (Sens: {p_sens:.1f}): Price/Rel Corr = {corr:.3f}")

if __name__ == "__main__":
    analyze()
