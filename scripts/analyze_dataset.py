import csv
import pandas as pd
import numpy as np
import os
# Paths
TRAIN_OUTPUT = "scripts/training_set.csv"
ANALYSIS_REPORT = "scripts/dataset_analysis.md"

def analyze():
    if not os.path.exists(TRAIN_OUTPUT):
        print(f"File {TRAIN_OUTPUT} not found.")
        return

    print(f"Analyzing {TRAIN_OUTPUT}...")
    df = pd.read_csv(TRAIN_OUTPUT)

    # 1. Basic Stats
    total_rows = len(df)
    unique_queries = df['query'].nunique()
    unique_items = df['item_id'].nunique()
    unique_personas = df['persona'].nunique()

    # 2. Score Distribution
    score_counts = df['relevance'].value_counts().sort_index()

    # 3. Variance Analysis (Representativeness)
    # Group by query and item_id, calculate the standard deviation of scores across personas
    variance_per_item = df.groupby(['query', 'item_id'])['relevance'].std()
    avg_std = variance_per_item.mean()

    # 4. Persona Diversity
    # Average score per persona to see if some are generally "happier" or "grumpier"
    persona_avg = df.groupby('persona')['relevance'].mean().sort_values()

    # 5. Generate Markdown Report
    with open(ANALYSIS_REPORT, "w", encoding="utf-8") as f:
        f.write("# L2R Dataset Analysis Report\n\n")
        f.write(f"## Overview\n")
        f.write(f"- **Total Records**: {total_rows}\n")
        f.write(f"- **Unique Queries**: {unique_queries}\n")
        f.write(f"- **Unique Products**: {unique_items}\n")
        f.write(f"- **Personas Simulated**: {unique_personas}\n\n")

        f.write(f"## Dataset Quality (Variance)\n")
        f.write(f"> [!NOTE]\n")
        f.write(f"> An average Standard Deviation of **{avg_std:.2f}** indicates how much personas disagree on the same item.\n")
        f.write(f"> Higher variance means the dataset is more 'Context-Aware'.\n\n")

        f.write(f"## Score Distribution\n")
        f.write("| Relevance Score | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        for score, count in score_counts.items():
            f.write(f"| {score} | {count} | {(count/total_rows)*100:.1f}% |\n")
        f.write("\n")

        f.write(f"## Persona Behavior (Avg Score)\n")
        f.write("| Persona | Avg Relevance |\n")
        f.write("| :--- | :--- |\n")
        for persona, avg in persona_avg.items():
            f.write(f"| {persona} | {avg:.2f} |\n")
        f.write("\n")

    print(f"Analysis complete. Report saved to {ANALYSIS_REPORT}")

if __name__ == "__main__":
    analyze()
