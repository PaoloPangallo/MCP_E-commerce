import asyncio
import os
import sys

# Aggiungi la root del progetto al path per gli import
sys.path.append(os.getcwd())

from app.services.rag.ltr_trainer import generate_training_data, train_ltr_model

async def run_training_flow():
    sample_queries = {
        "iphone 15 pro": [
            {"title": "Apple iPhone 15 Pro 256GB Titanio Naturale", "price": 1100, "trust_score": 0.95, "condition": "Nuovo"},
            {"title": "Custodia MagSafe per iPhone 15 Pro", "price": 50, "trust_score": 0.9, "condition": "Nuovo"},
            {"title": "iPhone 15 Pro Max 512GB", "price": 1300, "trust_score": 0.85, "condition": "Usato"},
            {"title": "Caricabatterie USB-C 20W Apple", "price": 25, "trust_score": 0.99, "condition": "Nuovo"},
            {"title": "Samsung Galaxy S24 Ultra", "price": 1000, "trust_score": 0.8, "condition": "Nuovo"}
        ],
        "sneakers jordan 1": [
            {"title": "Nike Air Jordan 1 Retro High OG Chicago", "price": 400, "trust_score": 0.98, "condition": "Nuovo"},
            {"title": "Jordan 1 Mid Shadow", "price": 150, "trust_score": 0.9, "condition": "Nuovo"},
            {"title": "Lacci per scarpe Jordan 1 Bianchi", "price": 10, "trust_score": 0.95, "condition": "Nuovo"},
            {"title": "Nike Dunk Low Panda", "price": 120, "trust_score": 0.85, "condition": "Nuovo"},
            {"title": "Jordan 4 Military Blue", "price": 300, "trust_score": 0.92, "condition": "Nuovo"}
        ],
        "playstation 5 console": [
            {"title": "Sony PlayStation 5 Console (Disc Edition)", "price": 499, "trust_score": 0.99, "condition": "Nuovo"},
            {"title": "Controller DualSense Wireless PS5", "price": 70, "trust_score": 0.95, "condition": "Nuovo"},
            {"title": "PS5 Digital Edition Console", "price": 399, "trust_score": 0.9, "condition": "Nuovo"},
            {"title": "Base di ricarica DualSense PS5", "price": 30, "trust_score": 0.98, "condition": "Nuovo"},
            {"title": "Xbox Series X Console", "price": 450, "trust_score": 0.8, "condition": "Nuovo"}
        ]
    }

    train_data_path = "tmp/ltr_training_set.jsonl"
    model_output_path = "app/services/rag/ltr_model.pkl"

    print("--- 1. Generazione dati sintetici con Ollama ---")
    await generate_training_data(sample_queries, train_data_path)

    print("\n--- 2. Addestramento modello LTR ---")
    await train_ltr_model(train_data_path, model_output_path)

    print(f"\nFlow completato. Modello salvato in {model_output_path}")

if __name__ == "__main__":
    asyncio.run(run_training_flow())
