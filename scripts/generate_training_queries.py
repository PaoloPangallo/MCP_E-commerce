import os
import json
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

# Configuration from .env
OLLAMA_CLOUD_HOST = os.getenv("OLLAMA_CLOUD_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_CLOUD_MODEL = os.getenv("OLLAMA_CLOUD_MODEL", "gpt-oss:120b")
OUTPUT_FILE = "scripts/generated_queries.json"

CATEGORIES = [
    "Electronics (Smartphone, Laptop, Audio, Camera)",
    "Fashion (Shoes, Watch, Handbags, Vintage Clothing, Jewelry)",
    "Home & Garden (Furniture, Tools, Kitchen, Decor, Yard)",
    "Collectibles (Lego, Cards, Antiques, Comics, Coins)",
    "Toys & Hobbies (Action Figures, Games, Model Trains)",
    "eBay Motors (Car Parts, Motorcycle Accessories, Tools)",
    "Media (Books, Movies, Music, Vinyl Records)",
    "Sporting Goods (Fitness, Outdoors, Cycling, Golf)"
]

async def generate_queries():
    print(f"Generating queries using {OLLAMA_CLOUD_MODEL}...")
    
    prompt = f"""Genera una lista di 60 query di ricerca per un sito di e-commerce (tipo eBay).
L'obiettivo è testare la sensibilità al brand: genera 30 COPPIE di query.
Ogni coppia deve contenere:
1. Una query con un BRAND famoso (es. 'iPhone', 'Nike shoes', 'Makita drill')
2. Una query GENERICA senza brand per lo stesso tipo di oggetto (es. 'Smartphone', 'Scarpe ginnastica', 'Trapano a batteria')

Le query devono essere brevi e coprire queste categorie: {', '.join(CATEGORIES)}.
Rispondi SOLO con un array JSON di stringhe alternando (Branded, Generica, Branded, Generica...)."""

    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OLLAMA_CLOUD_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # We use /api/chat if it follows Ollama's API
            url = f"{OLLAMA_CLOUD_HOST}/api/chat"
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            content = response.json().get("message", {}).get("content", "")
            # Clean JSON if any markdown is present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            queries = json.loads(content.strip())
            
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(queries, f, indent=2)
            
            print(f"Successfully generated {len(queries)} queries in {OUTPUT_FILE}")
            
    except Exception as e:
        print(f"Error generating queries: {e}")

if __name__ == "__main__":
    asyncio.run(generate_queries())
