import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

from app.services.parser import call_gemini, call_ollama, LLM_PROVIDER

logger = logging.getLogger(__name__)

# Prompt for NER extraction
NER_SYSTEM_PROMPT = """
Sei un esperto di Information Retrieval per e-commerce. 
Il tuo compito è estrarre attributi strutturati dal titolo di un prodotto eBay in formato JSON.
Devi estrarre:
- brand: il marchio del prodotto (es. Apple, Samsung, Nike)
- model: il modello specifico (es. iPhone 13 Pro, Galaxy S21)
- specs: un dizionario di caratteristiche tecniche (es. {"storage": "256GB", "color": "Midnight", "condition": "Usato"})
- category: una categoria macro (es. "Smartphone", "Scarpe", "Laptop")

IMPORTANTE: 
1. Rispondi SOLO in formato JSON valido.
2. Non aggiungere spiegazioni o testo extra.
3. Se un attributo non è presente, usa null.
4. Mantieni i valori delle specifiche concisi.
"""

def _build_batch_prompt(titles: List[str]) -> str:
    prompt = "Estrai gli attributi per i seguenti titoli di prodotti:\n\n"
    for i, title in enumerate(titles):
        prompt += f"{i+1}. {title}\n"
    prompt += "\nRestituisci una lista di oggetti JSON nel campo 'results'."
    return prompt

async def extract_attributes_batch(titles: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts structured attributes from a batch of product titles.
    """
    if not titles:
        return []

    prompt = _build_batch_prompt(titles)
    
    # We use the same configuration as parser.py
    try:
        if LLM_PROVIDER == "gemini":
            response = await call_gemini(f"{NER_SYSTEM_PROMPT}\n\n{prompt}")
        else:
            response = await call_ollama(prompt, system_prompt=NER_SYSTEM_PROMPT)

        if not response:
            return [{} for _ in titles]

        # Basic cleaning of markdown code blocks
        clean_response = response.strip()
        if "```json" in clean_response:
            clean_response = clean_response.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean_response:
            clean_response = clean_response.split("```")[-1].split("```")[0].strip()

        data = json.loads(clean_response)
        results = data.get("results", [])
        
        # Ensure we return the correct number of results
        if len(results) < len(titles):
            results.extend([{} for _ in range(len(titles) - len(results))])
        
        return results[:len(titles)]

    except Exception as e:
        logger.warning("NER batch extraction failed: %s", e)
        return [{} for _ in titles]

async def extract_attributes_single(title: str) -> Dict[str, Any]:
    """
    Extracts structured attributes from a single product title.
    """
    results = await extract_attributes_batch([title])
    return results[0] if results else {}
