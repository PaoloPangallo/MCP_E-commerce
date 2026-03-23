import sys
import asyncio
import json
import logging
import os
import re
import sqlite3
import csv
from typing import List, Dict, Any, Optional, Tuple
import httpx

# Add project root to path
sys.path.append(os.getcwd())

from app.services.ebay import search_items, get_item_details, init_http_client, close_http_client

# Configuration
OLLAMA_HOST = "http://localhost:11434/api/chat"
CONCURRENCY_LIMIT = 5
DEFAULT_FALLBACK = "llama3.2-vision:latest" 

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
GENERATED_QUERIES_FILE = os.path.join(SCRIPT_DIR, "generated_queries.json")
DB_PATH = os.path.join(SCRIPT_DIR, "l2r_simulator.db")
TRAIN_OUTPUT = os.path.join(SCRIPT_DIR, "training_set.csv")
MATRIX_OUTPUT = os.path.join(SCRIPT_DIR, "raw_matrix.csv")

# Serious queries for dataset generation (Fallback)
SERIOUS_QUERIES = [
    "iphone 15 pro",
    "sneakers jordan 1",
    "lego star wars millenium falcon",
    "macbook air m2",
    "vintage casio watch"
]

def load_queries() -> List[str]:
    """Loads queries from external file if exists, otherwise uses fallback."""
    if os.path.exists(GENERATED_QUERIES_FILE):
        try:
            with open(GENERATED_QUERIES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading generated queries: {e}")
    return SERIOUS_QUERIES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PERSONAS DEFINITIONS ---
PERSONAS = {
    "Spec Nerd": {
        "model": "llama3:8b",
        "system_prompt": "Sei un esperto tecnico ossessionato da specifiche hardware, benchmark e numeri di modello (SKU). Valuta i prodotti in base a RAM, CPU, risoluzione e velocità. Ignora il design o il marketing."
    },
    "Vintage Hunter": {
        "model": "mistral:7b",
        "system_prompt": "Sei un cercatore di oggetti vintage e autentici degli anni 80/90. Valuti positivamente prodotti originali, rarità e segni di autenticità storica. Penalizzi le imitazioni moderne."
    },
    "Designer": {
        "model": "gemma2:9b",
        "system_prompt": "Sei un designer industriale. Valuti i prodotti esclusivamente in base all'estetica, alla qualità dei materiali, al design minimalista e all'armonia cromatica."
    },
    "Budget Renovator": {
        "model": "phi3:mini",
        "system_prompt": "Sei un appassionato di fai-da-te con un budget limitato. Cerchi il miglior rapporto qualità-prezzo, preferisci l'usato se funzionale e risparmi su tutto ciò che non è essenziale."
    },
    "PC Master Race": {
        "model": "qwen2:7b",
        "system_prompt": "Sei un utente PC avanzato che disprezza i sistemi chiusi. Cerchi riparabilità, espandibilità e massime prestazioni. Valuti bene componenti SFX e workstation."
    },
    "Early Adopter": {
        "model": "llama3.1:8b",
        "system_prompt": "Vuoi solo l'ultima versione uscita. Se un prodotto è di una generazione precedente, lo scarti immediatamente. Cerchi parole come 'Nuovo', 'Versione 2024', 'Latest'."
    },
    "Practical DIYer": {
        "model": "mistral:7b-instruct",
        "system_prompt": "Ti serve un attrezzo che funzioni. Non ti importa del brand, ma della solidità e delle recensioni sulla durata. Ignora i prodotti troppo fragili o con troppa elettronica."
    },
    "Minimalist": {
        "model": "gemma:2b",
        "system_prompt": "Odi il disordine. Cerchi prodotti multifunzione, piccoli o che si integrano perfettamente in un ambiente pulito. Penalizzi prodotti ingombranti o con troppi fili."
    },
    "Value Hunter": {
        "model": "phi3:medium",
        "system_prompt": "Analizzi meticolosamente il prezzo per ogni caratteristica. Cerchi il 'sweet spot' dove la qualità è alta ma il prezzo non è ancora premium."
    },
    "Cheapskate": {
        "model": "qwen2:1.5b",
        "system_prompt": "Cerchi il prezzo PIÙ BASSO in assoluto. Non ti importa se è rotto, usato o senza scatola. Se costa poco, è un 4."
    },
    "Hobbyist Gamer": {
        "model": "tinyllama:latest",
        "system_prompt": "Sei un videogiocatore che cerca console e giochi per giocarci davvero, non per collezionarli. Valuti bene il prezzo e la giocabilità, accetti l'usato senza graffi."
    },
    "Aesthetic Homemaker": {
        "model": "mistral:7b",
        "system_prompt": "Cerchi oggetti per la casa che 'facciano atmosfera'. Colori pastello, legno naturale e forme organiche. Penalizzi la plastica nera e il metallo freddo."
    },
    "Refurbished Expert": {
        "model": "llama3.2-vision:latest",
        "system_prompt": "Ti fidi solo della roba ricondizionata professionale. Cerchi garanzie e 'Grade A'. Preferisci un iPhone usato garantito a uno nuovo economico."
    },
    "Luxury Buyer": {
        "model": "llama3.2-vision:latest",
        "system_prompt": "Vuoi solo il top del mercato. Pelle, metallo, loghi famosi. Se costa poco, probabilmente non è di qualità per te. Cerchi l'esclusività."
    },
    "Mint Collector": {
        "model": "llama3.2-vision:latest",
        "system_prompt": "Sei un collezionista maniacale. Solo scatole sigillate, mai aperte. Un graffio sulla confezione riduce il voto a zero. Cerchi 'New in Box', 'MISB'."
    }
}

class MultiAgentRanker:
    def __init__(self, semaphore: asyncio.Semaphore):
        self.semaphore = semaphore
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pairwise_cache (
                query TEXT,
                model TEXT,
                persona TEXT,
                item_a_id TEXT,
                item_b_id TEXT,
                winner_id TEXT,
                PRIMARY KEY (query, persona, item_a_id, item_b_id)
            )
        """)
        self.conn.commit()

    async def _get_available_models(self) -> List[str]:
        """Lists available models in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Use /api/tags to get the list of models
                url = OLLAMA_HOST.replace("/api/chat", "/api/tags")
                response = await client.get(url)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return [m["name"] for m in models]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        return []

    async def _call_ollama(self, model: str, system_prompt: str, user_prompt: str, available_models: List[str]) -> str:
        async with self.semaphore:
            # Fallback logic: if model is missing, use llama3:8b or the first available one
            actual_model = model
            if model not in available_models:
                fallback = DEFAULT_FALLBACK if DEFAULT_FALLBACK in available_models else (available_models[0] if available_models else None)
                if fallback and fallback != model:
                    logger.warning(f"Model {model} not found. Falling back to {fallback}")
                    actual_model = fallback
                elif not fallback:
                    logger.error("No models available in Ollama!")
                    return ""

            payload = {
                "model": actual_model,
                "messages": [
                    {"role": "system", "content": system_prompt + "\nIMPORTANT: You MUST respond only with a JSON object. No other text."},
                    {"role": "user", "content": user_prompt}
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 100}
            }
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=45.0)) as client:
                    response = await client.post(OLLAMA_HOST, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data.get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.error(f"Error calling {actual_model}: {e}")
                return ""

    async def compare_items(self, query: str, persona_name: str, item_a: Dict, item_b: Dict, available_models: List[str]) -> str:
        """Compares two items and returns the preferred item's ID."""
        persona = PERSONAS[persona_name]
        
        # Sort IDs to keep cache consistent (item_a_id < item_b_id)
        id_a, id_b = item_a.get("ebay_id", item_a.get("itemId")), item_b.get("ebay_id", item_b.get("itemId"))
        if not id_a or not id_b:
            return id_a or id_b

        if id_a > id_b:
            id_a, id_b = id_b, id_a
            item_a, item_b = item_b, item_a

        # Check cache
        self.cursor.execute("SELECT winner_id FROM pairwise_cache WHERE query=? AND persona=? AND item_a_id=? AND item_b_id=?",
                            (query, persona_name, id_a, id_b))
        row = self.cursor.fetchone()
        if row:
            return row[0]

        # Call LLM
        prompt_a = f"""PRODOTTO A:
- Titolo: {item_a.get('title')}
- Marca: {item_a.get('brand', 'N/A')}
- Prezzo: {item_a.get('price')} {item_a.get('currency')}
- Condizione: {item_a.get('condition', 'N/A')}
- Venditore: {item_a.get('seller_name', 'N/A')} (Feedback: {item_a.get('seller_rating', 'N/A')}%)
- Descrizione: {str(item_a.get('description', 'N/A'))[:500]}..."""

        prompt_b = f"""PRODOTTO B:
- Titolo: {item_b.get('title')}
- Marca: {item_b.get('brand', 'N/A')}
- Prezzo: {item_b.get('price')} {item_b.get('currency')}
- Condizione: {item_b.get('condition', 'N/A')}
- Venditore: {item_b.get('seller_name', 'N/A')} (Feedback: {item_b.get('seller_rating', 'N/A')}%)
- Descrizione: {str(item_b.get('description', 'N/A'))[:500]}..."""

        user_prompt = f"""Quale di questi due prodotti eBay preferiresti per la ricerca '{query}'?
Rispondi SOLO con questo formato JSON: {{"winner": "A", "reason": "breve spiegazione"}}

{prompt_a}

{prompt_b}"""

        for attempt in range(2):
            res_raw = await self._call_ollama(persona["model"], persona["system_prompt"], user_prompt, available_models)
            try:
                data = json.loads(res_raw)
                winner_val = str(data.get("winner", "")).strip()
                winner_char = None
                
                if winner_val.upper() in ['A', 'B']:
                    winner_char = winner_val.upper()
                else:
                    # Robust TITLE Matching
                    title_a = item_a.get('title', '').lower()
                    title_b = item_b.get('title', '').lower()
                    val_lower = winner_val.lower()
                    if title_a and (title_a in val_lower or val_lower in title_a):
                        winner_char = 'A'
                    elif title_b and (title_b in val_lower or val_lower in title_b):
                        winner_char = 'B'
                
                if winner_char:
                    winner_id = id_a if winner_char == 'A' else id_b
                    self.cursor.execute("INSERT OR REPLACE INTO pairwise_cache VALUES (?, ?, ?, ?, ?, ?)",
                                        (query, persona["model"], persona_name, id_a, id_b, winner_id))
                    self.conn.commit()
                    return winner_id
            except:
                pass
            
            # Simple Text Regex fallback
            match = re.search(r'"winner":\s*"([AB])"', res_raw, re.I) or re.search(r"WINNER:\s*([AB])", res_raw, re.I)
            if match:
                winner_char = match.group(1).upper()
                return id_a if winner_char == 'A' else id_b

            logger.warning(f"Retry {attempt+1} for {persona_name} on {query}. Response was: {res_raw[:50]}...")
        
        # Default fallback
        return id_a

    async def rank_items(self, query: str, items: List[Dict], persona_name: str, available_models: List[str]) -> List[str]:
        """Ranks items using a merge-sort approach for O(N log N) comparisons."""
        if len(items) <= 1:
            return [i.get("ebay_id", i.get("itemId")) for i in items if i.get("ebay_id", i.get("itemId"))]

        async def merge_sort(sub_items):
            if len(sub_items) <= 1:
                return sub_items
            
            mid = len(sub_items) // 2
            left = await merge_sort(sub_items[:mid])
            right = await merge_sort(sub_items[mid:])
            
            # Merge
            merged = []
            i = j = 0
            while i < len(left) and j < len(right):
                winner_id = await self.compare_items(query, persona_name, left[i], right[j], available_models)
                if winner_id == (left[i].get("ebay_id") or left[i].get("itemId")):
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    j += 1
            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged

        sorted_items = await merge_sort(items)
        return [i.get("ebay_id") or i.get("itemId") for i in sorted_items]

async def map_ranks_to_scores_granular(query: str, ranked_ids: List[str], persona_name: str, items_dict: Dict[str, Dict], sem: asyncio.Semaphore, available_models: List[str]) -> Dict[str, int]:
    """
    Maps ranks to unique scores (N-1 down to 0).
    Optionally identifies irrelevant items to set them to 0.
    """
    n = len(ranked_ids)
    if n == 0: return {}
    
    # Position based scoring (best = n-1, worst = 0)
    scores = {item_id: (n - 1 - i) for i, item_id in enumerate(ranked_ids)}
    
    # RELEVANCE CUTOFF: Ask the agent which items in the sorted list are actually irrelevant
    # We do this for the bottom half of the list to save time, or just skip if we want pure unique ranks
    # To follow the user's request "possibilità di mettere più volte 0", we'll check the relevance.
    
    persona = PERSONAS[persona_name]
    titles = []
    for item_id in ranked_ids:
        titles.append(f"- {items_dict[item_id].get('title')} (ID: {item_id})")
    
    titles_list = "\n".join(titles[-5:]) # Check only bottom 5 for irrelevance to be fast
    prompt = f"""In base alla tua personalità, quali di questi prodotti sono COMPLETAMENTE IRRILEVANTI per la query '{query}'?
Prodotti (dal peggiore al meno peggiore):
{titles_list}

Rispondi con una lista di ID separati da virgola che meritano VOTO 0. Se tutti sono minimamente rilevanti, rispondi 'NESSUNO'."""

    async with sem:
        payload = {
            "model": persona["model"] if persona["model"] in available_models else "llama3:8b",
            "messages": [
                {"role": "system", "content": persona["system_prompt"]},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.0}
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(OLLAMA_HOST, json=payload)
                if response.status_code == 200:
                    res_text = response.json().get("message", {}).get("content", "")
                    # Extract IDs
                    found_ids = [tid for tid in ranked_ids if tid in res_text]
                    for fid in found_ids:
                        scores[fid] = 0
        except Exception:
            pass # Fallback to unique ranks
            
    return scores

async def process_persona_and_save(ranker, query, items, items_dict, p_name, available_models, sem):
    """Worker task to process a single persona's ranking and save to CSV."""
    try:
        logger.info(f"Persona {p_name} starting ranking for: {query}")
        ranking = await ranker.rank_items(query, items, p_name, available_models)
        
        # Get granular scores with thresholding
        scores = await map_ranks_to_scores_granular(query, ranking, p_name, items_dict, sem, available_models)
        
        # WRITE TO CSV (Thread-safe append)
        with open(TRAIN_OUTPUT, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for item_id, s in scores.items():
                item = items_dict[item_id]
                writer.writerow([query, item_id, item.get('title'), f"{item.get('price')} {item.get('currency')}", p_name, s])
        
        logger.info(f"Persona {p_name} COMPLETED for query: {query}")
        return p_name, scores
    except Exception as e:
        logger.error(f"Error in persona {p_name} for {query}: {e}")
        return p_name, None

async def process_query(query: str, items: List[Dict]):
    logger.info(f"Processing query: {query} with {len(items)} items")
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    ranker = MultiAgentRanker(sem)
    
    available_models = await ranker._get_available_models()
    if not available_models:
        logger.error("No models available in Ollama. Skipping query.")
        return

    items_dict = { (item.get("ebay_id") or item.get("itemId")): item for item in items }
    
    # FETCH FULL DETAILS (including description) for each item in parallel
    logger.info(f"Fetching full details for {len(items)} items...")
    detail_tasks = [get_item_details(i_id) for i_id in items_dict.keys()]
    detail_results = await asyncio.gather(*detail_tasks)
    for detail in detail_results:
        if detail:
            i_id = detail.get("item_id")
            if i_id in items_dict:
                items_dict[i_id].update(detail)

    persona_names = list(PERSONAS.keys())

    # Create CSVs if they don't exist
    for path, row in [(TRAIN_OUTPUT, ["query", "item_id", "title", "price", "persona", "relevance"]), 
                      (MATRIX_OUTPUT, ["item_id"] + persona_names)]:
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    # PROCESS PERSONAS IN PARALLEL (limited by Semaphore)
    tasks = [process_persona_and_save(ranker, query, items, items_dict, p, available_models, sem) for p in persona_names]
    results = await asyncio.gather(*tasks)

    # Aggregate scores for raw_matrix.csv
    persona_scores_map = {p: s for p, s in results if s is not None}
    
    with open(MATRIX_OUTPUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i_id in items_dict.keys():
            row = [i_id]
            for p_name in persona_names:
                score = persona_scores_map.get(p_name, {}).get(i_id, "")
                row.append(score)
            writer.writerow(row)

    success_count = len(persona_scores_map)
    logger.info(f"Query {query} completed. Successful personas: {success_count}/{len(persona_names)}")

async def main():
    # Initialize files with headers if they don't exist
    if not os.path.exists(TRAIN_OUTPUT):
        with open(TRAIN_OUTPUT, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["query", "item_id", "title", "price", "persona", "relevance"])
    
    if not os.path.exists(MATRIX_OUTPUT):
        with open(MATRIX_OUTPUT, "w", newline="", encoding="utf-8") as f:
            headers = ["item_id"] + list(PERSONAS.keys())
            csv.writer(f).writerow(headers)

    await init_http_client()
    
    queries = load_queries()
    logger.info(f"Starting simulation with {len(queries)} queries.")
    
    # Count personas per query to identify partial completions
    query_persona_counts = {}
    if os.path.exists(TRAIN_OUTPUT):
        with open(TRAIN_OUTPUT, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row["query"]
                p = row["persona"]
                if q not in query_persona_counts:
                    query_persona_counts[q] = set()
                query_persona_counts[q].add(p)

    completed_queries = set()
    partial_queries = set()
    for q, p_set in query_persona_counts.items():
        if len(p_set) >= len(PERSONAS):
            completed_queries.add(q)
        else:
            partial_queries.add(q)
            logger.warning(f"Query '{q}' is incomplete ({len(p_set)}/{len(PERSONAS)} personas). Will re-process.")

    # Clean up partial queries from the CSV before starting
    if partial_queries:
        temp_file = TRAIN_OUTPUT + ".tmp"
        with open(TRAIN_OUTPUT, "r", encoding="utf-8") as f, \
             open(temp_file, "w", newline="", encoding="utf-8") as tf:
            reader = csv.reader(f)
            writer = csv.writer(tf)
            header = next(reader)
            writer.writerow(header)
            for row in reader:
                if row[0] not in partial_queries:
                    writer.writerow(row)
        os.replace(temp_file, TRAIN_OUTPUT)
        logger.info(f"Cleaned up partial results for: {partial_queries}")

    try:
        for query in queries:
            if query in completed_queries:
                logger.info(f"Skipping already completed query: {query}")
                continue

            logger.info(f"Fetching real eBay data for: {query}")
            # Mock parsed query for search_items
            parsed_query = {
                "original_query": query,
                "product": query,
                "brands": [],
                "constraints": [],
                "preferences": []
            }
            results = await search_items(parsed_query, limit=12)
            items = results.get("itemSummaries", [])
            
            if items:
                await process_query(query, items)
            else:
                logger.warning(f"No results for query: {query}")
                
    finally:
        await close_http_client()

if __name__ == "__main__":
    asyncio.run(main())
