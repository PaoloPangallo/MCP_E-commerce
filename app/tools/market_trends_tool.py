from typing import Any, Dict, Optional
import os
import logging
import asyncio
import re
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SERP_API_KEY = os.getenv("SERP_API_KEY")

def normalize_market_trends_arguments(action_input: Dict[str, Any]) -> Dict[str, Any]:
    query = action_input.get("query")
    if not query:
        raise ValueError("market_trends richiede una query (nome del prodotto).")
        
    # Pulizia del prefisso se l'agente passa la frase intera
    prefixes_to_strip = [
        "trend di mercato, statistiche e andamento prezzi medi online per:",
        "trend di mercato e analisi prezzi per:",
        "analisi dei trend di mercato per:",
        "mostrami i trend di mercato per:",
        "mostrami i trend di mercato e i prezzi medi online per:",
        "analisi mercato",
        "trend per",
        "prezzi per",
        "per:"
    ]
    
    clean_q = query.lower()
    for prefix in prefixes_to_strip:
        if clean_q.startswith(prefix):
            query = query[len(prefix):].strip()
            clean_q = query.lower()
            
    # Rimuoviamo eventuali caratteri superflui all'inizio
    query = re.sub(r'^[:\s\-]+', '', query).strip()
    
    return {"query": query}

async def fetch_google_shopping(query: str, api_key: str) -> Dict[str, Any]:
    from serpapi import GoogleSearch
    params = {
      "engine": "google_shopping",
      "q": query,
      "hl": "it",
      "gl": "it", # Italy for realistic local prices
      "api_key": api_key
    }
    
    try:
        # GoogleSearch non è asincrono di default, lo wrappiamo in to_thread
        def _search():
            search = GoogleSearch(params)
            return search.get_dict()
        
        results = await asyncio.to_thread(_search)
        shopping_results = results.get("shopping_results", [])
        
        if not shopping_results:
            return {"status": "no_data", "message": "Nessun risultato shopping trovato."}
            
        prices = []
        for item in shopping_results:
            price_str = item.get("price")
            if price_str:
                # Basic cleaning for Italian/EU format (e.g., "1.200,50 €" or "500,00 €")
                # Remove currency symbols and spaces
                clean_price = price_str.replace("€", "").replace("$", "").strip()
                # Remove thousands separator (.) and change decimal separator (,) to (.)
                if "," in clean_price and "." in clean_price:
                    clean_price = clean_price.replace(".", "").replace(",", ".")
                elif "," in clean_price:
                    clean_price = clean_price.replace(",", ".")
                
                try:
                    prices.append(float(clean_price))
                except ValueError:
                    continue
                    
        if not prices:
            return {"status": "no_data", "message": "Impossibile estrarre i prezzi."}
            
        avg_price = sum(prices) / len(prices)
        return {
            "status": "ok",
            "min_price": min(prices),
            "max_price": max(prices),
            "average_price": round(avg_price, 2),
            "samples": len(prices),
            "top_result": shopping_results[0]
        }
    except Exception as e:
        logger.error(f"Error fetching Google Shopping data: {e}")
        return {"status": "error", "error": str(e)}

async def fetch_google_trends(query: str, api_key: str) -> Dict[str, Any]:
    from serpapi import GoogleSearch
    
    # Per i trend, serve una query pulita e generica (es. "iPhone 14" invece di "iPhone 14 128GB Nero")
    trend_query = query
    # Rimuoviamo memoria (GB, TB)
    trend_query = re.sub(r'\d+(?:gb|tb)', '', trend_query, flags=re.IGNORECASE)
    # Rimuoviamo testo tra parentesi
    trend_query = re.sub(r'\(.*?\)', '', trend_query)
    # Rimuoviamo termini di condizione comuni
    trend_query = re.sub(r'\b(nuovo|usato|ricondizionato|originale|sbloccato|pari al nuovo)\b', '', trend_query, flags=re.IGNORECASE)
    # Pulizia spazi e punteggiatura
    trend_query = re.sub(r'[-\s]+', ' ', trend_query).strip()
    
    logger.info(f"Using trend_query: '{trend_query}' (Original: '{query}')")
    
    params = {
      "engine": "google_trends",
      "q": trend_query,
      "data_type": "TIMESERIES",
      "api_key": api_key
    }
    
    try:
        def _search():
            search = GoogleSearch(params)
            return search.get_dict()
            
        results = await asyncio.to_thread(_search)
        timeline = results.get("interest_over_time", {}).get("timeline_data", [])
        
        if not timeline:
             return {"status": "no_data", "message": "Nessun dato di trend trovato."}
             
        # Extract the values for the graph
        interest_graph = []
        for item in timeline:
            date_str = item.get("date", "")
            # Shorten date for better chart display (e.g. "Jan 1, 2023 - Jan 7, 2023" -> "Jan 23")
            match = re.search(r'([A-Z][a-z]{2})\s+\d+,\s+(\d{4})', date_str)
            if match:
                display_date = f"{match.group(1)} {match.group(2)[2:]}"
            else:
                display_date = date_str[:6]
                
            vals = item.get("values", [])
            for v in vals:
                try:
                    val = int(v.get("extracted_value", 0))
                    interest_graph.append({"date": display_date, "value": val})
                except ValueError:
                    pass
                    
        trend_values = [p["value"] for p in interest_graph]
        if not trend_values:
            return {"status": "no_data", "message": "Dati di trend vuoti."}
            
        current = trend_values[-1]
        previous = trend_values[0] if len(trend_values) > 1 else current
        
        trend_direction = "stabile"
        if current > previous * 1.15: # 15% growth for "in crescita"
            trend_direction = "in crescita"
        elif current < previous * 0.85: # 15% drop for "in calo"
            trend_direction = "in calo"
            
        return {
            "status": "ok",
            "current_interest": current,
            "trend_direction": trend_direction,
            "data_points": len(trend_values),
            "interest_graph": interest_graph
        }
        
    except Exception as e:
        logger.error(f"Error fetching Google Trends data: {e}")
        return {"status": "error", "error": str(e)}


async def execute_market_trends_tool(action_input: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    Analizza i trend di mercato (prezzi e interesse) per un prodotto tramite SerpApi.
    """
    clean = normalize_market_trends_arguments(action_input)
    query = clean["query"]
    
    logger.info(f"Market Trends Analysis for: {query}")
    
    if not SERP_API_KEY:
        logger.warning("SERP_API_KEY non configurata nel .env")
        return {
            "status": "error",
            "error": "SERP_API_KEY mancante. Configurala per usare questo tool."
        }
        
    # Eseguiamo le chiamate in parallelo
    shopping_task = fetch_google_shopping(query, SERP_API_KEY)
    trends_task = fetch_google_trends(query, SERP_API_KEY)
    
    shopping_data, trends_data = await asyncio.gather(shopping_task, trends_task)
    
    # Generate a "Verdict" (Verdetto)
    verdetto = "Analisi non disponibile."
    if shopping_data.get("status") == "ok" and trends_data.get("status") == "ok":
        avg = shopping_data["average_price"]
        trend = trends_data["trend_direction"]
        
        if trend == "in crescita":
            verdetto = "DOMANDA IN FORTE AUMENTO. Il prodotto è molto cercato in questo periodo. Se il prezzo è vicino al minimo (€{}), l'acquisto è ALTAMENTE CONSIGLIATO.".format(shopping_data["min_price"])
        elif trend == "in calo":
            verdetto = "DOMANDA IN DIMINUZIONE. L'interesse sta calando. Potresti trovare ottimi affari se non hai fretta, ma fai attenzione alla svalutazione futura."
        else:
            verdetto = "MERCATO STABILE. Prezzi e interesse sono costanti. Un acquisto sicuro se il prezzo rientra nella media di €{}.".format(avg)

    # Determine final status
    status = "ok"
    if shopping_data.get("status") == "error" and trends_data.get("status") == "error":
        status = "error"

    return {
        "status": status,
        "query": query,
        "shopping_data": shopping_data,
        "trends_data": trends_data,
        "verdetto": verdetto
    }
