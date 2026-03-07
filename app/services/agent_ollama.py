import json
import logging
import os
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1")

def search_products(query: str, max_price: float = None) -> str:
    """Cerca prodotti su eBay relativi alla richiesta dell'utente."""
    from app.services.ebay import search_items
    
    parsed_req = {"semantic_query": query}
    if max_price:
        parsed_req["constraints"] = [{"type": "price", "operator": "<=", "value": max_price}]
        
    risultati = search_items(parsed_req, limit=3)
    
    if not risultati:
        return json.dumps({"error": "Nessun prodotto trovato."})
        
    prodotti_estratti = [
        {"title": r.get('title'), "price": r.get('price'), "seller": r.get('seller_name')} 
        for r in risultati
    ]
    return json.dumps(prodotti_estratti)

def get_seller_trust(seller_name: str) -> str:
    """Recupera i dati e valuta il punteggio di affidabilità di un venditore eBay."""
    from app.services.feedback import get_seller_feedback
    from app.services.trust import compute_trust_score
    from app.services.nlp_sentiment import compute_sentiment_score
    
    feedbacks = get_seller_feedback(seller_name, limit=10)
    if not feedbacks:
        return json.dumps({"error": "Nessun feedback recente trovato per questo venditore."})
        
    sentiment = compute_sentiment_score(feedbacks)
    trust = compute_trust_score(feedbacks, sentiment_score=sentiment)
    
    return json.dumps({
        "seller": seller_name,
        "trust_score": round(trust, 2)
    })

AVAILABLE_TOOLS_MAP = {
    "search_products": search_products,
    "get_seller_trust": get_seller_trust
}

OLLAMA_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Cerca prodotti su eBay relativi alla richiesta dell'utente. Restituisce una lista di risultati tra cui titolo, prezzo e venditore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La parola chiave esatta del prodotto (es. 'scrivania pc', 'notebook lenovo')"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Prezzo massimo desiderato in euro. Non includere questo parametro se non specificato."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_seller_trust",
            "description": "Recupera i dati e valuta il punteggio di sicurezza e affidabilità di un venditore eBay specifico analizzando i feedback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_name": {
                        "type": "string",
                        "description": "Il nome esatto del venditore eBay (es. 'pcrepairita', 'lombardoshop')"
                    }
                },
                "required": ["seller_name"]
            }
        }
    }
]

def ask_ollama_agent(user_message: str) -> str:
    from dotenv import load_dotenv
    load_dotenv()
    
    global MODEL_NAME
    MODEL_NAME = os.getenv("OLLAMA_MODEL", MODEL_NAME)
    
    messages = [
        {"role": "system", "content": "Sei un assistente per lo shopping intelligente. Usa gli strumenti (tools) a tua disposizione per cercare prodotti e verificare le recensioni dei venditori. Ricorda di usare get_seller_trust quando hai trovato i venditori tramite search_products. Rispondi sempre in italiano, in modo chiaro e utile."},
        {"role": "user", "content": user_message}
    ]
    
    MAX_ITERATIONS = 5
    
    for iteration in range(MAX_ITERATIONS):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": OLLAMA_TOOLS_SCHEMA,
            "stream": False
        }
        
        try:
            response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=90)
            if response.status_code != 200:
                print(f"DEBUG: ERRORE API Ollama ({response.status_code}): {response.text}")
                return f"Errore API Ollama: {response.status_code}"
                
            data = response.json()
            message = data.get("message", {})
            
            # Aggiungiamo la risposta cruda del modello alla cronologia
            messages.append(message)
            
            # Caso A: il modello vuole chiamare uno o più tool
            tool_calls = message.get("tool_calls", [])
            
            if tool_calls:
                print(f"DEBUG [Iter {iteration+1}]: L'agente vuole usare {len(tool_calls)} tool(s).")
                for tool_call in tool_calls:
                    function_name = tool_call.get("function", {}).get("name")
                    arguments = tool_call.get("function", {}).get("arguments", {})
                    
                    print(f"DEBUG -> Eseguo Tool: {function_name}({arguments})")
                    
                    func_to_call = AVAILABLE_TOOLS_MAP.get(function_name)
                    if func_to_call:
                        try:
                            # Esegue la funzione
                            tool_result_json = func_to_call(**arguments)
                        except Exception as e:
                            tool_result_json = json.dumps({"error": f"Errore interno: {str(e)}"})
                    else:
                        tool_result_json = json.dumps({"error": f"Tool '{function_name}' sconosciuto."})
                        
                    print(f"DEBUG -> Risultato Tool: {tool_result_json}")
                    
                    # Restituiamo il risultato aggiungendolo come messaggio 'tool'
                    # ATTENZIONE: per alcuni server Ollama potrebbe esser necessario matchare il test_call id,
                    # ma per compatibilità base basta il ruolo "tool"
                    messages.append({
                        "role": "tool",
                        "content": tool_result_json,
                        "name": function_name  # Alcuni frontend vogliono anche func name
                    })
                
                # Il continue farà un giro postando la chat_history aggiornata con i risultati dei tool
                continue
                
            # Caso B: il modello ha fornito il testo finale
            content = message.get("content", "")
            return content
            
        except Exception as e:
            return f"Si è verificato un errore di connessione con Ollama: {str(e)}"

    return "C'è stato un errore: l'agente ha richiesto troppe operazioni consecutive senza chiudere la conversazione."
