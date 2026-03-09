import json
import logging
from typing import List, Dict, Optional, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import os
from app.core.config import MODEL_NAME

logger = logging.getLogger(__name__)

# Configurazione LLM per l'Explainer (simile a Planner)
explainer_llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.7, # Più creativo per l'eloquenza
    keep_alive=0
)

def explain_results(query: str, items: List[Dict], missing_info: List[str] = None, ebay_category: Dict[str, Any] = None) -> str:
    """Genera una spiegazione 'Eloquente' usando l'LLM, arricchita dai metadati di categoria."""
    if not items:
        # Arricchiamo il feedback se abbiamo info sulla categoria
        category_msg = ""
        if ebay_category:
            category_msg = f" nella categoria **{ebay_category.get('category_path')}**"
        
        readable_missing = f" (magari aggiungendo: {', '.join(missing_info)})" if missing_info else ""
        return f"### 🔍 Ricerca per: *{query}*{category_msg}\n\nPurtroppo non ho trovato risultati che corrispondano esattamente ai tuoi criteri su eBay{readable_missing}. Prova a variare i termini di ricerca o a essere meno specifico."

    # Prepariamo i dati per l'LLM (top 3 risultati)
    top_3 = []
    for it in items[:3]:
        top_3.append({
            "title": it.get("title"),
            "price": f"{it.get('price')} {it.get('currency')}",
            "seller": it.get("seller_name"),
            "trust": f"{round(it.get('trust_score', 0) * 100)}%",
            "features": it.get("explanations", [])
        })

    category_context = ""
    if ebay_category:
        category_context = f"\nEBAY CATEGORY: {ebay_category.get('category_path')}\nREQUIRED ASPECTS: {', '.join(ebay_category.get('required_aspects', []))}"

    prompt = f"""
Sei un Personal Shopper di alto livello per eBay. Il tuo compito è presentare i risultati della ricerca all'utente in modo ELEQUENTE, PROFESSIONALE e PERSUASIVO.

RICHIESTA UTENTE: "{query}"{category_context}
PRODOTTI TROVATI: {json.dumps(top_3, ensure_ascii=False)}

REGOLE DI RISPOSTA:
1. Tono caloroso e da esperto, non tecnico/robotico.
2. Formattazione pulita: usa titoli ###, grassetti e liste.
3. Spiega PERCHÉ il primo prodotto è la scelta migliore (bilancio prezzo/reputazione).
4. Se ci sono alternative valide (secondo/terzo), accennale brevemente.
5. Includi suggerimenti solo se mancano parametri chiave (tipo taglia o colore).
6. Lingua: ITALIANO.

RISPOSTA:
"""
    try:
        response = explainer_llm.invoke([
            SystemMessage(content="Sei un assistente allo shopping esperto ed eloquente. Scrivi in modo fluido ed elegante."),
            HumanMessage(content=prompt)
        ])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Errore LLM Explainer: {e}")
        # Fallback se l'LLM fallisce
        best = items[0]
        return f"### 🎯 Ho trovato quello che cercavi!\nIl prodotto migliore è **{best.get('title')}** a {best.get('price')} {best.get('currency')}. L'ho scelto perché il venditore ha un'affidabilità del {round(best.get('trust_score', 0)*100)}%."
