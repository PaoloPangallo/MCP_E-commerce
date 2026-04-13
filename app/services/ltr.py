import os
import logging
import json
import re
from typing import List, Dict, Any, Optional

from app.llm.client import call_llm

logger = logging.getLogger(__name__)

# Prompt avanzato con Few-Shot per distinguere Hardware da Ricambi/Accessori
RELEVANCE_PROMPT_TEMPLATE = """
Ti occupi di valutare la pertinenza di prodotti e-commerce per la query: "{query}".
Il tuo obiettivo è massimizzare la soddisfazione dell'utente che cerca un PRODOTTO COMPLETO E FUNZIONANTE (es. un telefono sciolto, un laptop), scartando ricambi, accessori o componenti (a meno che non siano esplicitamente richiesti).

### REGOLE DI VALUTAZIONE (Ritorna un JSON):
1. RELEVANCE (punteggio da -1.0 a 1.0):
    - 1.0: Match Perfetto (il dispositivo intero cercato).
    - 0.5: Match parziale o correlato (es. un modello diverso ma pertinente).
    - 0.0: Dubbio o poco correlato.
    - -1.0: DA SCARTARE ASSOLUTAMENTE. Include: Accessori (case, cover, caricatori), Parti di ricambio (schermi LCD, batterie, scocche), Scatole vuote, Manuali, Oggetti "finti" (dummy).

### LOGICA DI PREZZO E TITOLO:
- Sii estremamente sospettoso se il titolo parla di un dispositivo (es. "iPhone 12") ma il prezzo è quello di un accessorio (es. 20-50€). In tal caso, è quasi certamente un ricambio o un accessorio. Segnala come -1.0.

### ESEMPI:
- Query: "iPhone 12" | Prodotto: "iPhone 12 128GB" | 400€ -> {{ "relevance": 1.0, "motivation": "Prodotto corretto" }}
- Query: "iPhone 12" | Prodotto: "Vetro temperato per iPhone 12" | 10€ -> {{ "relevance": -1.0, "motivation": "Accessorio non richiesto" }}
- Query: "iPhone 12" | Prodotto: "Batteria iPhone 12" | 30€ -> {{ "relevance": -1.0, "motivation": "Ricambio non richiesto" }}

Dati in Input:
{items_list}

Ritorna ESCLUSIVAMENTE un JSON array di oggetti con: "id", "relevance", "value", "motivation".
"""

async def rerank_items(
    query: str, 
    items: List[Dict[str, Any]], 
    exclusions: Optional[List[str]] = None,
    rag_context: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Giudice LLM Unificato (Deep-Check 3.0). 
    Analizza Titolo, Specifiche tecniche dettagliate e Descrizione per una validazione totale.
    """
    if not items:
        return []

    # Configurazione esclusioni per il prompt
    excl_str = ", ".join(exclusions) if exclusions else "Nessuna"
    
    # Selezioniamo massimo 20 item per garantire focus e profondità (Reasoning richiede token)
    top_items = items[:20]
    items_list_parts = []
    
    for i, it in enumerate(top_items):
        title = str(it.get('title', '')).strip().replace('"', "'")
        price = it.get('price', 'N/A')
        cond = it.get('condition', 'N/A')
        
        # Dati Profondi (da Total Enrichment)
        specs = it.get('full_item_specifics', [])
        # Trasformiamo la lista di aspetti in una stringa leggibile
        specs_str = ", ".join([f"{s.get('localizedAspectName')}: {s.get('localizedAspectValue')}" for s in specs[:10]]) if specs else "N/A"
        
        desc = it.get('full_description', '')
        desc_snippet = desc[:1200].replace('\n', ' ') if desc else "Non disponibile"
        
        # Filtro RAG: integriamo il contesto storico se pertinente
        item_rag = []
        if rag_context:
            title_words = set(re.findall(r"\w+", title.lower()))
            for doc in rag_context:
                doc_text = doc.get("text", "").lower()
                if any(w in doc_text for w in title_words if len(w) > 3):
                    item_rag.append(doc.get("text"))
        rag_info = f" | RAG: {(' '.join(item_rag[:1]))[:200]}" if item_rag else ""
        
        items_list_parts.append(
            f"--- ID {i} ---\n"
            f"TITOLO: {title}\n"
            f"PREZZO: {price} | CONDIZIONE: {cond}\n"
            f"SPECIFICHE: {specs_str}\n"
            f"DESCRIZIONE: {desc_snippet}\n"
            f"{rag_info}"
        )
        
    items_list_str = "\n\n".join(items_list_parts)
    
    # Prompt con CHAIN-OF-THOUGHT (Ragionamento obbligatorio)
    prompt = f"""
Sei un esperto pignolo di e-commerce. Valuta la pertinenza dei prodotti per la query: "{query}"
VINCOLI DI ESCLUSIONE (L'utente NON vuole assolutamente): {excl_str}

### PROTOCOLLO DI VALUTAZIONE PER OGNI PRODOTTO:
1. **Analisi Critica (Reasoning)**: Leggi Titolo, Specifiche e Descrizione. 
   - Verifica se il prodotto è quello cercato (marca, categoria).
   - RICERCA ELEMENTI ESCLUSI: Cerca parole chiave o concetti legati alle esclusioni (es. se excl è 'zip', cerca 'zip', 'zipper', 'cerniera', 'zip-up' nella descrizione o specifiche).
2. **Veredetto di Rilevanza (-1.0 a 1.0)**:
   - 1.0: Match Perfetto e SENZA violazioni.
   - -1.0: VIOLAZIONE VINCOLI o Prodotto errato (es. ricambio/accessorio). Se trovi un elemento escluso (es. cerniera/zip), devi assegnare -1.0.

Ritorna un JSON array di oggetti con: "id", "reasoning", "relevance", "value", "motivation".
L'Analisi Critica va nel campo "reasoning". La sintesi del verdetto in "motivation".

Dati in Input:
{items_list_str}
""".strip()

    score_map = {}
    value_map = {}
    motivation_map = {}

    try:
        response, _ = await call_llm(prompt)
        if response:
            # Pulizia e parsing (estratto per brevità)
            if "```" in response:
                json_match = re.search(r"```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```", response, re.DOTALL | re.IGNORECASE)
                if json_match:
                    response = json_match.group(1)
            
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                response = response[start_idx:end_idx+1]
                
            scores = json.loads(response)
            if isinstance(scores, list):
                for s in scores:
                    item_id = s.get("id") if s.get("id") is not None else s.get("ID")
                    rel = s.get("relevance")
                    val = s.get("value", 0.5)
                    reason = s.get("reasoning", "")
                    motivation = s.get("motivation", "")
                    
                    if item_id is not None and rel is not None:
                        score_map[int(item_id)] = float(rel)
                        value_map[int(item_id)] = float(val)
                        # Combiniamo reasoning e motivation per l'utente
                        motivation_map[int(item_id)] = f"{motivation} ({reason[:150]}...)" if reason else motivation
    except Exception as e:
        logger.error("LTR | Expert Judge failed to parse response: %s", e)

    # ── APPLICA GLI SCORE E FILTRA ──
    final_items = []
    # La rilevanza è ora affidata interamente al Giudice LLM potenziato
    
    for i, item in enumerate(items):
        if i < len(top_items):
            rel_score = score_map.get(i)
            val_score = value_map.get(i, 0.5)
            motivation = motivation_map.get(i, "")
            
            # Fallback se l'LLM fallisce: usiamo segnali di categoria se presenti, 
            # altrimenti assumiamo una rilevanza cautelativa (0.5) invece di usare euristiche di testo.
            if rel_score is None:
                cat_low = str(item.get("category_name", "")).lower()
                is_part_cat = any(pk in cat_low for pk in ["parti", "componenti", "accessori", "parts"])
                rel_score = -0.5 if is_part_cat else 0.5
                val_score = 0.5
        else:
            # Item oltre i primi 20
            rel_score = 0.5
            val_score = 0.5
            motivation = "Non valutato profondamente"
            
        item['_ltr_score'] = rel_score
        item['value_score'] = val_score
        
        if motivation:
            item["_llm_motivation"] = motivation
            exps = list(item.get("explanations") or [])
            # Inseriamo il verdetto dell'esperto come prima spiegazione
            item["explanations"] = [f"⚖️ {motivation}"] + exps[:5]
            
        final_items.append(item)

    # ── FILTRO E ORDINAMENTO ──
    # Manteniamo solo prodotti con rilevanza positiva (> 0.2) per essere più selettivi
    final_list = [it for it in final_items if it.get('_ltr_score', 0) > 0.2]
    
    def sort_key(x):
        title = str(x.get("title", "")).lower()
        score = float(x.get("_ltr_score", 0))
        
        # Unica euristica residua richiesta dall'utente: telefoni sbloccati in fondo
        is_sbloccato = "sbloccato" in title
        
        # Priorità MASSIMA allo score LLM. La penalità 'sbloccato' è secondaria.
        return (score - (0.5 if is_sbloccato else 0.0), x.get("_final_score", 0))

    ranked_list = sorted(final_list, key=sort_key, reverse=True)
    
    logger.info("LTR | Expert Judge completato per '%s'. Mantengo %d/%d", query, len(ranked_list), len(items))
    return ranked_list
