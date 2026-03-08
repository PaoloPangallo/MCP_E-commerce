import json
import os
import logging
import time
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

SESSIONS_DIR = "app/logs/sessions"

def save_session_state(session_id: str, state: Dict[str, Any]):
    """Salva lo stato della sessione in un file JSON e un riepilogo MD."""
    if not os.path.exists(SESSIONS_DIR):
        os.makedirs(SESSIONS_DIR, exist_ok=True)
    
    # Save JSON for programmatic access
    json_path = os.path.join(SESSIONS_DIR, f"session_{session_id}.json")
    try:
        # Rimuoviamo oggetti non serializzabili e aggiungiamo timestamp
        serializable_state = {k: v for k, v in state.items() if k not in ["db_session", "user_obj"]}
        serializable_state["_last_updated"] = time.time()
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_state, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Errore salvataggio sessione JSON: {e}")

    # Save MD for human readability and "eloquence"
    md_path = os.path.join(SESSIONS_DIR, f"session_{session_id}.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 🧠 Memoria Sessione: {session_id}\n")
            f.write(f"Ultimo aggiornamento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## 🔍 Stato Ricerca Corrente\n")
            pq = state.get("parsed_query") or {}
            f.write(f"- **Prodotto:** {pq.get('product', 'N/A')}\n")
            f.write(f"- **Brand:** {', '.join(pq.get('brands', [])) or 'N/A'}\n")
            f.write(f"- **Query eBay:** `{pq.get('search_query', 'N/A')}`\n")
            f.write(f"- **Compatibilità:** {json.dumps(pq.get('compatibilities', {}))}\n\n")
            
            f.write(f"## 📊 Ultimi Risultati\n")
            results = state.get("results", [])
            f.write(f"- Trovati {len(results)} prodotti.\n")
            if results:
                f.write(f"- Esempio: {results[0].get('title')} ({results[0].get('price')} {results[0].get('currency')})\n")
            
            f.write(f"\n## 💭 Thinking Trace\n")
            for trace in state.get("thinking_trace", [])[-5:]:
                f.write(f"- {trace}\n")
    except Exception as e:
        logger.error(f"Errore salvataggio sessione MD: {e}")

def load_session_state(session_id: str) -> Dict[str, Any]:
    """Carica lo stato della sessione dal file JSON."""
    json_path = os.path.join(SESSIONS_DIR, f"session_{session_id}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Errore caricamento sessione: {e}")
    return {}
