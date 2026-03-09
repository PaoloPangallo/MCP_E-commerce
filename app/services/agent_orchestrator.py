import json
import logging
import time
import traceback
import re
from typing import Dict, Any, List

import requests

from app.services.agent_tools import AGENT_TOOLS_SCHEMA, TOOLS_MAP

from app.core.config import OLLAMA_CHAT_URL, MODEL_NAME

logger = logging.getLogger(__name__)

def _extract_tool_call_from_text(text: str) -> Dict[str, Any]:
    """
    Attempts to extract a tool call from text content.
    Handles:
    - JSON: {"name": "...", "parameters": {...}}
    - Python-style: name(arg1="val", arg2=123)
    """
    if not text:
        return None
    
    text = text.strip()

    # 1. PROVA JSON
    try:
        start = text.find("{")
        if start != -1:
            end = text.rfind("}") + 1
            possible_json = json.loads(text[start:end])
            
            # Formato diretto o array
            if isinstance(possible_json, list) and len(possible_json) > 0:
                obj = possible_json[0]
            else:
                obj = possible_json
                
            name = obj.get("name")
            args = obj.get("arguments") or obj.get("parameters") or {}
            
            if name:
                return {"function": {"name": name.replace("call_", ""), "arguments": args}}
    except:
        pass

    # 2. PROVA PYTHON-STYLE REGEX: name(arg="val")
    import re
    # Pattern per catturare nome(key="val", ...)
    func_match = re.search(r"(\w+)\s*\((.*)\)", text)
    if func_match:
        func_name = func_match.group(1)
        args_str = func_match.group(2)
        
        # Filtriamo nomi validi di tool
        from app.services.agent_tools import TOOLS_MAP
        if func_name in TOOLS_MAP:
            args = {}
            # Cerchiamo coppie key="val" o key='val'
            arg_matches = re.findall(r"(\w+)\s*=\s*['\"]([^'\"]+)['\"]", args_str)
            for k, v in arg_matches:
                args[k] = v
            
            # Se non ha trovato argomenti nominati, proviamo a estrarre tutto tra parentesi come 'query'
            if not args and len(args_str) > 2:
                clean_val = args_str.strip("'\" ")
                args["query"] = clean_val

            return {"function": {"name": func_name, "arguments": args}}

    # 3. PROVA NATURAL LANGUAGE (NLP Pattern): "Chiamo search_products con query '...'"
    from app.services.agent_tools import TOOLS_MAP
    # Pattern più flessibile per beccare "Chiamo search_products con query 'nike'" o "search_products con iphone"
    for tool_name in ["search_products", "parse_query"]:
        if f"`{tool_name}`" in text or tool_name in text:
            # Cerchiamo quello che sembra un parametro (testo tra virgolette o dopo prep)
            quote_match = re.search(r"['\"]([^'\"]{3,})['\"]", text)
            if quote_match:
                return {"function": {"name": tool_name, "arguments": {"query": quote_match.group(1)}}}
            
            # Se non ci sono virgolette, proviamo a estrarre sostantivi forti dopo certe keyword
            fallback_match = re.search(r"(?:query|con|per|cercando|su)\s+([a-zA-Z0-9\s]+)$", text, re.IGNORECASE)
            if fallback_match:
                return {"function": {"name": tool_name, "arguments": {"query": fallback_match.group(1).strip()}}}

    return None

from app.services.agent_graph import run_agent_graph

from app.services.session_persistence import save_session_state, load_session_state

def ask_agent_orchestrator(
    user_message: str, 
    history: List[Dict[str, str]],
    db_session: Any, 
    user_obj: Any,
    t0: float, 
    context: Dict[str, Any] = None,
    ecommerce_pipeline_func=None 
) -> Dict[str, Any]:
    """
    Nuovo Orchestratore basato su LangGraph con PERSISTENZA STATO.
    """
    session_id = str(user_obj.id) if user_obj and hasattr(user_obj, "id") else "guest"
    SESSION_TTL = 900 # 15 minuti
    
    # Tentiamo il recupero solo se il contesto è assente
    is_greeting = re.search(r"^(ciao|hi|buongiorno|salve|hey)\b", user_message.lower().strip())
    if not context and not (session_id == "guest" and is_greeting):
        persisted = load_session_state(session_id)
        if persisted and "_last_updated" in persisted:
            age = time.time() - persisted["_last_updated"]
            
            # Recuperiamo sempre se la sessione è "fresca" (age < TTL)
            if age < SESSION_TTL:
                logger.info(f"ORCHESTRATOR: Recovered context for session '{session_id}' (age: {int(age)}s).")
                context = persisted.get("parsed_query")
            else:
                logger.info(f"ORCHESTRATOR: Persisted context for '{session_id}' expired (age: {int(age)}s).")

    shared_state: Dict[str, Any] = {
        "results": [],
        "parsed_query": context or None,
        "thinking_trace": [],
        "rag_context": "",
        "metrics": {},
        "_timings": {},
        "seller_feedbacks": {},
        "db_session": db_session,
        "user_obj": user_obj,
    }
    
    try:
        # Chiamata al Grafo
        response_payload = run_agent_graph(
            user_message=user_message,
            history=history,
            shared_state=shared_state
        )
        
        # PERSISTENZA: salviamo lo stato aggiornato per la prossima volta
        if "shared_state" in response_payload:
            save_session_state(session_id, response_payload["shared_state"])
        
        # Aggiungiamo i timing finali se non presenti
        if "_timings" in response_payload:
             response_payload["_timings"]["total_s"] = round(time.time() - t0, 3)
             
        return response_payload

    except Exception as e:
        logger.error(f"Errore nel Grafo dell'Agente: {e}")
        traceback.print_exc()
        return {"error": str(e)}
