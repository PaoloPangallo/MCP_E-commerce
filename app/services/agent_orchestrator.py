import json
import logging
import time
import traceback
from typing import Dict, Any, List

import requests

from app.services.agent_tools import AGENT_TOOLS_SCHEMA, TOOLS_MAP

logger = logging.getLogger(__name__)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
import os
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1")

def ask_agent_orchestrator(
    user_message: str, 
    history: List[Dict[str, str]],
    db_session: Any, 
    user_obj: Any,
    t0: float, 
    context: Dict[str, Any] = None,
    ecommerce_pipeline_func=None 
) -> Dict[str, Any]:
    
    # -------------------------------------------------------------
    # LO STATO CONDIVISO (Sostituisce LangGraph!)
    # Questo state viene letto e aggiornato da tutti i tool chiamati.
    # Alla fine, lo passiamo formattato a React!
    # -------------------------------------------------------------
    shared_state: Dict[str, Any] = {
        "results": [],
        "parsed_query": context or None,
        "thinking_trace": [], # ui history trace
        "rag_context": "",
        "metrics": {},
        "_timings": {},
        "seller_feedbacks": {},
        "db_session": db_session,
        "user_obj": user_obj,
    }
    
    sys_prompt = (
        "Sei un assistente esperto per lo shopping su eBay.\n"
        "REGOLE AGENTICHE (TASSATIVE):\n"
        "1) FLUSSO: Esegui SEMPRE parse_query -> search_products -> (opzionali altri tool) -> explain_results.\n"
        "2) CHIARIMENTO: Se mancano dettagli fondamentali (taglia scarpe, memoria telefoni), usa 'request_user_clarification' e FERMATI.\n"
        "3) RICERCA: Se hai i dettagli, chiama 'parse_query'. Appena ricevi l'output, NON PARLARE, chiama SUBITO 'search_products'.\n"
        "4) FORMATO: Quando chiami un tool, il tuo messaggio deve avere il campo 'content' COMPLETAMENTE VUOTO. Non scrivere MAI JSON nel testo.\n"
        "5) CONTESTO: Usa il messaggio 'SITUAZIONE ATTUALE' per non fare domande ripetitive.\n"
        "6) CONCLUSIONE: Dopo 'search_products', chiudi sempre con 'explain_results' riassumendo i risultati e chiedendo se serve altro aiuto.\n"
    )
    
    messages = [{"role": "system", "content": sys_prompt}]
    
    # Historico (ultimi 6 scambi)
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
        
    # Inserimento dello stato attuale per prevenire loop e domande inutili
    if shared_state["parsed_query"]:
        context_msg = f"SITUAZIONE ATTUALE DELLA RICERCA: {json.dumps(shared_state['parsed_query'])}"
        messages.append({"role": "system", "content": context_msg})

    messages.append({"role": "user", "content": user_message})
    
    # Aumentiamo i cicli per permettere l'Agentic Multi-Step Reasoning
    MAX_ITER = 12 
    testo_agente = ""
    
    for iteration in range(MAX_ITER):
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": AGENT_TOOLS_SCHEMA,
            "stream": False
        }
        
        try:
            resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=120)
            if resp.status_code != 200:
                logger.error(f"Errore API Ollama: {resp.text}")
                return {"error": "LLM_ERROR"}
                
            json_resp = resp.json()
            message = json_resp.get("message", {})
            messages.append(message)
            
            # --- FALLBACK: Llama 3.1 8B puts JSON in content instead of tool_calls? ---
            tool_calls = message.get("tool_calls", [])
            content_text = message.get("content", "").strip()
            
            if not tool_calls and "{\"name\":" in content_text:
                try:
                    # Tenta di estrarre un JSON dal content se Ollama non l'ha mappato
                    start = content_text.find("{")
                    end = content_text.rfind("}") + 1
                    possible_json = json.loads(content_text[start:end])
                    if "name" in possible_json and "parameters" in possible_json:
                        tool_calls = [{
                            "function": {
                                "name": possible_json["name"],
                                "arguments": possible_json["parameters"]
                            }
                        }]
                        logger.info(f"Fallback: tool call trovato in content text.")
                except:
                    pass

            if tool_calls:
                logger.info(f"[Orchestrator Iter {iteration+1}] L'agente ha scelto {len(tool_calls)} tools.")
                
                for tool in tool_calls:
                    function_name = tool.get("function", {}).get("name")
                    args = tool.get("function", {}).get("arguments", {})
                    
                    if function_name in TOOLS_MAP:
                        executor = TOOLS_MAP[function_name]
                        try:
                            result_json = executor(**args, state=shared_state)
                        except Exception as e:
                            logger.error(f"Errore nel tool {function_name}: {e}")
                            result_json = json.dumps({"error": f"Tool error: {str(e)}"})
                    else:
                        result_json = json.dumps({"error": "Unknown tool."})
                        
                    messages.append({
                        "role": "tool",
                        "name": function_name,
                        "content": result_json
                    })
                    
                # Continue fa ripartire il ciclo e rimanda i risultati all'LLM (a meno che non abbia chiamato tool finali)
                is_done = False
                for tool in tool_calls:
                    fn_name = tool.get("function", {}).get("name")
                    if fn_name in ["explain_results", "request_user_clarification"]:
                        args = tool.get("function", {}).get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                args = {}
                        
                        if fn_name == "explain_results":
                            testo_agente = shared_state.get("_explanation", args.get("explanation", "Risultati elaborati."))
                        else:
                            testo_agente = args.get("question", "Potresti chiarire alcuni dettagli?")
                            
                        is_done = True
                        break
                        
                if is_done:
                    shared_state["_timings"]["total_s"] = round(time.time() - t0, 3)
                    return {
                        "parsed_query": shared_state.get("parsed_query") or {"semantic_query": user_message},
                        "ebay_query_used": user_message,
                        "results_count": len(shared_state.get("results", [])),
                        "saved_new_count": 0,
                        "results": shared_state.get("results", []),
                        "rag_context": shared_state.get("rag_context", ""),
                        "metrics": shared_state.get("metrics", {}),
                        "_timings": shared_state.get("_timings", {}),
                        "analysis": testo_agente,
                        "thinking_trace": shared_state.get("thinking_trace", [])
                    }
                    
                continue
            
            # --- FALLBACK DETERMINISTICO ---
            # Se l'LLM ha generato testo invece di chiamare un tool,
            # forziamo l'esecuzione dei tool mancanti nella pipeline.
            logger.info("L'Agente ha generato testo senza tool call. Attivo fallback deterministico.")
            
            llm_text = message.get("content", "")
            has_parsed = shared_state.get("parsed_query") is not None
            has_results = len(shared_state.get("results", [])) > 0
            
            # Se abbiamo parse ma non risultati → forza search_products
            if has_parsed and not has_results:
                logger.info("[Fallback] Forzo search_products...")
                parsed = shared_state["parsed_query"]
                search_query = (
                    parsed.get("semantic_query") 
                    or parsed.get("product") 
                    or user_message
                )
                try:
                    search_result = TOOLS_MAP["search_products"](
                        query=search_query, state=shared_state
                    )
                    shared_state["thinking_trace"].append("⚙ search forzato (fallback)")
                    logger.info(f"[Fallback] search_products eseguito: {len(shared_state.get('results', []))} risultati")
                except Exception as e:
                    logger.error(f"[Fallback] Errore in search_products: {e}")
            
            # Se abbiamo risultati → forza explain_results
            if shared_state.get("results"):
                try:
                    explain_query = user_message
                    TOOLS_MAP["explain_results"](
                        query=explain_query, state=shared_state
                    )
                    shared_state["thinking_trace"].append("⚙ explain forzato (fallback)")
                except Exception as e:
                    logger.error(f"[Fallback] Errore in explain_results: {e}")
            
            testo_agente = shared_state.get("_explanation", llm_text or "Risultati elaborati.")
            
            shared_state["_timings"]["total_s"] = round(time.time() - t0, 3)
            
            return {
                "parsed_query": shared_state.get("parsed_query") or {"semantic_query": user_message},
                "ebay_query_used": user_message,
                "results_count": len(shared_state.get("results", [])),
                "saved_new_count": 0,
                "results": shared_state.get("results", []),
                "rag_context": shared_state.get("rag_context", ""),
                "metrics": shared_state.get("metrics", {}),
                "_timings": shared_state.get("_timings", {}),
                "analysis": testo_agente,
                "thinking_trace": shared_state.get("thinking_trace", [])
            }
            
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

    # Se esaurisce tutte le 12 iterazioni 
    return {"error": "Too many agentic iterations"}
