import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

from app.core.config import PLANNER_MODEL, PARSER_MODEL, TOOL_CALLER_MODEL, NUM_CTX, NUM_GPU

def generate_plan(user_message: str, shared_state: Dict[str, Any], last_observation: str = None, history_context: str = "", tools_called: List[str] = None, worker_context: str = "") -> List[Dict[str, Any]]:
    """
    Usa l'LLM Planner con CONTESTO ESTESO e GPU per generare piani multi-step.
    """
    tools_called = tools_called or []
    planner_llm = ChatOllama(
        model=PLANNER_MODEL, 
        temperature=0,
        num_ctx=NUM_CTX,
        num_gpu=NUM_GPU,
        keep_alive=0 
    )
    
    tools_desc = """
    1. parse_query: Estrae filtri strutturati (prodotto, brand, prezzo).
    2. search_products: Cerca su eBay usando i filtri estratti.
    3. explain_results: Genera una risposta finale basata sui risultati di ricerca.
    4. request_user_clarification: Usare se la richiesta è troppo vaga.
    5. social_response: Risponde a saluti, ringraziamenti o domande generiche sulle capacità.
    6. get_seller_feedback: Analisi feedback venditore.
    7. rerank_products: Riordinamento risultati.
    8. END: Task concluso.
    """

    current_parsed = shared_state.get('parsed_query') or {}
    
    obs_context = f"\nLAST OBSERVATION:\n{last_observation}" if last_observation else ""
    history_sec = f"\nCURRENT SESSION LOGS:\n{history_context}" if history_context else ""
    worker_sec = f"\nWORKER_INSIGHTS (Internal tool logs):\n{worker_context}" if worker_context else ""
    called_sec = f"\nTOOLS ALREADY EXECUTED FOR THIS REQUEST: {', '.join(tools_called)}" if tools_called else ""
    
    prompt_template = """
    You are the CHIEF STRATEGIST for a premium eBay shopping assistant.
    Your goal is to generate a COMPLETE EXECUTION PLAN to fulfill the user's request.
    
    {history_sec}
    {worker_sec}
    {obs_context}
    
    PLANNING CONTEXT:
    - User Message: "{user_message}"
    - Parsed Data in Memory: {current_parsed_json}
    - Last Structured Result: {last_result_json}
    - Tools already called for THIS message: {tools_called_list}

    🔍 PLANNING ORIENTATION:
    1. EXAMPLES OF INTENT:
       - User: "Ciao, come stai?" -> Tool: social_response (Purely social)
       - User: "Ciao! Cercami delle scarpe Nike" -> Tool: parse_query (Shopping intent detected)
       - User: "Top! Ora cerca jeans Levi's" -> Tool: parse_query (New shopping intent after praise)
    
    2. THE GOLDEN RULE:
       - If any PRODUCT (jeans, sweater, pants) or BRAND (Nike, Levi's, CK) is mentioned, YOU MUST plan 'parse_query' first.
       - Use 'social_response' ONLY for empty talk, greetings, or "thank you" without new requests.

    3. SHOPPING FLOW:
       - Phase 1: 'parse_query' (Mandatory for new items).
       - Phase 2: 'request_user_clarification' (Only if Size/Gender is missing).
       - Phase 3: 'search_products' -> 'explain_results' (Only when info is complete).
    
    Response format: ONLY a JSON array of objects.
    Each object: {"tool_name": "...", "task_plan": "...", "expected_input": "..."}
    
    AVAILABLE TOOLS:
    - parse_query: Extract search filters from text.
    - search_products: Find items on eBay.
    - explain_results: Generate a report of found items.
    - request_user_clarification: Ask user for missing details like size.
    - social_response: Handle generic chat/greetings.
    - END: Finish the current task.
    """
    
    prompt = prompt_template.replace("{history_sec}", history_sec)\
                           .replace("{worker_sec}", worker_sec)\
                           .replace("{obs_context}", obs_context)\
                           .replace("{user_message}", user_message)\
                           .replace("{current_parsed_json}", json.dumps(current_parsed))\
                           .replace("{last_result_json}", json.dumps(shared_state.get('last_tool_output')) if shared_state.get('last_tool_output') else 'None')\
                           .replace("{tools_called_list}", ', '.join(tools_called) if tools_called else 'None')

    try:
        response = planner_llm.invoke([
            SystemMessage(content="You are an expert Planner. Return ONLY a valid JSON array. NEVER use Python-style booleans (True, False) or None. Use STRICT JSON (true, false, null). NO EXTRA TEXT."),
            HumanMessage(content=prompt)
        ])
        
        raw_text = response.content.strip()
        
        # Pulizia aggressiva del JSON
        json_text = raw_text
        if "```json" in raw_text:
            json_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            json_text = raw_text.split("```")[1].split("```")[0].strip()
        
        # Rimozione rumenta comune (apostrofi evasi male, trailing commas)
        json_text = json_text.replace("\\'", "'")
        import re
        
        # CORREZIONE BOOLEANI E NULL: Fix per modelli che confondono Python e JSON
        json_text = re.sub(r'\bTrue\b', 'true', json_text)
        json_text = re.sub(r'\bFalse\b', 'false', json_text)
        json_text = re.sub(r'\bNone\b', 'null', json_text)
        
        json_text = re.sub(r',\s*\]', "]", json_text) # Trailing commas in array
        json_text = re.sub(r',\s*\}', "}", json_text) # Trailing commas in objects
            
        plan_data = json.loads(json_text)
        
        # FLATTEN: Se l'LLM ha annidato degli array per errore, li appiattiamo
        if isinstance(plan_data, list):
            flattened = []
            for item in plan_data:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            plan_data = flattened
        
        if not isinstance(plan_data, list):
            plan_data = [plan_data]
            
        normalized_plan = []
        for step in plan_data:
            n_step = {
                "tool_name": step.get("tool_name") or step.get("tool") or step.get("function") or "END",
                "task_plan": step.get("task_plan") or step.get("reason") or step.get("instructions") or "No instructions",
                "expected_input": step.get("expected_input") or step.get("input") or "N/A"
            }
            normalized_plan.append(n_step)
            
        save_plan_to_file(user_message, normalized_plan, shared_state, last_observation)
        
        return normalized_plan
    except Exception as e:
        logger.error(f"Errore generazione piano: {e} | TEXT: {response.content if 'response' in locals() else 'N/A'}")
        return [{"tool_name": "END", "task_plan": f"Errore parsing JSON del piano: {str(e)}", "expected_input": ""}]

def save_plan_to_file(user_message: str, plan: List[Dict[str, Any]], shared_state: Dict[str, Any], last_obs: str = None):
    """Salva il piano in markdown con traccia del lavoro svolto."""
    try:
        plans_dir = "app/logs/plans"
        if not os.path.exists(plans_dir):
            os.makedirs(plans_dir, exist_ok=True)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plan_{timestamp}.md"
        filepath = os.path.join(plans_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# 🗺️ Piano di Esecuzione - {timestamp}\n\n")
            f.write(f"**Richiesta Utente:** {user_message}\n")
            
            parsed_data = shared_state.get('parsed_query') or {}
            if not isinstance(parsed_data, dict): parsed_data = {}
            
            f.write(f"**Query Ottimizzata:** {parsed_data.get('search_query', 'Non ancora disponibile')}\n\n")
            
            f.write("## 🎯 Progressi & Risultati\n")
            if last_obs:
                obs_preview = str(last_obs)[:500] + "..." if len(str(last_obs)) > 500 else str(last_obs)
                f.write(f"**Ultimo Risultato Tecnico:**\n> {obs_preview}\n\n")
            else:
                f.write("*Inizio del ciclo operativo o primo passo del piano.*\n\n")

            f.write("## 🛠️ Task Correnti per il Worker\n\n")
            for i, step in enumerate(plan):
                f.write(f"### {i+1}. Tool: `{step.get('tool_name')}`\n")
                f.write(f"- **🎯 Obiettivo:** {step.get('task_plan')}\n")
                f.write(f"- **📥 Input che deve usare il Worker:** `{step.get('expected_input')}`\n\n")
                
        logger.info(f"Piano salvato in {filepath}")
        
        files = [os.path.join(plans_dir, f) for f in os.listdir(plans_dir) if f.endswith(".md")]
        if len(files) > 20:
            files.sort(key=os.path.getmtime)
            for old_file in files[:-20]:
                try:
                    os.remove(old_file)
                except:
                    pass
    except Exception as e:
        logger.error(f"Errore salvataggio piano su file: {e}")

CONVERSATIONAL_MODEL = os.getenv("CONVERSATIONAL_MODEL", "mistral-nemo:12b")

def get_parser_llm():
    """Modello specializzato per il parsing testuale con GPU."""
    return ChatOllama(
        model=PARSER_MODEL, 
        temperature=0,
        num_ctx=NUM_CTX,
        num_gpu=NUM_GPU,
        keep_alive=0
    )

def get_tool_caller_llm():
    """Modello specializzato per tool calling con GPU."""
    from app.services.agent_tools import AGENT_TOOLS_SCHEMA
    return ChatOllama(
        model=TOOL_CALLER_MODEL, 
        temperature=0,
        num_ctx=NUM_CTX,
        num_gpu=NUM_GPU,
        keep_alive=0
    ).bind_tools(AGENT_TOOLS_SCHEMA)

def get_conversational_llm():
    """Modello orientato alla conversazione per risposte sociali e spiegazioni."""
    from app.services.agent_tools import AGENT_TOOLS_SCHEMA
    return ChatOllama(
        model=CONVERSATIONAL_MODEL, 
        temperature=0.3, # Leggera creatività per l'eloquenza
        num_ctx=NUM_CTX,
        num_gpu=NUM_GPU,
        keep_alive=0
    ).bind_tools(AGENT_TOOLS_SCHEMA)
