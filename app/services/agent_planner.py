import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# Configurazione modelli (personalizzabili via env)
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "mistral-nemo:12b")
PARSER_MODEL = os.getenv("PARSER_MODEL", "qwen2.5-coder:7b")
TOOL_CALLER_MODEL = os.getenv("TOOL_CALLER_MODEL", "qwen2.5-coder:7b")

# Parametri estesi (GPU & Context)
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "1"))

def generate_plan(user_message: str, shared_state: Dict[str, Any], last_observation: str = None, history_context: str = "") -> List[Dict[str, Any]]:
    """
    Usa l'LLM Planner con CONTESTO ESTESO e GPU per generare piani multi-step.
    """
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
    
    prompt = f"""
    You are the CHIEF STRATEGIST for a premium eBay shopping assistant.
    Your goal is to generate a COMPLETE EXECUTION PLAN (sequence of steps) to fulfill the user's request.
    
    {history_sec}
    {obs_context}
    
    CURRENT STATE (Extracted Data): {json.dumps(current_parsed)}
    
    🔍 STRATEGY RULES:
    1. ANALYZE INTENT: 
       - If greeting/thanks/capabilities: plan 'social_response'. 
       - IMPORTANT: If CURRENT STATE has a product/query (welcome back scenario), YOU SHOULD ALSO plan 'explain_results' to show consistency.
    2. REFINING / UPDATING: If the user provides a NEW constraint (e.g., "color gold", "size 44", "cheaper") related to the topic in CURRENT STATE, plan:
       - 'parse_query' (to merge the new detail)
       - 'search_products'
       - 'explain_results'
    3. TOPIC CHANGE: If the user mentions a DIFFERENT product category (e.g., Jeans -> Shoes), plan a fresh 'parse_query' and IGNORE old context.
    4. SEARCH: If you have a valid 'search_query' for the active topic and need results, plan 'search_products'.
    5. CLARIFICATION: Only call 'request_user_clarification' if BOTH the user message and CURRENT STATE provide zero specific guidance on style/model (e.g., just "shoes" with no brand and no previous context). If a brand like 'Adidas' is present, DO NOT clarify, just search.
    6. EXPLAIN: If you have updated results in the shared_state/history, ALWAYS plan 'explain_results' as the final step before END.
    7. COMPLETION: Once 'explain_results' has been successfully executed, plan ONLY [{{"tool_name": "END", "task_plan": "Goal reached" }}].
    
    Response format: ONLY a JSON array of objects.
    Each object MUST have: 'tool_name', 'task_plan', 'expected_input'.
    
    AVAILABLE TOOLS: {tools_desc}
    USER MESSAGE: {user_message}
    """

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
