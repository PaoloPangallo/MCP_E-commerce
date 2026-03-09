import json
import logging
from typing import Annotated, Dict, Any, List, TypedDict, Union

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os
import re
from datetime import datetime

from app.services.agent_tools import AGENT_TOOLS_SCHEMA, TOOLS_MAP
from app.services.agent_planner import (
    generate_plan, get_parser_llm, get_tool_caller_llm, 
    get_conversational_llm, PLANNER_MODEL
)

logger = logging.getLogger(__name__)

# Configurazione del modello (legacy, mantenuta per compatibilità se serve)
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:latest") 
llm = ChatOllama(model=MODEL_NAME, temperature=0).bind_tools(AGENT_TOOLS_SCHEMA)

class AgentState(TypedDict):
    """Lo stato del Grafo: messaggi + dati dell'e-commerce + Piano."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    shared_state: Dict[str, Any]
    plan: List[Dict[str, Any]]
    current_step_idx: int
    execution_logs: Annotated[List[str], lambda x, y: x + y]
    next_step: str 
    fresh_search_done: bool 
    iteration_count: int
    tools_called: Annotated[List[str], lambda x, y: x + y]

def planner_node(state: AgentState):
    """
    RE-PLANNER: Genera o aggiorna il piano di esecuzione.
    In questa versione, cerchiamo di ottenere un piano COMPLETO per ridurre i turnaround.
    """
    messages = state["messages"]
    user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break
            
    execution_history = "\n".join(state.get("execution_logs", []))
    last_tool_result = state.get("last_observation")
    
    # --- INGESTIONE LOG WORKER (Informed Planning) ---
    # Leggiamo l'ultimo report dettagliato per dare al planner visione completa del lavoro del worker
    worker_context = ""
    try:
        import glob
        logs_dir = "app/logs/workers"
        if os.path.exists(logs_dir):
            files = glob.glob(os.path.join(logs_dir, "worker_*.md"))
            if files:
                latest_worker_log = max(files, key=os.path.getmtime)
                with open(latest_worker_log, "r", encoding="utf-8") as f:
                    # Prendiamo solo le prime righe (input/output) per non sovraccaricare il context
                    worker_context = f.read()
                    # Limitiamo la dimensione del log per il prompt
                    if len(worker_context) > 2000:
                        worker_context = worker_context[:2000] + "... [log troncato]"
    except Exception as e:
        logger.warning(f"Errore caricamento worker logs: {e}")

    # --- KILL-SWITCH DETERMINISTICO ---
    last_log = state.get("execution_logs", [])
    if last_log and "TOOL RESULT (explain_results)" in last_log[-1]:
        logger.info("STOP DETERMINISTICO: Risultati già spiegati.")
        return {"plan": [{"tool_name": "END", "task_plan": "Task completato", "expected_input": "N/A"}]}

    # --- CHECK RISULTATI GIÀ PRESENTI (Anti-Loop / Resume) ---
    # Saltiamo alla spiegazione solo SE abbiamo risultati E abbiamo effettivamente cercato in questo ciclo.
    if state["shared_state"].get("results") and "search_products" in state.get("tools_called", []):
        logger.info("PLANNER: Risultati già presenti. Salto alla spiegazione.")
        return {
            "shared_state": state["shared_state"],
            "plan": [
                {"tool_name": "explain_results", "task_plan": "Generate final markdown report from found products.", "expected_input": user_msg},
                {"tool_name": "END", "task_plan": "Task completed.", "expected_input": "N/A"}
            ],
            "current_step_idx": 0,
            "execution_logs": state.get("execution_logs", []) + ["PLANNER: Detected results, skipping to explain."]
        }

    logger.info(f"PLANNER ({PLANNER_MODEL}): Generazione piano strategico...")
    plan = generate_plan(
        user_msg, 
        state["shared_state"], 
        last_observation=last_tool_result,
        history_context=execution_history,
        tools_called=state.get("tools_called", []),
        worker_context=worker_context
    )
    
    # DETERMISTIC LOOP BREAKER: 
    # Se parse_query è già stato fatto per questa query, lo rimuoviamo dal nuovo piano
    if "parse_query" in state.get("tools_called", []):
        plan = [s for s in plan if s["tool_name"] != "parse_query"]
        if not plan: # Se il piano diventa vuoto, forziamo una conclusione o una ricerca
            if state["shared_state"].get("results"):
                plan = [{"tool_name": "explain_results", "task_plan": "Mostra i risultati esistenti.", "expected_input": "N/A"}]
            else:
                plan = [{"tool_name": "search_products", "task_plan": "Cerca con i dati estratti.", "expected_input": "N/A"}]

    return {
        "shared_state": state["shared_state"],
        "plan": plan,
        "current_step_idx": 0,
        "execution_logs": [f"PLANNER: Generato piano con {len(plan)} passi."]
    }

def router(state: AgentState):
    """Gestisce il flusso decide se tornare all'executor, resettare col planner o finire."""
    plan = state.get("plan", [])
    idx = state.get("current_step_idx", 0)
    
    # Check stop immediato se non c'è piano o siamo alla fine
    if not plan or idx >= len(plan):
        return END

    # Se il passo corrente è END, terminiamo
    if plan[idx].get("tool_name") == "END":
        return END

    # Kill-switch se abbiamo fatto troppe iterazioni (sicurezza estrema)
    if state.get("iteration_count", 0) >= 10:
        logger.warning("ROUTER: Raggiunto limite massimo di iterazioni (10). Forza END.")
        return END

    # REATTIVITÀ (Informed Planning): Se l'ultimo tool era parse_query, torniamo al planner
    # per fargli valutare i risultati del parsing (es. missing_info dalla categoria eBay).
    last_tools = state.get("tools_called", [])
    if last_tools and last_tools[-1] == "parse_query":
        logger.info("ROUTER: Re-evaluating strategy after parse_query.")
        return "planner"

    # Se il piano ha ancora passi, andiamo all'executor per il passo idx
    return "executor"

def executor_node(state: AgentState):
    """Esegue il passo corrente del piano recuperando i parametri ottimali."""
    plan = state.get("plan", [])
    idx = state.get("current_step_idx", 0)
    
    if idx >= len(plan):
        return {"next_step": END}
    
    step = plan[idx]
    tool_name = step.get("tool_name")
    
    if tool_name == "END":
        return {
            "current_step_idx": idx + 1,
            "execution_logs": ["EXECUTOR: Raggiunto END."]
        }
    
    # ESTRAZIONE MESSAGGIO UTENTE
    user_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_msg = msg.content
            break
    
    log_entry = f"EXECUTOR ({idx+1}/{len(plan)}): Chiamata tool {tool_name}"
    logger.info(log_entry)
    
    new_shared_state = state["shared_state"].copy()
    new_shared_state.setdefault("thinking_trace", []).append(f"⚙️ {log_entry}")
    
    # SMART MODEL SWITCH: Conversational for text, Coder for structures
    verbal_tools = ["social_response", "explain_results", "request_user_clarification"]
    
    # Valida se il tool esiste nella toolbox
    valid_tools = ["parse_query", "search_products", "explain_results", "request_user_clarification", "social_response", "END"]
    if tool_name not in valid_tools:
        logger.warning(f"EXECUTOR: Tool halluncinated by planner: {tool_name}. Skipping.")
        return {
            "execution_logs": state.get("execution_logs", []) + [f"SKIP: Tool unknown {tool_name}"],
            "current_step_idx": idx + 1
        }

    if tool_name in verbal_tools:
        logger.info(f"EXECUTOR: Using Conversational model for verbal tool '{tool_name}'")
        tool_caller = get_conversational_llm()
    else:
        tool_caller = get_tool_caller_llm()
    
    current_parsed = new_shared_state.get("parsed_query") or {}
    
    execution_history = "\n".join(state.get("execution_logs", []))
    
    prompt = f"""
    You are a technical EBAY SHOPPING EXPERT.
    CURRENT TASK: {step.get('task_plan')}
    
    OPERATIONAL HISTORY:
    {execution_history}
    
    MANDATORY PARAMETER RULES:
    1. If calling 'parse_query': use ONLY the original user message: "{user_msg}"
    2. If calling 'search_products':
       - If "{current_parsed.get('semantic_query', '')}" is NOT empty, USE IT as 'query'.
       - If "{current_parsed.get('semantic_query', '')}" IS empty, STOP and ask to call 'parse_query' first. DO NOT invent attributes like Size or Color.
    3. If calling 'explain_results': use: query="{current_parsed.get('semantic_query') or user_msg}"
    
    CURRENT STATE: {json.dumps(current_parsed)}
    ORIGINAL USER REQUEST: "{user_msg}"
    
    Output ONLY the JSON Tool Call for '{tool_name}'. No conversation.
    """
    
    messages = [
        SystemMessage(content=f"You are a shopping worker assistant. Invoke tool: {tool_name}."),
        HumanMessage(content=prompt)
    ]
    
    response = tool_caller.invoke(messages)
    
    # Robust tool call extraction
    if not (hasattr(response, "tool_calls") and response.tool_calls):
        from app.services.agent_orchestrator import _extract_tool_call_from_text
        extracted = _extract_tool_call_from_text(response.content)
        if extracted:
            response.tool_calls = [{
                "name": extracted["function"]["name"],
                "args": extracted["function"]["arguments"],
                "id": f"f_{os.urandom(2).hex()}",
                "type": "tool_call"
            }]
        elif tool_name in verbal_tools and response.content.strip():
            # FALLBACK: If verbal tool and model just gave text, wrap it as a tool call
            logger.info(f"EXECUTOR: Verbal tool '{tool_name}' returned plain text, wrapping it.")
            response.tool_calls = [{
                "name": tool_name,
                "args": {"response": response.content.strip()},
                "id": f"f_{os.urandom(2).hex()}",
                "type": "tool_call"
            }]
        response.content = ""
    
    # SAFETY: Ensure parse_query always has the original query
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "parse_query":
                tc["args"] = {"query": user_msg}

    return {
        "messages": [response],
        "shared_state": new_shared_state,
        "execution_logs": [log_entry],
        "current_step_idx": idx + 1
    }

def execute_tools(state: AgentState):
    """Nodo custom per eseguire i tool e aggiornare lo shared_state."""
    logger.info("ENTERING execute_tools node")
    messages = state.get("messages", [])
    if not messages:
        logger.warning("execute_tools: No messages found in state.")
        return {"messages": []}
    
    # Cerchiamo l'ultimo messaggio AI con tool_calls (potrebbe non essere l'ultimo assoluto)
    last_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_message = msg
            break
            
    if not last_message:
        logger.warning("execute_tools: Last AIMessage has no tool_calls, skipping.")
        return {"messages": []}

    tool_outputs = []
    executed_names = []
    result = None # Inizializzazione di sicurezza
    
    # Prepariamo la cartella dei log del worker
    worker_logs_dir = "app/logs/workers"
    if not os.path.exists(worker_logs_dir):
        os.makedirs(worker_logs_dir, exist_ok=True)

    # Usiamo una copia dello stato condiviso
    new_shared_state = state["shared_state"].copy()
    pre_state_snapshot = json.dumps(new_shared_state, indent=2, default=str)

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        from app.services.agent_tools import TOOLS_MAP
        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            executed_names.append(name)
            args = tool_call["args"]
            
            # Recuperiamo il piano per questo tool dal messaggio originale (se possibile)
            current_plan_desc = "N/A"
            if state.get("plan"):
                current_plan_desc = state["plan"][0].get("task_plan", "No specific instructions")

            logger.info(f"Executing tool: {name} with args: {args}")
            
            try:
                # ESECUZIONE TOOL
                result = TOOLS_MAP[name](state=new_shared_state, **args)
                
                # LOGGING DETTAGLIATO FLUSSO DATI
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                log_file = os.path.join(worker_logs_dir, f"worker_{name}_{timestamp}.md")
                with open(log_file, "w", encoding="utf-8") as wf:
                    wf.write(f"# 🛠️ Worker Report: {name}\n\n")
                    wf.write(f"**Timestamp:** {timestamp}\n")
                    wf.write(f"**Obiettivo Planner:** {current_plan_desc}\n\n")
                    wf.write(f"## 📥 Input (Arguments)\n```json\n{json.dumps(args, indent=2, default=str)}\n```\n\n")
                    wf.write(f"## 🧠 Stato Pre-Esecuzione\n```json\n{pre_state_snapshot}\n```\n\n")
                    wf.write(f"## 📤 Output (Result)\n```json\n{result}\n```\n\n")
                    wf.write(f"## 🏁 Stato Post-Esecuzione (Aggiornato)\n```json\n{json.dumps(new_shared_state, indent=2, default=str)}\n```\n")
                
                tool_outputs.append(ToolMessage(
                    tool_call_id=tool_call.get("id", "call_none"),
                    name=name,
                    content=str(result)
                ))
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                tool_outputs.append(ToolMessage(
                    tool_call_id=tool_call.get("id", "call_err"),
                    name=name,
                    content=json.dumps({"error": str(e)})
                ))
    iterations = state.get("iteration_count", 0) + 1
    new_logs = []
    for out in tool_outputs:
        new_logs.append(f"TOOL RESULT ({out.name}): {str(out.content)[:200]}...")

    return {
        "messages": tool_outputs, 
        "shared_state": new_shared_state, 
        "tools_called": executed_names,
        "iteration_count": iterations,
        "execution_logs": new_logs,
        "last_observation": str(tool_outputs[-1].content) if tool_outputs else "",
        "last_tool_output": result if tool_outputs else None # Dati grezzi per il planner
    }

# Costruzione del Grafo
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("planner")

# 1. Il Planner genera il piano ed entra nell'executor
workflow.add_edge("planner", "executor")

# 2. L'Executor chiama i Tools
workflow.add_edge("executor", "tools")

# 3. DOPO i tools, il router decide: continuiamo il piano o abbiamo finito?
workflow.add_conditional_edges(
    "tools",
    router,
    {
        "executor": "executor",
        "planner": "planner",
        END: END
    }
)

# Compilazione
app_graph = workflow.compile()

def run_agent_graph(user_message: str, history: List[Dict[str, str]], shared_state: Dict[str, Any]):
    """Entry point per l'integrazione con FastAPI."""
    formatted_history = []
    for h in history[-8:]:
        role = h.get("role")
        content = h.get("content", "").strip()
        if not content: continue
        
        if role == "user":
            formatted_history.append(HumanMessage(content=content))
        elif role == "assistant":
            # Pulizia per messaggi tecnici
            clean_content = content
            if content.startswith("{") and ("status" in content or "explanation" in content):
                try:
                    data = json.loads(content)
                    clean_content = data.get("explanation") or data.get("social_message") or data.get("question") or content
                except:
                    pass
            
            # Se dopo la pulizia è ancora un JSON o vuoto, lo saltiamo per non sporcare
            if clean_content and not (clean_content.startswith("{") and "name" in clean_content):
                formatted_history.append(AIMessage(content=clean_content))
            
    formatted_history.append(HumanMessage(content=user_message))
    
    shared_state["last_tool_output"] = None # Reset per evitare inquinamento tra messaggi
    shared_state["worker_logs"] = [] # Reset log fisici per questo run
    
    # Eseguiamo il Grafo
    final_state = app_graph.invoke({
        "messages": formatted_history,
        "shared_state": shared_state,
        "plan": [],
        "current_step_idx": 0,
        "execution_logs": [],
        "tools_called": [],
        "iteration_count": 0,
        "fresh_search_done": False,
        "next_step": ""
    })
    
    # Estraiamo l'ultima risposta analizzando solo i messaggi NUOVI di questo ciclo
    testo_agente = ""
    s_state = final_state.get("shared_state", {})
    
    # Identifichiamo i messaggi generati solo in questo run (o ultimi 5 per sicurezza)
    messages_to_scan = final_state["messages"]
    
    explain_msg = ""
    clarify_msg = ""
    social_msg = ""
    
    # Scansioniamo TUTTI i messaggi per trovare le risposte dei tool
    for msg in reversed(messages_to_scan):
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) and (msg.content.startswith("{") or msg.content.startswith("[")) else {}
                if msg.name == "explain_results":
                    explain_msg = data.get("explanation") or (msg.content if not isinstance(data, dict) else "")
                elif msg.name == "request_user_clarification":
                    clarify_msg = data.get("question") or data.get("clarification")
                elif msg.name == "social_response":
                    social_msg = data.get("response") or data.get("social_message")
                
                # Se abbiamo trovato qualcosa, possiamo fermarci per quel tipo
                if explain_msg or clarify_msg or social_msg:
                    if not (isinstance(msg.content, str) and "None" in msg.content):
                         break
            except:
                # Fallback se non è JSON
                if msg.name == "explain_results": explain_msg = msg.content
                elif msg.name == "social_response": social_msg = msg.content
                elif msg.name == "request_user_clarification": clarify_msg = msg.content

    # Fallback estremo: se non abbiamo trovato nei ToolMessages, cerchiamo nell'ultimo AIMessage pulito
    if not explain_msg and not social_msg and not clarify_msg:
        for msg in reversed(messages_to_scan):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                testo_agente = msg.content.strip()
                break
    
    if explain_msg:
        testo_agente = explain_msg.replace("RESPONSE:", "").strip()
    elif social_msg:
        testo_agente = social_msg
    elif clarify_msg:
        testo_agente = clarify_msg
        
    # Se ancora vuoto, proviamo a recuperare dallo shared_state
    if not testo_agente:
        testo_agente = s_state.get("_explanation") or s_state.get("social_message") or ""

    # Se non c'è una risposta testuale ma abbiamo risultati, forziamo una spiegazione base
    if not testo_agente and s_state.get("results"):
        testo_agente = f"### Risultati della ricerca\nHo trovato {len(s_state['results'])} prodotti che corrispondono alla tua richiesta."

    return {
        "parsed_query": s_state.get("parsed_query") or {"semantic_query": user_message},
        "ebay_query_used": user_message,
        "results_count": len(s_state.get("results", [])),
        "saved_new_count": 0,
        "results": s_state.get("results", []),
        "rag_context": s_state.get("rag_context", ""),
        "metrics": s_state.get("metrics", {}),
        "explanation": testo_agente,
        "analysis": testo_agente,
        "thinking_trace": s_state.get("thinking_trace", []),
        "_timings": s_state.get("_timings", {}),
        "shared_state": s_state  # Aggiunto per persistenza
    }
