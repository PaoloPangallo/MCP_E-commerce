import json
import logging
from typing import Annotated, Dict, Any, List, TypedDict, Union

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os
import re

from app.services.agent_tools import AGENT_TOOLS_SCHEMA, TOOLS_MAP

logger = logging.getLogger(__name__)

# Configurazione del modello
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:latest") 
llm = ChatOllama(model=MODEL_NAME, temperature=0).bind_tools(AGENT_TOOLS_SCHEMA)

class AgentState(TypedDict):
    """Lo stato del Grafo: messaggi + dati dell'e-commerce."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    shared_state: Dict[str, Any]
    next_step: str 
    fresh_search_done: bool 
    iteration_count: int
    tools_called: List[str]

def parse_input(state: AgentState):
    """Nodo iniziale di parsing con memoria del contesto."""
    from app.services.agent_tools import tool_parse_query
    
    # PULIZIA: Ogni nuovo turno puliamo i risultati precedenti
    state["shared_state"]["results"] = []
    messages = state["messages"]
    
    # Identifichiamo l'ultimo messaggio dell'utente e il contesto del prodotto (ultimi 3 messaggi max)
    user_msg = ""
    product_context = ""
    
    # Priorità: se l'ultima ricerca aveva un prodotto, usiamo quello come contesto
    last_parsed = state["shared_state"].get("parsed_query") or {}
    if last_parsed.get("product"):
        product_context = str(last_parsed["product"]).capitalize()

    # Guardiamo indietro solo per un raggio limitato per evitare stale context
    lookback_window = messages[-6:] # Ultime 3 coppie circa
    
    for msg in reversed(lookback_window):
        if isinstance(msg, HumanMessage) and not user_msg:
            user_msg = msg.content
        
        if not product_context:
            content = msg.content.lower()
            # Lista prodotti estesa
            strong_products = ["iphone", "samsung", "macbook", "scarpe", "scarpa", "shoes", "nike", "maglione", "felpa", "laptop", "jeans", "pantaloni", "zaino", "orologio"]
            for p in strong_products:
                if p in content:
                    product_context = p.capitalize()
                    break
            
            if not product_context:
                # Pattern più preciso per modelli tech, evitando falsi positivi con 's' singola (che beccava Levis 501 -> s 501)
                model_match = re.search(r"\b(iphone|samsung|galaxy\s*s|macbook)\s*(\d+\s*(pro|max|mini|plus)?)", content)
                if model_match:
                    product_context = model_match.group(0)

    # LOGICA DI PERTINENZA MIGLIORATA
    parser_query = user_msg
    is_new_topic = False
    
    # Se il nuovo messaggio contiene un prodotto forte DIVERSO da quello in memoria, cambiamo topic
    new_product_triggers = ["scarpe", "scarpa", "nike", "adidas", "maglione", "felpa", "tv", "monitor", "pc", "laptop", "computer", "iphone", "samsung", "macbook", "jeans", "pantaloni", "zaino", "orologio"]
    
    # Se abbiamo già un contesto (es. "Scarpe"), e il nuovo messaggio dice ancora "scarpe", NON è un nuovo topic.
    current_product = (product_context or "").lower()
    
    # Identifichiamo prompt di "nuovo inizio" (es. "ciao", "cerca dei", "trovami")
    reset_signals = ["ciao", "cerca", "trovami", "fammi vedere", "cerco", "nuova ricerca"]
    has_reset_signal = any(re.search(rf"\b{s}\b", user_msg.lower()) for s in reset_signals)

    possible_new_trigger = next((t for t in new_product_triggers if t in user_msg.lower()), None)
    if possible_new_trigger:
        # Se c'è un reset signal + un prodotto, è SEMPRE un nuovo topic (anche se stessa categoria)
        if has_reset_signal:
            is_new_topic = True
            logger.info(f"New topic triggered by reset signal: {user_msg}")
        # Altrimenti, se il trigger trovato è già contenuto nel contesto corrente (o viceversa), continuiamo lo stesso topic
        elif current_product and (possible_new_trigger in current_product or current_product in possible_new_trigger):
            is_new_topic = False
            logger.info(f"Continuing same topic: {current_product}")
        else:
            is_new_topic = True
            logger.info(f"Switching to new topic in query: {user_msg}")
    
    # Se è un nuovo topic, resettiamo il parsed_query nello shared_state per evitare "memory" di filtri vecchi
    if is_new_topic:
        state["shared_state"]["parsed_query"] = {}
        logger.info("Cleared previous parsed context for new topic.")

    if product_context and not is_new_topic:
        # Iniettiamo il contesto se è un raffinamento (es. "numero 43", "nero", "più economico")
        refinement_keywords = ["vorrei", "prefer", "meglio", "più", "numero", "taglia", "taglie", "misura", "misure", "lunghezza", "lunghezze", "colore", "per", "con", "baggy", "stile", "classico", "neri", "bianchi", "blu", "rossi", "verde", "un", "una", "dei", "delle"]
        if len(user_msg.split()) < 15 or any(re.search(rf"\b{x}\b", user_msg.lower()) for x in refinement_keywords):
            # Non iniettiamo se la parola chiave di contesto è già presente nel messaggio 
            message_low = user_msg.lower()
            context_low = product_context.lower()
            
            # Se il contesto non è già esplicitamente nel messaggio, lo iniettiamo
            if context_low not in message_low:
                # Puliamo il user_msg da noise pre-iniezione
                clean_msg = user_msg
                for noise in ["ho chiesto", "vorrei", "cercavo", "precedentemente", "invece", "per favore", "fammi vedere", "puoi cercare"]:
                    clean_msg = re.sub(rf"\b{noise}\b", "", clean_msg, flags=re.IGNORECASE).strip()
                
                parser_query = f"{product_context} {clean_msg}"
                logger.info(f"Context Injection: {parser_query}")
    
    if user_msg:
        # Eseguiamo il parse (questo aggiorna lo stato e fa user profiling)
        result = tool_parse_query(state=state["shared_state"], query=parser_query)
        
        import uuid
        call_id = f"init_{uuid.uuid4().hex[:8]}"
        dummy_ai = AIMessage(content="", tool_calls=[{"name": "parse_query", "args": {"query": parser_query}, "id": call_id, "type": "tool_call"}])
        tool_res = ToolMessage(tool_call_id=call_id, name="parse_query", content=str(result))
        
        return {
            "messages": [dummy_ai, tool_res],
            "shared_state": state["shared_state"], 
            "fresh_search_done": False, 
            "iteration_count": 0, 
            "tools_called": ["parse_query"]
        }
    
    return {"shared_state": state["shared_state"], "fresh_search_done": False, "iteration_count": 0, "tools_called": []}

def call_model(state: AgentState):
    """Nodo che interroga l'LLM con estrazione forzata dei tool."""
    messages = state["messages"]
    it = state.get("iteration_count", 0) + 1
    
    # System Prompt STRETTO: No chiacchiere, solo tool
    sys_prompt = (
        "You are an expert PREMIUM eBay Shopping Assistant.\n"
        "STRICT MANDATORY TOOL CHAIN:\n"
        "1. Parameters are parsed automatically.\n"
        "2. You MUST perform a new search for every turn using the appropriate tool.\n"
        "3. You MUST format the final answer using the explainer tool. Never use text summaries.\n"
        "\n"
        "RULES:\n"
        "- NO CONVERSATIONAL FILLER. No 'Ciao', no 'Certamente'. Just execute the tools.\n"
        "- NEVER respond with text descriptions of products. Use ONLY the results formatting tool.\n"
        "- NEVER assume results from previous turns are still valid. Always search again if requested.\n"
        "- NEVER pass a 'results' parameter manually. The tools handle data internally.\n"
        "- Language: Italian."
    )
    messages = [SystemMessage(content=sys_prompt)] + messages

    # RECOVERY/NUDGE: Se l'agente sta balbettando o ha saltato uno step, lo raddrizziamo
    executed_tools = state.get("tools_called", [])
    if "search_products" in executed_tools and "explain_results" not in executed_tools:
        nudge = "Data is ready. Now format the final response for the user using the explainer tool. DO NOT use text."
        messages = messages + [SystemMessage(content=nudge)]

    elif it >= 1 and not state.get("fresh_search_done", False):
        last_tool = next((msg for msg in reversed(messages) if isinstance(msg, ToolMessage)), None)
        if last_tool and "status" in last_tool.content and "parsed" in last_tool.content:
            nudge = "Parameters have been successfully extracted. Proceed with the search now. Do NOT skip this step."
            messages = messages + [SystemMessage(content=nudge)]

    response = llm.invoke(messages)
    
    # FIX AUTO-CLEAR: Se ci sono tool call (native o estratte), svuotiamo il testo 
    # per evitare che l'utente veda allucinazioni "mentre" il tool lavora.
    if hasattr(response, "tool_calls") and response.tool_calls:
        response.content = ""
    else:
        from app.services.agent_orchestrator import _extract_tool_call_from_text
        extracted = _extract_tool_call_from_text(response.content)
        if extracted:
            response.tool_calls = [{
                "name": extracted["function"]["name"],
                "args": extracted["function"]["arguments"],
                "id": f"f_{os.urandom(2).hex()}",
                "type": "tool_call"
            }]
            response.content = "" # Svuota il testo allucinato se abbiamo estratto il tool

    return {"messages": [response], "iteration_count": it}

def recovery_node(state: AgentState):
    """Nodo che punisce l'allucinazione e istruisce l'agente."""
    nudge = SystemMessage(content="CRITICAL ERROR: Stop hallucinating data. Call `search_products` IMMEDIATELY with real parameters. Do not respond with text tables.")
    return {"messages": [nudge]}

def execute_tools(state: AgentState):
    """Nodo custom per eseguire i tool e aggiornare lo shared_state."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_outputs = []

    # Tracciamo i tool chiamati in questo turno
    executed_names = []
    search_executed = False
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            executed_names.append(name)
            args = tool_call["args"]
            
            if name == "search_products":
                search_executed = True
                state["shared_state"]["thinking_trace"].append("✔ search & ingestion completed")

            logger.info(f"Executing tool: {name} with args: {args}")
            
            if name in TOOLS_MAP:
                try:
                    result = TOOLS_MAP[name](state=state["shared_state"], **args)
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
            else:
                tool_outputs.append(ToolMessage(
                    tool_call_id=tool_call.get("id", "call_unk"),
                    name=name,
                    content="Unknown tool."
                ))
                
    return {
        "messages": tool_outputs, 
        "fresh_search_done": state.get("fresh_search_done", False) or search_executed,
        "tools_called": state.get("tools_called", []) + executed_names
    }

def router(state: AgentState):
    """Decide se andare avanti con i tool o finire con controllo di qualità."""
    messages = state["messages"]
    last_message = messages[-1]
    shared_state = state["shared_state"]
    fresh_search = state.get("fresh_search_done", False)
    it = state.get("iteration_count", 0)
    
    # Tracciamo se explain_results è stato chiamato
    executed_tools = state.get("tools_called", [])
    explanation_done = "explain_results" in executed_tools

    if it >= 10:
        return END

    # Se ci sono tool calls, andiamo a execute_tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # GUARDRAIL ANTI-ALLUCINAZIONE
    parsed = shared_state.get("parsed_query", {})
    is_product_query = bool(parsed.get("product") or parsed.get("semantic_query"))
    
    # Se abbiamo una query ma non abbiamo fatto la ricerca, continuiamo
    if is_product_query and not fresh_search:
        content = last_message.content.lower()
        if any(x in content for x in ["€", "euro", "price", "disponibile"]) and len(content) > 20:
             logger.warning("HALLUCINATION DETECTED (Results without search). Routing to recovery.")
             return "recovery"
        return "agent"

    # Se abbiamo fatto la ricerca ma NO spiegazione, NON possiamo finire (l'agente sta "saltando" lo step)
    if fresh_search and not explanation_done and is_product_query:
        logger.warning("FORCING EXPLANATION: Search done but explain_results missing. Redirecting to agent.")
        return "agent"
    
    return END

# Costruzione del Grafo
workflow = StateGraph(AgentState)

workflow.add_node("parser", parse_input)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_node("recovery", recovery_node)

workflow.set_entry_point("parser")

workflow.add_edge("parser", "agent")
workflow.add_edge("recovery", "agent")

workflow.add_conditional_edges(
    "agent",
    router,
    {
        "tools": "tools",
        "agent": "agent",
        "recovery": "recovery",
        END: END
    }
)

workflow.add_edge("tools", "agent")

# Compilazione
app_graph = workflow.compile()

def run_agent_graph(user_message: str, history: List[Dict[str, str]], shared_state: Dict[str, Any]):
    """Entry point per l'integrazione con FastAPI."""
    formatted_history = []
    # Puliamo la history per evitare che tool_calls passati "sporcano" la logica del nuovo giro
    for h in history[-8:]:
        if h["role"] == "user":
            formatted_history.append(HumanMessage(content=h["content"]))
        elif h["role"] == "assistant":
            # Se il messaggio precedente era un JSON di un tool, lo trasformiamo in un messaggio AI pulito
            # per non confondere il modello nel turno successivo
            content = h["content"]
            if content.startswith("{") and "name" in content:
                # È un vecchio tool call testuale, passiamolo come assistente che chiede qualcosa
                try:
                    data = json.loads(content)
                    if "parameters" in data:
                        content = data["parameters"].get("question", content)
                except:
                    pass
            formatted_history.append(AIMessage(content=content))
            
    formatted_history.append(HumanMessage(content=user_message))
    
    # Eseguiamo il Grafo
    final_state = app_graph.invoke({
        "messages": formatted_history,
        "shared_state": shared_state
    })
    
    # Estraiamo l'ultima risposta (il testo dell'analisi)
    testo_agente = ""
    s_state = final_state.get("shared_state", {})
    
    # Diamo priorità assoluta a explain_results se presente
    explain_msg = ""
    clarify_msg = ""
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, ToolMessage):
            # ESTRAZIONE PULITA DEL CONTENUTO (da JSON a stringa)
            try:
                data = json.loads(msg.content)
                if msg.name == "explain_results":
                    explain_msg = data.get("explanation", msg.content)
                elif msg.name == "request_user_clarification":
                    clarify_msg = data.get("question", msg.content)
            except:
                if msg.name == "explain_results":
                   explain_msg = msg.content
                elif msg.name == "request_user_clarification":
                    clarify_msg = msg.content
    
    if explain_msg:
        testo_agente = explain_msg
    elif clarify_msg:
        testo_agente = clarify_msg
    else:
        # Fallback al contenuto AIMessage se non ci sono tool messages utili (saltando tool calls)
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                text = msg.content.strip()
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    continue
                if text.startswith("{") and "name" in text:
                    continue
                testo_agente = text
                break

    # Se non c'è una risposta testuale ma abbiamo risultati, forziamo una spiegazione base
    if not testo_agente and s_state.get("results"):
        testo_agente = f"## Results for your search\nI found {len(s_state['results'])} items matching your request."

    return {
        "parsed_query": s_state.get("parsed_query") or {"semantic_query": user_message},
        "ebay_query_used": user_message,
        "results_count": len(s_state.get("results", [])),
        "saved_new_count": 0,
        "results": s_state.get("results", []),
        "rag_context": s_state.get("rag_context", ""),
        "metrics": s_state.get("metrics", {}),
        "analysis": testo_agente,
        "thinking_trace": s_state.get("thinking_trace", []),
        "_timings": s_state.get("_timings", {})
    }
