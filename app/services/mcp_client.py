import os
import sys
import json
import logging
import time
from contextlib import AsyncExitStack
from typing import List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Usa ChatOllama che supporta nativamente il tool calling!
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import OLLAMA_CHAT_URL, MODEL_NAME
from app.services.session_persistence import save_session_state, load_session_state

logger = logging.getLogger(__name__)

class MCPOrchestrator:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self.available_tools = []
        # Fallback al modello che sa usare i tool. Mistral-nemo o qwen2.5-coder sono consigliati.
        tool_model = os.getenv("TOOL_CALLER_MODEL", MODEL_NAME)
        self.llm = ChatOllama(
            model=tool_model,
            base_url=OLLAMA_CHAT_URL.replace("/api/chat", ""),
            temperature=0
        )

    async def connect_to_server(self, script_path: str):
        # Retrieve the venv python path reliably
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        python_exe = os.path.join(base_dir, "venv", "Scripts", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = sys.executable # fallback
            
        env = os.environ.copy()
        env["PYTHONPATH"] = base_dir # Ensure app. modules can be found
        
        server_params = StdioServerParameters(
            command=python_exe,
            args=[script_path],
            env=env
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.sessions.append(session)
        
        # Recupero nativo dei Tool esposti tramite MCP!
        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            # Mappa per far capire i tool di MCP a LangChain/Ollama
            langchain_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            self.available_tools.append({
                "session": session,
                "mcp_tool": tool,
                "langchain_format": langchain_tool
            })
            logger.info(f"MCP CLIENT: Caricato tool '{tool.name}' dal server '{os.path.basename(script_path)}'")

    async def __aenter__(self):
        # Percorsi ai server MCP
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        await self.connect_to_server(os.path.join(base_dir, "mcp_servers", "ebay_engine.py"))
        await self.connect_to_server(os.path.join(base_dir, "mcp_servers", "ai_analyst.py"))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.exit_stack.aclose()

    async def dispatch_tool_call(self, tool_name: str, tool_args: dict) -> str:
        for tool_info in self.available_tools:
            if tool_info["mcp_tool"].name == tool_name:
                session = tool_info["session"]
                try:
                    logger.info(f"MCP CLIENT: Eseguo tool nativo '{tool_name}' con args: {tool_args}")
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    if result.isError:
                        return f"Errore server MCP: {result.content}"
                    content_texts = [c.text for c in result.content if c.type == "text"]
                    return "\n".join(content_texts)
                except Exception as e:
                    logger.error(f"MCP CLIENT: Eccezione in '{tool_name}': {e}")
                    return f"Errore locale: {e}"
        return f"Tool sconosciuto: {tool_name}"

    async def _is_new_topic(self, current_message: str, previous_context: str) -> bool:
        """Usa una chiamata LLM velocissima per capire se l'utente ha cambiato argomento."""
        if not previous_context:
            return True
            
        prompt = f"""
Sei un classificatore di intenti shopping.
Messaggio precedente dell'utente o argomento: "{previous_context}"
Nuovo messaggio dell'utente: "{current_message}"

L'utente sta parlando dello STESSO prodotto/argomento o ha CAMBIATO completamente soggetto (es. da scarpe a computer)?
Rispondi SOLO con una parola: "SAME" o "NEW".
"""
        try:
            # Usiamo temperature 0 per massima coerenza
            res = await self.llm.ainvoke([HumanMessage(content=prompt)])
            text = res.content.strip().upper()
            logger.info(f"TOPIC DETECTOR: Decisione = {text} (basata su context: '{previous_context}')")
            return "NEW" in text
        except Exception as e:
            logger.error(f"Errore nel topic detector: {e}")
            return True # Fallback safer

    async def process_request(self, user_text: str, history: List[Dict[str, str]] = None, context: dict = None) -> Dict[str, Any]:
        tools_for_llm = [t["langchain_format"] for t in self.available_tools]
        
        # Colleghiamo fisicamente i tools nativi al modello
        llm_with_tools = self.llm.bind_tools(tools_for_llm)
        
        system_prompt = f"""
Sei l'Assistente Shopping ufficiale basato sul Model Context Protocol (MCP).
Hai accesso a dei tools sicuri per cercare prodotti su eBay e calcolare la fiducia dei venditori.

RIFERIMENTO TECNICO (RAG): Se qui sotto appaiono dati tecnici (taglie, prezzi reali, modelli), USALI per dare consigli precisi e non sbagliare le conversioni (es. taglia W vs L).
Dati RAG: {json.dumps(context.get("rag_context", "Nessun dato RAG disponibile"))}

REGOLE FONDAMENTALI:
1. Se l'utente chiede di cercare qualcosa, DEVI CHIAMARE IL TOOL `search_products`.
2. DIVIETO ASSOLUTO DI INVENTARE RISULTATI O PRODOTTI: Non scrivere mai finti articoli nel testo!
3. GESTIONE VINCOLI: Se l'utente mette limiti (es. prezzo < 100€), rispettali rigorosamente. Se il tool restituisce 0 risultati, puoi RIFREQUENTARE la ricerca impostando `allow_relaxed=True`, ma DEVI spiegare all'utente: "Purtroppo non ho trovato nulla sotto i 100€, ecco i risultati più vicini..."
4. Se mancano delle informazioni, usa il tool `parse_intent` per analizzare la query.
5. Rispondi SEMPRE in italiano e in modo cortese.

Contesto Sessione Corrente (Strutturato): {json.dumps(context)}
"""
        messages = [SystemMessage(content=system_prompt)]
        
        if history:
            from langchain_core.messages import AIMessage
            # Limit history to last 6 messages to avoid context overflow
            for h in history[-6:]:
                role = h.get("role", "")
                content = h.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                    
        messages.append(HumanMessage(content=user_text))
        
        trace = []
        max_iterations = 5
        final_answer = ""
        
        # Agente "ReAct" nativo (senza json creati a mano!)
        for i in range(max_iterations):
            logger.info(f"MCP CLIENT: Iterazione LLM {i+1}...")
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            
            import uuid
            
            tool_calls = getattr(response, 'tool_calls', [])
            
            # Fallback for models that output JSON instead of native tool calls
            if not tool_calls and response.content:
                text = response.content.strip()
                if "name" in text and "{" in text and "}" in text:
                    try:
                        start = text.find("{")
                        end = text.rfind("}") + 1
                        possible_json = text[start:end]
                        # Handling code blocks inside the chunk (e.g. ```json...```)
                        obj = json.loads(possible_json)
                        if isinstance(obj, dict) and "name" in obj:
                            t_name = obj.get("name")
                            t_args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or {}
                            tool_calls.append({
                                "name": t_name,
                                "args": t_args,
                                "id": f"call_{uuid.uuid4().hex}"
                            })
                            # Pulisci il JSON dal messaggio
                            response.content = text.replace(possible_json, "").replace("```json", "").replace("```", "").strip()
                    except Exception as e:
                        logger.warning(f"Failed to parse text as fallback tool call: {e}")

            if not tool_calls:
                final_answer = response.content
                break
                
            for tool_call in tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                trace.append(f"🔧 Chiamata Tool Naturale: {t_name}")
                
                t_result = await self.dispatch_tool_call(t_name, t_args)
                trace.append(f"✅ Risultato da {t_name}: estratto con successo.")
                
                # Formattiamo la risposta per l'inbox dell'LLM
                messages.append({
                    "role": "tool",
                    "name": t_name,
                    "content": t_result[:1500] + "...(troncato)" if len(t_result) > 1500 else t_result,
                    "tool_call_id": tool_call.get("id", "123")
                })
                
                # Langchain might complain if the message is 'tool' but we didn't send a tool_calls chunk.
                # To be completely safe and avoid missing tool_call_id errors on specific models,
                # we just override the system behavior and treat it as a new generation after the tool.
                if not getattr(response, 'tool_calls', []):
                    # If it was a mock tool call, we update the previous AIMessage to look like one.
                    response.content = response.content.replace(text, "")
                    setattr(response, 'tool_calls', tool_calls)

                
        return {
            "agent_response": final_answer,
            "thinking_trace": trace,
            "messages": [m.dict() if hasattr(m, 'dict') else m for m in messages]
        }

async def ask_mcp_orchestrator(
    user_message: str, 
    history: List[Dict[str, str]],
    db_session: Any, 
    user_obj: Any,
    t0: float, 
    context: Dict[str, Any] = None,
    ecommerce_pipeline_func=None,
    reset_context: bool = False,
    session_id: str = None
) -> Dict[str, Any]:
    
    # Identità per persistenza (usiamo session_id se fornito, altrimenti user_id o guest)
    persistance_id = session_id or (str(user_obj.id) if user_obj and hasattr(user_obj, "id") else "guest")
    
    if reset_context:
        context = {}
    elif not context:
        persisted = load_session_state(persistance_id)
        if persisted and "_last_updated" in persisted:
            age = time.time() - persisted["_last_updated"]
            if age < 900:
                context = persisted.get("context", {})
            else:
                context = {}
    else:
        # Il context è già arrivato dal frontend, lo usiamo così com'è
        pass

    # Avviamo il nuovo cervello MCP!
    async with MCPOrchestrator() as orchestrator:
        mcp_result = await orchestrator.process_request(user_message, history=history, context=context)
        
    payload = {
        "results": [],
        "parsed_query": context,
        "thinking_trace": mcp_result.get("thinking_trace", []),
        "analysis": mcp_result.get("agent_response", "Ricerca terminata!"),
        "_timings": {"total_s": round(time.time() - t0, 3)}
    }
    
    # Cerchiamo se l'LLM ha deciso di fare una ricerca, analizzando anche l'intent
    search_query = None
    for msg in reversed(mcp_result["messages"]):
        # Messaggi di tipo AIMessage con tool_calls in Langchain
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                t_name = tc.get("name")
                if t_name == "search_products":
                    search_query = tc.get("args", {}).get("semantic_query")
                    break
                elif t_name == "parse_intent" and not search_query:
                    search_query = tc.get("args", {}).get("query")
        if search_query:
            break
            
    # Fallback super-aggressivo se il Modello locale fa i capricci decidendo di chattare a vuoto invece di usare "search_products"
    if not search_query:
        msg_lower = user_message.lower()
        ans_lower = payload["analysis"].lower()
        
        search_triggers = ["cercami", "cerca ", "trovami", "orologio", "felpa", "scarpe", "pantalon", "jeans", "giacca", "alimentatore", "notebook", "taglia", "euro"]
        llm_triggers = ["sto cercando", "chiamo il tool", "search_products", "ecco i risultati", "risultati della ricerca"]
        
        force_search = any(t in msg_lower for t in search_triggers) or any(t in ans_lower for t in llm_triggers)
        
        # Evitiamo di attivare il fallback per domande puramente generiche come "quali sono i tuoi task"
        if force_search and len(msg_lower.split()) > 2:
            logger.warning(f"MCP CLIENT: Il modello ha evaso l'uso del tool. Forzatura manuale della ricerca per '{user_message}'")
            
            # Uniamo il vecchio contesto alla nuova frase per preservare il soggetto della ricerca solo se stiamo rifinendo
            old_subject = context.get("semantic_query", "") if isinstance(context, dict) else ""
            
            # Utilizziamo l'IA per decidere se mantenere il contesto della ricerca precedente
            old_subject = context.get("semantic_query", "") if isinstance(context, dict) else ""
            
            is_new_topic = await orchestrator._is_new_topic(user_message, old_subject)
            
            if old_subject and not is_new_topic:
                search_query = f"{old_subject} {user_message}".strip()
                payload["thinking_trace"].append(f"🧠 AI Context: Raffinamento ricerca precedente ('{old_subject}')")
            else:
                search_query = user_message
                if old_subject:
                    payload["thinking_trace"].append("🔄 AI Context: Cambio argomento rilevato, reset filtri.")
                else:
                    payload["thinking_trace"].append("🆕 AI Context: Nuova ricerca avviata.")
            
            # Iniettiamo stringhe nel Thinking Trace così il Frontend non crasha e non scarta la UI di ricerca
            payload["thinking_trace"].append("🛠️ Chiamata Tool Forzata (Fallback): search_products")
            payload["thinking_trace"].append(f"🔍 Query generata per eBay: {search_query}")
    # Se ha fatto una ricerca, avviamo la Pipeline completa per avere DB, Trust e Metriche
    if search_query and ecommerce_pipeline_func:
        logger.info(f"MCP CLIENT: Intercettata search_products. Avvio pipeline e-commerce per '{search_query}'")
        try:
            full_results = ecommerce_pipeline_func(search_query, db_session, user_obj, t0)
            
            # Uniamo i risultati della pipeline robusta al payload dell'Orchestratore MCP
            payload["results"] = full_results.get("results", [])
            payload["results_count"] = len(payload["results"])
            payload["rag_context"] = full_results.get("rag_context", "")
            payload["metrics"] = full_results.get("metrics", {})
            payload["_timings"].update(full_results.get("_timings", {}))
            
            # Aggiorniamo il contesto per la memoria
            context = full_results.get("parsed_query", context)
        except Exception as e:
            logger.error(f"Errore nella pipeline e-commerce: {e}")
            payload["error"] = str(e)
                
    if session_id and db_session:
        from app.models.chat import ChatSession, ChatMessage
        try:
            # 1. Assicuriamoci che la sessione esista
            session = db_session.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                session = ChatSession(
                    id=session_id, 
                    title=user_message[:50], 
                    user_id=getattr(user_obj, "id", None)
                )
                db_session.add(session)
            
            # 2. Salviamo il messaggio dell'utente
            user_msg_db = ChatMessage(
                session_id=session_id,
                role="user",
                content=user_message,
                payload=None
            )
            db_session.add(user_msg_db)

            # 3. Salviamo il messaggio dell'assistente con i dati strutturati
            assistant_msg_db = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=payload["analysis"],
                payload=json.dumps({
                    "results": payload.get("results", []),
                    "thinking_trace": payload.get("thinking_trace", []),
                    "metrics": payload.get("metrics"),
                    "rag_context": payload.get("rag_context"),
                    "parsed_query": context # Salviamo anche il contesto estratto
                }, ensure_ascii=False)
            )
            db_session.add(assistant_msg_db)
            db_session.commit()
            logger.info(f"DB PERSISTENCE: Salvata interazione per sessione '{session_id}'")
        except Exception as e:
            logger.error(f"Errore nel salvataggio DB della chat: {e}")
            db_session.rollback()

    save_session_state(persistance_id, {"context": context, "_last_updated": time.time()})
    
    return payload
