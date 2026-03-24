import logging
from typing import Dict, Any, Annotated, List, Optional
from pydantic import Field

from app.mcp.core import mcp, _db_context, _build_context, _tool_error
from app.llm.client import call_llm
from app.utils.text import clean_text as _clean_text

logger = logging.getLogger(__name__)

def _build_conversation_prompt(
    query: str,
    conversation_history: List[Dict[str, str]],
    context_info: str,
    custom_instructions: str,
) -> str:
    lines = [
        "Sei ebayGPT, un assistente e-commerce amichevole e competente.",
    ]

    if custom_instructions:
        lines.append(f"REGOLA PRIORITÀ ASSOLUTA (rispetta sempre):\n{custom_instructions}\n")

    lines += [
        "Regola: rispondi in modo naturale, conciso, come un esperto di shopping con un amico.",
        "Regola: usa 'tu' informale, tono caldo e diretto.",
        "Regola: NON essere prolisso. 1-3 frasi per domande semplici.",
        "Regola: NON offrire liste di azioni a meno che non ti venga esplicitamente richiesto.",
    ]

    if conversation_history:
        lines.append("\n--- STORICO CONVERSAZIONE ---")
        for turn in conversation_history:
            role = turn.get("role", "")
            content = str(turn.get("content", "")).strip()
            if role == "user":
                lines.append(f"Utente: {content}")
            elif role == "assistant":
                lines.append(f"Assistente: {content}")
        lines.append("--- FINE STORICO ---\n")
    elif context_info:
        lines.append(f"\nContesto delle ultime richieste: {context_info}\n")

    lines.append(f"Utente: {query}")
    lines.append("Assistente:")

    return "\n".join(lines)


@mcp.tool(
    name="conversation",
    description="Generatore di risposte conversazionali libere in caso nessuna operazione su eBay sia esplicitamente pre-richiesta.",
)
async def conversation(
    query: Annotated[str, Field(description="La frase o la domanda scritta dall'utente")],
    llm_engine: Annotated[str, Field(description="L'engine LLM preferito (es 'ollama')")] = "ollama",
    session_id: Annotated[str, Field(description="ID di sessione utente (opzionale)")] = "",
    context_info: Annotated[str, Field(description="Contesto testuale estratto dalla memoria")] = "",
    conversation_history: Annotated[Optional[List[Dict[str, str]]], Field(description="Lista di oggetti role/content precedenti")] = None
) -> Dict[str, Any]:
    try:
        with _db_context() as db:
            context = _build_context(db=db, llm_engine=llm_engine, session_id=session_id)
            
            custom_instructions = ""
            if context.user and getattr(context.user, "custom_instructions", None):
                custom_instructions = str(context.user.custom_instructions)

            hist = conversation_history or []
            
            prompt = _build_conversation_prompt(
                query=query,
                conversation_history=hist,
                context_info=context_info,
                custom_instructions=custom_instructions,
            )

            result_text, _ = await call_llm(prompt=prompt, llm_engine=llm_engine)
            answer = _clean_text(result_text) if result_text else "Non sono riuscito a generare una risposta."

            payload = {
                "status": "ok",
                "query": query,
                "answer": answer,
                "summary": str(answer),
                "_backend": "mcp"
            }
            return payload
    except Exception as exc:
        logger.exception("MCP conversation failed")
        return _tool_error(query=query, error=str(exc))
