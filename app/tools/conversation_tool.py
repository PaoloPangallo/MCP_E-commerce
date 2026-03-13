from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from app.services.parser import call_gemini, call_ollama


class ToolContextLike(Protocol):
    db: Any
    user: Optional[object]
    llm_engine: str


from typing import Any, Dict

from app.utils.text import clean_text as _clean_text
# Using shared _clean_text from app.utils.text


def normalize_conversation_arguments(action_input: Dict[str, Any], fallback_query: str = "") -> Dict[str, Any]:
    query = _clean_text(action_input.get("query") or fallback_query)
    if not query:
        raise ValueError("conversation richiede una query non vuota.")
    return {"query": query}


def _call_conversation_llm(prompt: str, llm_engine: str) -> str:
    engine = _clean_text(llm_engine).lower() or "ollama"

    if engine == "gemini":
        return _clean_text(call_gemini(prompt))
    if engine == "ollama":
        return _clean_text(call_ollama(prompt))

    return ""


def execute_conversation_tool(action_input: Dict[str, Any], context: ToolContextLike) -> Dict[str, Any]:
    clean = normalize_conversation_arguments(action_input)

    custom_instructions = ""
    if context.user and getattr(context.user, "custom_instructions", None):
        custom_instructions = f"REGOLA 0 (PRIORITÀ ASSOLUTA - PREFERENZE DELL'UTENTE):\n{context.user.custom_instructions}\n\nDevi RISPETTARE ASSOLUTAMENTE la regola 0 (es. se ti chiede una lingua specifica, DEVI usarla per tutta la risposta).\n\n"

    prompt = (
        "Sei ebayGPT, un assistente e-commerce.\n"
        f"{custom_instructions}"
        "Regola 1: NON essere prolisso, rispondi con 1-2 frasi al massimo.\n"
        "Regola 2: NON offrire liste di azioni a meno che non ti venga esplicitamente richiesto.\n"
        "Regola 3: Sii amichevole ma vai dritto al punto.\n\n"
        f"Contesto delle ultime richieste dell'utente: {clean.get('context_info', 'Nessuno')}\n"
        f"Messaggio utente: {clean['query']}"
    )

    answer = _call_conversation_llm(prompt, getattr(context, "llm_engine", "ollama"))

    if answer:
        return {
            "status": "ok",
            "query": clean["query"],
            "answer": answer,
        }

    return {
        "status": "error",
        "query": clean["query"],
        "error": "Non riesco a generare una risposta conversazionale in questo momento.",
        "answer": "",
    }