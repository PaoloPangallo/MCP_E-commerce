from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from app.services.parser import call_gemini, call_ollama, call_ollama_cloud


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


async def _call_conversation_llm(prompt: str, llm_engine: str) -> str:
    engine = _clean_text(llm_engine).lower() or "ollama"

    if engine == "gemini":
        return _clean_text(await call_gemini(prompt))
    if engine == "ollama":
        return _clean_text(await call_ollama(prompt))
    if engine == "ollama_cloud":
        return _clean_text(await call_ollama_cloud(prompt))

    # Fallback to local ollama or empty if really unknown
    return _clean_text(await call_ollama(prompt)) or ""


async def execute_conversation_tool(action_input: Dict[str, Any], context: ToolContextLike) -> Dict[str, Any]:
    clean = normalize_conversation_arguments(action_input)

    custom_instructions = ""
    if context.user and getattr(context.user, "custom_instructions", None):
        custom_instructions = f"REGOLA 0 (PRIORITÀ ASSOLUTA - PREFERENZE DELL'UTENTE):\n{context.user.custom_instructions}\n\nDevi RISPETTARE ASSOLUTAMENTE la regola 0 (es. se ti chiede una lingua specifica, DEVI usarla per tutta la risposta).\n\n"

    prompt = (
        "Sei ebayGPT, un esperto assistente per lo shopping online su eBay. Il tuo obiettivo è essere utile, chiaro e preciso.\n\n"
        f"{custom_instructions}"
        "REGOLE DI RISPOSTA:\n"
        "1. Inserisci SEMPRE uno spazio dopo ogni punto (.) o virgola (,).\n"
        "2. Se l'utente ti saluta (es. 'Ciao', 'Hola'), rispondi in modo cordiale e conciso, invitandolo a chiederti supporto per acquisti, confronti tecnici o analisi venditori.\n"
        "3. Non usare il grassetto (**) per intere frasi, usalo solo per evidenziare termini chiave.\n"
        "4. Mantieni un tono professionale ma caloroso.\n\n"
        f"Contesto precedente: {clean.get('context_info', 'Nessuno')}\n"
        f"Messaggio dell'utente: {clean['query']}"
    )

    answer = await _call_conversation_llm(prompt, getattr(context, "llm_engine", "ollama"))

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