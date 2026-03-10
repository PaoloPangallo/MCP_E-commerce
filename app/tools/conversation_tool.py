from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from app.services.parser import call_gemini, call_ollama


class ToolContextLike(Protocol):
    db: Any
    user: Optional[object]
    llm_engine: str


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


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

    prompt = (
        "Sei ebayGPT, un assistente e-commerce in italiano.\n"
        "Rispondi in modo naturale e utile.\n"
        "Non inventare dati di prodotti se non ne hai.\n\n"
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