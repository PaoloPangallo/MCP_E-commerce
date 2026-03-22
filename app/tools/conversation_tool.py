from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

import app.llm.client as llm_client


class ToolContextLike(Protocol):
    db: Any
    user: Optional[object]
    llm_engine: str


from app.utils.text import clean_text as _clean_text


def normalize_conversation_arguments(action_input: Dict[str, Any], fallback_query: str = "") -> Dict[str, Any]:
    query = _clean_text(action_input.get("query") or fallback_query)
    if not query:
        raise ValueError("conversation richiede una query non vuota.")
    return {"query": query, "context_info": action_input.get("context_info", ""),
            "conversation_history": action_input.get("conversation_history") or []}


async def _call_conversation_llm(prompt: str, llm_engine: str) -> str:
    engine = _clean_text(llm_engine).lower() or "ollama"

    result, _ = await llm_client.call_llm(prompt=prompt, llm_engine=engine)
    return _clean_text(result) if result else "Non sono riuscito a generare una risposta."


def _build_conversation_prompt(
    query: str,
    conversation_history: List[Dict[str, str]],
    context_info: str,
    custom_instructions: str,
) -> str:
    """Costruisce un prompt multi-turn usando la conversation_history strutturata."""

    lines = [
        "Sei ebayGPT, un assistente e-commerce amichevole e competente.",
    ]

    if custom_instructions:
        lines.append(
            f"REGOLA PRIORITÀ ASSOLUTA (rispetta sempre):\n{custom_instructions}\n"
        )

    lines += [
        "Regola: rispondi in modo naturale, conciso, come un esperto di shopping con un amico.",
        "Regola: usa 'tu' informale, tono caldo e diretto.",
        "Regola: NON essere prolisso. 1-3 frasi per domande semplici.",
        "Regola: NON offrire liste di azioni a meno che non ti venga esplicitamente richiesto.",
    ]

    # Se abbiamo una history strutturata, la usiamo
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
        # Fallback al vecchio context_info se non c'è history
        lines.append(f"\nContesto delle ultime richieste: {context_info}\n")

    lines.append(f"Utente: {query}")
    lines.append("Assistente:")

    return "\n".join(lines)


async def execute_conversation_tool(action_input: Dict[str, Any], context: ToolContextLike) -> Dict[str, Any]:
    clean = normalize_conversation_arguments(action_input)

    custom_instructions = ""
    if context.user and getattr(context.user, "custom_instructions", None):
        custom_instructions = str(context.user.custom_instructions)

    conversation_history: List[Dict[str, str]] = clean.get("conversation_history") or []
    context_info: str = clean.get("context_info", "")

    prompt = _build_conversation_prompt(
        query=clean["query"],
        conversation_history=conversation_history,
        context_info=context_info,
        custom_instructions=custom_instructions,
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