from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from app.services.parser import call_gemini_async, call_ollama_async


class ToolContextLike(Protocol):
    db: Any
    user: Optional[object]
    llm_engine: str


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


async def _call_conversation_llm_async(prompt: str, llm_engine: str) -> str:
    engine = _clean_text(llm_engine).lower() or "ollama"
    if engine == "gemini":
        return _clean_text(await call_gemini_async(prompt))
    if engine == "ollama":
        return _clean_text(await call_ollama_async(prompt))
    return ""


async def execute_conversation_tool(action_input: Dict[str, Any], context: ToolContextLike) -> Dict[str, Any]:
    query = _clean_text(action_input.get("query"))
    if not query:
        return {"status": "error", "error": "Query missing"}

    prompt = f"Sei un assistente e-commerce. Rispondi a: {query}"
    answer = await _call_conversation_llm_async(prompt, getattr(context, "llm_engine", "ollama"))

    return {"status": "ok", "query": query, "answer": answer} if answer else {"status": "error", "answer": ""}