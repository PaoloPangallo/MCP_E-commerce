import logging
from typing import Dict, Any, Annotated
from pydantic import Field

from app.mcp.core import mcp, _db_context, _build_context, _tool_error
from app.tools import execute_conversation_tool

logger = logging.getLogger(__name__)

@mcp.tool(
    name="conversation",
    description="Generatore di risposte conversazionali libere in caso nessuna operazione su eBay sia esplicitamente pre-richiesta.",
)
async def conversation(
    query: Annotated[str, Field(description="La frase o la domanda scritta dall'utente")],
    llm_engine: Annotated[str, Field(description="L'engine LLM preferito (es 'ollama')")] = "ollama",
    session_id: Annotated[str, Field(description="ID di sessione utente (opzionale)")] = ""
) -> Dict[str, Any]:
    try:
        with _db_context() as db:
            context = _build_context(db=db, llm_engine=llm_engine, session_id=session_id)
            payload = await execute_conversation_tool({"query": query}, context)

            if not isinstance(payload, dict):
                payload = {"result": payload}

            payload.setdefault("status", "ok")
            payload["_backend"] = "mcp"
            return payload
    except Exception as exc:
        logger.exception("MCP conversation failed")
        return _tool_error(query=query, error=str(exc))
