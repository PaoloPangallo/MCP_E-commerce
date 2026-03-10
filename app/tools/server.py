from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from mcp.server.fastmcp import FastMCP

from app.services.memory_service import (
    get_session_memory,
    get_user_memory,
    search_semantic_memory,
)

from app.tools import (
    execute_conversation_tool,
    execute_profile_tool,
    execute_search_tool,
    execute_seller_tool,
)

logger = logging.getLogger(__name__)


# ============================================================
# MCP CONFIG
# ============================================================

@dataclass
class MCPDependencies:
    db_factory: Optional[Callable[[], Any]] = None
    user_resolver: Optional[Callable[[Any], Optional[object]]] = None


@dataclass
class MCPToolContext:
    db: Any
    user: Optional[object] = None
    llm_engine: str = "ollama"


_DEPS = MCPDependencies()

mcp = FastMCP("mcp-ecommerce-agent")


# ============================================================
# CONFIGURATION
# ============================================================

def configure_mcp(
    db_factory: Optional[Callable[[], Any]] = None,
    user_resolver: Optional[Callable[[Any], Optional[object]]] = None,
) -> None:
    """
    Collega MCP al backend reale.
    """
    _DEPS.db_factory = db_factory
    _DEPS.user_resolver = user_resolver


# ============================================================
# UTILS
# ============================================================

def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return json.dumps({"error": "serialization_error"})


def _get_db() -> Any:
    if _DEPS.db_factory is None:
        return None
    return _DEPS.db_factory()


def _close_db(db: Any) -> None:
    if db is None:
        return

    close = getattr(db, "close", None)

    if callable(close):
        try:
            close()
        except Exception:
            logger.warning("Failed closing DB session")


def _build_context(db: Any, llm_engine: str = "ollama") -> MCPToolContext:

    return MCPToolContext(
        db=db,
        user=None,
        llm_engine=(llm_engine or "ollama").strip().lower(),
    )


def _tool_error(**kwargs: Any) -> str:

    payload = {"status": "error", **kwargs}

    return _safe_json(payload)


# ============================================================
# MCP TOOLS
# ============================================================

@mcp.tool(
    name="search_products",
    description="Cerca prodotti e-commerce usando la pipeline completa.",
)
def search_products(query: str, llm_engine: str = "ollama") -> str:

    db = None

    try:

        db = _get_db()

        context = _build_context(db, llm_engine)

        payload = execute_search_tool(
            {"query": query},
            context,
        )

        return _safe_json(payload)

    except Exception as exc:

        logger.exception("MCP search_products failed")

        return _tool_error(query=query, error=str(exc))

    finally:

        _close_db(db)


@mcp.tool(
    name="analyze_seller",
    description="Analizza un venditore e-commerce recuperando feedback e trust score.",
)
def analyze_seller(
    seller_name: str,
    page: int = 1,
    limit: int = 10,
) -> str:

    db = None

    try:

        db = _get_db()

        context = _build_context(db)

        payload = execute_seller_tool(
            {
                "seller_name": seller_name,
                "page": page,
                "limit": limit,
            },
            context,
        )

        return _safe_json(payload)

    except Exception as exc:

        logger.exception("MCP analyze_seller failed")

        return _tool_error(
            seller_name=seller_name,
            error=str(exc),
        )

    finally:

        _close_db(db)


@mcp.tool(
    name="conversation",
    description="Gestisce messaggi conversazionali generici.",
)
def conversation(query: str, llm_engine: str = "ollama") -> str:

    db = None

    try:

        db = _get_db()

        context = _build_context(db, llm_engine)

        payload = execute_conversation_tool(
            {"query": query},
            context,
        )

        return _safe_json(payload)

    except Exception as exc:

        logger.exception("MCP conversation failed")

        return _tool_error(query=query, error=str(exc))

    finally:

        _close_db(db)


@mcp.tool(
    name="profile_query",
    description="Analizza una query utente e restituisce un profilo strutturato.",
)
def profile_query(query: str) -> str:

    try:

        payload = execute_profile_tool({"query": query})

        return _safe_json(payload)

    except Exception as exc:

        logger.exception("MCP profile_query failed")

        return _tool_error(query=query, error=str(exc))


# ============================================================
# MCP RESOURCES
# ============================================================

@mcp.resource("catalog://tools")
def tools_catalog() -> str:

    return _safe_json(
        {
            "tools": [
                "search_products",
                "analyze_seller",
                "conversation",
                "profile_query",
            ]
        }
    )


@mcp.resource("profile://query/{text}")
def query_profile_resource(text: str) -> str:

    try:

        payload = execute_profile_tool({"query": text})

        return _safe_json(payload)

    except Exception as exc:

        logger.exception("profile resource failed")

        return _tool_error(query=text, error=str(exc))


# ============================================================
# MEMORY RESOURCES
# ============================================================

@mcp.resource("memory://session/{user_key}")
def session_memory_resource(user_key: str) -> str:

    try:

        data = get_session_memory(user_key)

        return _safe_json(
            {
                "user_key": user_key,
                "session_memory": data,
            }
        )

    except Exception as exc:

        logger.exception("session memory failed")

        return _tool_error(user_key=user_key, error=str(exc))


@mcp.resource("memory://long-term/{user_key}")
def long_term_memory_resource(user_key: str) -> str:

    db = None

    try:

        db = _get_db()

        memory = get_user_memory(db, user_key)

        return _safe_json(
            {
                "user_key": user_key,
                "memory": memory,
            }
        )

    except Exception as exc:

        logger.exception("long term memory failed")

        return _tool_error(user_key=user_key, error=str(exc))

    finally:

        _close_db(db)


@mcp.resource("memory://semantic/{query}")
def semantic_memory_resource(query: str) -> str:

    try:

        results = search_semantic_memory(query)

        return _safe_json(
            {
                "query": query,
                "results": results,
            }
        )

    except Exception as exc:

        logger.exception("semantic memory failed")

        return _tool_error(query=query, error=str(exc))


# ============================================================
# MCP PROMPTS
# ============================================================

@mcp.prompt(name="search_assistant_prompt")
def search_assistant_prompt(query: str) -> str:

    return f"""
Sei un assistente e-commerce.

Usa il tool `search_products` per trovare prodotti rilevanti.

Query utente:
{query}

Rispondi in modo sintetico e utile.
""".strip()


@mcp.prompt(name="seller_assistant_prompt")
def seller_assistant_prompt(seller_name: str) -> str:

    return f"""
Sei un assistente che analizza venditori e-commerce.

Usa il tool `analyze_seller`.

Venditore:
{seller_name}

Riassumi feedback e trust score.
""".strip()


@mcp.prompt(name="conversation_assistant_prompt")
def conversation_assistant_prompt(query: str) -> str:

    return f"""
Sei ebayGPT.

Usa il tool `conversation` per rispondere.

Messaggio utente:
{query}
""".strip()