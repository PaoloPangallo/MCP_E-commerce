from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from mcp.server.fastmcp import FastMCP

from app.services.parser import parse_query_service
from app.services.search_pipeline import run_search_pipeline
from app.services.seller_pipeline import run_seller_pipeline
from app.tools import (
    execute_conversation_tool,
    execute_compare_tool,
    execute_item_details_tool,
    execute_shipping_costs_tool,
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


def configure_mcp(
    db_factory: Optional[Callable[[], Any]] = None,
    user_resolver: Optional[Callable[[Any], Optional[object]]] = None,
) -> None:
    _DEPS.db_factory = db_factory
    _DEPS.user_resolver = user_resolver
    logger.info("MCP configured | db_factory=%s | user_resolver=%s", bool(db_factory), bool(user_resolver))


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        logger.warning("JSON serialization failed: %s", exc)
        return json.dumps(
            {
                "status": "error",
                "error": "serialization_error",
            },
            ensure_ascii=False,
        )


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
        except Exception as exc:
            logger.warning("Failed to close DB session: %s", exc)


def _build_context(db: Any, llm_engine: str = "ollama", session_id: Optional[str] = None) -> MCPToolContext:
    user = None
    if session_id and _DEPS.user_resolver:
        try:
            user = _DEPS.user_resolver(session_id)
        except Exception as exc:
            logger.warning("Failed to resolve user for session_id=%s: %s", session_id, exc)

    return MCPToolContext(
        db=db,
        user=user,
        llm_engine=(llm_engine or "ollama").strip().lower(),
    )


def _tool_error(**kwargs: Any) -> str:
    payload = {"status": "error", **kwargs}
    return _safe_json(payload)


# ============================================================
# NORMALIZERS
# ============================================================

def _normalize_search_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    results = raw.get("results") or raw.get("items") or []
    top_result = results[0] if results else None

    return {
        "status": "ok" if results else "no_results",
        "query": raw.get("query"),
        "results_count": raw.get("results_count", len(results)),
        "results": results,
        "top_result": top_result,
        "analysis": raw.get("analysis"),
        "metrics": raw.get("metrics"),
        "rag_context": raw.get("rag_context"),
        "raw": raw,
    }


def _normalize_seller_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    feedbacks = raw.get("feedbacks") or raw.get("feedback") or []
    count = raw.get("count", len(feedbacks))

    if raw.get("status"):
        status = raw["status"]
    else:
        status = "ok" if count > 0 else "no_data"

    return {
        "status": status,
        "seller_name": raw.get("seller_name"),
        "count": count,
        "feedbacks": feedbacks,
        "trust_score": raw.get("trust_score"),
        "sentiment_score": raw.get("sentiment_score"),
        "error": raw.get("error"),
        "raw": raw,
    }

def _normalize_item_details_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": raw.get("status", "ok"),
        "item_id": raw.get("item_id"),
        "data": raw.get("data"),
        "error": raw.get("error"),
        "message": raw.get("message"),
        "raw": raw,
    }


def _normalize_shipping_costs_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": raw.get("status", "ok"),
        "item_id": raw.get("item_id"),
        "data": raw.get("data"),
        "error": raw.get("error"),
        "raw": raw,
    }

# ============================================================
# MCP TOOLS
# ============================================================

@mcp.tool(
    name="search_products",
    description=(
        "Cerca prodotti e-commerce usando la pipeline completa: parsing query, "
        "retrieval, reranking, trust scoring e analisi finale. "
        "Se include_shipping=true, calcola automaticamente anche i costi di spedizione "
        "per il primo risultato, risparmiando un passaggio aggiuntivo."
    ),
)
def search_products(query: str, include_shipping: bool = False, session_id: str = "") -> str:

    db = None

    try:

        logger.info("MCP TOOL search_products START")

        db = _get_db()

        context = _build_context(db=db, session_id=session_id)
        
        raw = run_search_pipeline(
            query=query,
            db=db,
            user=context.user,
            llm_engine="ollama"
        )

        normalized = _normalize_search_output(raw)

        if include_shipping and normalized.get("top_result"):
            top_item_id = normalized["top_result"].get("ebay_id")
            if top_item_id:
                try:
                    logger.info("MCP TOOL search_products - fetching shipping for top item %s", top_item_id)
                    shipping_raw = execute_shipping_costs_tool(
                        {
                            "item_id": top_item_id,
                            "country_code": "IT",
                            "zip_code": "",
                        },
                        context,
                    )
                    shipping_norm = _normalize_shipping_costs_output(shipping_raw)
                    normalized["top_result"]["shipping_info"] = shipping_norm.get("data")
                    normalized["top_result"]["shipping_status"] = shipping_norm.get("status")
                except Exception as e:
                    logger.warning("Auto-shipping fetch failed in search pipeline: %s", e)

        normalized["_backend"] = "mcp"

        logger.info("MCP TOOL search_products END")

        return _safe_json(normalized)

    except Exception as exc:

        logger.exception("MCP search_products failed")

        return _tool_error(
            query=query,
            error=str(exc)
        )

    finally:

        _close_db(db)


@mcp.tool(
    name="analyze_seller",
    description=(
        "Analizza un venditore e-commerce recuperando feedback, trust score "
        "e sentiment score."
    ),
)
def analyze_seller(seller_name: str, page: int = 1, limit: int = 10, session_id: str = "") -> str:
    db = None
    try:
        db = _get_db()
        raw = run_seller_pipeline(
            seller_name=seller_name,
            page=page,
            limit=limit,
        )
        normalized = _normalize_seller_output(raw)
        normalized["_backend"] = "mcp"
        return _safe_json(normalized)
    except Exception as exc:
        logger.exception("MCP analyze_seller failed")
        return _tool_error(seller_name=seller_name, error=str(exc))
    finally:
        _close_db(db)


@mcp.tool(
    name="profile_query",
    description=(
        "Analizza una query utente e restituisce un profilo strutturato "
        "utile per capire brand, prezzo, taglia, categoria e altri vincoli."
    ),
)
def profile_query(query: str, session_id: str = "") -> str:
    try:
        parsed = parse_query_service(query)
        return _safe_json(
            {
                "status": "ok",
                "query": query,
                "parsed": parsed,
                "_backend": "mcp",
            }
        )
    except Exception as exc:
        logger.exception("MCP profile_query failed")
        return _tool_error(query=query, error=str(exc))


@mcp.tool(
    name="conversation",
    description=(
        "Gestisce messaggi conversazionali generici quando non serve usare "
        "tool e-commerce specifici."
    ),
)
def conversation(query: str, llm_engine: str = "ollama", session_id: str = "") -> str:
    db = None
    try:
        db = _get_db()
        context = _build_context(db=db, llm_engine=llm_engine, session_id=session_id)
        payload = execute_conversation_tool({"query": query}, context)

        if not isinstance(payload, dict):
            payload = {"result": payload}

        payload.setdefault("status", "ok")
        payload["_backend"] = "mcp"

        return _safe_json(payload)

    except Exception as exc:
        logger.exception("MCP conversation failed")
        return _tool_error(query=query, error=str(exc))
    finally:
        _close_db(db)


@mcp.tool(
    name="compare_products",
    description=(
        "Confronta più prodotti e-commerce cercando ognuno in parallelo e "
        "restituisce una matrice di confronto (prezzo, trust, rilevanza, condizione) "
        "con il prodotto vincitore e la motivazione. "
        "Input: lista di query separate da virgola o punto e virgola, ad esempio "
        "'iphone 13, samsung galaxy s22'. Min 2, max 4 query."
    ),
)
def compare_products(queries: str, llm_engine: str = "ollama", session_id: str = "") -> str:
    """
    queries: stringa con le query separate da virgola o punto e virgola.
    Esempio: 'nike air max, adidas ultraboost'
    """
    import asyncio
    from app.services.compare_pipeline import run_compare_pipeline

    db = None
    try:
        # Parse the comma/semicolon-separated query string
        sep_queries = [
            q.strip()
            for q in queries.replace(";", ",").split(",")
            if q.strip()
        ]

        if len(sep_queries) < 2:
            return _tool_error(
                error="Fornisci almeno 2 query separate da virgola per confrontare i prodotti.",
                example="iphone 13, samsung galaxy s22",
            )

        db = _get_db()
        context = _build_context(db=db, llm_engine=llm_engine, session_id=session_id)

        logger.info("MCP TOOL compare_products START | queries=%s", queries)

        result = execute_compare_tool(
            {"queries": queries},
            context
        )

        result["_backend"] = "mcp"
        logger.info("MCP TOOL compare_products END")
        return _safe_json(result)


    except Exception as exc:
        logger.exception("MCP compare_products failed")
        return _tool_error(queries=queries, error=str(exc))
    finally:
        _close_db(db)


@mcp.tool(
    name="get_item_details",
    description=(
        "Recupera i dettagli estesi, descrizione e specifiche tecniche di un prodotto "
        "conoscendo il suo ID eBay (item_id)."
    ),
)
def get_item_details(item_id: str, session_id: str = "") -> str:
    db = None
    try:
        db = _get_db()
        context = _build_context(db=db, session_id=session_id)
        logger.info("MCP TOOL get_item_details START | item_id=%s", item_id)
        
        result = execute_item_details_tool(
            {"item_id": item_id},
            context
        )
        normalized = _normalize_item_details_output(result)
        normalized["_backend"] = "mcp"
        
        logger.info("MCP TOOL get_item_details END")
        return _safe_json(normalized)
    except Exception as exc:
        logger.exception("MCP get_item_details failed")
        return _tool_error(item_id=item_id, error=str(exc))
    finally:
        _close_db(db)


@mcp.tool(
    name="get_shipping_costs",
    description=(
        "Recupera i costi e le opzioni di spedizione precisi per un determinato oggetto "
        "(item_id) verso un CAP (zip_code) e Paese (country_code)."
    ),
)
def get_shipping_costs(item_id: str, country_code: str = "IT", zip_code: str = "", session_id: str = "") -> str:
    db = None
    try:
        db = _get_db()
        context = _build_context(db=db, session_id=session_id)
        logger.info("MCP TOOL get_shipping_costs START | item_id=%s", item_id)
        
        result = execute_shipping_costs_tool(
            {
                "item_id": item_id,
                "country_code": country_code,
                "zip_code": zip_code,
            },
            context
        )
        normalized = _normalize_shipping_costs_output(result)
        normalized["_backend"] = "mcp"
        
        logger.info("MCP TOOL get_shipping_costs END")
        return _safe_json(normalized)
    except Exception as exc:
        logger.exception("MCP get_shipping_costs failed")
        return _tool_error(item_id=item_id, error=str(exc))
    finally:
        _close_db(db)


# ============================================================
# MCP RESOURCES
# ============================================================

@mcp.resource("catalog://tools")
def tools_catalog() -> str:
    return _safe_json(
        {
            "tools": [
                {
                    "name": "search_products",
                    "description": "Ricerca prodotti e-commerce",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "analyze_seller",
                    "description": "Analizza venditore e feedback",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "seller_name": {"type": "string"},
                            "page": {"type": "integer", "default": 1},
                            "limit": {"type": "integer", "default": 10},
                        },
                        "required": ["seller_name"],
                    },
                },
                {
                    "name": "profile_query",
                    "description": "Profila query utente",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "conversation",
                    "description": "Risponde a messaggi conversazionali generici",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "llm_engine": {"type": "string", "default": "ollama"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "compare_products",
                    "description": "Confronta più prodotti e-commerce",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "queries": {"type": "string"},
                            "llm_engine": {"type": "string", "default": "ollama"},
                        },
                        "required": ["queries"],
                    },
                },
                {
                    "name": "get_item_details",
                    "description": "Recupera i dettagli estesi di un prodotto (item_id)",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                        },
                        "required": ["item_id"],
                    },
                },
                {
                    "name": "get_shipping_costs",
                    "description": "Recupera i costi di spedizione precisi per un oggetto (item_id)",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "country_code": {"type": "string", "default": "IT"},
                            "zip_code": {"type": "string", "default": ""},
                        },
                        "required": ["item_id"],
                    },
                },
            ]
        }
    )


@mcp.resource("profile://query/{text}")
def query_profile_resource(text: str) -> str:
    try:
        parsed = parse_query_service(text)
        return _safe_json(
            {
                "query": text,
                "parsed": parsed,
            }
        )
    except Exception as exc:
        logger.exception("MCP query profile resource failed")
        return _safe_json(
            {
                "query": text,
                "error": str(exc),
            }
        )


@mcp.resource("memory://session/{user_key}")
def session_memory_resource(user_key: str) -> str:
    return _safe_json(
        {
            "user_key": user_key,
            "session_memory": {
                "recent_queries": [],
                "recent_sellers": [],
                "recent_products": [],
            },
            "note": "Collega qui il tuo memory service reale se vuoi esporre stato sessione.",
        }
    )


@mcp.resource("memory://long-term/{user_key}")
def long_term_memory_resource(user_key: str) -> str:
    return _safe_json(
        {
            "user_key": user_key,
            "long_term_memory": {
                "user_preferences": {},
                "previous_searches": [],
                "user_behaviour": {},
            },
            "note": "Collega qui Redis/DB/vector store per memoria persistente.",
        }
    )


# ============================================================
# MCP PROMPTS
# ============================================================

@mcp.prompt(name="search_assistant_prompt")
def search_assistant_prompt(query: str) -> str:
    return f"""
Sei un assistente e-commerce.
Usa il tool `search_products` per cercare prodotti rilevanti per questa richiesta:

Query utente: {query}

Rispondi in modo sintetico, utile e concreto.
""".strip()


@mcp.prompt(name="seller_assistant_prompt")
def seller_assistant_prompt(seller_name: str) -> str:
    return f"""
Sei un assistente che analizza affidabilità di venditori e-commerce.
Usa il tool `analyze_seller` per questo venditore:

Venditore: {seller_name}

Riassumi feedback, trust score e possibili criticità.
""".strip()