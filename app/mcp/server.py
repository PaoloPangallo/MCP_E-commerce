from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from mcp.server.fastmcp import FastMCP

from app.services.parser import parse_query_service
from app.services.search_pipeline import run_search_pipeline
from app.services.seller_pipeline import run_seller_pipeline
from app.models.user import User
from app.tools import (
    execute_conversation_tool,
    execute_compare_tool,
    execute_item_details_tool,
    execute_shipping_costs_tool,
    execute_similar_items_tool,
    execute_metadata_tool,
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
    llm_engine: str = "gemini"


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


from contextlib import contextmanager

@contextmanager
def _db_context():
    """Context manager garantisce la chiusura della sessione DB anche in caso di eccezione."""
    db = _get_db()
    try:
        yield db
    finally:
        _close_db(db)


def resolve_user_by_id(user_id_str: str) -> Optional[User]:
    """Default user resolver that fetches user from DB by ID string."""
    if not user_id_str:
        return None
    
    db = _get_db()
    if not db:
        return None
        
    try:
        user_id = int(user_id_str)
        user = db.query(User).filter(User.id == user_id).first()
        return user
    except (ValueError, Exception) as exc:
        logger.warning("Failed to resolve user by ID '%s': %s", user_id_str, exc)
        return None
    finally:
        _close_db(db)


def _build_context(db: Any, llm_engine: str = "gemini", session_id: Optional[str] = None) -> MCPToolContext:
    user = None
    
    # Try configured resolver first
    if session_id and _DEPS.user_resolver:
        try:
            user = _DEPS.user_resolver(session_id)
        except Exception as exc:
            logger.warning("Failed to resolve user for session_id=%s via custom resolver: %s", session_id, exc)

    # Fallback to default ID resolver if still None
    if user is None and session_id:
        user = resolve_user_by_id(session_id)

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


def _normalize_similar_items_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": raw.get("status", "ok"),
        "item_id": raw.get("item_id"),
        "results": raw.get("results", []),
        "results_count": raw.get("results_count", 0),
        "error": raw.get("error"),
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
        "Cerca prodotti e-commerce usando la pipeline completa di search, ranking e trust. "
        "Se include_shipping=true, calcola automaticamente i costi di spedizione "
        "per il primo risultato."
    ),
)
async def search_products(query: str, include_shipping: bool = False, session_id: str = "") -> str:
    try:
        logger.info("MCP TOOL search_products START")

        with _db_context() as db:
            context = _build_context(db=db, session_id=session_id)

            raw = await run_search_pipeline(
                query=query,
                db=db,
                user=context.user,
                llm_engine=context.llm_engine,
                session_id=session_id,
            )

            normalized = _normalize_search_output(raw)

            if include_shipping and normalized.get("top_result"):
                top_item_id = normalized["top_result"].get("ebay_id")
                if top_item_id:
                    try:
                        logger.info("MCP TOOL search_products - fetching shipping for top item %s", top_item_id)
                        shipping_raw = await execute_shipping_costs_tool(
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


@mcp.tool(
    name="analyze_seller",
    description="Analizza un venditore e-commerce usando feedback, trust score e sentiment.",
)
async def analyze_seller(seller_name: str, page: int = 1, limit: int = 10, session_id: str = "") -> str:
    try:
        with _db_context() as db:
            raw = await run_seller_pipeline(
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


@mcp.tool(
    name="profile_query",
    description=(
        "Analizza una query utente e restituisce un profilo strutturato "
        "utile per capire brand, prezzo, taglia, categoria e altri vincoli."
    ),
)
async def profile_query(query: str, session_id: str = "") -> str:
    try:
        parsed = await parse_query_service(query, session_id=session_id, context_info=session_id) # Using session_id as anchor for history
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
        "Risponde a messaggi conversazionali generici quando non serve usare tool eBay specifici."
    ),
)
async def conversation(query: str, llm_engine: str = "ollama", session_id: str = "") -> str:
    try:
        with _db_context() as db:
            context = _build_context(db=db, llm_engine=llm_engine, session_id=session_id)
            payload = await execute_conversation_tool({"query": query}, context)

            if not isinstance(payload, dict):
                payload = {"result": payload}

            payload.setdefault("status", "ok")
            payload["_backend"] = "mcp"

            return _safe_json(payload)

    except Exception as exc:
        logger.exception("MCP conversation failed")
        return _tool_error(query=query, error=str(exc))


@mcp.tool(
    name="compare_products",
    description=(
        "Confronta più prodotti e-commerce cercando ognuno in parallelo. "
        "Restituisce una matrice di confronto (prezzo, trust, rilevanza, condizione) "
        "con il prodotto vincitore e la motivazione. "
        "Input: lista di query separate da virgola o punto e virgola, ad esempio "
        "'iphone 13, samsung galaxy s22'. Min 2, max 4 query."
    ),
)
async def compare_products(queries: str, llm_engine: str = "ollama", session_id: str = "") -> str:
    """
    queries: stringa con le query separate da virgola o punto e virgola.
    Esempio: 'nike air max, adidas ultraboost'
    """
    from app.services.compare_pipeline import run_compare_pipeline

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

        with _db_context() as db:
            context = _build_context(db=db, llm_engine=llm_engine, session_id=session_id)

            logger.info("MCP TOOL compare_products START | queries=%s", queries)

            result = await execute_compare_tool(
                {"queries": queries},
                context
            )

            result["_backend"] = "mcp"
            logger.info("MCP TOOL compare_products END")
            return _safe_json(result)

    except Exception as exc:
        logger.exception("MCP compare_products failed")
        return _tool_error(queries=queries, error=str(exc))


@mcp.tool(
    name="get_item_details",
    description=(
        "Recupera i dettagli estesi, descrizione e specifiche tecniche di un prodotto "
        "conoscendo il suo ID eBay (item_id)."
    ),
)
async def get_item_details(item_id: str, session_id: str = "") -> str:
    try:
        with _db_context() as db:
            context = _build_context(db=db, session_id=session_id)
            logger.info("MCP TOOL get_item_details START | item_id=%s", item_id)
            
            result = await execute_item_details_tool(
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


@mcp.tool(
    name="get_shipping_costs",
    description=(
        "Recupera i costi e le opzioni di spedizione precisi per un determinato oggetto "
        "(item_id) verso un CAP (zip_code) e Paese (country_code)."
    ),
)
async def get_shipping_costs(item_id: str, country_code: str = "IT", zip_code: str = "", session_id: str = "") -> str:
    try:
        with _db_context() as db:
            context = _build_context(db=db, session_id=session_id)
            logger.info("MCP TOOL get_shipping_costs START | item_id=%s", item_id)
            
            result = await execute_shipping_costs_tool(
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


@mcp.tool(
    name="get_similar_items",
    description=(
        "Recupera prodotti simili o correlati per un oggetto (item_id)."
    )
)
async def get_similar_items(item_id: str, session_id: str = "") -> str:
    try:
        with _db_context() as db:
            context = _build_context(db=db, session_id=session_id)
            logger.info("MCP TOOL get_similar_items START | item_id=%s", item_id)
            
            result = await execute_similar_items_tool(
                {"item_id": item_id},
                context
            )
            normalized = _normalize_similar_items_output(result)
            normalized["_backend"] = "mcp"
            
            logger.info("MCP TOOL get_similar_items END")
            return _safe_json(normalized)
    except Exception as exc:
        logger.exception("MCP get_similar_items failed")
        return _tool_error(item_id=item_id, error=str(exc))

@mcp.tool(
    name="get_marketplace_metadata",
    description=(
        "Recupera i metadata delle policy eBay per un marketplace: condizioni articolo, "
        "politiche di reso, struttura listino (varianti). "
        "Specifica policy_type tra 'item_conditions', 'return_policies', 'listing_structure'."
    )
)
async def get_marketplace_metadata(policy_type: str = "item_conditions", marketplace_id: str = "", category_id: str = "", session_id: str = "") -> str:
    try:
        with _db_context() as db:
            context = _build_context(db=db, session_id=session_id)
            logger.info("MCP TOOL get_marketplace_metadata START | policy_type=%s", policy_type)
            
            result = await execute_metadata_tool(
                {
                    "policy_type": policy_type,
                    "marketplace_id": marketplace_id,
                    "category_id": category_id or None,
                },
                context
            )
            
            result["_backend"] = "mcp"
            
            logger.info("MCP TOOL get_marketplace_metadata END")
            return _safe_json(result)
    except Exception as exc:
        logger.exception("MCP get_marketplace_metadata failed")
        return _tool_error(policy_type=policy_type, error=str(exc))

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
                {
                    "name": "get_similar_items",
                    "description": "Recupera prodotti simili o correlati per un oggetto (item_id)",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                        },
                        "required": ["item_id"],
                    },
                },
                {
                    "name": "get_marketplace_metadata",
                    "description": "Recupera i metadata delle policy eBay per un marketplace",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "policy_type": {"type": "string", "default": "item_conditions"},
                            "marketplace_id": {"type": "string", "default": ""},
                            "category_id": {"type": "string", "default": ""},
                        },
                    },
                },
            ]
        }
    )


@mcp.resource("profile://query/{text}")
async def query_profile_resource(text: str) -> str:
    try:
        parsed = await parse_query_service(text)
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
    from app.services.memory_service import get_session_memory
    memory = get_session_memory(user_key)
    return _safe_json(
        {
            "user_key": user_key,
            "session_memory": {
                "recent_queries": memory.get("recent_queries", []),
                "recent_sellers": memory.get("recent_sellers", []),
                "recent_products": memory.get("history", []),
            },
            "note": "Session memory fetched from Redis.",
        }
    )


@mcp.resource("memory://long-term/{user_key}")
def long_term_memory_resource(user_key: str) -> str:
    db = _get_db()
    user_preferences = {}
    if db:
        try:
            user = resolve_user_by_id(user_key)
            if user:
                user_preferences = {
                    "favorite_brands": user.favorite_brands,
                    "price_preference": user.price_preference
                }
        finally:
            _close_db(db)
            
    return _safe_json(
        {
            "user_key": user_key,
            "long_term_memory": {
                "user_preferences": user_preferences,
                "previous_searches": [],
                "user_behaviour": {},
            },
            "note": "Long-term memory user preferences fetched from DB.",
        }
    )


# ============================================================
# MCP PROMPTS
# ============================================================

@mcp.prompt(name="search_assistant_prompt")
def search_assistant_prompt(query: str) -> str:
    return f"""
Sei un assistente e-commerce intelligente in grado di utilizzare session memory, 
long-term memory (es. preferenze utente come favorite_brands e price_preference) 
e diversi strumenti per assistere l'utente.

Usa il tool `search_products` per cercare prodotti rilevanti basandoti sulla seguente richiesta:

Query utente: {query}

Passi da seguire se applicabili:
1. Controlla le history e memory per personalizzare i risultati.
2. Controlla e confronta i risultati per fornire le migliori opzioni.
3. Se menzionati, fai riferimento ai costi di spedizione.
Rispondi in modo sintetico, utile e concreto.
""".strip()


@mcp.prompt(name="shopping_expert_prompt")
def shopping_expert_prompt(query: str) -> str:
    return f"""
Sei un Shopping Expert Agent molto accurato ed esauriente.
Segui attentamente questo flusso quando rispondi all'utente:

Query utente: {query}

FLUSSO CONSIGLIATO:
1. Esegui `profile_query` per capire esattamente intenti, categoria, budget e preferenze della query.
2. Esegui `search_products` (o più ricerche mirate) tenendo in considerazione questi attributi.
3. Se l'utente chiede informazioni su regole, condizioni, resi o politiche del marketplace, usa `get_marketplace_metadata`.
4. Esegui `compare_products` passando gli ID o titoli dei risultati migliori trovati per fornire un'analisi dettagliata.
5. (Opzionale) Ottieni ulteriori dettagli o oggetti simili se la richiesta dell'utente è vaga.

Sintetizza i risultati con una chiara raccomandazione finale.
""".strip()


@mcp.prompt(name="seller_assistant_prompt")
def seller_assistant_prompt(seller_name: str) -> str:
    return f"""
Sei un assistente che analizza affidabilità di venditori e-commerce.
Usa il tool `analyze_seller` per questo venditore:

Venditore: {seller_name}

Riassumi feedback, trust score e possibili criticità.
""".strip()