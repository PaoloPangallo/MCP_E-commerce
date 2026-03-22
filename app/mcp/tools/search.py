import logging
from typing import Dict, Any, Annotated
from pydantic import Field

from app.mcp.core import mcp, _db_context, _build_context, _tool_error
from app.mcp.normalizers import _normalize_search_output, _normalize_shipping_costs_output
from app.services.search_pipeline import run_search_pipeline
from app.services.parser import parse_query_service
from app.tools import execute_shipping_costs_tool, execute_compare_tool

logger = logging.getLogger(__name__)

@mcp.tool(
    name="search_products",
    description=(
        "Cerca prodotti e-commerce usando la pipeline completa di search, ranking e trust. "
        "Se include_shipping=true, calcola automaticamente i costi di spedizione "
        "per il primo risultato."
    ),
)
async def search_products(
    query: Annotated[str, Field(description="Query di ricerca testuale per il prodotto (es. 'iphone 13 nero', 'cappello estivo da uomo')")],
    include_shipping: Annotated[bool, Field(description="Se impostato a true, calcola dinamicamente le spese di spedizione per la top hit")] = False,
    session_id: Annotated[str, Field(description="Identificativo utente di sessione (fornito dal sistema)")] = ""
) -> Dict[str, Any]:
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
                        shipping_raw = await execute_shipping_costs_tool(
                            {"item_id": top_item_id, "country_code": "IT", "zip_code": ""},
                            context,
                        )
                        shipping_norm = _normalize_shipping_costs_output(shipping_raw)
                        normalized["top_result"]["shipping_info"] = shipping_norm.get("data")
                        normalized["top_result"]["shipping_status"] = shipping_norm.get("status")
                    except Exception as e:
                        logger.warning("Auto-shipping fetch failed in search pipeline: %s", e)

            normalized["_backend"] = "mcp"
            logger.info("MCP TOOL search_products END")
            return normalized

    except Exception as exc:
        logger.exception("MCP search_products failed")
        return _tool_error(query=query, error=str(exc))


@mcp.tool(
    name="profile_query",
    description=(
        "Analizza NLP una query utente ed estrae un object strutturato "
        "con brand, entità prodotto, range di prezzo e categoria."
    ),
)
async def profile_query(
    query: Annotated[str, Field(description="La query scritta in input dall'utente")],
    session_id: Annotated[str, Field(description="ID di sessione utente")] = ""
) -> Dict[str, Any]:
    try:
        parsed = await parse_query_service(query, session_id=session_id, context_info=session_id)
        return {
            "status": "ok",
            "query": query,
            "parsed": parsed,
            "_backend": "mcp",
        }
    except Exception as exc:
        logger.exception("MCP profile_query failed")
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
async def compare_products(
    queries: Annotated[str, Field(description="Stringa con le query da confrontare separate da virgola (es. 'nike air max, adidas ultraboost')")],
    llm_engine: Annotated[str, Field(description="Engine LLM da utilizzare per l'analisi (es. 'ollama')")] = "ollama",
    session_id: Annotated[str, Field(description="ID di sessione utente")] = ""
) -> Dict[str, Any]:
    try:
        sep_queries = [q.strip() for q in queries.replace(";", ",").split(",") if q.strip()]

        if len(sep_queries) < 2:
            return _tool_error(
                error="Fornisci almeno 2 query separate da virgola per confrontare i prodotti.",
                example="iphone 13, samsung galaxy s22",
            )

        with _db_context() as db:
            context = _build_context(db=db, llm_engine=llm_engine, session_id=session_id)
            logger.info("MCP TOOL compare_products START | queries=%s", queries)
            result = await execute_compare_tool({"queries": queries}, context)
            result["_backend"] = "mcp"
            logger.info("MCP TOOL compare_products END")
            return result
    except Exception as exc:
        logger.exception("MCP compare_products failed")
        return _tool_error(queries=queries, error=str(exc))
