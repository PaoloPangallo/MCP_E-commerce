"""
compare_tool.py

Wrapper for the compare_pipeline service to be used by both MCP and 
the local agent executor.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from app.services.compare_pipeline import run_compare_pipeline

logger = logging.getLogger(__name__)


def execute_compare_tool(action_input: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Executes the product comparison pipeline.
    
    Args:
        action_input: Dict containing 'queries' (comma-separated string).
        context: ToolContext containing DB session and optionally llm_engine.
        
    Returns:
        Structured comparison result.
    """
    queries_str = action_input.get("queries") or ""
    llm_engine = getattr(context, "llm_engine", "ollama")
    db = context.db

    # Parse queries (same logic as in server.py)
    sep_queries = [
        q.strip()
        for q in queries_str.replace(";", ",").split(",")
        if q.strip()
    ]

    if len(sep_queries) < 2:
        return {
            "status": "error",
            "error": "Fornisci almeno 2 query separate da virgola per confrontare i prodotti.",
            "example": "iphone 13, samsung galaxy s22",
        }

    logger.info("Executing compare_tool | queries=%s", sep_queries)

    # Since this is usually called from synchronous code (MCP or local executor),
    # we run the async pipeline synchronously.
    # Note: If called from an async context, this might need care with loop management.
    try:
        # Check if we are already in an event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # This is tricky if called from an async function that isn't awaited.
            # But the local executor and MCP server.py use sync wrappers.
            # In our case, EbayReactAgent.execute (local) is async, so we'll need an async version too.
            # For now, let's provide a sync-friendly wrapper that calls asyncio.run if no loop.
            return asyncio.run(run_compare_pipeline(
                queries=sep_queries,
                db=db,
                llm_engine=llm_engine
            ))
        else:
            return asyncio.run(run_compare_pipeline(
                queries=sep_queries,
                db=db,
                llm_engine=llm_engine
            ))
    except Exception as exc:
        logger.exception("compare_tool execution failed")
        return {
            "status": "error",
            "error": str(exc),
            "queries": sep_queries
        }
