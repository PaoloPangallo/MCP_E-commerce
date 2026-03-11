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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def _run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    run_compare_pipeline(
                        queries=sep_queries,
                        db=db,
                        llm_engine=llm_engine
                    )
                )
            finally:
                new_loop.close()

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_in_new_loop)
                return future.result()
        else:
            return _run_in_new_loop()

    except Exception as exc:
        logger.exception("compare_tool execution failed")
        return {
            "status": "error",
            "error": str(exc),
            "queries": sep_queries
        }

