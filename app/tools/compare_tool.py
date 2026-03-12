from __future__ import annotations

import logging
from typing import Any, Dict

from app.services.compare_pipeline import run_compare_pipeline

logger = logging.getLogger(__name__)


async def execute_compare_tool(action_input: Dict[str, Any], context: Any) -> Dict[str, Any]:
    queries_str = action_input.get("queries") or ""
    llm_engine = getattr(context, "llm_engine", "ollama")
    db = context.db

    sep_queries = [
        q.strip()
        for q in queries_str.replace(";", ",").split(",")
        if q.strip()
    ]

    if len(sep_queries) < 2:
        return {
            "status": "error",
            "error": "Fornisci almeno 2 query separate da virgola.",
        }

    logger.info("Executing compare_tool | queries=%s", sep_queries)

    try:
        return await run_compare_pipeline(
            queries=sep_queries,
            db=db,
            llm_engine=llm_engine
        )
    except Exception as exc:
        logger.exception("compare_tool execution failed")
        return {
            "status": "error",
            "error": str(exc),
        }
