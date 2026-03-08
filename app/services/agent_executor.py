from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from sqlalchemy.orm import Session

from app.services.search_pipeline import run_search_pipeline
from app.services.seller_pipeline import run_seller_pipeline

logger = logging.getLogger(__name__)


async def run_agent(
    query: str,
    db: Session,
    user: Optional[object] = None,
    llm_engine: str = "ollama",
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream semplice stile agent:
    - emette step di thinking/action
    - esegue search pipeline completa
    - opzionalmente analizza il seller top
    """

    if not query or not query.strip():
        yield {
            "type": "final",
            "results": [],
            "error": "Query vuota"
        }
        return

    yield {
        "step": 1,
        "type": "thought",
        "message": "Analizzo la query"
    }

    await asyncio.sleep(0.05)

    yield {
        "step": 2,
        "type": "action",
        "message": "Cerco prodotti su eBay"
    }

    try:
        # eseguo la pipeline bloccante fuori dal loop async
        payload = await asyncio.to_thread(
            run_search_pipeline,
            query,
            db,
            user,
            llm_engine,
        )
    except Exception as e:
        logger.exception("run_search_pipeline failed: %s", e)

        yield {
            "step": 3,
            "type": "observation",
            "message": "Errore durante la ricerca"
        }

        yield {
            "type": "final",
            "results": [],
            "error": str(e),
        }
        return

    results = payload.get("results") or []
    results_count = payload.get("results_count", len(results))

    yield {
        "step": 3,
        "type": "observation",
        "message": f"Trovati {results_count} risultati"
    }

    await asyncio.sleep(0.05)

    top = results[0] if results else None

    if top and top.get("seller_name"):
        seller_name = top["seller_name"]

        yield {
            "step": 4,
            "type": "action",
            "message": f"Analizzo venditore {seller_name}"
        }

        try:
            seller_data = await asyncio.to_thread(
                run_seller_pipeline,
                seller_name,
                1,
                10,
            )

            # arricchisco il top result senza toccare troppo il contratto
            top["seller_analysis"] = {
                "seller_name": seller_data.get("seller_name"),
                "trust_score": seller_data.get("trust_score"),
                "sentiment_score": seller_data.get("sentiment_score"),
                "feedback_count": seller_data.get("count"),
            }

            yield {
                "step": 5,
                "type": "observation",
                "message": "Trust score calcolato"
            }

        except Exception as e:
            logger.warning("run_seller_pipeline failed for %s: %s", seller_name, e)

            yield {
                "step": 5,
                "type": "observation",
                "message": "Analisi venditore non disponibile"
            }

    yield {
        "type": "final",
        "results": results,
        "payload": payload,
    }