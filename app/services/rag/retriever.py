from typing import List, Dict, Optional
import logging

from app.services.rag.qdrant_store import search as qdrant_search

logger = logging.getLogger(__name__)


def retrieve_context(
    query: str,
    k: int = 5,
    doc_type: Optional[str] = None,
    per_source: Optional[int] = None,
    k0: int = 60
) -> List[Dict]:
    """
    Hybrid retrieval: utilizza esclusivamente Qdrant (Dense+Sparse+RRF)
    per il recupero del contesto.
    """
    query = (query or "").strip()
    if not query:
        return []

    take = per_source if per_source is not None else k

    try:
        docs = qdrant_search(query, k=take, doc_type=doc_type)
        if docs:
            for d in docs:
                d["_sources"] = ["vector", "bm25"]
                d["_source"] = "hybrid"
            return docs
    except Exception as exc:
        logger.error(
            "Qdrant retrieval failed: %s", exc
        )

    return []