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
    Hybrid retrieval: usa Qdrant (Dense+Sparse+RRF) con fallback automatico su FAISS
    quando Qdrant non è disponibile (dev locale, Docker spento, ecc.).
    """
    query = (query or "").strip()
    if not query:
        return []

    take = per_source if per_source is not None else k

    # --- Tentativo primario: Qdrant ---
    try:
        docs = qdrant_search(query, k=take, doc_type=doc_type)
        if docs:
            for d in docs:
                d["_sources"] = ["vector", "bm25"]
                d["_source"] = "hybrid"
            return docs
    except Exception as exc:
        logger.warning(
            "Qdrant retrieval failed (%s). Falling back to FAISS local index.", exc
        )

    # --- Fallback: FAISS locale ---
    try:
        from app.services.rag.vector_store import search as faiss_search
        docs = faiss_search(query, k=take, doc_type=doc_type)
        for d in docs:
            d["_sources"] = ["vector"]
            d["_source"] = "faiss_fallback"
        return docs
    except Exception as exc2:
        logger.warning("FAISS fallback also failed: %s", exc2)
        return []