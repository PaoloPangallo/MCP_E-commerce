from typing import List, Dict, Optional

from app.services.rag.qdrant_store import search as qdrant_search

def retrieve_context(
    query: str,
    k: int = 5,
    doc_type: Optional[str] = None,
    per_source: Optional[int] = None,
    k0: int = 60
) -> List[Dict]:
    """
    Hybrid retrieval now uses Qdrant natively.
    Qdrant handles Dense + Sparse vectors and RRF fusion internally.
    """
    query = (query or "").strip()
    if not query:
        return []

    take = per_source if per_source is not None else k

    # qdrant_search already executes a FusionQuery (RRF) internally 
    # and filters by doc_type.
    docs = qdrant_search(query, k=take, doc_type=doc_type)
    
    # We add _sources for backward compatibility with the explainer
    # Since it's hybrid from Qdrant, we just tag it as both.
    for d in docs:
        d["_sources"] = ["vector", "bm25"]
        d["_source"] = "hybrid"
        
    return docs