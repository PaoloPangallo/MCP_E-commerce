from typing import List, Dict
from .vector_store import search as vector_search
from .bm25_store import search as bm25_search


def _rrf(rank: int, k0: int = 60) -> float:
    # Reciprocal Rank Fusion: 1 / (k0 + rank)
    return 1.0 / (k0 + rank)


def retrieve_context(query: str, k: int = 5, k0: int = 60) -> List[Dict]:
    """
    Hybrid retrieval con RRF fusion tra:
      - FAISS semantic search
      - BM25 lexical search

    Ritorna documenti unici (key = text) con:
      - _rrf_score (fusion)
      - _sources (["vector","bm25"])
      - _similarity / _bm25_score se presenti
    """

    vector_docs = vector_search(query, k=k)
    bm25_docs = bm25_search(query, k=k)

    merged: Dict[str, Dict] = {}

    # RRF from vector ranks
    for rank, d in enumerate(vector_docs, start=1):
        key = (d.get("text") or "").strip()
        if not key:
            continue

        if key not in merged:
            merged[key] = dict(d)
            merged[key]["_sources"] = ["vector"]
            merged[key]["_rrf_score"] = 0.0
        else:
            if "vector" not in merged[key].get("_sources", []):
                merged[key]["_sources"].append("vector")

        merged[key]["_rrf_score"] += _rrf(rank, k0=k0)

        # label source (compatibilità con tuo debug)
        merged[key]["_source"] = "hybrid"

    # RRF from bm25 ranks
    for rank, d in enumerate(bm25_docs, start=1):
        key = (d.get("text") or "").strip()
        if not key:
            continue

        if key not in merged:
            merged[key] = dict(d)
            merged[key]["_sources"] = ["bm25"]
            merged[key]["_rrf_score"] = 0.0
        else:
            if "bm25" not in merged[key].get("_sources", []):
                merged[key]["_sources"].append("bm25")

        merged[key]["_rrf_score"] += _rrf(rank, k0=k0)
        merged[key]["_source"] = "hybrid"

    results = list(merged.values())

    # sort by fused score
    results.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)

    # keep top-k
    return results[:k]