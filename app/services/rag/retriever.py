from typing import List, Dict, Optional

from .vector_store import search as vector_search
from .bm25_store import search as bm25_search


def _rrf(rank: int, k0: int = 60) -> float:
    return 1.0 / (k0 + rank)


def retrieve_context(
    query: str,
    k: int = 5,
    doc_type: Optional[str] = None,
    per_source: Optional[int] = None,
    k0: int = 60
) -> List[Dict]:
    """
    Hybrid retrieval:
      - vector_search + bm25_search
      - RRF fusion
      - dedup by doc_id (fallback text)
      - optional doc_type filter

    per_source:
      - if None => use k
      - else fetch per_source from each source before fusion
    """

    query = (query or "").strip()
    if not query:
        return []

    take = per_source if per_source is not None else k

    vector_docs = vector_search(query, k=take, doc_type=doc_type)
    bm25_docs = bm25_search(query, k=take, doc_type=doc_type)

    merged: Dict[str, Dict] = {}

    def key_of(d: Dict) -> str:
        return d.get("doc_id") or (d.get("text") or "").strip()

    # vector contribution
    for rank, d in enumerate(vector_docs, start=1):
        key = key_of(d)
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
        merged[key]["_source"] = "hybrid"

    # bm25 contribution
    for rank, d in enumerate(bm25_docs, start=1):
        key = key_of(d)
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
    results.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)

    return results[:k]