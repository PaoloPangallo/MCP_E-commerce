from typing import List, Dict

from .vector_store import search as vector_search
from .bm25_store import search as bm25_search


def retrieve_context(query: str, k: int = 5) -> List[Dict]:

    vector_docs = vector_search(query, k=k)
    bm25_docs = bm25_search(query, k=k)

    merged = {}

    for d in vector_docs:

        key = d.get("text")

        merged[key] = d
        merged[key]["_source"] = "vector"

    for d in bm25_docs:

        key = d.get("text")

        if key not in merged:

            merged[key] = d
            merged[key]["_source"] = "bm25"

    return list(merged.values())[:k]