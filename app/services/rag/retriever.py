from typing import List, Dict
from .vector_store import search


def retrieve_context(query: str, k: int = 5) -> List[Dict]:
    """
    Recupera documenti rilevanti per la query.
    """

    docs = search(query, k=k)

    return docs