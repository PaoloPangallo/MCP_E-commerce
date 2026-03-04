import faiss
import numpy as np
from typing import List, Dict

from app.services.rag.embedding import embed

# dimensione embedding MiniLM
DIM = 384

# indice FAISS
_index = faiss.IndexFlatL2(DIM)

# document store
_documents: List[Dict] = []


def add_documents(texts: List[str], metadata: List[Dict]):
    """
    Aggiunge documenti al vector store.
    """


    vectors = []

    for text in texts:
        vec = embed(text)

        if vec is not None:
            vectors.append(vec)

    if not vectors:
        return

    vectors = np.vstack(vectors)

    _index.add(vectors)

    _documents.extend(metadata)


def search(query: str, k: int = 5):


    q = embed(query)

    if q is None:
        return []

    q = np.expand_dims(q, axis=0)

    distances, ids = _index.search(q, k)

    results = []

    for i in ids[0]:
        if i < len(_documents):
            results.append(_documents[i])

    return results