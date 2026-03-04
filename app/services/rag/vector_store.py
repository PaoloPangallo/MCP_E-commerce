import faiss
import numpy as np
from typing import List, Dict

from app.services.rag.embedding import embed

DIM = 384

# cosine similarity → inner product
_index = faiss.IndexFlatIP(DIM)

_documents: List[Dict] = []


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def add_documents(texts: List[str], metadata: List[Dict]):

    if not texts or not metadata:
        return

    vectors = []
    metas = []

    for text, meta in zip(texts, metadata):

        vec = embed(text)

        if vec is None:
            continue

        vec = np.asarray(vec, dtype=np.float32)

        if vec.shape[0] != DIM:
            continue

        vec = _normalize(vec)

        vectors.append(vec)
        metas.append(meta)

    if not vectors:
        return

    vectors = np.vstack(vectors)

    _index.add(vectors)

    _documents.extend(metas)


def search(query: str, k: int = 5):

    if len(_documents) == 0:
        return []

    q = embed(query)

    if q is None:
        return []

    q = np.asarray(q, dtype=np.float32)

    if q.shape[0] != DIM:
        return []

    q = _normalize(q)
    q = np.expand_dims(q, axis=0)

    scores, ids = _index.search(q, k)

    results = []

    for score, idx in zip(scores[0], ids[0]):

        if idx < 0 or idx >= len(_documents):
            continue

        doc = dict(_documents[idx])

        doc["_similarity"] = float(score)

        results.append(doc)

    return results