import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Optional

from app.services.rag.embedding import embed

DIM = 384

INDEX_PATH = "rag_index.faiss"
META_PATH = "rag_metadata.pkl"

# cosine similarity → inner product
_index = faiss.IndexFlatIP(DIM)

_documents: List[Dict] = []

# Dedup per testo (coerente con retriever key = doc["text"])
_seen_texts = set()


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _rebuild_seen_texts():
    global _seen_texts
    _seen_texts = set()

    for d in _documents:
        t = (d.get("text") or "").strip()
        if t:
            _seen_texts.add(t)


def load_index():
    global _index, _documents

    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)

    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            _documents = pickle.load(f)

    _rebuild_seen_texts()


def save_index():
    faiss.write_index(_index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(_documents, f)


def add_documents(texts: List[str], metadata: List[Dict]):
    """
    Aggiunge documenti al FAISS store.
    - Dedup basato sul campo 'text' (coerente con retriever.merge)
    - Forza meta['text'] = text per coerenza
    - Salva su disco automaticamente
    """
    if not texts or not metadata:
        return

    vectors = []
    metas = []
    added = 0

    for text, meta in zip(texts, metadata):

        if not text:
            continue

        text = " ".join(str(text).split()).strip()
        if not text:
            continue

        # Dedup globale per testo
        if text in _seen_texts:
            continue

        vec = embed(text)
        if vec is None:
            continue

        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape[0] != DIM:
            continue

        vec = _normalize(vec)

        # forza coerenza schema
        meta = dict(meta or {})
        meta["text"] = text

        vectors.append(vec)
        metas.append(meta)

        _seen_texts.add(text)
        added += 1

    if not vectors:
        return

    vectors = np.vstack(vectors)

    _index.add(vectors)
    _documents.extend(metas)

    if added > 0:
        save_index()


def search(query: str, k: int = 5) -> List[Dict]:
    if len(_documents) == 0:
        return []

    query = (query or "").strip()
    if not query:
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

    results: List[Dict] = []

    for score, idx in zip(scores[0], ids[0]):

        if idx < 0 or idx >= len(_documents):
            continue

        doc = dict(_documents[idx])
        doc["_similarity"] = float(score)
        results.append(doc)

    return results


# AUTO LOAD ON IMPORT
load_index()