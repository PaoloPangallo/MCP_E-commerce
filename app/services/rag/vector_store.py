import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Optional

from app.services.rag.embedding import embed
from app.services.rag.schemas import make_doc_id

DIM = 384

INDEX_PATH = "rag_index.faiss"
META_PATH = "rag_metadata.pkl"

_index = faiss.IndexFlatIP(DIM)

_documents: List[Dict] = []
_seen_doc_ids = set()


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _rebuild_seen():
    global _seen_doc_ids
    _seen_doc_ids = set()

    for d in _documents:
        did = d.get("doc_id")
        if did:
            _seen_doc_ids.add(did)
        else:
            # fallback: compute from text if old docs
            t = (d.get("text") or "").strip()
            if t:
                _seen_doc_ids.add(make_doc_id(t))


def load_index():
    global _index, _documents

    if os.path.exists(INDEX_PATH):
        _index = faiss.read_index(INDEX_PATH)

    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            _documents = pickle.load(f)

    _rebuild_seen()


def save_index():
    faiss.write_index(_index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(_documents, f)


def add_documents(texts: List[str], metadata: List[Dict]):
    """
    - Dedup by doc_id
    - Force meta['text'] and meta['doc_id']
    - Persist index
    """
    if not texts or not metadata:
        return

    vectors = []
    metas = []
    added = 0

    for text, meta in zip(texts, metadata):

        if not text:
            continue

        clean = " ".join(str(text).split()).strip()
        if not clean:
            continue

        m = dict(meta or {})
        m["text"] = clean
        m["doc_id"] = m.get("doc_id") or make_doc_id(clean)

        # dedup
        if m["doc_id"] in _seen_doc_ids:
            continue

        vec = embed(clean)
        if vec is None:
            continue

        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape[0] != DIM:
            continue

        vec = _normalize(vec)

        vectors.append(vec)
        metas.append(m)

        _seen_doc_ids.add(m["doc_id"])
        added += 1

    if not vectors:
        return

    vectors = np.vstack(vectors)

    _index.add(vectors)
    _documents.extend(metas)

    if added > 0:
        save_index()


def search(query: str, k: int = 5, doc_type: Optional[str] = None) -> List[Dict]:
    """
    Returns top-k from FAISS. If doc_type is provided, filters results by meta['type'].
    Filtering is post-search (fast enough for your scale).
    """
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

    # oversample a bit if we need to filter
    kk = k * 5 if doc_type else k

    scores, ids = _index.search(q, kk)

    results: List[Dict] = []

    for score, idx in zip(scores[0], ids[0]):

        if idx < 0 or idx >= len(_documents):
            continue

        doc = dict(_documents[idx])
        doc["_similarity"] = float(score)

        if doc_type and doc.get("type") != doc_type:
            continue

        results.append(doc)

        if len(results) >= k:
            break

    return results


# auto-load
load_index()