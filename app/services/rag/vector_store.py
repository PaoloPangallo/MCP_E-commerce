import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Optional

from app.services.rag.embedding import embed
from app.services.rag.schemas import make_doc_id

DIM = 384

class VectorStore:
    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.index_path = f"rag_index_{collection_name}.faiss"
        self.meta_path = f"rag_metadata_{collection_name}.pkl"
        
        self.index = faiss.IndexFlatIP(DIM)
        self.documents: List[Dict] = []
        self.seen_doc_ids = set()
        
        self.load()

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def _rebuild_seen(self):
        self.seen_doc_ids = set()
        for d in self.documents:
            did = d.get("doc_id")
            if did:
                self.seen_doc_ids.add(did)
            else:
                t = (d.get("text") or "").strip()
                if t:
                    self.seen_doc_ids.add(make_doc_id(t))

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.documents = pickle.load(f)
        self._rebuild_seen()

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.documents, f)

    def add_documents(self, texts: List[str], metadata: List[Dict]):
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

            if m["doc_id"] in self.seen_doc_ids:
                continue

            vec = embed(clean)
            if vec is None:
                continue

            vec = np.asarray(vec, dtype=np.float32)
            if vec.shape[0] != DIM:
                continue

            vec = self._normalize(vec)
            vectors.append(vec)
            metas.append(m)

            self.seen_doc_ids.add(m["doc_id"])
            added += 1

        if not vectors:
            return

        vectors = np.vstack(vectors)
        self.index.add(vectors)
        self.documents.extend(metas)

        if added > 0:
            self.save()

    def search(self, query: str, k: int = 5, doc_type: Optional[str] = None) -> List[Dict]:
        if len(self.documents) == 0:
            return []

        query = (query or "").strip()
        if not query:
            return []

        q = embed(query)
        if q is None:
            return []

        q = np.asarray(q, dtype=np.float32)
        q = self._normalize(q)
        q = np.expand_dims(q, axis=0)

        kk = k * 5 if doc_type else k
        scores, ids = self.index.search(q, kk)

        results: List[Dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = dict(self.documents[idx])
            doc["_similarity"] = float(score)
            if doc_type and doc.get("type") != doc_type:
                continue
            results.append(doc)
            if len(results) >= k:
                break
        return results

# Singleton for default collection to maintain backward compatibility
_default_store = VectorStore("default")

def add_documents(texts: List[str], metadata: List[Dict]):
    _default_store.add_documents(texts, metadata)

def search(query: str, k: int = 5, doc_type: Optional[str] = None) -> List[Dict]:
    return _default_store.search(query, k, doc_type)

def load_index():
    _default_store.load()

def save_index():
    _default_store.save()
