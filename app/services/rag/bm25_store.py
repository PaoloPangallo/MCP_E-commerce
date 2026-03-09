from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
from app.services.rag.schemas import make_doc_id

class BM25Store:
    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.documents: List[Dict] = []
        self.corpus: List[List[str]] = []
        self.bm25 = None
        self.seen_doc_ids = set()

    def _tokenize(self, text: str):
        return text.lower().split()

    def add_documents(self, texts: List[str], metadata: List[Dict]):
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
            tokens = self._tokenize(clean)
            if not tokens:
                continue
            self.corpus.append(tokens)
            self.documents.append(m)
            self.seen_doc_ids.add(m["doc_id"])
            added += 1
        if added > 0:
            self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, k: int = 5, doc_type: Optional[str] = None):
        if not self.bm25:
            return []
        query = (query or "").strip()
        if not query:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )
        results = []
        for idx, score in ranked:
            doc = dict(self.documents[idx])
            doc["_bm25_score"] = float(score)
            if doc_type and doc.get("type") != doc_type:
                continue
            results.append(doc)
            if len(results) >= k:
                break
        return results

# Singleton for default collection to maintain backward compatibility
_default_bm25 = BM25Store("default")

def add_documents(texts: List[str], metadata: List[Dict]):
    _default_bm25.add_documents(texts, metadata)

def search(query: str, k: int = 5, doc_type: Optional[str] = None):
    return _default_bm25.search(query, k, doc_type)