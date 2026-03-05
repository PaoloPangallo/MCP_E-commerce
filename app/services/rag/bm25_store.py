from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
from app.services.rag.schemas import make_doc_id

_documents: List[Dict] = []
_corpus: List[List[str]] = []
_bm25 = None

_seen_doc_ids = set()


def _tokenize(text: str):
    return text.lower().split()


def add_documents(texts: List[str], metadata: List[Dict]):

    global _bm25
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

        # dedup by doc_id
        if m["doc_id"] in _seen_doc_ids:
            continue

        tokens = _tokenize(clean)
        if not tokens:
            continue

        _corpus.append(tokens)
        _documents.append(m)

        _seen_doc_ids.add(m["doc_id"])
        added += 1

    if added > 0:
        _bm25 = BM25Okapi(_corpus)


def search(query: str, k: int = 5, doc_type: Optional[str] = None):

    if not _bm25:
        return []

    query = (query or "").strip()
    if not query:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = _bm25.get_scores(tokens)

    ranked = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in ranked:

        doc = dict(_documents[idx])
        doc["_bm25_score"] = float(score)

        if doc_type and doc.get("type") != doc_type:
            continue

        results.append(doc)

        if len(results) >= k:
            break

    return results