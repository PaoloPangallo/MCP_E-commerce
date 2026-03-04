from rank_bm25 import BM25Okapi
from typing import List, Dict

_documents: List[Dict] = []
_corpus: List[List[str]] = []
_bm25 = None


def _tokenize(text: str):
    return text.lower().split()


def add_documents(texts: List[str], metadata: List[Dict]):

    global _bm25

    for text, meta in zip(texts, metadata):

        tokens = _tokenize(text)

        _corpus.append(tokens)
        _documents.append(meta)

    _bm25 = BM25Okapi(_corpus)


def search(query: str, k: int = 5):

    if not _bm25:
        return []

    tokens = _tokenize(query)

    scores = _bm25.get_scores(tokens)

    ranked = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in ranked[:k]:

        doc = dict(_documents[idx])
        doc["_bm25_score"] = float(score)

        results.append(doc)

    return results