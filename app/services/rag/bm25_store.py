from rank_bm25 import BM25Okapi
from typing import List, Dict

_documents: List[Dict] = []
_corpus: List[List[str]] = []
_bm25 = None

_seen_texts = set()


def _tokenize(text: str):
    return text.lower().split()


def add_documents(texts: List[str], metadata: List[Dict]):

    global _bm25

    added = 0

    for text, meta in zip(texts, metadata):

        if not text:
            continue

        # dedup
        if text in _seen_texts:
            continue

        tokens = _tokenize(text)

        if not tokens:
            continue

        _corpus.append(tokens)
        _documents.append(meta)

        _seen_texts.add(text)
        added += 1

    if added > 0:
        _bm25 = BM25Okapi(_corpus)


def search(query: str, k: int = 5):

    if not _bm25:
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

    for idx, score in ranked[:k]:

        doc = dict(_documents[idx])
        doc["_bm25_score"] = float(score)

        results.append(doc)

    return results