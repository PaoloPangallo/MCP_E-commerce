from functools import lru_cache

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """Load the model only when reranking actually needs it."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def cross_rerank(query, items):

    if not items:
        return items

    pairs = [
        (query, item.get("title", ""))
        for item in items
    ]

    scores = _get_model().predict(pairs)

    for item, score in zip(items, scores):
        item["_cross_score"] = float(score)

    items.sort(
        key=lambda x: x["_cross_score"],
        reverse=True
    )

    return items
