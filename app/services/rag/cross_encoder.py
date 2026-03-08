from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def cross_rerank(query, items):

    if not items:
        return items

    pairs = [
        (query, item.get("title", ""))
        for item in items
    ]

    scores = model.predict(pairs)

    for item, score in zip(items, scores):
        item["_cross_score"] = float(score)

    items.sort(
        key=lambda x: x["_cross_score"],
        reverse=True
    )

    return items