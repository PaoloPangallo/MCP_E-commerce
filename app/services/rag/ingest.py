from typing import List, Dict

from app.services.rag.qdrant_store import add_documents
from app.services.rag.chunking import chunk_text
from app.services.nlp_sentiment import extract_sentiment_label


def ingest_seller_feedback(
    seller_name: str,
    feedbacks: List[Dict],
    max_docs: int = 20
) -> int:
    """
    Converte i feedback venditore in documenti RAG e li indicizza.
    """

    if not seller_name or not feedbacks:
        return 0

    texts: List[str] = []
    metas: List[Dict] = []

    for fb in feedbacks[:max_docs]:

        comment = (fb.get("comment") or fb.get("text") or "").strip()
        if not comment:
            continue

        rating = fb.get("rating") or fb.get("type") or fb.get("value")
        ts = fb.get("time") or fb.get("date") or fb.get("timestamp")

        doc_text = f"Seller {seller_name} feedback: {comment}"
        if rating:
            doc_text += f" Rating: {rating}."

        doc_text = " ".join(doc_text.split()).strip()
        if not doc_text:
            continue

        # Ask LLM for exact sentiment once per feedback
        sentiment_label = extract_sentiment_label(comment)

        chunks = chunk_text(doc_text, chunk_size=150, overlap=30)
        
        for i, chunk in enumerate(chunks):
            meta = {
                "text": chunk,
                "type": "seller_feedback",
                "seller": seller_name,
                "rating": rating,
                "time": ts,
                "source": "seller_feedback",
                "sentiment_label": sentiment_label,
            }

            texts.append(chunk)
            metas.append(meta)

    if not texts:
        return 0

    add_documents(texts, metas)

    return len(texts)