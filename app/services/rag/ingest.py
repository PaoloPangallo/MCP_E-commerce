from typing import List, Dict

from .vector_store import add_documents as vector_add

# bm25_store potrebbe non avere add_documents: lo gestiamo in try/except
try:
    from .bm25_store import add_documents as bm25_add
except Exception:
    bm25_add = None


def ingest_seller_feedback(
    seller_name: str,
    feedbacks: List[Dict],
    max_docs: int = 20
) -> int:
    """
    Converte i feedback venditore in "documenti" RAG e li indicizza.
    Ritorna quanti documenti sono stati indicizzati.
    """

    if not seller_name or not feedbacks:
        return 0

    texts: List[str] = []
    metas: List[Dict] = []

    for fb in feedbacks[:max_docs]:

        # prova a trovare un testo utile nel feedback
        comment = (fb.get("comment") or fb.get("text") or "").strip()
        if not comment:
            continue

        rating = fb.get("rating") or fb.get("type") or fb.get("value")
        ts = fb.get("time") or fb.get("date") or fb.get("timestamp")

        # Documento "pulito" e informativo (utile al retrieval semantico)
        doc_text = f"Seller {seller_name} feedback: {comment}"
        if rating:
            doc_text += f" Rating: {rating}."

        meta = {
            # IMPORTANTISSIMO: serve al tuo retrieve_context (key = d.get('text'))
            "text": doc_text,

            # campi utili per filtri/interpretazione
            "seller": seller_name,
            "rating": rating,
            "time": ts,
            "source": "seller_feedback",
        }

        texts.append(doc_text)
        metas.append(meta)

    if not texts:
        return 0

    # indicizza nel vector store (FAISS)
    vector_add(texts, metas)

    # indicizza anche in BM25 se disponibile
    if bm25_add is not None:
        try:
            bm25_add(texts, metas)
        except Exception:
            pass

    return len(texts)