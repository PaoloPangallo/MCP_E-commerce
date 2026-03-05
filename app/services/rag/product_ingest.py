from typing import List, Dict

from app.services.rag.vector_store import add_documents
from app.services.rag.bm25_store import add_documents as bm25_add


# set per evitare duplicati
_seen_products = set()


def ingest_products(items: List[Dict]):

    texts = []
    metas = []

    for item in items:

        ebay_id = item.get("ebay_id")

        if not ebay_id:
            continue

        # -------------------------
        # DEDUP
        # -------------------------
        if ebay_id in _seen_products:
            continue

        _seen_products.add(ebay_id)

        title = item.get("title")
        price = item.get("price")
        seller = item.get("seller_name")
        condition = item.get("condition")

        if not title:
            continue

        # documento semantico
        text = f"""
        Product: {title}.
        Seller: {seller}.
        Price: {price}.
        Condition: {condition}.
        """

        text = " ".join(text.split())

        meta = {
            "text": text,
            "type": "product",
            "title": title,
            "seller": seller,
            "price": price,
            "ebay_id": ebay_id,
        }

        texts.append(text)
        metas.append(meta)

    if not texts:
        return

    # vector search
    add_documents(texts, metas)

    # lexical search
    bm25_add(texts, metas)