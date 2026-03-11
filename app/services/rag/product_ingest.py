from typing import List, Dict

from app.services.rag.qdrant_store import add_documents
from app.services.rag.schemas import make_doc_id
from app.services.rag.chunking import chunk_text

_seen_products = set()


def ingest_products(items: List[Dict]):

    texts = []
    metas = []

    for item in items:

        ebay_id = item.get("ebay_id")
        if not ebay_id:
            continue

        if ebay_id in _seen_products:
            continue
        _seen_products.add(ebay_id)

        title = item.get("title")
        price = item.get("price")
        seller = item.get("seller_name")
        condition = item.get("condition")

        if not title:
            continue

        text = f"Product: {title}. Seller: {seller}. Price: {price}. Condition: {condition}."
        text = " ".join(text.split()).strip()

        chunks = chunk_text(text, chunk_size=150, overlap=30)
        
        for i, chunk in enumerate(chunks):
            meta = {
                "doc_id": make_doc_id(chunk),
                "text": chunk,
                "type": "product",
                "title": title,
                "seller": seller,
                "price": price,
                "condition": condition,
                "ebay_id": ebay_id,
                "source": "product_ingest",
            }
    
            texts.append(chunk)
            metas.append(meta)

    if not texts:
        return

    add_documents(texts, metas)