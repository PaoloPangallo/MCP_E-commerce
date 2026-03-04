from app.services.rag.vector_store import add_documents
from app.services.rag.retriever import retrieve_context

texts = [
    "spedizione velocissima",
    "venditore affidabile",
    "oggetto come descritto"
]

meta = [
    {"text": t} for t in texts
]

add_documents(texts, meta)

print(retrieve_context("venditore veloce"))