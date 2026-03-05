import traceback

from app.services.parser import parse_query_service
from app.services.ebay import search_items
from app.services.rag.product_ingest import ingest_products
from app.services.rag import retrieve_context
from app.services.rag.reranker import rerank_products


query = "scarpe"

print("\n======================")
print("TEST SEARCH PIPELINE")
print("======================\n")


# ============================================================
# 1) PARSER
# ============================================================

try:
    print("1️⃣ PARSER")

    parsed = parse_query_service(
        query,
        use_llm=True,
        include_meta=True
    )

    print("Parsed query:")
    print(parsed)

except Exception:
    print("❌ Parser error")
    traceback.print_exc()
    exit()


# ============================================================
# 2) EBAY SEARCH
# ============================================================

try:
    print("\n2️⃣ EBAY SEARCH")

    items = search_items(
        parsed_query=parsed,
        limit=10
    )

    items = items or []

    print("Items found:", len(items))

    if items:
        print("\nSample item:")
        print(items[0])

except Exception:
    print("❌ eBay error")
    traceback.print_exc()
    exit()


# ============================================================
# 3) RAG INGEST
# ============================================================

try:
    print("\n3️⃣ INGEST PRODUCTS")

    ingest_products(items)

    print("✅ ingest ok")

except Exception:
    print("❌ ingest error")
    traceback.print_exc()


# ============================================================
# 4) RAG RETRIEVAL
# ============================================================

try:
    print("\n4️⃣ RETRIEVE CONTEXT")

    docs = retrieve_context(query, k=5)

    print("Documents retrieved:", len(docs))

    if docs:
        print("\nSample context doc:")
        print(docs[0])

except Exception:
    print("❌ retrieve error")
    traceback.print_exc()


# ============================================================
# 5) RERANK
# ============================================================

try:
    print("\n5️⃣ RERANK")

    reranked = rerank_products(query, items)

    print("Reranked items:", len(reranked))

except Exception:
    print("❌ rerank error")
    traceback.print_exc()


print("\n======================")
print("TEST FINISHED")
print("======================")