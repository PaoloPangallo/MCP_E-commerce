import os
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from app.services.parser import OLLAMA_URL, OLLAMA_API_KEY, call_llm
from app.services.rag.qdrant_store import _get_qdrant, COLLECTION_NAME

async def verify():
    print("--- Parser Config Verification ---")
    print(f"OLLAMA_API_KEY: {OLLAMA_API_KEY}")
    print(f"OLLAMA_URL: {OLLAMA_URL}")
    
    # Test auto-upgrade logic in call_llm (without actually calling it)
    # We can inspect what call_llm would do by looking at how it was modified
    
    print("\n--- Qdrant Config Verification ---")
    try:
        from qdrant_client import QdrantClient
        client, sparse = _get_qdrant()
        # If it's a cloud client, it should have a different internal state than a local one
        # Local client uses a storage manager, cloud client uses a grpc/rest client
        print(f"Qdrant Client: {client}")
        # In QdrantClient, if it's local, it often has a 'location' or similar attribute
        if hasattr(client, '_client') and hasattr(client._client, 'rest_uri'):
            print(f"Qdrant seems to be remote: {client._client.rest_uri}")
        else:
            print("Qdrant seems to be local or initialized differently.")
    except Exception as e:
        print(f"Qdrant init failed (expected if QDRANT_URL is not actually reachable): {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(verify())
