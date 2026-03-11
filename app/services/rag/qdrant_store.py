import os
import uuid
from typing import List, Dict, Optional
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding

from app.services.rag.embedding import embed_batch, embed
from app.services.rag.schemas import make_doc_id

import threading

QDRANT_PATH = "qdrant_storage"
COLLECTION_NAME = "ecommerce_rag"
DENSE_DIM = 384

_client: Optional[QdrantClient] = None
_sparse_model: Optional[SparseTextEmbedding] = None
_init_lock = threading.Lock()

def _get_qdrant() -> tuple[QdrantClient, SparseTextEmbedding]:
    global _client, _sparse_model
    if _client is not None and _sparse_model is not None:
        return _client, _sparse_model
        
    with _init_lock:
        if _client is not None and _sparse_model is not None:
            return _client, _sparse_model
            
        _client = QdrantClient(path=QDRANT_PATH)
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        try:
            if not _client.collection_exists(COLLECTION_NAME):
                _client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=DENSE_DIM,
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            modifier=models.Modifier.IDF
                        )
                    }
                )
                # Create indexing for the 'type' field
                _client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="type",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning("Error initializing Qdrant collection: %s", e)
            
        return _client, _sparse_model

def add_documents(texts: List[str], metadata: List[Dict]):
    if not texts or not metadata:
        return

    client, sparse_model = _get_qdrant()
    points = []
    
    dense_ebs = embed_batch(texts)
    sparse_ebs = list(sparse_model.embed(texts))
    
    for i, (text, meta) in enumerate(zip(texts, metadata)):
        clean = " ".join(str(text).split()).strip()
        if not clean:
            continue
            
        m = dict(meta or {})
        m["text"] = clean
        # Doc ID is used as the Point ID (UUID format)
        doc_id_str = m.get("doc_id") or make_doc_id(clean)
        m["doc_id"] = doc_id_str
        
        # Convert string doc_id (which might be an MD5 hash) to a proper UUID for Qdrant
        # If it's not a valid UUID string, we can hash it to a UUID
        try:
            point_id = str(uuid.UUID(doc_id_str))
        except ValueError:
            import hashlib
            h = hashlib.md5(doc_id_str.encode("utf-8")).hexdigest()
            point_id = str(uuid.UUID(h))
            
        dense_vec = dense_ebs[i].tolist()
        sparse_vec = sparse_ebs[i]
        
        points.append(
            models.PointStruct(
                id=point_id,
                payload=m,
                vector={
                    "dense": dense_vec,
                    "sparse": models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist()
                    )
                }
            )
        )
        
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

def search(query: str, k: int = 5, doc_type: Optional[str] = None) -> List[Dict]:
    """
    Search combining dense and sparse vectors via Reciprocal Rank Fusion (handled inside retriever or natively by Qdrant if using Prefetch).
    Here we do a simple hybrid query using Qdrant's query API with Prefetch logic.
    """
    query = (query or "").strip()
    if not query:
        return []
    
    client, sparse_model = _get_qdrant()
    
    q_dense = embed(query)
    q_sparse = list(sparse_model.embed([query]))[0]
    
    filter_cond = None
    if doc_type:
        filter_cond = models.Filter(
            must=[
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=doc_type)
                )
            ]
        )
        
    # Hybrid search in Qdrant using RRF
    prefetch = [
        models.Prefetch(
            query=q_dense.tolist(),
            using="dense",
            limit=k * 2,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=q_sparse.indices.tolist(),
                values=q_sparse.values.tolist()
            ),
            using="sparse",
            limit=k * 2,
        ),
    ]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        query_filter=filter_cond,
        with_payload=True,
    )

    docs = []
    for point in results.points:
        doc = dict(point.payload)
        doc["_rrf_score"] = point.score
        docs.append(doc)
        
    return docs
