from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache


# ============================================================
# MODEL LOAD
# ============================================================

_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# SINGLE EMBEDDING (CACHED)
# ============================================================

@lru_cache(maxsize=10000)
def embed(text: str):

    if not text:
        return np.zeros(384, dtype="float32")

    try:

        vec = _model.encode(
            text,
            normalize_embeddings=True
        )

        return np.array(vec).astype("float32")

    except Exception:

        return np.zeros(384, dtype="float32")


# ============================================================
# BATCH EMBEDDING (FAST)
# ============================================================

def embed_batch(texts):

    if not texts:
        return []

    try:

        vectors = _model.encode(
            texts,
            normalize_embeddings=True
        )

        return [np.array(v).astype("float32") for v in vectors]

    except Exception:

        return [np.zeros(384, dtype="float32") for _ in texts]