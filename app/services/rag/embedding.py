import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

# ============================================================
# GPU CONFIG & MODEL LOAD
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
# Usiamo un modello multilingue per supportare meglio l'Italiano su eBay
_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
_model.to(device)


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