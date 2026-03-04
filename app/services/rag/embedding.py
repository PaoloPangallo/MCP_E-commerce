from sentence_transformers import SentenceTransformer
import numpy as np

# modello piccolo e veloce
_model = SentenceTransformer("all-MiniLM-L6-v2")


from functools import lru_cache

@lru_cache(maxsize=10000)
def embed(text: str):

    if not text:
        return None

    vec = _model.encode(text, normalize_embeddings=True)

    return np.array(vec).astype("float32")