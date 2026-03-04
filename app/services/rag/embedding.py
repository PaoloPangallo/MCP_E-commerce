from sentence_transformers import SentenceTransformer
import numpy as np

# modello piccolo e veloce
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str) -> np.ndarray:
    """
    Converte una stringa in embedding vector.
    """
    if not text:
        return None

    vec = _model.encode(text)

    return np.array(vec).astype("float32")