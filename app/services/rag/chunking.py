from typing import List

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """
    Subdivide large text into smaller overlapping chunks (by words).
    Default size is small to keep embeddings highly focused (around ~150 words).
    """
    if not text:
        return []
        
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += (chunk_size - overlap)
        
    return chunks
