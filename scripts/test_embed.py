import os
import sys
import numpy as np

# Aggiungi la root del progetto al path per gli import
sys.path.append(os.getcwd())

from app.services.rag.embedding import embed

def test_embed():
    q = "iphone 15 pro"
    i = "Apple iPhone 15 Pro 256GB Titanio Naturale"
    
    q_emb = embed(q)
    i_emb = embed(i)
    
    print(f"Query embedding shape: {q_emb.shape}, non-zero: {np.count_nonzero(q_emb)}")
    print(f"Item embedding shape: {i_emb.shape}, non-zero: {np.count_nonzero(i_emb)}")
    
    sim = float(np.dot(q_emb, i_emb))
    print(f"Similarity: {sim}")

if __name__ == "__main__":
    test_embed()
