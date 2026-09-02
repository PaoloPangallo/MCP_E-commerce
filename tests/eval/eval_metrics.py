import math
from typing import List

def mean_reciprocal_rank(retrieved_lists: List[List[str]], relevant_items: List[str]) -> float:
    """
    Calcola il Mean Reciprocal Rank (MRR) per un set di query processate.
    
    :param retrieved_lists: Lista di liste contenenti gli ID degli item recuperati dal RAG ordinati per rilevanza calcolata.
    :param relevant_items: La controparte attesa (ground truth). L'ID dell'oggetto che l'utente davvero cercava per ogni query.
                           (Dimensione delle due liste esternamente deve combaciare).
    :return: MRR aggregato come float (1.0 = Max, 0.0 = Min)
    """
    if not retrieved_lists or len(retrieved_lists) != len(relevant_items):
        raise ValueError("Le due liste devono avere la stessa lunghezza")

    mrr_sum = 0.0
    for r_list, rel_item in zip(retrieved_lists, relevant_items):
        try:
            # Lavoriamo in 1-based index per il Rango
            rank = r_list.index(rel_item) + 1
            mrr_sum += 1.0 / rank
        except ValueError:
            # Il rel_item non è tra i risultati recuperati!
            mrr_sum += 0.0

    return mrr_sum / len(retrieved_lists)

def ndcg_at_k(retrieved_relevance_scores: List[List[float]], k: int = 5) -> float:
    """
    Calcola il Normalized Discounted Cumulative Gain (NDCG) a profondita' K.
    Serve a valutare se, per query simili, il sistema Reranker/Agente ha messo ai primi posti i prodotti migliori.
    
    :param retrieved_relevance_scores: Lista di metriche di 'rilevanza oggettiva' (es. ground_truth_scoring 0,1,2,3) 
                                       nell'ordine esatto in cui l'agente li ha restituiti.
    :param k: Cutoff top K.
    :return: nDCG medio per tutto il batch.
    """
    def dcg(scores_r):
        return sum(rel / math.log2(i + 1) for i, rel in enumerate(scores_r, start=1))
        
    ndcg_sum = 0.0
    
    for scores in retrieved_relevance_scores:
        top_k_scores = scores[:k]
        
        # dcg standard dell'algoritmo
        actual_dcg = dcg(top_k_scores)
        
        # dcg ideale: e' lo stesso pacchetto di score ma ordinato in base decrescente al 100%
        ideal_scores = sorted(scores, reverse=True)[:k]
        ideal_dcg = dcg(ideal_scores)
        
        if ideal_dcg > 0:
            ndcg_sum += actual_dcg / ideal_dcg
            
    return ndcg_sum / len(retrieved_relevance_scores) if retrieved_relevance_scores else 0.0

if __name__ == '__main__':
    # Test Sanity
    example_retrieved = [
        ["item_A", "item_B", "item_C"], # La query 1 ha come risposta attesa 'item_B', quindi rank = 2, RR = 0.5
        ["item_X", "item_Y", "item_Z"]  # La query 2 ha come risposta attesa 'item_X', quindi rank = 1, RR = 1.0
    ]
    gt_items = ["item_B", "item_X"]
    
    print(f"MRR Test: {mean_reciprocal_rank(example_retrieved, gt_items)}") # (0.5 + 1.0) / 2 = 0.75
    
    # NDCG assume di dare punteggi in [0 = non pertinente, 3 = molto pertinente]
    # Se il rag propone [Non Pertinente, Perfetto, CosiCosi]
    rag_scores = [[0, 3, 1]] 
    print(f"NDCG@3 Test: {ndcg_at_k(rag_scores, k=3):.3f}")
