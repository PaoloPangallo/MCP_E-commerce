import math
from typing import List


def precision_at_k(relevant: List[int], k: int):

    if k == 0:
        return 0

    relevant_k = relevant[:k]

    return sum(relevant_k) / k


def recall_at_k(relevant: List[int], total_relevant: int, k: int):

    if total_relevant == 0:
        return 0

    relevant_k = relevant[:k]

    return sum(relevant_k) / total_relevant


def dcg_at_k(scores: List[int], k: int):

    dcg = 0

    for i, rel in enumerate(scores[:k]):

        dcg += rel / math.log2(i + 2)

    return dcg


def ndcg_at_k(scores: List[int], k: int):

    dcg = dcg_at_k(scores, k)

    ideal = sorted(scores, reverse=True)

    idcg = dcg_at_k(ideal, k)

    if idcg == 0:
        return 0

    return dcg / idcg