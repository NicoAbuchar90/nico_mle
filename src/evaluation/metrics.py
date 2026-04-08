import numpy as np


def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    recommended_at_k = recommended[:k]
    hits = len(set(recommended_at_k) & set(relevant))
    return hits / k


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    """Fraction of relevant items found in top-K recommendations."""
    if not relevant:
        return 0.0
    recommended_at_k = recommended[:k]
    hits = len(set(recommended_at_k) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: list, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    recommended_at_k = recommended[:k]
    dcg = sum(
        1 / np.log2(i + 2)
        for i, item in enumerate(recommended_at_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
