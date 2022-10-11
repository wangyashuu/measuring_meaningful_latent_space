import torch
from .mig import calc_mutual_infos


"""
Modularity metric
Based on "Learning Deep Disentangled Embeddings With the F-Statistic Loss"

Adapted from: https://github.com/google-research/disentanglement_lib
"""

def modularity(factors, codes):
    """Computes the modularity from mutual information."""
    # Mutual information has shape [num_codes, num_factors].
    mutual_infos = calc_mutual_infos(codes, factors)
    squared_mi = torch.square(mutual_infos)
    max_squared_mi = torch.max(squared_mi, axis=1)[0]
    numerator = torch.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] - 1.0)
    delta = numerator / denominator
    modularity_score = 1.0 - delta
    index = max_squared_mi == 0.0
    modularity_score[index] = 0.0
    return torch.mean(modularity_score)
