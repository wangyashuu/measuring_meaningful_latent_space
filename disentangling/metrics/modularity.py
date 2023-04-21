import numpy as np
from .mi import get_mutual_infos


"""
Modularity metric
Based on "Learning Deep Disentangled Embeddings With the F-Statistic Loss"

Adapted from: https://github.com/google-research/disentanglement_lib
"""


def modularity(
    factors, codes, estimator="ksg", discrete_factors=False, **kwargs
):
    """Computes the modularity from mutual information."""
    # Mutual information has shape [num_codes, num_factors].
    mutual_infos = get_mutual_infos(
        codes, factors, discrete_factors=discrete_factors, estimator=estimator
    )
    squared_mi = mutual_infos**2
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] - 1.0)
    delta = numerator / denominator
    modularity_score = 1.0 - delta
    index = max_squared_mi == 0.0
    modularity_score[index] = 0.0
    return np.mean(modularity_score)
