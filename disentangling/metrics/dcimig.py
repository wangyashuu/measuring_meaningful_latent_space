import numpy as np

from .mi import get_mutual_infos, get_entropies

"""
Implementation of DCIMIG.
Based on "How to Not Measure Disentanglement"
"""


def dcimig(factors, codes, estimator='ksg', continuous_factors=True, n_bins=10):
    """
    Computes the dcimig

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
        continuous_factors: if the factors are continous
        n_bins: the number of bins use for discretization
    Returns:
        score
    """

    n_factors, n_codes = factors.shape[1], codes.shape[1]
    # shape: n_codes, n_factors
    mutual_infos = get_mutual_infos(codes, factors, estimator=estimator)

    mutual_infos_normalized = np.zeros((n_codes, n_factors))
    for c in range(n_codes):
        sorted = np.sort(mutual_infos[c, :]) # sort by each factor c
        max_idx = np.argmax(mutual_infos[c, :])
        gap = sorted[-1] - sorted[-2]
        mutual_infos_normalized[c, max_idx] = gap

    gap_sum = 0
    for f in range(n_factors):
        gap_sum += np.max(mutual_infos_normalized[:, f])

    factor_entropy = np.sum(get_entropies(factors))
    dcimig_score = gap_sum / factor_entropy
    return dcimig_score
