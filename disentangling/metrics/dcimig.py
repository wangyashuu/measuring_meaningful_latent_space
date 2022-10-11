import torch
from .mig import calc_mutual_infos, calc_entropy

"""
Implementation of DCIMIG.
Based on "How to Not Measure Disentanglement"
"""

def dcimig(factors, codes, continuous_factors=True, nb_bins=10):
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
    mutual_infos = calc_mutual_infos(codes, factors) # n_codes, n_factors

    mutual_infos_normalized = torch.zeros(n_codes, n_factors)
    for c in range(n_codes):
        sorted = torch.sort(mutual_infos[c, :])[0] # sort by each factor c
        max_idx = torch.argmax(mutual_infos[c, :])
        gap = sorted[-1] - sorted[-2]
        mutual_infos_normalized[c, max_idx] = gap

    gap_sum = 0
    for f in range(n_factors):
        gap_sum += torch.max(mutual_infos_normalized[:, f])

    factor_entropy = torch.sum(calc_entropy(factors))
    dcimig_score = gap_sum / factor_entropy
    return dcimig_score