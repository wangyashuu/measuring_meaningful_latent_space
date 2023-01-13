import numpy as np
from .mi import get_mutual_infos, get_entropies

"""Mutual Information Gap from the beta-TC-VAE paper.
Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""

def mig(factors, codes, epsilon=1e-8):
    """
    Compute MIG

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    mutual_infos = get_mutual_infos(codes, factors)
    # sort mi for each factor
    sorted = np.sort(mutual_infos, axis=0)[::-1, :]
    entropies = get_entropies(factors)
    score = np.mean((sorted[0, :] - sorted[1, :]) / (entropies + epsilon))
    return score
