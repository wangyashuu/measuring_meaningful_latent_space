import sklearn.metrics
import torch
import numpy as np
from .mig import calc_mutual_infos, calc_entropy

from .modularity import modularity


def dcii_d(factors, codes, epsilon=10e-8):
    """
    Compute disentanglement

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    mutual_infos = calc_mutual_infos(codes, factors).cpu().numpy()
    mutual_infos_normed = mutual_infos / (
        np.sum(mutual_infos, axis=1, keepdims=True) + epsilon
    )
    mi_max = np.max(mutual_infos_normed, axis=1, keepdims=True)
    # score for each code
    scores = (1.0 / (factors.shape[1] - 1)) * np.sum(
        mi_max**2 - mutual_infos_normed**2, axis=1
    )
    score = np.mean(scores)
    modul = modularity(factors, codes)
    return score


def dcii_c(factors, codes, epsilon=10e-8):
    """
    Compute completeness

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    mutual_infos = calc_mutual_infos(codes, factors).cpu().numpy()
    mutual_infos_normed = mutual_infos / (
        np.sum(mutual_infos, axis=0, keepdims=True) + epsilon
    )
    mi_max = np.max(mutual_infos_normed, axis=0, keepdims=True)
    # scores for each factor
    scores = (1.0 / (codes.shape[1] - 1)) * np.sum(
        mi_max**2 - mutual_infos_normed**2, axis=0
    )
    score = np.mean(scores)

    return score


def dcii_i(factors, codes, epsilon=10e-8):
    """
    Compute informativeness

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    mutual_infos = calc_mutual_infos(codes, factors).cpu().numpy()
    mi_max = np.max(mutual_infos, axis=0)
    entropy = calc_entropy(factors, discretize=True).cpu().numpy()
    score = np.mean(mi_max / (entropy + epsilon))
    return score
