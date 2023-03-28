import sklearn.metrics
import torch
import numpy as np

from .mi import get_mutual_infos, get_entropies, get_captured_mis


def dcii_d(factors, codes, epsilon=10e-8, mi_improved=True):
    """
    Compute disentanglement

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    estimator = "ksg" if mi_improved else "bins"
    mutual_infos = get_mutual_infos(
        codes, factors, estimator=estimator, normalized=True
    )

    # get scores for each code
    mi_max = np.max(mutual_infos, axis=1, keepdims=True)
    errorness = np.sqrt(
        (1.0 / (factors.shape[1] - 1))
        * (np.sum(mutual_infos**2, axis=1, keepdims=True) - mi_max**2)
    )
    correctness = mi_max
    scores = correctness - errorness
    score = np.mean(scores)
    return score


def dcii_c(factors, codes, epsilon=10e-8, mi_improved=True):
    """
    Compute completeness

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    estimator = "ksg" if mi_improved else "bins"
    mutual_infos = get_mutual_infos(
        codes, factors, estimator=estimator, normalized=True
    )

    # get scores for each factor
    mi_max = np.max(mutual_infos, axis=0, keepdims=True)
    errorness = np.sqrt(
        (1.0 / (codes.shape[1] - 1))
        * (np.sum(mutual_infos**2, axis=0, keepdims=True) - mi_max**2)
    )
    correctness = mi_max
    scores = correctness - errorness
    score = np.mean(scores)
    return score


def dcii_i(factors, codes, epsilon=10e-8, mi_improved=True):
    """
    Compute informativeness

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # discrete_factors, discrete_codes?
    # mutual_info matrix (n_codes, n_factors)
    estimator = "ksg" if mi_improved else "bins"
    captured = (
        get_captured_mis(codes, factors, estimator="ksg")
        if mi_improved
        else np.max(
            get_mutual_infos(codes, factors, estimator=estimator),
            axis=0,
        )
    )
    entropies = get_entropies(factors, estimator=estimator)
    captured[captured > entropies] = entropies[captured > entropies]
    score = np.mean(captured / (entropies + epsilon))
    return score
