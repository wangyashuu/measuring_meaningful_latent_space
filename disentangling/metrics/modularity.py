"""Modularity metric from `Learning Deep Disentangled Embeddings With the F-Statistic Loss <https://arxiv.org/abs/1802.05312>`.

Part of code is adapted from `disentanglement lib <https://github.com/google-research/disentanglement_lib>`.
"""
from typing import Union, List
import numpy as np

from disentangling.utils.mi import get_mutual_infos


def modularity_scores_from_mutual_infos(mutual_infos: np.ndarray) -> np.ndarray:
    """Computes the modularity scores from mutual information.

    Adapted from `disentanglement lib <https://github.com/google-research/disentanglement_lib>`

    Args:
        mutual_infos (np.ndarray): [Shape (n_codes, n_factors)] mutual information matrix where ij entry represents mutual information between code i and factor j

    Returns:
        scores (np.ndarray): [Shape (n_codes, )] a list where each represents modularity score per code.
    """
    squared_mi = mutual_infos**2
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] - 1.0)
    delta = numerator / denominator
    modularity_score = 1.0 - delta
    index = max_squared_mi == 0.0
    modularity_score[index] = 0.0
    return modularity_score


def modularity(
    factors: np.ndarray,
    codes: np.ndarray,
    estimator: str = "ksg",
    discrete_factors: Union[List[bool], bool] = False,
    **kwargs
) -> float:
    """Compute modularity score of given factors and codes.

    Args:
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        estimator (str, optional): String in ["ksg", "bins", "mine"], each represents different method to estimate mutual information, see more in `disentangling.utils.mi`. Default: "ksg".
        discrete_factors (Union[List[bool], bool]): It implies if each factor is discrete. Default: True.

    Returns:
        score (float): an overall modularity score.
    """

    mutual_infos = get_mutual_infos(
        codes, factors, discrete_factors=discrete_factors, estimator=estimator
    )
    modularity_scores = modularity_scores_from_mutual_infos(mutual_infos)
    return np.mean(modularity_scores)
