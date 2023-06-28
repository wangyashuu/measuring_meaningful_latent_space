"""3CharM, aka DCIMIG, from `How to Not Measure Disentanglement <https://arxiv.org/abs/1910.05587>`.

Part of code is adapted from `Supervised Disentanglement Metrics <https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics>`.
"""
from typing import Union, List
import numpy as np

from disentangling.utils.mi import get_mutual_infos, get_entropies


def _dcimig_from_mutual_infos(mutual_infos: np.ndarray) -> np.ndarray:
    """Computes the DCIMIG scores from mutual information.

    Part of code is adapted from `Supervised Disentanglement Metrics <https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics>`.

    Args:
        mutual_infos (np.ndarray): [Shape (n_codes, n_factors)] mutual information matrix where ij entry represents mutual information between code i and factor j

    Returns:
        scores (np.ndarray): [Shape (n_factors, )] a list where each represents DCIMIG score per factor?.
    """

    n_codes, n_factors = mutual_infos.shape
    mutual_infos_normalized = np.zeros((n_codes, n_factors))
    for c in range(n_codes):
        sorted = np.sort(mutual_infos[c, :])  # sort by each factor c
        max_idx = np.argmax(mutual_infos[c, :])
        gap = sorted[-1] - sorted[-2]
        mutual_infos_normalized[c, max_idx] = gap

    scores = np.max(mutual_infos_normalized, axis=0)
    return scores


def dcimig(
    factors: np.ndarray,
    codes: np.ndarray,
    estimator: str = "ksg",
    discrete_factors: Union[List[bool], bool] = False,
    epsilon: float = 1e-10,
    **kwargs
) -> float:
    """Compute DCIMIG score

    Args:
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        estimator (str, optional): String in ["ksg", "bins", "mine"], each represents different method to estimate mutual information, see more in `disentangling.utils.mi`. Default: "ksg".
        discrete_factors (Union[List[bool], bool]): It implies if each factor is discrete. Default: True.
        epsilon (float, optional): Epsilon (the very small number) used in calculation. Default: 1e-10.

    Returns:
        score (float): the overall MIG-sup score
    """
    mutual_infos = get_mutual_infos(
        codes, factors, estimator=estimator, discrete_factors=discrete_factors
    )
    gaps = _dcimig_from_mutual_infos(mutual_infos)

    factor_entropy = np.sum(get_entropies(factors, discrete=discrete_factors))
    dcimig_score = np.sum(gaps) / (factor_entropy + epsilon)
    return dcimig_score
