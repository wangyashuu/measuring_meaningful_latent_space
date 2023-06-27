"""MIG supplement (MIG-sup) from `Progressive Learning and Disentanglement of Hierarchical Representations <https://arxiv.org/abs/2002.10549>`."""
from typing import Union, List
import numpy as np

from disentangling.utils.mi import get_mutual_infos, get_entropies


def mig_sup(
    factors: np.ndarray,
    codes: np.ndarray,
    estimator: str = "ksg",
    discrete_factors: Union[List[bool], bool] = False,
    epsilon: float = 1e-10,
    **kwargs
) -> float:
    """Compute MIG-sup score

    Args:
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        estimator (str, optional): String in ["ksg", "bins", "mine"], each represents different method to estimate mutual information, see more in `disentangling.utils.mi`. Default: "ksg".
        discrete_factors (Union[List[bool], bool]): It implies if each factor is discrete. Default: True.
        epsilon (float, optional): Epsilon (the very small number) used in calculation. Default: 1e-10.

    Returns:
        score (float): the overall MIG-sup score
    """
    # mutual_info matrix (n_codes, n_factors)
    mutual_infos = get_mutual_infos(
        codes, factors, discrete_factors=discrete_factors, estimator=estimator
    )
    # sort mi for each codes
    sorted = np.sort(mutual_infos, axis=1)[:, ::-1]
    entropies = get_entropies(codes, discrete=False, estimator=estimator)
    score = np.mean((sorted[:, 0] - sorted[:, 1]) / (entropies + epsilon))
    return score
