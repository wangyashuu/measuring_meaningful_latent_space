import numpy as np

from .mi import get_mutual_infos, get_entropies


def mig_sup(
    factors,
    codes,
    estimator="ksg",
    epsilon=1e-10,
    discrete_factors=False,
    **kwargs
):
    """
    Compute MIG-sup

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    mutual_infos = get_mutual_infos(
        codes, factors, discrete_factors=discrete_factors, estimator=estimator
    )
    # sort mi for each codes
    sorted = np.sort(mutual_infos, axis=1)[:, ::-1]
    entropies = get_entropies(codes)
    score = np.mean((sorted[:, 0] - sorted[:, 1]) / (entropies + epsilon))
    return score
