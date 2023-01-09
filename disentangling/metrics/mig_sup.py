import numpy as np

from .mig import calc_entropy, calc_mutual_infos


def mig_sup(factors, codes, epsilon=1e-10):
    """
    Compute MIG-sup

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    mutual_infos = calc_mutual_infos(codes, factors)
    # sort mi for each codes
    sorted = np.sort(mutual_infos, axis=1)[:, ::-1]
    entropy = calc_entropy(codes)
    score = np.mean((sorted[:, 0] - sorted[:, 1]) / (entropy + epsilon))
    return score
