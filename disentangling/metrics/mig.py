import sklearn.metrics
import numpy as np


def np_discretize(xs, n_bins=30):
    """Discretization based on histograms."""
    discretized = np.digitize(xs, np.histogram(xs, n_bins)[1][:-1])
    return discretized


def calc_mutual_info(x, y, n_bins=50, discretize_x=True, discretize_y=False):
    """
    more info about calc mutual info:
    1. calculate mutual info between float
    - https://www.researchgate.net/post/Can_anyone_help_with_calculating_the_mutual_information_between_for_float_number,
    - https://bitbucket.org/szzoli/ite/src/master/
    2. implementation
    - https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    """
    normalized_x = np_discretize(x, n_bins) if discretize_x else x
    normalized_y = np_discretize(y, n_bins) if discretize_y else y
    return sklearn.metrics.mutual_info_score(normalized_x, normalized_y)


def calc_mutual_infos(codes, factors):
    n_codes = codes.shape[1]
    n_factors = factors.shape[1]
    m = np.zeros((n_codes, n_factors))
    for i in range(n_codes):
        for j in range(n_factors):
            m[i, j] = calc_mutual_info(codes[:, i], factors[:, j])
    return m


def calc_entropy(factors, discretize=False):
    n_factors = factors.shape[1]
    h = np.zeros((n_factors,))
    for i in range(n_factors):
        h[i] = calc_mutual_info(
            factors[:, i],
            factors[:, i],
            discretize_x=discretize,
            discretize_y=discretize,
        )
    return h


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
    mutual_infos = calc_mutual_infos(codes, factors)
    # sort mi for each factor
    sorted = np.sort(mutual_infos, axis=0)[::-1, :]
    entropy = calc_entropy(factors)
    score = np.mean((sorted[0, :] - sorted[1, :]) / (entropy + epsilon))
    return score
