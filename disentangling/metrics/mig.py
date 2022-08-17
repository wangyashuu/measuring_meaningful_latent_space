# https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/mig.py
# https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics/blob/main/src/metrics/mig.py
# https://github.com/rtqichen/beta-tcvae/blob/master/metric_helpers/mi_metric.py

import sklearn
import numpy as np


def calc_mutual_info(x, y, n_bins=20):
    """
    more info about calc mutual info:
    1. calculate mutual info between float
    - https://www.researchgate.net/post/Can_anyone_help_with_calculating_the_mutual_information_between_for_float_number,
    - https://bitbucket.org/szzoli/ite/src/master/
    2. implementation
    - https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    """
    discretize = lambda a: np.digitize(a, np.histogram(a, n_bins)[1][:-1])
    # x_bin_edges = np.histogram(x, n_bins)[1]
    # x_discretized =
    # y_bin_edges = np.histogram(y, n_bins)[1]
    return sklearn.metrics.mutual_info_score(disretize(x), discretize(y))


def calc_mutual_infos(codes, factors):
    n_codes = codes.shape[1]
    n_factors = factors.shape[1]
    m = np.zeros((n_codes, n_factors))
    for i in range(n_codes):
        for j in range(n_factors):
            m[i, j] = calc_mutual_info(codes[:, i], factors[:, j])
    return m


def calc_entropy(factors):
    n_factors = factors.shape[1]
    h = np.zeros(n_factors)
    for i in range(n_factors):
        h[i] = calc_mutual_info(factors[:, i], factors[:, i])


def mig(factors, codes):
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
    sorted = np.sort(mutual_infos, axis=0)[::-1]
    entropy = calc_entropy(factors)
    score = np.mean((sorted[0, :] - sorted[1, :]) / entropy)
    return score
