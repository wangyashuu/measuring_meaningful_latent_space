# https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/mig.py
# https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics/blob/main/src/metrics/mig.py
# https://github.com/rtqichen/beta-tcvae/blob/master/metric_helpers/mi_metric.py

import sklearn.metrics
import torch


def histogram(xs, bins):
    # https://github.com/pytorch/pytorch/issues/69519
    # Like torch.histogram, but works with cuda
    min, max = xs.min().item(), xs.max().item()
    # counts = torch.histc(xs, bins, min=min, max=max).to(xs.device)
    boundaries = torch.linspace(min, max, bins + 1).to(xs.device)
    return boundaries


def calc_mutual_info(x, y, n_bins=20):
    """
    more info about calc mutual info:
    1. calculate mutual info between float
    - https://www.researchgate.net/post/Can_anyone_help_with_calculating_the_mutual_information_between_for_float_number,
    - https://bitbucket.org/szzoli/ite/src/master/
    2. implementation
    - https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    """
    # TODO: why -1
    discretize = lambda a: torch.bucketize(a, histogram(a, n_bins)[:-1])
    return sklearn.metrics.mutual_info_score(
        discretize(x).cpu(), discretize(y).cpu()
    )


def calc_mutual_infos(codes, factors):
    n_codes = codes.shape[1]
    n_factors = factors.shape[1]
    m = torch.zeros(n_codes, n_factors).to(codes.device)
    for i in range(n_codes):
        for j in range(n_factors):
            m[i, j] = calc_mutual_info(codes[:, i], factors[:, j])
    return m


def calc_entropy(factors):
    n_factors = factors.shape[1]
    h = torch.zeros(n_factors).to(factors.device)
    for i in range(n_factors):
        h[i] = calc_mutual_info(factors[:, i], factors[:, i])
    return h


def mig(factors, codes, epsilon=10e-8):
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
    # ::-1 reverse not support: https://github.com/pytorch/pytorch/issues/59786
    # torch.searchsorted(): input value tensor is non-contiguous https://github.com/pytorch/pytorch/issues/52743
    # https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
    sorted = torch.sort(mutual_infos, dim=0, descending=True)[0]
    entropy = calc_entropy(factors)
    score = torch.mean((sorted[0, :] - sorted[1, :]) / (entropy + epsilon))
    return score