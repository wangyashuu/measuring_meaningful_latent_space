# https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/mig.py
# https://github.com/ubisoft/ubisoft-laforge-disentanglement-metrics/blob/main/src/metrics/mig.py
# https://github.com/rtqichen/beta-tcvae/blob/master/metric_helpers/mi_metric.py

import torch

from .mig import *

def mig_sup(factors, codes, epsilon=10e-8):
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
    sorted = torch.sort(mutual_infos, dim=1, descending=True)[0]
    entropy = calc_entropy(codes)
    score = torch.mean((sorted[:, 0] - sorted[:, 1]) / (entropy + epsilon))
    return score
