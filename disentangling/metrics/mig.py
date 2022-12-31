import sklearn.metrics
import torch
import numpy as np


def histogram(xs, bins):
    # https://github.com/pytorch/pytorch/issues/69519
    # Like torch.histogram, but works with cuda
    min, max = xs.min().item(), xs.max().item()
    # counts = torch.histc(xs, bins, min=min, max=max).to(xs.device)
    boundaries = torch.linspace(min, max, bins + 1).to(xs.device)
    return boundaries


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
    # TODO: why -1
    discretize = lambda a: torch.tensor(np_discretize(a.cpu().numpy(), n_bins))
    # discretize = lambda a: torch.bucketize(a, histogram(a, n_bins)[:-1])
    normalized_x = discretize(x) if discretize_x else x
    normalized_y = discretize(y) if discretize_y else y
    return sklearn.metrics.mutual_info_score(
        normalized_x.detach().cpu(), normalized_y.detach().cpu()
    )


def calc_mutual_infos(codes, factors):
    n_codes = codes.shape[1]
    n_factors = factors.shape[1]
    m = np.zeros((n_codes, n_factors))  # torch.zeros.to(codes.device)
    for i in range(n_codes):
        for j in range(n_factors):
            m[i, j] = calc_mutual_info(codes[:, i], factors[:, j])
    return torch.from_numpy(m)


def calc_entropy(factors, discretize=False):
    n_factors = factors.shape[1]
    # h = torch.zeros(n_factors).to(factors.device)
    h = np.zeros((n_factors,))
    for i in range(n_factors):
        h[i] = calc_mutual_info(
            factors[:, i],
            factors[:, i],
            discretize_x=discretize,
            discretize_y=discretize,
        )
    return torch.from_numpy(h)


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
