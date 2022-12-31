import torch
from sklearn.decomposition import PCA
import numpy as np

"""
representation_functions
m1: a code capture no more than one factor
c1: a factor is captured by no more than one code
"""


def m0c0i0(factors):
    latents = generate_factors(factors.shape[0], factors.shape[1])
    return latents


def m0c0i1(factors):
    # TODO
    pass


def m0c1i0(factors):
    # m0: BUT a code capture more than one factor
    # c1: a factor is captured by no more than one code
    # by duplicated factors + modify code to loss info
    n_factors_per_latent = 2
    n_factors_half = factors.shape[1] // n_factors_per_latent
    original = factors[:, :n_factors_half]
    factors[:, n_factors_half:] = original
    latents = torch.clone(original)
    means = latents.float().mean(0)
    latents[latents < means] = -1
    return latents


def m0c1i1(factors):
    # m0: BUT a code capture more than one factor
    # c1: a factor is captured by no more than one code
    # by duplicated factors
    n_factors_per_latent = 2
    n_factors_half = factors.shape[1] // n_factors_per_latent
    original = factors[:, :n_factors_half]
    factors[:, n_factors_half:] = original
    latents = original
    return latents


def m1c0i0(factors):
    # m1: a code capture no more than one factor
    # c0: BUT a factor is captured by more than one code
    # by duplicated codes + modify code to loss info
    x = factors
    latents = torch.hstack([x, x])
    means = latents.float().mean(0)
    latents[latents < means] = -1
    return latents


def m1c0i1(factors):
    # m1: a code capture no more than one factor
    # c0: BUT a factor is captured by more than one code
    # by duplicated codes
    x = factors
    latents = torch.hstack([x, x])
    return latents


def m1c1i0(factors):
    latents = torch.clone(factors)
    means = latents.float().mean(0)
    latents[latents < means] = -1
    return latents


def m1c1i1(factors):
    latents = torch.clone(factors)
    return latents


def m0c1_PCA(factors):
    batch_size, n_factors = factors.shape
    n_factors_per_latent = 2
    n_latents = n_factors // n_factors_per_latent
    latents = torch.zeros(
        (batch_size, n_latents), dtype=factors.dtype, device=factors.device
    )
    for i in range(n_latents):
        target_factors = factors[:, i * 2 : (i + 1) * 2].cpu()
        # print(a.shape, latents[:, i].shape)
        latents[:, i] = torch.from_numpy(
            PCA(n_components=1).fit_transform(target_factors)
        ).reshape(-1)
    return latents


def generate_factors(batch_size, n_factor_dims, mu=0, sigma=10):
    size = (batch_size, n_factor_dims)
    factors = np.random.normal(mu, sigma, size).astype(int)
    return torch.from_numpy(factors)


def run_metric(metric, representation_function, batch_size=12000, n_factors=8):
    factors = generate_factors(batch_size, n_factor_dims=4)
    latents = representation_function(factors)
    score = metric(factors, latents)
    if torch.is_tensor(score):
        return score.cpu()
    elif type(score) is dict and torch.is_tensor(next(iter(score.values()))):
        return {k: score[k].cpu() for k in score}
    return score
