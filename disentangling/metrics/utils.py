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
    latents = np.copy(original)
    means = latents.mean(0)
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
    latents = np.hstack([x, x])
    means = latents.mean(0)
    latents[latents < means] = -1
    return latents


def m1c0i1(factors):
    # m1: a code capture no more than one factor
    # c0: BUT a factor is captured by more than one code
    # by duplicated codes
    x = factors
    latents = np.hstack([x, x])
    return latents


def m1c1i0(factors):
    latents = np.copy(factors)
    means = latents.mean(0)
    latents[latents < means] = -1
    return latents


def m1c1i1(factors):
    latents = np.copy(factors)
    return latents


def m0c1_PCA(factors):
    batch_size, n_factors = factors.shape
    n_factors_per_latent = 2
    n_latents = n_factors // n_factors_per_latent
    latents = np.zeros((batch_size, n_latents))
    for i in range(n_latents):
        target_factors = factors[:, i * 2 : (i + 1) * 2]
        # print(a.shape, latents[:, i].shape)
        latents[:, i] = (
            PCA(n_components=1).fit_transform(target_factors).reshape(-1)
        )
    return latents


def generate_factors(batch_size, n_factors=4, mu=0, sigma=10):
    shape = (batch_size, n_factors)
    factors = np.random.normal(mu, sigma, shape).astype(int)
    return factors


def run_metric(metric, representation_function, batch_size=12000, n_factors=4):
    factors = generate_factors(batch_size, n_factors)
    latents = representation_function(factors)
    score = metric(factors, latents)
    return score
