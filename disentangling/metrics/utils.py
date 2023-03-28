import numpy as np
from sklearn.decomposition import PCA


"""
representation_functions
m1: a code capture no more than one factor
c1: a factor is captured by no more than one code
"""


def m0c0i0(factors): # n_factors = 2, n_codes = 2
    latents = generate_factors(factors.shape[0], factors.shape[1])
    return latents


def m0c0i1(factors): # n_factors = 2, n_codes = 2
    batch_size, n_factors = factors.shape
    n_latents = n_factors
    train_size = batch_size // 2
    train_factors = factors[:train_size]
    test_factors = factors[train_size:]
    model = PCA()
    model.fit(train_factors)
    latents = model.transform(test_factors)
    return latents


def m0c1i0(factors):  # n_factors = 4, n_codes = 2
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


def m0c1i1(factors): # n_factors = 4, n_codes = 2
    # m0: BUT a code capture more than one factor
    # c1: a factor is captured by no more than one code
    # by duplicated factors
    n_factors_per_latent = 2
    n_factors_half = factors.shape[1] // n_factors_per_latent
    original = factors[:, :n_factors_half]
    factors[:, n_factors_half:] = original
    latents = original
    return latents


def m1c0i0(factors): # n_factors = 2, n_codes = 4
    # m1: a code capture no more than one factor
    # c0: BUT a factor is captured by more than one code
    # by duplicated codes + modify code to loss info
    x = factors
    latents = np.hstack([x, x])
    means = latents.mean(0)
    latents[latents < means] = -1
    return latents


def m1c0i1(factors): # n_factors = 2, n_codes = 4
    # m1: a code capture no more than one factor
    # c0: BUT a factor is captured by more than one code
    # by duplicated codes
    x = factors
    latents = np.hstack([x, x])
    return latents


def m1c1i0(factors): # n_factors = 2, n_codes = 2
    latents = np.copy(factors)
    means = latents.mean(0)
    latents[latents < means] = -1
    return latents


def m1c1i1(factors): # n_factors = 2, n_codes = 2
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


def run_metric(metric, representation_function, batch_size=50000):
    n_factors = 2
    if representation_function == m0c0i1:
        # generate double size of factors
        # in representation_function: half for train, half for get codes.
        factors = generate_factors(batch_size * 2, n_factors)
        latents = representation_function(factors)
        score = metric(factors[batch_size:], latents)
    else:
        if representation_function in [m0c1i0, m0c1i1]:
            n_factors = 4
        factors = generate_factors(batch_size, n_factors)
        latents = representation_function(factors)
        score = metric(factors, latents)
    return score
