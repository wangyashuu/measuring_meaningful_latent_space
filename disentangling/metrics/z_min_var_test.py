import math
import numpy as np

from .z_min_var import z_min_var
from .utils import generate_factors

n_factors = 4


def test_m0c0i0():
    def sample_factors(batch_size):
        return generate_factors(batch_size, n_factors).cpu().numpy()

    def factor2code(factors):
        return generate_factors(factors.shape[0], n_factors).cpu().numpy()

    score = z_min_var(
        n_votes=10000,
        batch_size=128,
        sample_factors=sample_factors,
        factor2code=factor2code,
        n_global_items=30000,
    )
    # TODO: analysis
    assert math.isclose(score, 1 / n_factors, abs_tol=0.2)  # 0 vs 1/n_factors


def test_m0c0i1():
    # TODO
    pass
    # def sample_factors(batch_size):
    #     return generate_factors(batch_size, n_factor_dims=2).cpu().numpy()

    # def factor2code(factors):
    #     return (
    #         generate_factors(factors.shape[0], n_factor_dims=2).cpu().numpy()
    #     )

    # score = z_min_var(
    #     n_votes=10000,
    #     batch_size=128,
    #     sample_factors=sample_factors,
    #     factor2code=factor2code,
    #     n_global_items=30000,
    # )
    # assert math.isclose(score, 1, abs_tol=0.2)


def test_m0c1i0():
    n_factors = 4

    def sample_factors(batch_size):
        factors = generate_factors(batch_size, n_factors).cpu().numpy()
        return np.hstack([factors, factors])

    n_factors_per_latent = 2
    def factor2code(factors):
        n_factors_half = factors.shape[1] // n_factors_per_latent
        latents = (
            factors[:, :n_factors_half] + factors[:, n_factors_half:]
        ) / n_factors_per_latent
        latents[latents < 0] = -1
        return latents

    score = z_min_var(
        n_votes=10000,
        batch_size=128,
        sample_factors=sample_factors,
        factor2code=factor2code,
        n_global_items=30000,
    )
    # TODO
    assert math.isclose(score, 1/n_factors_per_latent, abs_tol=0.2)  # 0 vs 0.4997


def test_m0c1i1():
    def sample_factors(batch_size):
        factors = generate_factors(batch_size, 2).cpu().numpy()
        return np.hstack([factors, factors])

    n_factors_per_latent = 2
    def factor2code(factors):
        n_factors_half = factors.shape[1] // n_factors_per_latent
        latents = (
            factors[:, :n_factors_half] + factors[:, n_factors_half:]
        ) / n_factors_per_latent
        return latents

    score = z_min_var(
        n_votes=10000,
        batch_size=128,
        sample_factors=sample_factors,
        factor2code=factor2code,
        n_global_items=30000,
    )
    assert math.isclose(score, 1/n_factors_per_latent, abs_tol=0.2)  # 0 vs 0.5068


def test_m1c0i0():
    def sample_factors(batch_size):
        factors = generate_factors(batch_size, n_factors).cpu().numpy()
        return factors

    def factor2code(factors):
        x = factors
        latents = np.hstack([x, x])
        latents[latents < 0] = -1
        return latents

    score = z_min_var(
        n_votes=10000,
        batch_size=128,
        sample_factors=sample_factors,
        factor2code=factor2code,
        n_global_items=30000,
    )
    assert math.isclose(score, 1, abs_tol=0.2)  # 1


def test_m1c0i1():
    def sample_factors(batch_size):
        factors = generate_factors(batch_size, n_factors).cpu().numpy()
        return factors

    def factor2code(factors):
        x = factors
        return np.hstack([x, x])

    score = z_min_var(
        n_votes=10000,
        batch_size=128,
        sample_factors=sample_factors,
        factor2code=factor2code,
        n_global_items=30000,
    )
    assert math.isclose(score, 1, abs_tol=0.2)  # 1


def test_m1c1i0():
    def sample_factors(batch_size):
        return generate_factors(batch_size, n_factors).cpu().numpy()

    def factor2code(factors):
        latents = np.copy(factors)
        latents[latents < 0] = -1
        return latents

    score = z_min_var(
        n_votes=10000,
        batch_size=128,
        sample_factors=sample_factors,
        factor2code=factor2code,
        n_global_items=30000,
    )
    assert math.isclose(score, 1, abs_tol=0.2)  # 1 vs 0.4968


def test_m1c1i1():
    def sample_factors(batch_size):
        return generate_factors(batch_size, n_factor_dims=2).cpu().numpy()

    def factor2code(factors):
        return np.copy(factors)

    score = z_min_var(
        n_votes=10000,
        batch_size=128,
        sample_factors=sample_factors,
        factor2code=factor2code,
        n_global_items=30000,
    )
    assert math.isclose(score, 1, abs_tol=0.2)  # 1
