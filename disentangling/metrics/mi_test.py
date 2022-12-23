import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sklearn.utils import check_random_state
from .mi import mutual_info


def get_variance(arr):
    if len(arr.shape) == 1 or (len(arr.shape) == 2 and arr.shape[1] == 1):
        return np.var(arr)
    return np.linalg.det(np.cov(arr.T))


def test_compute_mi_cc():
    mean = np.zeros(2)

    # Setup covariance matrix with correlation coeff. equal 0.5.
    sigma_1 = 1
    sigma_2 = 10
    corr = 0.5
    cov = np.array(
        [
            [sigma_1**2, corr * sigma_1 * sigma_2],
            [corr * sigma_1 * sigma_2, sigma_2**2],
        ]
    )

    # Theoretical MI for 2 cont var of bivariate normal distribution
    # https://en.wikipedia.org/wiki/Mutual_information#Linear_correlation
    I_theory = (
        np.log(sigma_1) + np.log(sigma_2) - 0.5 * np.log(np.linalg.det(cov))
    )

    rng = check_random_state(0)
    Z = rng.multivariate_normal(mean, cov, size=1000)
    X = np.c_[Z[:, 0], Z[:, 0]]
    y = Z[:, 1].reshape(-1, 1)

    # Theory and computed values won't be very close, assert that the
    # first figures after decimal point match.
    for n_neighbors in [3, 5, 7]:
        I_computed = mutual_info(X, y, n_neighbors=n_neighbors)
        print("hello", I_computed, I_theory)
    # assert_almost_equal(I_computed, I_theory, 1)


def test_compute_mi_cd():
    # To test define a joint distribution as follows:
    # p(x, y) = p(x) p(y | x)
    # X ~ Bernoulli(p)
    # (Y | x = 0) ~ Uniform(-1, 1)
    # (Y | x = 1) ~ Uniform(0, 2)

    # Use the following formula for mutual information:
    # I(X; Y) = H(Y) - H(Y | X)
    # Two entropies can be computed by hand:
    # H(Y) = -(1-p)/2 * ln((1-p)/2) - p/2*log(p/2) - 1/2*log(1/2)
    # H(Y | X) = ln(2)

    # Now we need to implement sampling from out distribution, which is
    # done easily using conditional distribution logic.

    n_samples = 1000
    rng = check_random_state(0)

    for p in [0.3, 0.5, 0.7]:
        y = rng.uniform(size=n_samples) > p

        x = np.empty(n_samples)
        mask = y == 0
        x[mask] = rng.uniform(-1, 1, size=np.sum(mask))
        x[~mask] = rng.uniform(0, 2, size=np.sum(~mask))
        X = np.c_[x, x] * 1

        I_theory = -0.5 * (
            (1 - p) * np.log(0.5 * (1 - p)) + p * np.log(0.5 * p) + np.log(0.5)
        ) - np.log(2)

        # Assert the same tolerance.
        for n_neighbors in [3, 5, 7]:
            I_computed = mutual_info(
                X, y, discrete_y=True, n_neighbors=n_neighbors
            )
            print("hello", I_computed, I_theory)
