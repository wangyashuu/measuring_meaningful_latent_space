import math
import numpy as np

from sklearn.utils import check_random_state
from .mi import get_mutual_info_by_ksg, get_entropy


def test_entropy():
    n_samples = 1000000
    n_dims = 1

    x = (np.random.randn(n_samples, n_dims) * 10).astype(int)
    x_probs = np.unique(x, return_counts=True)[1] / n_samples
    x_h_real = -np.sum(x_probs * np.log(x_probs))
    x_h_hat = get_mutual_info_by_ksg(x, x)
    assert math.isclose(x_h_real, x_h_hat, abs_tol=0.5)

    y1 = np.copy(x)
    y2 = (np.random.randn(n_samples, n_dims) * 10).astype(int)
    y3 = (np.random.randn(n_samples, n_dims) * 10).astype(int)
    y4 = (np.random.randn(n_samples, n_dims) * 10).astype(int)
    y = np.c_[y1, y2, y3, y4]
    y_probs = np.unique(y, axis=0, return_counts=True)[1] / n_samples
    y_h_real = -np.sum(y_probs * np.log(y_probs))
    y_h_hat = get_mutual_info_by_ksg(y, y)
    assert math.isclose(y_h_real, y_h_hat, abs_tol=3)

    z = np.c_[x, y]
    z_probs = np.unique(z, axis=0, return_counts=True)[1] / n_samples
    z_h_real = -np.sum(z_probs * np.log(z_probs))
    z_h_hat = get_mutual_info_by_ksg(z, z)
    assert math.isclose(z_h_real, z_h_hat, abs_tol=3)

    # I(x; (y1, y2)) = H(x) + H(y1, y2) - H(x, y1, y2)???
    mi_real = x_h_real + y_h_real - z_h_real
    mi_hat = x_h_hat + y_h_hat - z_h_hat
    assert math.isclose(mi_real, mi_hat, abs_tol=1)
    mi_hat_ksg = get_mutual_info_by_ksg(x, y)
    assert math.isclose(mi_real, mi_hat_ksg, abs_tol=1)


def test_compute_mi_cc():
    mean = np.zeros(2)

    # Setup covariance matrix with correlation coeff. equal 0.5.
    sigma_1 = 10
    sigma_2 = 50
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
        I_computed = get_mutual_info(X, y, n_neighbors=n_neighbors)
        assert math.isclose(I_computed, I_theory, abs_tol=0.03)


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
        y = np.c_[y, y]

        I_theory = -0.5 * (
            (1 - p) * np.log(0.5 * (1 - p)) + p * np.log(0.5 * p) + np.log(0.5)
        ) - np.log(2)

        # Assert the same tolerance.
        for n_neighbors in [3, 5, 7]:
            I_computed = get_mutual_info(
                X, y, discrete_y=True, n_neighbors=n_neighbors
            )
            assert math.isclose(I_computed, I_theory, abs_tol=0.03)
