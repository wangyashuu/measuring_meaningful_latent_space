"""Mutual information estimation by Kraskov estimator (based on `Estimating Mutual Information <https://arxiv.org/abs/cond-mat/0305641>`)

The implementation is adapted from `scikit-learn <https://github.com/scikit-learn/scikit-learn/blob/82df48934eba1df9a1ed3be98aaace8eada59e6e/sklearn/feature_selection/_mutual_info.py#>`.
The main change is that we extended to high dimensions, so it is able to compute mutual information between vector $[c_1, c_2, ...c_n]$ and variable $z_1$ (originally can only estimate mutual information between two variables $c_1$ and $z_1$).

More related Kraskov estimator implementation:
- https://github.com/paulbrodersen/entropy_estimators/blob/789863a1cf327fb4e2bfdcc100ab67a1f6755ba2/entropy_estimators/continuous.py#L225


# True mutual information is non-negative, the return value will get the max bewteen the estimation and 0
"""
from typing import Callable
import numpy as np

from scipy.special import digamma, gamma
from sklearn.neighbors import NearestNeighbors, KDTree
from .funcs import atleast_2d, default_transform


def add_random_noise(X, epsilon=1e-10):
    # Add small noise to continuous features as advised in Kraskov et. al.
    return X + epsilon * np.random.rand(*X.shape)


def get_nearest_neighbors_distances(
    X: np.ndarray,
    n_neighbors=3,
    distance_metric="chebyshev",
):
    # it gives nearest `n_neighbors` points,
    # include the point itself and `n_neighbors - 1` neighbors
    # from paper: there are k − 1 other points at smaller distances.
    nn = NearestNeighbors(metric=distance_metric, n_neighbors=n_neighbors)
    nn.fit(X)
    # the distance to the k-th nearest neighbor
    radius = nn.kneighbors()[0]
    # why next floating point after radius[:, -1] and before 0
    radius = np.nextafter(radius[:, -1], 0)
    return radius


def mutual_info_cc(X, y, distance_metric="chebyshev", n_neighbors=3):
    """Estimate mutual information between continuous vector or variable $X$ and continuous vector or  variable $y$."""
    n = X.shape[0]
    Xy = np.c_[X, y]

    radius = get_nearest_neighbors_distances(
        Xy, distance_metric=distance_metric, n_neighbors=n_neighbors
    )

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    x_tree = KDTree(X, metric=distance_metric)
    nx = x_tree.query_radius(X, radius, count_only=True)
    nx = np.array(nx) - 1.0

    y_tree = KDTree(y, metric=distance_metric)
    ny = y_tree.query_radius(y, radius, count_only=True)
    ny = np.array(ny) - 1.0

    mi = (
        digamma(n)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )
    return max(0, mi)


def mutual_info_cd(X, y, n_neighbors=3):
    """Estimate mutual information between continuous vector or variable $X$ and discrete variable $y$."""
    n_samples = X.shape[0]
    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()
    for label in np.unique(y, axis=0):
        mask = (y == label).all(axis=1)
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(X[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]

    X_masked = X[mask]
    radius = radius[mask]

    tree = KDTree(X_masked)
    m_all = tree.query_radius(
        X_masked, radius, count_only=True, return_distance=False
    )
    m_all = np.array(m_all) - 1.0

    mi = (
        digamma(n_samples)
        + np.mean(digamma(k_all))
        - np.mean(digamma(label_counts))
        - np.mean(digamma(m_all + 1))
    )
    return max(0, mi)


def estimate_mutual_info(
    X: np.ndarray,
    y: np.ndarray,
    discrete_y: bool = False,
    n_neighbors: int = 3,
    distance_metric: str = "chebyshev",
    transform: Callable[[np.ndarray], np.ndarray] = default_transform,
    epsilon: float = 1e-10,
    **kwargs,
) -> float:
    """Estimate mutual information between X and u using K-nearest neighbors.

    Based on `Estimating Mutual Information <https://arxiv.org/abs/cond-mat/0305641>`

    Args:
        X (np.ndarray): A target cluster.
        y (np.ndarray): Another target cluster which should have the same number of sample as `X`.
        discrete_y (bool, optional): If y is discrete.
        n_neighbors (int, optional): The number of neighbors. Default: 3
        distance_metric (str, optional): The distance metric when calculating distance in nearest neighbors. See support metric in `sklearn.neighbors.NearestNeighbors`. Default: "chebyshev".
        transform (Callable[[np.ndarray], np.ndarray], optional): Cloud transform function. Default: default_transform (scale with mean).
        epsilon (float, optional): The very smallest number used in calculation. Default: 1e-10.

    Returns:
        float: The mutual information estimated by k nearest neighbors.
    """

    X = atleast_2d(X)
    y = atleast_2d(y)

    # If X'=F(X) and Y'=G(Y) are homeomorphisms, then I(X,Y)=I(X',Y')
    # possible transform: rank, std, log
    if transform:
        X = transform(X)
        y = transform(y)

    X = add_random_noise(X, epsilon=epsilon)

    if discrete_y:
        return mutual_info_cd(X, y, n_neighbors=n_neighbors)
    else:
        y = add_random_noise(y, epsilon=epsilon)
        return mutual_info_cc(
            X, y, n_neighbors=n_neighbors, distance_metric=distance_metric
        )


### get entropy


def get_entropy_by_ksg(
    X: np.ndarray,
    n_neighbors: int = 3,
    distance_metric: str = "chebyshev",
    transform: Callable[[np.ndarray], np.ndarray] = default_transform,
) -> float:

    """Estimate entropy of X using K-nearest neighbors.

    Based on `Estimating Mutual Information <https://arxiv.org/abs/cond-mat/0305641>`

    Args:
        X (np.ndarray): A target cluster.
        n_neighbors (int, optional): The number of neighbors. Default: 3
        distance_metric (str, optional): The distance metric when calculating distance in nearest neighbors. See support metric in `sklearn.neighbors.NearestNeighbors`. Default: "chebyshev".
        transform (Callable[[np.ndarray], np.ndarray], optional): Cloud transform function. Default: default_transform (scale with mean).

    Returns:
        float: The entropy estimated by k nearest neighbors.
    """

    X = atleast_2d(X)
    if transform:
        X = transform(X)
    X = add_random_noise(X)

    n_samples, n_dims = X.shape
    distances = get_nearest_neighbors_distances(
        X, distance_metric=distance_metric, n_neighbors=n_neighbors
    )
    if distance_metric == "minkowski":  # l2 norm, p = 2
        vub = np.pi ** (n_dims / 2.0) / gamma(1 + n_dims / 2.0)
    elif distance_metric == "chebyshev":  # l_inf norm, p = inf
        vub = 2**n_dims

    return (
        digamma(n_samples)
        - digamma(n_neighbors)
        + np.log(vub)
        + n_dims * np.mean(np.log(2 * distances[distances > 0]))
    )
