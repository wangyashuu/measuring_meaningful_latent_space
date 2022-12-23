import numpy as np
from sklearn.feature_selection import mutual_info_regression
from entropy_estimators import continuous

from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors, KDTree

"""
adapted from https://github.com/scikit-learn/scikit-learn/blob/82df48934eba1df9a1ed3be98aaace8eada59e6e/sklearn/feature_selection/_mutual_info.py#
for compute MI between vector and variable
more references:
1. https://github.com/paulbrodersen/entropy_estimators/blob/789863a1cf327fb4e2bfdcc100ab67a1f6755ba2/entropy_estimators/continuous.py#L225
2. https://github.com/scikit-learn/scikit-learn/blob/82df48934eba1df9a1ed3be98aaace8eada59e6e/sklearn/feature_selection/_mutual_info.py
"""

"""
True mutual information is non-negative, the return value will get the max bewteen the estimation and 0
"""


def atleast_2d(arr):
    if len(arr.shape) == 1:
        return arr.reshape(arr.shape[0], 1)
    return arr


def mutual_info_cc(X, y, n_neighbors=3):
    X = atleast_2d(X)
    y = atleast_2d(y)
    n = X.shape[0]

    Xy = np.c_[X, y]

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)

    nn.fit(Xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(X, metric="chebyshev")
    nx = kd.query_radius(X, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric="chebyshev")
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0

    mi = (
        digamma(n)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )
    return max(0, mi)


def mutual_info_cd(X, y, n_neighbors=3):
    n_samples = X.shape[0]

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()
    for label in np.unique(y):
        mask = y == label
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

    kd = KDTree(X_masked)
    m_all = kd.query_radius(
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


def mutual_info(X, y, discrete_y=False, n_neighbors=3):
    if discrete_y:
        return mutual_info_cd(X, y, n_neighbors=n_neighbors)
    else:
        return mutual_info_cc(X, y, n_neighbors=n_neighbors)
