import numpy as np
import torch
from scipy.special import digamma, gamma
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.metrics import mutual_info_score


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
    if torch.is_tensor(arr):
        arr = arr.cpu().numpy()
    if len(arr.shape) == 1:
        return arr.reshape(arr.shape[0], 1)
    return arr


def add_random_noise(X, epsilon=1e-12):
    # Add small noise to continuous features as advised in Kraskov et. al.
    return X + epsilon * np.random.rand(*X.shape)


def get_nearest_neighbors_distances(
    X,
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
    n = X.shape[0]
    Xy = np.c_[X, y]

    radius = get_nearest_neighbors_distances(
        Xy, distance_metric=distance_metric, n_neighbors=n_neighbors
    )

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(X, metric=distance_metric)
    nx = kd.query_radius(X, radius, count_only=True)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric=distance_metric)
    ny = kd.query_radius(y, radius, count_only=True)
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

    # if X_masked.shape[0] == 0:
    #     m_all = 0
    # else:
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


def get_mutual_info_by_ksg(
    X, y, discrete_x=False, discrete_y=False, n_neighbors=3, transform=None
):
    X = atleast_2d(X)
    y = atleast_2d(y)

    # If X'=F(X) and Y'=G(Y) are homeomorphisms, then I(X,Y)=I(X',Y')
    if transform:  # rank, std, log
        X = transform(X)
        y = transform(y)

    X = add_random_noise(X)

    if discrete_y:
        return mutual_info_cd(X, y, n_neighbors=n_neighbors)
    else:
        y = add_random_noise(y)
        return mutual_info_cc(X, y, n_neighbors=n_neighbors)


### get entropy


def get_entropy_by_ksg(
    X, n_neighbors=3, distance_metric="chebyshev", transform=None
):
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
        # refs:
        # - https://github.com/mutualinfo/mutual_info/blob/20b40e1faafbb78aadadfe37b4154833ff92fd1d/mutual_info/mutual_info.py#L140
        # - https://github.com/BiuBiuBiLL/NPEET_LNC/blob/82f29263bf745aded3d6b577894d229c946092b8/lnc.py#L217
        vub = 2**n_dims
    # TODO: check if distances*2
    return (
        digamma(n_samples)
        - digamma(n_neighbors)
        + np.log(vub)
        + n_dims * np.mean(np.log(distances))
    )


### get mutual info by discrete cloud into bins


def np_discretize(xs, n_bins=30):
    """Discretization based on histograms."""
    discretized = np.digitize(xs, np.histogram(xs, n_bins)[1][:-1])
    return discretized


def get_mutual_info_by_bins(x, y, discrete_x=True, discrete_y=True, n_bins=50):
    """
    more info about calc mutual info:
    1. calculate mutual info between float
    - https://www.researchgate.net/post/Can_anyone_help_with_calculating_the_mutual_information_between_for_float_number,
    - https://bitbucket.org/szzoli/ite/src/master/
    2. implementation
    - https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    """
    discretized_x = x if discrete_x else np_discretize(x, n_bins)
    discretized_y = y if discrete_y else np_discretize(y, n_bins)
    return mutual_info_score(discretized_x, discretized_y)


## exported


def get_entropies(clouds, *args, discrete=True, **kwargs):
    n = clouds.shape[1]
    if type(discrete) is bool:
        discrete = np.full((n,), discrete)

    entropies = [
        get_entropy(clouds[:, i], discrete=discrete[i], *args, **kwargs)
        for i in range(n)
    ]
    return np.array(entropies)


def get_entropy(cloud, *args, discrete=True, **kwargs):
    # return get_entropy_by_ksg(cloud, *args, **kwargs)
    return get_mutual_info(
        cloud, cloud, discrete_x=discrete, discrete_y=discrete, *args, **kwargs
    )


def get_mutual_info(x, y, *args, estimator="bins", **kwargs):
    if estimator == "bins":
        return get_mutual_info_by_bins(x, y, *args, **kwargs)
    elif estimator == "ksg":
        return get_mutual_info_by_ksg(x, y, *args, **kwargs)


def get_mutual_infos(
    codes,
    factors,
    estimator="bins",
    discrete_codes=False,
    discrete_factors=True,
    normalized=False,
    *args,
    **kwargs
):
    n_codes = codes.shape[1]
    n_factors = factors.shape[1]
    if type(discrete_codes) is bool:
        discrete_codes = np.full((n_codes,), discrete_codes)
    if type(discrete_factors) is bool:
        discrete_factors = np.full((n_factors,), discrete_factors)
    m = np.zeros((n_codes, n_factors))
    for i in range(n_codes):
        for j in range(n_factors):
            discrete_x = discrete_codes[i]
            discrete_y = discrete_factors[j]
            m[i, j] = get_mutual_info(
                codes[:, i],
                factors[:, j],
                discrete_x=discrete_x,
                discrete_y=discrete_y,
                estimator=estimator,
                *args,
                **kwargs
            )
    if normalized:
        code_entropies = get_entropies(
            codes, discrete=discrete_codes, estimator=estimator
        )
        factor_entropies = get_entropies(
            factors, discrete=discrete_factors, estimator=estimator
        )
        normalize_value = (
            code_entropies.reshape(-1, 1) + factor_entropies.reshape(1, -1)
        ) / 2
        m = m / normalize_value
        m[m > 1] = 1.0  # minor estimator error, might induced by random noise
    return m


def get_captured_mi_from_factor(
    codes, factor, discrete_factor=True, estimator="naive", *args, **kwargs
):
    # I(C; z)
    if estimator == "naive":
        factor = factor.reshape(-1, 1)
        h_codes = get_mutual_info(
            codes, codes, discrete_y=discrete_factor, estimator="ksg"
        )
        h_factor = get_mutual_info(
            factor, factor, discrete_y=discrete_factor, estimator="ksg"
        )
        xy = np.c_[codes, factor]
        h_xy = get_mutual_info(
            xy, xy, discrete_y=discrete_factor, estimator="ksg"
        )
        return max(h_codes + h_factor - h_xy, 0)
    elif estimator == "ksg":
        return get_mutual_info(
            codes,
            factor,
            discrete_y=discrete_factor,
            estimator="ksg",
            *args,
            **kwargs
        )


def get_captured_mis(codes, factors, *args, **kwargs):
    n_codes = codes.shape[1]
    n_factors = factors.shape[1]
    discrete_factors = np.full((n_factors,), False)

    captured_mis = [
        get_captured_mi_from_factor(
            codes,
            factors[:, j],
            discrete_factor=discrete_factors[j],
            *args,
            **kwargs
        )
        for j in range(n_factors)
    ]
    return np.array(captured_mis)
