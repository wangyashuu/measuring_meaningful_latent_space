"""Estimate mutual information and entropy."""
from typing import Union, List
import numpy as np
from sklearn.metrics import mutual_info_score

from .mine_estimator import estimate_mutual_info as estimate_by_mine
from .bins_estimator import estimate_mutual_info as estimate_by_bins
from .ksg_estimator import estimate_mutual_info as estimate_by_ksg
from .funcs import is_1d


def cache_fn(fn):
    cache = dict()

    def cached_fn(x, y, *args, **kwargs):
        use_cache = kwargs.pop("use_cache", True)
        key_args = dict(x=x[-64:], y=y[-64:], args=args, kwargs=kwargs)
        key = hash(str(key_args))
        if (not use_cache) or (key not in cache):
            r = fn(x, y, *args, **kwargs)
            cache[key] = r
        return cache[key]

    cached_fn.cache = cache
    return cached_fn


def _estimate_mutual_info(
    x: np.ndarray,
    y: np.ndarray,
    *args,
    estimator: str = "bins",
    discrete_x: bool = False,
    discrete_y: bool = False,
    **kwargs,
) -> float:
    """Estimate mutual information.

    Args:
        x (np.ndarray): A target cluster.
        y (np.ndarray): Another target cluster which should have the same number of sample as `x`.
        estimator (str, optional): String in ["bins", "ksg", "mine"] describe the method to estimate the mutual information. Default: "bins".
        discrete_x (bool, optional): If x is discrete.
        discrete_y (bool, optional): If y is discrete.

    Returns:
        float: The mutual information between x and y.
    """
    ## if both x and y are discrete and they are both 1-d array, calculate mutual information directly.
    if discrete_x and discrete_y and is_1d(x) and is_1d(y):
        return mutual_info_score(x.reshape(-1), x.reshape(-1))
    if estimator == "bins":
        return estimate_by_bins(
            x, y, *args, discrete_x=discrete_x, discrete_y=discrete_y, **kwargs
        )
    elif estimator == "ksg":
        return estimate_by_ksg(x, y, *args, **kwargs)
    elif estimator == "mine":
        return estimate_by_mine(x, y, *args, **kwargs)
    else:
        raise NotImplementedError(f"get_mutual_info estimator = {estimator}")


get_mutual_info = cache_fn(_estimate_mutual_info)


def get_mutual_infos(
    codes: np.ndarray,
    factors: np.ndarray,
    estimator: str = "bins",
    discrete_codes: (Union[List[bool], bool]) = False,
    discrete_factors: (Union[List[bool], bool]) = True,
    normalized: bool = False,
    *args,
    **kwargs,
) -> np.ndarray:
    """Compute the mutual information matrix between two high-dimensional cluster.

    Args:
        codes (np.ndarray): [Shape (n_samples, n_dims)] A target high-dimensional cluster with any number of samples.
        factors (np.ndarray): [Shape (n_samples, n_dims)] Another target high-dimensional cluster that should have same number of samples as `codes`.
        discrete_codes (Union[List[bool], bool]): implies if each code is discrete. Default: False.
        discrete_factors (Union[List[bool], bool]): implies if each factor is discrete. Default: True.
        normalized (bool): Normalize the mutual information to 0 and 1.

    Returns:
        np.ndarray: A mutual information matrix where ij entry represents the mutual information between i dimension in codes and j dimension in factos.
    """

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
                **kwargs,
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


def get_entropies(
    clouds: np.ndarray,
    *args,
    discrete: Union[List[bool], bool] = True,
    **kwargs,
):
    """Compute the entropies of a 2-d cluster.

    Args:
        clouds (np.ndarray): [Shape (n_samples, n_dims)] A target high-dimensional cluster with any number of samples.
        discrete (Union[List[bool], bool]): implies if each dimension is discrete. Default: False.

    Returns:
        np.ndarray: A 1-d array where i entry represents the entropy of i dimension in clouds.
    """

    n = clouds.shape[1]
    if type(discrete) is bool:
        discrete = np.full((n,), discrete)

    entropies = [
        get_entropy(clouds[:, i], discrete=discrete[i], *args, **kwargs)
        for i in range(n)
    ]
    return np.array(entropies)


def get_entropy(cloud, *args, discrete=False, **kwargs):
    """Estimate entropy.

    Args:
        x (np.ndarray): A target cluster.
        discrete (bool, optional): If cloud is discrete.
        kwargs: pass to `get_mutual_info`

    Returns:
        float: The mutual information between x and y.
    """
    # return get_entropy_by_ksg(cloud, *args, **kwargs)
    return get_mutual_info(
        cloud, cloud, discrete_x=discrete, discrete_y=discrete, *args, **kwargs
    )
