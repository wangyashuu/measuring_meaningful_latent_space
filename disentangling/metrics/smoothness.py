from typing import Union, List, Optional
import numpy as np
from sklearn.neighbors import KDTree

from disentangling.utils.mi import get_mutual_infos, get_entropies


def atleast_2d(arr):
    """Reshape to 2-d array if it is a 1-d array."""
    if len(arr.shape) == 1:
        return arr.reshape(arr.shape[0], 1)
    return arr


def get_tp(y_truth, y_pred):
    """Return a set of the true positive elements between two set."""
    tp = set(y_truth) & set(y_pred)
    return tp


def precision_score(y_truth, y_pred, eps=1e-16):
    """Calculate the precision score for two clusters."""
    tp = get_tp(y_truth, y_pred)
    precision = len(tp) / (len(y_pred) + eps)
    return precision


def cluster_for_discrete(x):
    """Cluster each categories of x"""
    classes = np.unique(x)
    index_map = {c: np.where(x == c)[0] for c in classes}
    rs = [index_map[x_i.item()] for x_i in x]
    return rs


def smoothness_for_pairs(x, y, k=3, discrete=False, tight=False):
    # expect all y.neighbors in x.neighbors
    x = atleast_2d(x)
    y = atleast_2d(y)
    if discrete:
        x_indices = cluster_for_discrete(x)
    else:
        x_tree = KDTree(x)
        dist, x_indices = x_tree.query(x, k)
        if not tight:
            x_indices = x_tree.query_radius(x, dist[:, k - 1])
    y_tree = KDTree(y)
    y_indices = y_tree.query(y, k, return_distance=False)

    precisions = []
    for i, (x_ind, y_ind) in enumerate(zip(x_indices, y_indices)):
        precisions.append(precision_score(x_ind[x_ind != i], y_ind[1:]))
    return np.mean(precisions)


def get_top_indices(relationship, threshold=0.5):
    n_codes, n_factors = relationship.shape
    sorted_matrix = np.sort(relationship, axis=0)[::-1]
    sorted_indices = np.argsort(relationship, axis=0)[::-1]
    indicators = np.cumsum(sorted_matrix, axis=0) > threshold

    top_indices = []
    for i in range(n_factors):
        indices = np.where(indicators[:, i])[0]
        if len(indices) == 0:
            top_indices.append(np.arange(n_codes))
        else:
            n_tops = indices[0] + 1
            top_indices.append(sorted_indices[:, i][np.arange(n_tops)])
    return top_indices


def smoothness(
    factors: np.ndarray,
    codes: np.ndarray,
    n_neighbors: Optional[int] = None,
    discrete_factors: Union[List[bool], bool] = False,
) -> np.ndarray:
    """Compute smoothness score for each factor.

    Args:
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        discrete_factors (Union[List[bool], bool]): implies if each factor is discrete. Default: False.
        test_size (float, optional): The rate of test samples, between 0 and 1. Default: 0.3.
        random_state (Union[float, None], optional): Seed of random generator for sampling test data. Default: None.

    Returns:
        scores (np.ndarray): [Shape (n_factors, )] A list where each represents smoothness score for each factor.
    """

    # get target factor and code
    mi_matrix = get_mutual_infos(
        codes, factors, estimator="ksg", discrete_factors=discrete_factors
    )
    entropies = get_entropies(
        factors, estimator="ksg", discrete=discrete_factors
    )
    factor_indices = np.arange(mi_matrix.shape[1])
    code_indices = np.argmax(mi_matrix, axis=0)
    # code_indices = get_top_indices(mi_matrix / entropies)

    # fix parameters
    proportions = np.exp(-entropies)
    if n_neighbors is None:
        n_neighbors = (factors.shape[0] * proportions).astype(int)
    if type(n_neighbors) is int:
        n_neighbors = [n_neighbors] * factors.shape[1]
    if type(discrete_factors) is bool:
        discrete_factors = [discrete_factors] * factors.shape[1]

    scores = []
    for f_idx, c_idx, k, discrete_factor in zip(
        factor_indices, code_indices, n_neighbors, discrete_factors
    ):
        f = factors[:, f_idx]
        c = codes[:, c_idx]
        scores.append(smoothness_for_pairs(f, c, k, discrete=discrete_factor))

    scores_corrected = (np.array(scores) - proportions) / (1 - proportions)

    results = {
        **{f"smoothness_{i}": s for i, s in enumerate(scores_corrected)},
    }
    return results


from .dci import dci_collect_relationship


def corrcoef_for_pairs(x, y):
    return np.corrcoef(x, y)[1][0]


def smoothness_for_comparison(
    factors: np.ndarray,
    codes: np.ndarray,
    n_neighbors: Optional[int] = None,
    discrete_factors: Union[List[bool], bool] = False,
):
    """The method is for research, it calculates mutual inforamtions, feature importance, correlation coefficient and smoothness for comparison.

    Args:
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        discrete_factors (Union[List[bool], bool]): implies if each factor is discrete. Default: True.
        test_size (float, optional): The rate of test samples, between 0 and 1. Default: 0.3.
        random_state (Union[float, None], optional): Seed of random generator for sampling test data. Default: None.

    Returns:
        scores (dict): All scores mentioned above where dict key is the name and dict value is the score
    """

    # get target factor and code
    mi_matrix = get_mutual_infos(
        codes, factors, estimator="ksg", discrete_factors=discrete_factors
    )
    entropies = get_entropies(
        factors, estimator="ksg", discrete=discrete_factors
    )
    factor_indices = np.arange(mi_matrix.shape[1])
    code_indices = np.argmax(mi_matrix, axis=0)
    # code_indices = get_top_indices(mi_matrix / entropies)

    # fix parameters
    proportions = np.exp(-entropies)
    if n_neighbors is None:
        n_neighbors = (factors.shape[0] * proportions).astype(int)
    if type(n_neighbors) is int:
        n_neighbors = [n_neighbors] * factors.shape[1]
    if type(discrete_factors) is bool:
        discrete_factors = [discrete_factors] * factors.shape[1]

    scores = []
    corrcoefs = []
    mis = []
    importance_matrix, _, _ = dci_collect_relationship(
        factors, codes, discrete_factors=discrete_factors
    )
    im_max = np.max(importance_matrix, axis=0)
    im_idx = np.argmax(importance_matrix, axis=0)
    ims = []
    tight_scores = []
    for f_idx, c_idx, k, discrete_factor in zip(
        factor_indices, code_indices, n_neighbors, discrete_factors
    ):
        f = factors[:, f_idx]
        c = codes[:, c_idx]
        scores.append(smoothness_for_pairs(f, c, k, discrete=discrete_factor))
        tight_scores.append(
            smoothness_for_pairs(f, c, k, discrete=discrete_factor, tight=True)
        )
        corrcoefs.append(corrcoef_for_pairs(f, c))
        mis.append(mi_matrix[c_idx, f_idx])
        ims.append(importance_matrix[c_idx, f_idx])

    scores_corrected = (np.array(scores) - proportions) / (1 - proportions)
    tight_scores_corrected = (np.array(tight_scores) - proportions) / (
        1 - proportions
    )
    results = {
        **{f"smoothness_{i}": s for i, s in enumerate(scores)},
        **{f"smoothness_tight_{i}": s for i, s in enumerate(tight_scores)},
        **{f"corrected_{i}": s for i, s in enumerate(scores_corrected)},
        **{
            f"tight_corrected_{i}": s
            for i, s in enumerate(tight_scores_corrected)
        },
        **{f"corrcoef_{i}": c for i, c in enumerate(corrcoefs)},
        **{f"mi_{i}": c for i, c in enumerate(mis)},
        **{f"im_{i}": c for i, c in enumerate(ims)},
        **{f"im_max_{i}": c for i, c in enumerate(im_max)},
        **{f"im_idx_{i}": c for i, c in enumerate(im_idx)},
        **{f"mi_idx_{i}": c for i, c in enumerate(code_indices)},
    }
    return results
