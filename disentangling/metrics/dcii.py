import numpy as np

from ..utils.mi import get_mutual_infos, get_entropies, get_captured_mis


def dcii(
    factors,
    codes,
    epsilon=1e-10,
    mi_improved=True,
    discrete_factors=False,
    **kwargs
):
    # mutual_info matrix (n_codes, n_factors)
    estimator = "ksg" if mi_improved else "bins"
    mutual_infos = get_mutual_infos(
        codes,
        factors,
        estimator=estimator,
        normalized=True,
        discrete_factors=discrete_factors,
    )
    d_score = _dcii_disentanglement(mutual_infos)
    c_score = _dcii_completeness(mutual_infos)
    i_score = dcii_i(factors, codes, epsilon=epsilon, mi_improved=mi_improved)
    return dict(
        disentanglement=d_score, completeness=c_score, informativeness=i_score
    )


def _dcii_disentanglement(mutual_infos):
    # mutual_info (n_codes, n_factors)
    # get scores for each code
    n_factors = mutual_infos.shape[1]
    mi_max = np.max(mutual_infos, axis=1, keepdims=True)
    errorness = np.sqrt(
        (1.0 / (n_factors - 1))
        * (np.sum(mutual_infos**2, axis=1, keepdims=True) - mi_max**2)
    )
    correctness = mi_max
    scores = correctness - errorness
    score = np.mean(scores)
    return score


def _dcii_completeness(mutual_infos):
    # mutual_info (n_codes, n_factors)
    # get scores for each factor
    n_codes = mutual_infos.shape[0]
    mi_max = np.max(mutual_infos, axis=0, keepdims=True)
    errorness = np.sqrt(
        (1.0 / (n_codes - 1))
        * (np.sum(mutual_infos**2, axis=0, keepdims=True) - mi_max**2)
    )
    correctness = mi_max
    scores = correctness - errorness
    score = np.mean(scores)
    return score


def dcii_d(
    factors,
    codes,
    epsilon=1e-10,
    mi_improved=True,
    discrete_factors=False,
    **kwargs
):
    """
    Compute disentanglement

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    estimator = "ksg" if mi_improved else "bins"
    mutual_infos = get_mutual_infos(
        codes,
        factors,
        estimator=estimator,
        normalized=True,
        discrete_factors=discrete_factors,
    )
    return _dcii_disentanglement(mutual_infos)


def dcii_c(
    factors,
    codes,
    epsilon=1e-10,
    mi_improved=True,
    discrete_factors=False,
    **kwargs
):
    """
    Compute completeness

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    estimator = "ksg" if mi_improved else "bins"
    mutual_infos = get_mutual_infos(
        codes,
        factors,
        estimator=estimator,
        normalized=True,
        discrete_factors=discrete_factors,
    )
    return _dcii_completeness(mutual_infos)


def dcii_i(
    factors,
    codes,
    epsilon=1e-10,
    mi_improved=True,
    discrete_factors=False,
    **kwargs
):
    """
    Compute informativeness

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # discrete_codes?
    # mutual_info matrix (n_codes, n_factors)
    estimator = "ksg" if mi_improved else "bins"
    captured = (
        get_captured_mis(codes, factors, estimator="ksg")
        if mi_improved
        else np.max(
            get_mutual_infos(codes, factors, estimator=estimator),
            axis=0,
        )
    )
    entropies = get_entropies(
        factors, estimator=estimator, discrete=discrete_factors
    )
    captured[captured > entropies] = entropies[captured > entropies]
    score = np.mean(captured / (entropies + epsilon))
    return score
