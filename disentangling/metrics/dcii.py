import numpy as np

from disentangling.utils.mi import (
    get_mutual_infos,
    get_entropies,
    get_mutual_info,
)


def estimate_mi_codes_and_factor(
    codes,
    factor,
    estimate_by="mine",
    estimator=None,
    discrete_factor=False,
):
    # I(C; z)
    if estimate_by == "mine":
        if codes.shape[1] < 4:
            return get_mutual_info(codes, factor, estimator="ksg")
        return get_mutual_info(codes, factor, estimator="mine")
    else:
        params = dict(estimator=estimator, discrete_y=discrete_factor)
        mis = [
            get_mutual_info(codes[:, i], factor, **params)
            for i in range(codes.shape[1])
        ]
        if estimate_by == "max":
            return np.max(mis)
        elif estimate_by == "sum":
            return np.sum(mis)
    raise NotImplementedError(
        f"get_captured_mi_from_factor estimate_by = {estimate_by}"
    )


def get_captured_mis(
    codes,
    factors,
    discrete_factors=False,
    estimate_mi_codes_and_factor_by="mine",
    estimator=None,
):
    n_codes = codes.shape[1]
    n_factors = factors.shape[1]
    if type(discrete_factors) is bool:
        discrete_factors = np.full((n_factors,), discrete_factors)

    captured_mis = [
        estimate_mi_codes_and_factor(
            codes,
            factors[:, i],
            estimate_by=estimate_mi_codes_and_factor_by,
            estimator=estimator,
            discrete_factor=discrete_factor,
        )
        for i, discrete_factor in enumerate(discrete_factors)
    ]
    return np.array(captured_mis)


def exclusive_rate(target, axis=0, keepdims=True):
    n = target.shape[axis]
    aggregate_params = dict(axis=axis, keepdims=keepdims)
    max_item = np.max(target, **aggregate_params)
    correctness = max_item
    errorness = np.sqrt(
        1 / (n - 1) * (np.sum(target**2, **aggregate_params) - max_item**2)
    )
    return correctness - errorness


def _disentanglement(mutual_infos):
    n_codes, n_factors = mutual_infos.shape
    scored_mutual_infos = np.zeros((n_codes, n_factors))
    for c in range(n_codes):
        target = mutual_infos[c, :]
        max_idx = np.argmax(target)
        scored_mutual_infos[c, max_idx] = exclusive_rate(target)

    scored = np.minimum(np.sum(scored_mutual_infos, axis=0), 1)
    return np.sum(scored) / n_factors


def _completeness(mutual_infos):
    n_codes, n_factors = mutual_infos.shape
    return np.sum(exclusive_rate(mutual_infos, axis=0)) / n_factors


def dcii(
    factors,
    codes,
    epsilon=1e-10,
    mi_improved=True,
    discrete_factors=False,
    estimate_mi_codes_and_factor_by="mine",
):
    # mutual_info matrix (n_codes, n_factors)
    estimator = "ksg" if mi_improved else "bins"
    mutual_infos = get_mutual_infos(
        codes,
        factors,
        estimator=estimator,
        discrete_factors=discrete_factors,
    )
    captured = get_captured_mis(
        codes,
        factors,
        estimate_mi_codes_and_factor_by=estimate_mi_codes_and_factor_by,
        estimator=estimator,
        discrete_factors=discrete_factors,
    )
    normalized_mutual_infos = mutual_infos / captured
    d_score = _disentanglement(normalized_mutual_infos)
    c_score = _completeness(normalized_mutual_infos)
    i_score = dcii_i(factors, codes, epsilon=epsilon, mi_improved=mi_improved)
    return dict(
        disentanglement=d_score,
        completeness=c_score,
        informativeness=i_score,
    )


def dcii_d(
    factors,
    codes,
    epsilon=1e-10,
    mi_improved=True,
    discrete_factors=False,
    estimate_mi_codes_and_factor_by="mine",
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
        discrete_factors=discrete_factors,
    )
    captured = get_captured_mis(
        codes,
        factors,
        estimate_mi_codes_and_factor_by=estimate_mi_codes_and_factor_by,
        estimator=estimator,
        discrete_factors=discrete_factors,
    )
    normalized_mutual_infos = mutual_infos / captured
    d_score = _disentanglement(normalized_mutual_infos)
    return d_score


def dcii_c(
    factors,
    codes,
    epsilon=1e-10,
    mi_improved=True,
    discrete_factors=False,
    estimate_mi_codes_and_factor_by="mine",
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
        discrete_factors=discrete_factors,
    )
    captured = get_captured_mis(
        codes,
        factors,
        estimate_mi_codes_and_factor_by=estimate_mi_codes_and_factor_by,
        estimator=estimator,
        discrete_factors=discrete_factors,
    )
    normalized_mutual_infos = mutual_infos / captured
    c_score = _completeness(normalized_mutual_infos)
    return c_score


def dcii_i(
    factors,
    codes,
    epsilon=1e-10,
    mi_improved=True,
    discrete_factors=False,
    estimate_mi_codes_and_factor_by="mine",
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
    captured = get_captured_mis(
        codes,
        factors,
        estimate_mi_codes_and_factor_by=estimate_mi_codes_and_factor_by,
        estimator=estimator,
        discrete_factors=discrete_factors,
    )

    entropies = get_entropies(
        factors, estimator=estimator, discrete=discrete_factors
    )
    captured[captured > entropies] = entropies[captured > entropies]
    score = np.mean(captured / (entropies + epsilon))
    return score
