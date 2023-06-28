from typing import Union, List
import numpy as np

from disentangling.utils.mi import (
    get_mutual_infos,
    get_entropies,
    get_mutual_info,
)


def estimate_mi_codes_and_factor(
    codes: np.ndarray,
    factor: np.ndarray,
    discrete_factor: bool = False,
    estimate_by: str = "mine",
    estimator: str = "ksg",
) -> float:
    """Calculate mutual infors between codes $c$ and factor $z_j$ I(c; z_j).

    Args:
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        factor (np.ndarray): [Shape (batch_size, )] On dimension in factor representation.
        discrete_factors (bool): It implies if the factor is discrete. Default: True.
        estimate_by (str, optional): String in ["mine", "max", "sum"] The method of estimating the mutual information between codes $c$ and factor $z_j$, I(C; z_j), where "mine" will use a neural network to estimate, "max" will approximate $I(c;z_j)$ as the maximum entry $\max_i I(c_i; z_j)", and "sum" will approximate $I(c;z_j)$ as sum $\sum_i I(c_i; z_j)". Default: "mine".
        estimator (str, optional): String in ["ksg", "bins", "mine"], each represents different method to estimate mutual information (see more in `disentangling.utils.mi`). If $I(c;z_j)$ is not estimated by mine, we need the estimated mutual information I(c_i;z_j) to approximate the value, see more in above. Default: "ksg".

    Returns:
        scores (dict):  Dictionary where
            - "d" represents average disentanglement score,
            - "c" represents average completeness score,
            - "i" represents average informativeness score.
    """

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
    codes: np.ndarray,
    factors: np.ndarray,
    discrete_factors: Union[List[bool], bool] = False,
    estimate_mi_codes_and_factor_by: str = "mine",
    estimator: str = "ksg",
) -> float:
    """Calculate mutual infors between codes $c$ and factor $z_j$ I(c; z_j), for each dimension in factors.

    Args:
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        discrete_factors (Union[List[bool], bool]): It implies if each factor is discrete. Default: True.
        estimate_mi_codes_and_factor_by (str, optional): String in ["mine", "max", "sum"] The method of estimating the mutual information between codes $c$ and factor $z_j$, I(C; z_j), where "mine" will use a neural network to estimate, "max" will approximate $I(c;z_j)$ as the maximum entry $\max_i I(c_i; z_j)", and "sum" will approximate $I(c;z_j)$ as sum $\sum_i I(c_i; z_j)". Default: "mine".
        estimator (str, optional): String in ["ksg", "bins", "mine"], each represents different method to estimate mutual information (see more in `disentangling.utils.mi`). If $I(c;z_j)$ is not estimated by mine, we need the estimated mutual information I(c_i;z_j) to approximate the value, see more in above. Default: "ksg".

    Returns:
        scores (dict):  Dictionary where
            - "d" represents average disentanglement score,
            - "c" represents average completeness score,
            - "i" represents average informativeness score.
    """
    n_codes = codes.shape[1]
    n_factors = factors.shape[1]
    if type(discrete_factors) is bool:
        discrete_factors = np.full((n_factors,), discrete_factors)

    captured_mis = [
        estimate_mi_codes_and_factor(
            codes,
            factors[:, i],
            discrete_factor=discrete_factor,
            estimate_by=estimate_mi_codes_and_factor_by,
            estimator=estimator,
        )
        for i, discrete_factor in enumerate(discrete_factors)
    ]
    return np.array(captured_mis)


def exclusive_rate(
    target: np.ndarray, axis: int = 0, keepdims: bool = True
) -> Union[np.ndarray, float]:
    """Calculate exclusive rate.

    $$ \max target - \sqrt \sum target_i, \text{ where} i \neq \argmax target $$

    Args:
        target (np.ndarray): An 1-d array or 2-d array.
        axis (int): The axis to compute the exclusive rate along with.
        keepdims (bool): If keep the dims.

    Returns:
        (Union[np.ndarray, float]): the exclusive rate along the given axis.
    """

    n = target.shape[axis]
    aggregate_params = dict(axis=axis, keepdims=keepdims)
    max_item = np.max(target, **aggregate_params)
    correctness = max_item
    errorness = np.sqrt(
        1 / (n - 1) * (np.sum(target**2, **aggregate_params) - max_item**2)
    )
    return correctness - errorness


def _disentanglement(mutual_infos: np.ndarray) -> float:
    """Compute the disentanglement scores from mutual information [Shape (n_codes, n_factors)]."""
    n_codes, n_factors = mutual_infos.shape
    scored_mutual_infos = np.zeros((n_codes, n_factors))
    for c in range(n_codes):
        target = mutual_infos[c, :]
        max_idx = np.argmax(target)
        scored_mutual_infos[c, max_idx] = exclusive_rate(target)

    scored = np.minimum(np.sum(scored_mutual_infos, axis=0), 1)
    return np.sum(scored) / n_factors


def _completeness(mutual_infos: np.ndarray) -> float:
    """Compute the completeness scores from mutual information."""
    n_codes, n_factors = mutual_infos.shape
    return np.sum(exclusive_rate(mutual_infos, axis=0)) / n_factors


def dcii(
    factors: np.ndarray,
    codes: np.ndarray,
    discrete_factors: Union[List[bool], bool] = False,
    estimator: str = "ksg",
    estimate_mi_codes_and_factor_by: str = "mine",
    epsilon: float = 1e-10,
) -> dict:
    """Calculate DCII scores.

    Args:
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        discrete_factors (Union[List[bool], bool]): It implies if each factor is discrete. Default: True.
        estimator (str, optional): String in ["ksg", "bins", "mine"], each represents different method to estimate mutual information, see more in `disentangling.utils.mi`. Default: "ksg".
        estimate_mi_codes_and_factor_by (str, optional): estimator of estimating the mutual information between codes $c$ and factor $z_j$, I(C; z_j). Default: "mine".
        epsilon (float, optional): Epsilon (the very small number) used in calculation. Default: 1e-10.

    Returns:
        scores (dict):  Dictionary where
            - "d" represents average disentanglement score,
            - "c" represents average completeness score,
            - "i" represents average informativeness score.
    """
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
    normalized_mutual_infos = mutual_infos / (captured + epsilon)
    d_score = _disentanglement(normalized_mutual_infos)
    c_score = _completeness(normalized_mutual_infos)
    i_score = dcii_i(factors, codes, epsilon=epsilon, estimator=estimator)
    return dict(
        disentanglement=d_score,
        completeness=c_score,
        informativeness=i_score,
    )


def dcii_d(
    factors: np.ndarray,
    codes: np.ndarray,
    discrete_factors: Union[List[bool], bool] = False,
    estimator: str = "ksg",
    estimate_mi_codes_and_factor_by: str = "mine",
    epsilon: float = 1e-10,
) -> float:
    """Calculate DCII(D) score.

    Args:
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        discrete_factors (Union[List[bool], bool]): It implies if each factor is discrete. Default: True.
        estimator (str, optional): String in ["ksg", "bins", "mine"], each represents different method to estimate mutual information, see more in `disentangling.utils.mi`. Default: "ksg".
        estimate_mi_codes_and_factor_by (str, optional): estimator of estimating the mutual information between codes $c$ and factor $z_j$, I(C; z_j). Default: "mine".
        epsilon (float, optional): Epsilon (the very small number) used in calculation. Default: 1e-10.

    Returns:
        score (float): Average disentanglement score.
    """
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
    normalized_mutual_infos = mutual_infos / (captured + epsilon)
    d_score = _disentanglement(normalized_mutual_infos)
    return d_score


def dcii_c(
    factors: np.ndarray,
    codes: np.ndarray,
    discrete_factors: Union[List[bool], bool] = False,
    estimator: str = "ksg",
    estimate_mi_codes_and_factor_by: str = "mine",
    epsilon: float = 1e-10,
) -> float:
    """Calculate DCII(C) score.

    Args:
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        discrete_factors (Union[List[bool], bool]): It implies if each factor is discrete. Default: True.
        estimator (str, optional): String in ["ksg", "bins", "mine"], each represents different method to estimate mutual information, see more in `disentangling.utils.mi`. Default: "ksg".
        estimate_mi_codes_and_factor_by (str, optional): estimator of estimating the mutual information between codes $c$ and factor $z_j$, I(C; z_j). Default: "mine".
        epsilon (float, optional): Epsilon (the very small number) used in calculation. Default: 1e-10.

    Returns:
        score (float): Average completeness score.
    """
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
    normalized_mutual_infos = mutual_infos / (captured + epsilon)
    c_score = _completeness(normalized_mutual_infos)
    return c_score


def dcii_i(
    factors: np.ndarray,
    codes: np.ndarray,
    discrete_factors: Union[List[bool], bool] = False,
    estimator: str = "ksg",
    estimate_mi_codes_and_factor_by: str = "mine",
    epsilon: float = 1e-10,
) -> float:
    """Calculate DCII(I) score.

    Args:
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        discrete_factors (Union[List[bool], bool]): It implies if each factor is discrete. Default: True.
        estimator (str, optional): String in ["ksg", "bins", "mine"], each represents different method to estimate mutual information, see more in `disentangling.utils.mi`. Default: "ksg".
        estimate_mi_codes_and_factor_by (str, optional): estimator of estimating the mutual information between codes $c$ and factor $z_j$, I(C; z_j). Default: "mine".
        epsilon (float, optional): Epsilon (the very small number) used in calculation. Default: 1e-10.

    Returns:
        score (float): Average informativeness score.
    """
    # discrete_codes?
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
