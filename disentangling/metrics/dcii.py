import sklearn.metrics
import torch
import numpy as np

from .mig import calc_mutual_infos, calc_entropy
from .mi import mutual_info, atleast_2d
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

from mutual_info.mutual_info import mutual_information

from sklearn.feature_selection._mutual_info import _compute_mi_cc
from entropy_estimators import continuous


# def mutual_info_func(improved=True):
#     if improved:
#         return mutual_info
#     else:
#         return calc_mutual_info


def get_entropies(factors):
    n_factors = factors.shape[1]
    entropies = [
        mutual_info(factors[:, i], factors[:, i]) for i in range(n_factors)
    ]
    return np.array(entropies)


def get_continuous_mi(x, y, epsilon=1e-10, k=3):
    x = atleast_2d(x)
    y = atleast_2d(y)
    x = x + epsilon * np.random.rand(*x.shape)
    y = y + epsilon * np.random.rand(*y.shape)
    return continuous.get_mi(x, y, k=k)


def get_mutual_infos(codes, factors):
    codes = codes.cpu().numpy()
    factors = factors.cpu().numpy()
    n_codes = codes.shape[1]
    n_factors = factors.shape[1]
    m = np.zeros((n_codes, n_factors))  # torch.zeros.to(codes.device)
    for i in range(n_codes):
        for j in range(n_factors):
            # i = 0
            # j = 0

            # haha = mutual_info_regression(x, y.reshape(-1))
            # print(haha)
            m[i, j] = mutual_info(codes[:, i], factors[:, j], discrete_y=True)
    return m


# def get_entropy


def dcii_d(factors, codes, epsilon=10e-8, mi_improved=True):
    """
    Compute disentanglement

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)

    mutual_infos = (
        get_mutual_infos(codes, factors)
        if mi_improved
        else calc_mutual_infos(codes, factors).cpu().numpy()
    )
    codes = codes.cpu().numpy()
    factors = factors.cpu().numpy()

    n_codes = codes.shape[1]
    hx = continuous.get_h(codes[:, 0].reshape(-1, 1), k=3)
    # I(c; Z) if mi_improved
    normalize_value = (
        np.array(
            [
                [mutual_info(codes[:, i], factors, discrete_y=True)]
                for i in range(n_codes)
            ]
        )
        if mi_improved
        else np.sum(mutual_infos, axis=1, keepdims=True)
    )

    mutual_infos_normed = mutual_infos / (normalize_value + epsilon)
    mi_max = np.max(mutual_infos_normed, axis=1, keepdims=True)
    # score for each code
    scores = (1.0 / (factors.shape[1] - 1)) * np.sum(
        mi_max**2 - mutual_infos_normed**2, axis=1
    )
    score = np.mean(scores)
    return score


def dcii_c(factors, codes, epsilon=10e-8, mi_improved=True):
    """
    Compute completeness

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    mutual_infos = (
        get_mutual_infos(codes, factors)
        if mi_improved
        else calc_mutual_infos(codes, factors).cpu().numpy()
    )
    n_factors = factors.shape[1]
    # I(C; z) if mi_improved
    normalize_value = (
        np.array(
            [[mutual_info(codes, factors[:, j]) for j in range(n_factors)]]
        )
        if mi_improved
        else np.sum(mutual_infos, axis=0, keepdims=True)
    )

    mutual_infos_normed = mutual_infos / (normalize_value + epsilon)
    mi_max = np.max(mutual_infos_normed, axis=0, keepdims=True)
    # scores for each factor
    scores = (1.0 / (codes.shape[1] - 1)) * np.sum(
        mi_max**2 - mutual_infos_normed**2, axis=0
    )
    score = np.mean(scores)

    return score


def dcii_i(factors, codes, epsilon=10e-8, mi_improved=True):
    """
    Compute informativeness

    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        score
    """
    # mutual_info matrix (n_codes, n_factors)
    mutual_infos = (
        get_mutual_infos(codes, factors)
        if mi_improved
        else calc_mutual_infos(codes, factors).cpu().numpy()
    )
    # I(C; z)
    n_factors = factors.shape[1]
    captured = (
        np.array(
            [[mutual_info(codes, factors[:, j]) for j in range(n_factors)]]
        )
        if mi_improved
        else np.max(mutual_infos, axis=0)
    )
    entropy = (
        get_entropies(factors)
        if mi_improved
        else calc_entropy(factors, discretize=True).cpu().numpy()
    )
    score = np.mean(captured / (entropy + epsilon))
    return score
