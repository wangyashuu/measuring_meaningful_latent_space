"""SAP score from `Variational Inference of Disentangled Latent Concepts from Unlabeled Observations <https://arxiv.org/abs/1711.00848>`.

Part of Code is adapted from `disentanglement lib<https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/sap_score.py>`.
"""

from typing import Union, List
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def label_transformer(train_data):
    classes = np.unique(train_data)

    def transform(new_data):
        nonlocal classes
        new_classes = np.array(list(set(np.unique(new_data)) - set(classes)))
        classes = np.hstack([classes, new_classes])
        indices = (new_data == classes.reshape(-1, 1)).argmax(axis=0)
        return indices

    return transform


def get_score_matrix(
    codes: np.ndarray,
    factors: np.ndarray,
    discrete_factors: Union[List[bool], bool] = True,
    test_size: float = 0.2,
    random_state: Union[float, None] = None,
    **kwargs,
) -> np.ndarray:

    """Compute the relationship matrix for SAP.

    Part of Code is adapted from `disentanglement lib<https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/sap_score.py>`.

    Args:
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        discrete_factors (Union[List[bool], bool]): implies if each factor is discrete. Default: True.
        test_size (float, optional): The rate of test samples, between 0 and 1. Default: 0.3.
        random_state (Union[float, None], optional): Seed of random generator for sampling test data. Default: None.

    Returns:
        score matrix (np.ndarray): score matrix where ij entry represents the relationship between code i and factor j
    """

    n_factors, n_codes = factors.shape[1], codes.shape[1]
    if type(discrete_factors) is not list:
        discrete_factors = [discrete_factors] * n_factors
    X_train, X_test, y_train, y_test = train_test_split(
        codes, factors, test_size=test_size, random_state=random_state
    )
    score_matrix = np.zeros([n_codes, n_factors])
    for j, discrete_factor in enumerate(discrete_factors):
        y_j = y_train[:, j]
        if discrete_factor:
            y_j_test = y_test[:, j]
            transform = label_transformer(y_j)
            y_j_encoded = transform(y_j)[:, np.newaxis]
            y_j_test_encoded = transform(y_j_test)[:, np.newaxis]

        for i in range(n_codes):
            x_i = X_train[:, i]
            if discrete_factor:
                x_i_test = X_test[:, i]
                classifier = XGBClassifier(tree_method="gpu_hist")
                classifier.fit(
                    x_i[:, np.newaxis].astype(np.float32), y_j_encoded
                )
                pred = classifier.predict(
                    x_i_test[:, np.newaxis].astype(np.float32)
                )
                score_matrix[i, j] = np.mean(pred == y_j_test_encoded)
            else:
                cov_x_i_y_j = np.cov(x_i, y_j, ddof=1)
                var_x_i_y_j = cov_x_i_y_j[0, 1] ** 2
                var_x = cov_x_i_y_j[0, 0]
                var_y = cov_x_i_y_j[1, 1]
                if var_x > 1e-10:
                    score_matrix[i, j] = var_x_i_y_j * 1.0 / (var_x * var_y)
                else:
                    score_matrix[i, j] = 0.0
    return score_matrix


def sap(
    factors: np.ndarray,
    codes: np.ndarray,
    discrete_factors: Union[List[bool], bool] = True,
    test_size: float = 0.2,
    random_state: Union[float, None] = None,
    **kwargs,
) -> float:
    """Compute SAP score.

    Args:
        codes (np.ndarray): [Shape (batch_size, n_codes)] The latent codes.
        factors (np.ndarray): [Shape (batch_size, n_factors)] The real generative factors.
        discrete_factors (Union[List[bool], bool]): implies if each factor is discrete. Default: True.
        test_size (float, optional): The rate of test samples, between 0 and 1. Default: 0.3.
        random_state (Union[float, None], optional): Seed of random generator for sampling test data. Default: None.

    Returns:
        score (float): SAP score
    """
    matrix = get_score_matrix(
        codes,
        factors,
        discrete_factors=discrete_factors,
        test_size=test_size,
        random_state=random_state,
    )
    sorted = np.sort(matrix, axis=0)
    score = np.mean(sorted[-1, :] - sorted[-2, :])
    return score
