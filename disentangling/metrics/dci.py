import scipy
import numpy as np
from sklearn import ensemble

"""
Implementation of Disentanglement, Completeness and Informativeness.
Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).

Adapted from: https://github.com/google-research/disentanglement_lib
"""


def dci(factors, codes):
    """
    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        scores: Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    factors = factors.cpu().numpy()
    codes = codes.cpu().numpy()
    batch_size = factors.shape[0]
    train_size = int(batch_size * 0.8)
    x_train, y_train = codes[:train_size, :].T, factors[:train_size, :].T
    x_test, y_test = codes[train_size:, :].T, factors[train_size:, :].T
    return _compute_dci(x_train, y_train, x_test, y_test)


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(
        mus_train, ys_train, mus_test, ys_test
    )
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    scores["i"] = test_err
    scores["d"] = disentanglement(importance_matrix)
    scores["c"] = completeness(importance_matrix)
    return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(
        shape=[num_codes, num_factors], dtype=np.float64
    )
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = ensemble.GradientBoostingClassifier()
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1.0 - scipy.stats.entropy(
        importance_matrix.T + 1e-11, base=importance_matrix.shape[1]
    )


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1.0 - scipy.stats.entropy(
        importance_matrix + 1e-11, base=importance_matrix.shape[0]
    )


def completeness(importance_matrix):
    """ "Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.0:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)
