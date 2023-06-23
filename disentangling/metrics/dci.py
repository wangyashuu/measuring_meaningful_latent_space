import scipy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

# from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor


def dci_ad_hoc_model(model_name="XGBClassifier"):
    if model_name == "XGBClassifier":
        return XGBClassifier(tree_method="gpu_hist")
    elif model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier()
    elif model_name == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(
            max_iter=10, early_stopping=False
        )
    return NotImplementedError(f"dci_ad_hoc_model model_name = {model_name}")


"""
Implementation of Disentanglement, Completeness and Informativeness.
Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).

Adapted from: https://github.com/google-research/disentanglement_lib
"""


def label_transformer(train_data):
    classes = np.unique(train_data)

    def transform(new_data):
        nonlocal classes
        new_classes = np.array(list(set(np.unique(new_data)) - set(classes)))
        classes = np.hstack([classes, new_classes])
        indices = (new_data == classes.reshape(-1, 1)).argmax(axis=0)
        return indices

    return transform


def dci(factors, codes, test_size=0.3, random_state=None, **kwargs):
    return dci_gpu(
        factors,
        codes,
        test_size=test_size,
        random_state=random_state,
        **kwargs,
    )


def dci_importance_matrix(
    factors,
    codes,
    test_size=0.3,
    random_state=None,
    discrete_factors=True,
    **kwargs,
):
    """
    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        scores: Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    n_factors = factors.shape[1]
    if type(discrete_factors) is bool:
        discrete_factors = [discrete_factors] * n_factors
    n_codes = codes.shape[1]
    x_train, x_test, y_train, y_test = train_test_split(
        codes, factors, test_size=test_size, random_state=random_state
    )
    importances, train_accuracies, test_accuracies = [], [], []
    for i, discrete_factor in enumerate(discrete_factors):
        if discrete_factor:
            transform = label_transformer(y_train[:, i])
            y_train_encoded = transform(y_train[:, i])
            y_test_encoded = transform(y_test[:, i])
            model = dci_ad_hoc_model()
            model.fit(x_train, y_train_encoded)
            importances.append(np.abs(model.feature_importances_))
            train_accuracies.append(model.score(x_train, y_train_encoded))
            test_accuracies.append(model.score(x_test, y_test_encoded))
        else:
            model = XGBRegressor(tree_method="gpu_hist")
            model.fit(x_train, y_train[:, i])
            importances.append(np.abs(model.feature_importances_))
            train_accuracies.append(model.score(x_train, y_train[:, i]))
            test_accuracies.append(model.score(x_test, y_test[:, i]))

    importance_matrix = np.stack(importances, axis=1)
    return importance_matrix


def dci_gpu(
    factors,
    codes,
    test_size=0.3,
    random_state=None,
    discrete_factors=True,
    **kwargs,
):
    """
    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        scores: Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    n_factors = factors.shape[1]
    if type(discrete_factors) is bool:
        discrete_factors = [discrete_factors] * n_factors
    n_codes = codes.shape[1]
    x_train, x_test, y_train, y_test = train_test_split(
        codes, factors, test_size=test_size, random_state=random_state
    )
    importances, train_accuracies, test_accuracies = [], [], []
    for i, discrete_factor in enumerate(discrete_factors):
        if discrete_factor:
            transform = label_transformer(y_train[:, i])
            y_train_encoded = transform(y_train[:, i])
            y_test_encoded = transform(y_test[:, i])
            model = dci_ad_hoc_model()
            model.fit(x_train, y_train_encoded)
            importances.append(np.abs(model.feature_importances_))
            train_accuracies.append(model.score(x_train, y_train_encoded))
            test_accuracies.append(model.score(x_test, y_test_encoded))
        else:
            model = XGBRegressor(tree_method="gpu_hist")
            model.fit(x_train, y_train[:, i])
            importances.append(np.abs(model.feature_importances_))
            train_accuracies.append(model.score(x_train, y_train[:, i]))
            test_accuracies.append(model.score(x_test, y_test[:, i]))

    importance_matrix = np.stack(importances, axis=1)
    train_accuracy = np.mean(train_accuracies)
    test_accuracy = np.mean(test_accuracies)

    d_score = disentanglement(importance_matrix)
    c_score = completeness(importance_matrix)
    i_score = test_accuracy
    return dict(d=d_score, c=c_score, i=i_score, i_train=train_accuracy)


def dci_from_lib(factors, codes, test_size=0.3, random_state=None, **kwargs):
    """
    Args:
        factors: the real generative factors (batch_size, factor_dims).
        codes: the latent codes (batch_size, code_dims).
    Returns:
        scores: Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    x_train, x_test, y_train, y_test = train_test_split(
        codes, factors, test_size=test_size, random_state=random_state
    )
    return _compute_dci(x_train.T, y_train.T, x_test.T, y_test.T)


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}

    importance_matrix, train_err, test_err = compute_importance_gbt(
        mus_train, ys_train, mus_test, ys_test
    )
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    d = disentanglement(importance_matrix)
    c = completeness(importance_matrix)
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = d
    scores["completeness"] = c
    scores["i"] = test_err
    scores["d"] = d
    scores["c"] = c
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
        model = GradientBoostingClassifier()
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
