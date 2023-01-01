import numpy as np
from sklearn import svm

"""
Implementation of the SAP score.
Based on "Variational Inference of Disentangled Latent Concepts from Unlabeled
Observations" (https://openreview.net/forum?id=H1kG7GZAW), Section 3.

Reference Implementation: https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/sap_score.py

"""


def get_score_matrix(codes, factors, discreted_factor=False):
    n_codes = codes.shape[1]
    n_factors = factors.shape[1]
    batch_size = codes.shape[0]
    train_size = int(batch_size * 0.8)
    X_train, y_train = codes[:train_size], factors[:train_size]
    X_test, y_test = codes[train_size:], factors[train_size:]
    score_matrix = np.zeros([n_codes, n_factors])
    for i in range(n_codes):
        for j in range(n_factors):
            x_i = X_train[:, i]
            y_j = y_train[:, j]
            if discreted_factor:
                x_i_test = X_test[:, i]
                y_j_test = y_test[:, j]
                classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                classifier.fit(x_i[:, np.newaxis], y_j)
                pred = classifier.predict(x_i_test[:, np.newaxis])
                score_matrix[i, j] = np.mean(pred == y_j_test)
            else:
                cov_x_i_y_j = np.cov(x_i, y_j, ddof=1)
                var_x_i_y_j = cov_x_i_y_j[0, 1] ** 2
                var_x = cov_x_i_y_j[0, 0]
                var_y = cov_x_i_y_j[1, 1]
                if var_x > 1e-12:
                    score_matrix[i, j] = var_x_i_y_j * 1.0 / (var_x * var_y)
                else:
                    score_matrix[i, j] = 0.0
    return score_matrix


def sap(factors, codes):
    factors = factors.cpu().numpy()
    codes = codes.cpu().numpy()
    matrix = get_score_matrix(codes, factors)
    sorted = np.sort(matrix, axis=0)
    score = np.mean(sorted[-1, :] - sorted[-2, :])
    return score
