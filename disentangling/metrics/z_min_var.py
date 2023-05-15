import numpy as np

"""
Implementation of FactorVAE metric.
Based on "Disentangling by Factorising"
Reference Implementation: https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/factor_vae.py
"""


def z_min_var(
    sample_factors,
    factor2code,
    n_votes=1000,
    batch_size=256,
    n_global_items=70000,
):
    global_codes = factor2code(sample_factors(n_global_items))
    global_variances = np.var(global_codes, axis=0, ddof=1)

    size, n_codes = global_codes.shape
    active_dims = global_variances > 0

    train_votes = get_votes(
        n_votes,
        batch_size,
        active_dims,
        global_variances,
        sample_factors,
        factor2code,
    )

    classifier = np.argmax(train_votes, axis=0)
    # after training, the classifier tells you for each code, it capture which factor.
    code_indices = np.arange(n_codes)
    train_accuracy = (
        np.sum(train_votes[classifier, code_indices])
        * 1.0
        / np.sum(train_votes)
    )

    eval_votes = get_votes(
        n_votes,
        batch_size,
        active_dims,
        global_variances,
        sample_factors,
        factor2code,
    )

    eval_accuracy = (
        np.sum(eval_votes[classifier, code_indices]) * 1.0 / np.sum(eval_votes)
    )
    return eval_accuracy


def get_votes(
    n_votes,
    batch_size,
    active_dims,
    global_variances,
    sample_factors,
    factor2code,
):
    f = sample_factors(10)
    c = factor2code(f)
    n_factors = f.shape[1]
    n_codes = c.shape[1]
    votes = np.zeros((n_factors, n_codes))

    for i in range(n_votes):
        # fix a random selected factor
        fixed_index = np.random.randint(low=0, high=n_factors)
        factors = sample_factors(batch_size)
        factors[:, fixed_index] = factors[0, fixed_index]

        # get the index of the lowest variance in codes
        codes = factor2code(factors)
        local_variances = np.var(codes, axis=0, ddof=1)
        argmin = np.argmin(
            local_variances[active_dims] / global_variances[active_dims]
        )

        votes[fixed_index, argmin] += 1

    return votes
