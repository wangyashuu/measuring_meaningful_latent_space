"""Kim metric from `Disentangling by Factorising <https://arxiv.org/abs/1802.05983>`

Adapted from: https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/factor_vae.py
"""
from typing import Callable, Optional
import numpy as np


def z_min_var(
    sample_factors: Callable[[Optional[int]], np.ndarray],
    factor2code: Callable[[np.ndarray], np.ndarray],
    n_votes: int = 1000,
    batch_size: int = 256,
    n_global_items: int = 70000,
) -> float:
    """Compute Kim metric.

    Args:
        sample_factors (Callable[[Optional[int]], np.ndarray]): Function to generate the corresponding number of factor samples.
        factor2code (Callable[[np.ndarray], np.ndarray]): Function to encode factors to latents.
        n_votes (int, optional): The number of repeat votes. Default: 1000.
        batch_size (int, optional): The batch size of generate factor samples in a vote. Default: 256.
        n_global_items (int, optional): The number of global factor samples to generate (The global factor samples are used to calculate the global factor variance. Default: 70000.

    Returns:
        score (float): The overall Kim metric.
    """
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
    n_votes: int,
    batch_size: int,
    active_dims: np.ndarray,
    global_variances: np.ndarray,
    sample_factors: Callable[[Optional[int]], np.ndarray],
    factor2code: Callable[[np.ndarray], np.ndarray],
):
    """Generate the votes used in Kim metric calculation.

    Args:
        n_votes (int): The number of repeat votes.
        batch_size (int): The batch size of generate factor samples in a vote.
        active_dims (np.ndarray): [Shape (n_factors, )] A 1-d array to tell if each dimension in a factor representation is active.
        global_variances (np.ndarray): [Shape (n_factors, )] A 1-d array to tell the variance of each dimension in a factor representation.
        sample_factors (Callable[[Optional[int]], np.ndarray]): Function to generate the corresponding number of factor samples.
        factor2code (Callable[[np.ndarray], np.ndarray]): Function to encode factors to latents.

    Returns:
        votes (np.ndarray): [Shape (n_factors, n_codes)] A matrix where ij entry represents the number of votes of code i when predicting factor j
    """
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
