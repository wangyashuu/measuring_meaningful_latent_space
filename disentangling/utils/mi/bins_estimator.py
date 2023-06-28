import numpy as np
from sklearn.metrics import mutual_info_score


def np_discretize(xs, n_bins=30):
    """Discretization based on histograms."""
    discretized = np.digitize(xs, np.histogram(xs, n_bins)[1][:-1])
    return discretized


def estimate_mutual_info(
    x: np.ndarray,
    y: np.ndarray,
    discrete_x: bool = True,
    discrete_y: bool = True,
    n_bins: int = 30,
):
    """Estimate mutual information by split clouds into bins.

    Args:
        x (np.ndarray): [Shape (n_samples, )] A target cloud with any number of samples.
        y (np.ndarray): [Shape (n_samples, )] Another target cloud that should have same number of samples as x.
        discrete_x (bool, optional): If cloud x is discrete. Default: True.
        discrete_y (bool, optional): If cloud y is discrete. Default: True.
        n_bins (int, optional): The number of bins to split into. Default: 30.

    Returns:
        float: The mutual information estimated by spliting clouds into bins.
    """

    # more info about calc mutual info between float.
    # - https://www.researchgate.net/post/Can_anyone_help_with_calculating_the_mutual_information_between_for_float_number,
    # - https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy

    discretized_x = x if discrete_x else np_discretize(x, n_bins)
    discretized_y = y if discrete_y else np_discretize(y, n_bins)
    return mutual_info_score(discretized_x, discretized_y)
