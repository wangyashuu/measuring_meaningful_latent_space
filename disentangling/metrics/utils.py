
"""This module is used to generate test case for metrics.

A small explaination about the name of representation_functions
- m1: A code capture no more than one factor. m0: A code capture two factors.
- c1: A factor is captured by no more than one code. c0: a factor is captured by two codes,
- i1: All information of factors is captured. i0: Half of information of factors is captured.
"""
from typing import Callable, Union, List
import numpy as np
import itertools

n_classes = 9

def generate_code_classes(n_classes_per_code, shuffle=False):
    n_codes = len(n_classes_per_code)
    iter = itertools.product(*[np.arange(n_c) for n_c in n_classes_per_code])
    classes = np.array(list(iter))
    if shuffle:
        re_indices = np.arange(len(classes))
        np.random.shuffle(re_indices)
        classes = classes[re_indices]
    return classes


def get_class_index_of(factors, classes):
    indices = np.stack(
        [(factors == classes[i]).all(axis=1) for i in range(len(classes))],
        axis=1,
    )
    indices = np.argmax(indices, axis=1)
    return indices


def information_pipeline(factors, informativeness=0.6, noised=False):
    """Filter the information."""
    if noised:
        p = np.random.uniform(0, 1, size=factors.shape)
        noised_factors = np.copy(factors)
        for i in range(factors.shape[1]):
            noised = p[:, i] > informativeness
            noised_factors[:, i][noised] = np.random.randint(
                0, n_classes, size=(noised.sum(),)
            )
        return noised_factors
    else:
        passed_factors = np.copy(factors)
        for i in range(factors.shape[1]):
            classes = np.unique(factors[:, i])
            n_droped_classes = int(informativeness * len(classes))
            passed = factors[:, i] > n_droped_classes
            passed_factors[:, i] = passed * factors[:, i]
        return passed_factors


def m0c0i1(factors):
    """Return the encode process which will encode n factors into n codes"""
    classes = np.unique(factors, axis=0)
    entangled_factors = classes
    entangled_classes, inverse = np.unique(
        entangled_factors, axis=0, return_inverse=True
    )
    n_classes_per_factor = [
        len(np.unique(entangled_classes[:, i]))
        for i in range(entangled_classes.shape[1])
    ]
    entangled_code_classes = generate_code_classes(
        n_classes_per_factor, shuffle=True
    )
    return lambda factors: entangled_code_classes[
        get_class_index_of(factors, entangled_classes)
    ]


def m0c1i1(factors):
    """Generate the encode process from the factors, and the encode process which will encode first two factors into one code."""
    # m0: BUT a code capture more than one factor
    # c1: a factor is captured by no more than one code
    classes = np.unique(factors, axis=0)

    entangled_factors = classes[:, 0:2]
    entangled_classes, inverse = np.unique(
        entangled_factors, axis=0, return_inverse=True
    )
    entangled_code_classes = generate_code_classes([len(entangled_classes)])

    def encode(factors):
        entangled_codes = entangled_code_classes[
            get_class_index_of(factors[:, 0:2], entangled_classes)
        ]
        other_codes = factors[:, 2:]
        return np.hstack([entangled_codes, other_codes])
    return encode


def m1c0i1(factors):
    """Generate the encode process from the factors, and the encode process which will encode first factor into two code."""
    # m1: a code capture no more than one factor
    # c0: BUT a factor is captured by more than one code
    classes = np.unique(factors, axis=0)
    entangled_factors = classes[:, 0:1]
    entangled_classes, inverse = np.unique(
        entangled_factors, axis=0, return_inverse=True
    )
    entangled_code_classes = generate_code_classes([3, 3])

    def encode(factors):
        entangled_codes = entangled_code_classes[
            get_class_index_of(factors[:, 0:1], entangled_classes)
        ]
        other_codes = factors[:, 1:]
        return np.hstack([entangled_codes, other_codes])

    return encode


def m1c1i1(factors):
    """Generate the encode process from the factors, and the encode process which will encode each factor into separate code."""
    return lambda factors: np.copy(factors)


def generate_factors(batch_size, n_factors=2):
    """Generate factor samples of size as `batch_size`, and the dimension of factor is n_factors."""
    factors = [
        np.random.randint(0, n_classes, size=(batch_size, 1))
        for i in range(n_factors)
    ]
    factors = np.hstack(factors)
    return factors


def sample_factors(
    disentanglement, completeness, informativeness=True, batch_size=50000
):
    """Generate factor samples according to the presence of disentanglement, completeness and informativeness.
    
    When it is not disentangled, the dimension of factor will be set to 3, otherwise 2.
    Since in the not disentangled case, the first two factor will be encoded into one code. Set the dimension to 3 so the dimension of codes can be more than one.
    """
    n_factors = 3 if not disentanglement and completeness else 2
    factors = generate_factors(batch_size, n_factors=n_factors)
    return factors


def get_representation_function(
    factors: np.ndarray,
    disentanglement: bool,
    completeness: bool,
    informativeness: bool,
):
    """Get the encode process according to the presence of disentanglement, completeness, informativeness.

    Args:
        factors (np.ndarray): The global factors.
        disentanglement (bool): Indicate the encoding process disentangle or not.
        completeness (bool): Indicate the encoding process complete or not.
        informativeness (bool): Indicate the encoding process informative or not.
    
    Returns:
        Callable[np.ndarray, np.ndarray]: The desired encoding process.
    """
    if disentanglement and completeness:
        encode = m1c1i1(factors)
    elif disentanglement and not completeness:
        encode = m1c0i1(factors)
    elif not disentanglement and completeness:
        encode = m0c1i1(factors)
    elif not disentanglement and not completeness:
        encode = m0c0i1(factors)
    if informativeness:
        return encode
    return lambda factors: encode(information_pipeline(factors))


def get_scores(
    metric: Callable,
    disentanglement: bool,
    completeness: bool,
    informativeness: bool,
    discrete_factors: Union[List[bool], bool]=False,
    batch_size=50000,
):
    """Run the metric of a desired encoding process indicated by disentanglement, completeness, informativeness.
    
    Args:
        metric (Callable): The target metric.
        disentanglement (bool): Indicate the encoding process disentangle or not.
        completeness (bool): Indicate the encoding process complete or not.
        informativeness (bool): Indicate the encoding process informative or not.
        discrete_factors (Union[List[bool], bool]): It implies if each factor is discrete. Default: True.
        batch_size (int): The batch size of generated factor samples.
    
    Returns:
        Any: The return values of the target metric where the input is the random generative factors and the latent codes encoded by the desired encoding process.
    """

    if metric.__name__ == "z_min_var":
        sample_func = lambda batch_size=10000: sample_factors(
            disentanglement, completeness, informativeness, batch_size
        )
        encode_func = get_representation_function(
            sample_func(10000),
            disentanglement,
            completeness,
            informativeness,
        )
        scores = metric(sample_factors=sample_func, factor2code=encode_func)
        return scores
    factors = sample_factors(disentanglement, completeness, informativeness, batch_size=batch_size)
    encode = get_representation_function(
        factors, disentanglement, completeness, informativeness
    )
    codes = encode(factors)
    scores = metric(factors, codes, discrete_factors=discrete_factors)
    return scores
