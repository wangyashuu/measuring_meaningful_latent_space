import numpy as np
from sklearn.decomposition import PCA


"""
representation_functions
m1: a code capture no more than one factor
c1: a factor is captured by no more than one code
"""

n_classes = 9
import itertools

# 00 -> 0
# 01, 10 -> 1, 2
# 02, 11, 20 -> 3, 4, 5
# 03, 12, 21, 30 -> 6, 7, 8, 9
# 04, 13, 22, 31, 40 -> 10, 11, 12, 13, 14
# 1+2+ ... (i+j)  i+j


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


def information_pipeline(factors, informativeness=0.6):
    passed_factors = np.copy(factors)
    for i in range(factors.shape[1]):
        classes = np.unique(factors[:, i])
        n_droped_classes = int(informativeness * len(classes))
        passed = factors[:, i] > n_droped_classes
        passed_factors[:, i] = passed * factors[:, i]
    return passed_factors
    # p = np.random.uniform(0, 1, size=factors.shape)
    # passed_factors = (p < informativeness) * factors
    # return passed_factors

    # noised_factors = np.copy(factors)
    # for i in range(factors.shape[1]):
    #     noised = p[:, i] > informativeness
    #     noised_factors[:, i][noised] = np.random.randint(
    #         0, n_classes, size=(noised.sum(),)
    #     )
    # return noised_factors


def m0c0i1(factors):  # n_factors = 2, n_codes = 2
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


def m0c1i1(factors):  # n_factors = 3, n_codes = 2
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

    # classes_0 = np.unique(classes[:, 0])
    # classes_1 = np.unique(classes[:, 1])
    # def encode(factors):
    #     i = get_class_index_of(factors[:, 0:1], classes_0)
    #     j = get_class_index_of(factors[:, 1:2], classes_1)
    #     rows_to_counts = np.vectorize(lambda temp: np.arange(temp).sum())
    #     k = (rows_to_counts(i + j + 1) + i).reshape(-1, 1)
    #     return np.hstack([k, factors[:, 2:]])
    return encode


def m1c0i1(factors):  # n_factors = 2, n_codes = 3
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


def m1c1i1(factors):  # n_factors = 2, n_codes = 2
    return lambda factors: np.copy(factors)


def generate_factors(batch_size, n_factors=2):
    factors = [
        np.random.randint(0, n_classes, size=(batch_size, 1))
        for i in range(n_factors)
    ]
    factors = np.hstack(factors)
    return factors


def sample_factors(
    disentanglement, completeness, informativeness=True, batch_size=50000
):
    n_factors = 3 if not disentanglement and completeness else 2
    factors = generate_factors(batch_size, n_factors=n_factors)
    return factors


def get_representation_function(
    factors,
    disentanglement,
    completeness,
    informativeness,
):
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
    metric,
    disentanglement,
    completeness,
    informativeness,
    discrete_factors=False,
    batch_size=50000,
):
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
    factors = sample_factors(disentanglement, completeness, informativeness)
    encode = get_representation_function(
        factors, disentanglement, completeness, informativeness
    )
    codes = encode(factors)
    scores = metric(factors, codes, discrete_factors=discrete_factors)
    return scores
