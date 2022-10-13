import math

from .utils import run_metric, m0c0, m0c1, m1c0, m1c1, m0c1_duplicated_factors
from .modularity import modularity


def test_m0c0():
    score = run_metric(modularity, m0c0)
    assert math.isclose(score, 0, abs_tol=0.05)


# def test_m0c1():
#     score = run_metric(modularity, m0c1)
#     assert math.isclose(score, 1, abs_tol=0.05)


def test_m0c1_duplicated_factors():
    score = run_metric(modularity, m0c1_duplicated_factors)
    assert not math.isclose(score, 1, abs_tol=0.2)


def test_m1c0():
    score = run_metric(modularity, m1c0)
    assert math.isclose(score, 1, abs_tol=0.05)


def test_m1c1():
    score = run_metric(modularity, m1c1)
    assert math.isclose(score, 1, abs_tol=0.05)