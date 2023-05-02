import math

from .z_min_var import z_min_var
from .utils import get_scores


def test_m0c0i0():
    score = get_scores(z_min_var, False, False, False)
    assert math.isclose(score, 0.5, abs_tol=0.2)  # 1/n_factors


def test_m0c0i1():
    score = get_scores(z_min_var, False, False, True)
    assert math.isclose(score, 0.5, abs_tol=0.1)  # 0.5440


def test_m0c1i0():
    score = get_scores(z_min_var, False, True, False)
    assert math.isclose(score, 0.6, abs_tol=0.2)  # TODO: 0.464 vs 0.673


def test_m0c1i1():
    score = get_scores(z_min_var, False, True, True)
    assert math.isclose(score, 0.6, abs_tol=0.2)  # 0.6739


def test_m1c0i0():
    score = get_scores(z_min_var, True, False, False)
    assert math.isclose(score, 0.8, abs_tol=0.2)  # TODO: 0.7693 vs 1


def test_m1c0i1():
    score = get_scores(z_min_var, True, False, True)
    assert math.isclose(score, 1.0, abs_tol=0.2)  # 1


def test_m1c1i0():
    score = get_scores(z_min_var, True, True, False)
    assert math.isclose(score, 0.8, abs_tol=0.2)  # TODO: 0.681 vs 1


def test_m1c1i1():
    score = get_scores(z_min_var, True, True, True)
    assert math.isclose(score, 1.0, abs_tol=0.1)  # 1
