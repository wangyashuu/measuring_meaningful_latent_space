import math

from .utils import get_scores
from .sap import sap


def test_m0c0i0():
    score = get_scores(sap, False, False, False, discrete_factors=True)
    assert math.isclose(score, 0.0, abs_tol=0.2)


def test_m0c0i1():
    score = get_scores(sap, False, False, True, discrete_factors=True)
    assert math.isclose(score, 0.0, abs_tol=0.1)


def test_m0c1i0():
    score = get_scores(sap, False, True, False, discrete_factors=True)
    assert math.isclose(score, 0.3, abs_tol=0.1)


def test_m0c1i1():
    score = get_scores(sap, False, True, True, discrete_factors=True)
    assert math.isclose(score, 1.0, abs_tol=0.2)


def test_m1c0i0():
    score = get_scores(sap, True, False, False, discrete_factors=True)
    assert math.isclose(score, 0.3, abs_tol=0.1)


def test_m1c0i1():
    score = get_scores(sap, True, False, True, discrete_factors=True)
    assert math.isclose(score, 0.5, abs_tol=0.1)


def test_m1c1i0():
    score = get_scores(sap, True, True, False, discrete_factors=True)
    assert math.isclose(score, 0.3, abs_tol=0.1)


def test_m1c1i1():
    score = get_scores(sap, True, True, True, discrete_factors=True)
    assert math.isclose(score, 1.0, abs_tol=0.2)  # 0.8923
