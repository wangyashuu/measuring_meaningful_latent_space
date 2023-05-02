import math

from .utils import get_scores
from .sap import sap


def test_m0c0i0():
    score = get_scores(sap, False, False, False, discreted_factors=True)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0 0.0036


def test_m0c0i1():
    score = get_scores(sap, False, False, True, discreted_factors=True)
    assert not math.isclose(score, 1, abs_tol=0.2)  # 0.0036


def test_m0c1i0():
    score = get_scores(sap, False, True, False, discreted_factors=True)
    assert not math.isclose(score, 1, abs_tol=0.2)  # 0.5299


def test_m0c1i1():
    score = get_scores(sap, False, True, True, discreted_factors=True)
    assert math.isclose(score, 1.0, abs_tol=0.2)  # 0.8899


def test_m1c0i0():
    score = get_scores(sap, True, False, False, discreted_factors=True)
    assert math.isclose(score, 0.25, abs_tol=0.1)  # 0.2732


def test_m1c0i1():
    score = get_scores(sap, True, False, True, discreted_factors=True)
    assert math.isclose(score, 0.5, abs_tol=0.1)  # 0.4459


def test_m1c1i0():
    score = get_scores(sap, True, True, False, discreted_factors=True)
    assert math.isclose(score, 0.6, abs_tol=0.1)  # 0.532


def test_m1c1i1():
    score = get_scores(sap, True, True, True, discreted_factors=True)
    assert math.isclose(score, 1.0, abs_tol=0.2)  # 0.8923
