import math

from .utils import get_scores
from .mig import mig


def test_m0c0i0():
    score = get_scores(mig, False, False, False)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.0286


def test_m0c0i1():
    score = get_scores(mig, False, False, True)
    # use bins mi, the score is unstable (0-.9)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.0209


def test_m0c1i0():
    score = get_scores(mig, False, True, False)
    assert math.isclose(score, 0.5, abs_tol=0.2)  # 0.5770


def test_m0c1i1():
    score = get_scores(mig, False, True, True)
    assert math.isclose(score, 1.0, abs_tol=0.2)  # 1.0023


def test_m1c0i0():
    score = get_scores(mig, True, False, False)
    assert math.isclose(score, 0.25, abs_tol=0.1)  # 0.2771


def test_m1c0i1():
    score = get_scores(mig, True, False, True)
    assert math.isclose(score, 0.5, abs_tol=0.1)  # 0.5023


def test_m1c1i0():
    score = get_scores(mig, True, True, False)
    assert math.isclose(score, 0.6, abs_tol=0.1)  # 0.5504


def test_m1c1i1():
    score = get_scores(mig, True, True, True)
    assert math.isclose(score, 1.0, abs_tol=0.2)  # 1.0072
