import math

from .utils import get_scores
from .dcimig import dcimig


def test_m0c0i0():
    score = get_scores(dcimig, False, False, False)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.0131


def test_m0c0i1():
    score = get_scores(dcimig, False, False, True)
    # use bins mi, the score is unstable (0-.9) else (0-.25)
    assert math.isclose(score, 0, abs_tol=0.2) # 0.0104


def test_m0c1i0():
    score = get_scores(dcimig, False, True, False)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.1849


def test_m0c1i1():
    score = get_scores(dcimig, False, True, True)
    assert math.isclose(score, 0.3, abs_tol=0.2)  # 0.3380


def test_m1c0i0():
    score = get_scores(dcimig, True, False, False)
    assert not math.isclose(score, 0.3, abs_tol=0.2)  # TODO: 0.3930


def test_m1c0i1():
    score = get_scores(dcimig, True, False, True)
    assert math.isclose(score, 0.9, abs_tol=0.2)  # TODO: 0.9954 vs 0.7570


def test_m1c1i0():
    score = get_scores(dcimig, True, True, False)
    assert math.isclose(score, 0.5, abs_tol=0.2)  # 0.5486


def test_m1c1i1():
    score = get_scores(dcimig, True, True, True)
    assert math.isclose(score, 1.0, abs_tol=0.1)  # 0.9919
