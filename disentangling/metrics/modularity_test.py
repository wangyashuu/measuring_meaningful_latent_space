import math

from .utils import get_scores
from .modularity import modularity


def test_m0c0i0():
    score = get_scores(modularity, False, False, False)
    assert not math.isclose(
        score, 1.0, abs_tol=0.1
    )  # TODO: vary 0.3677 0.1387 0.2794 0.3473


def test_m0c0i1():
    score = get_scores(modularity, False, False, True)
    # use bins mi, the score is unstable (0-.9)
    # TODO: 0.1644 0.1058 0.1456 0.4358
    assert math.isclose(score, 0, abs_tol=0.2)


def test_m0c1i0():
    score = get_scores(modularity, False, True, False)
    assert math.isclose(score, 0.6, abs_tol=0.2)  # 0.7527


def test_m0c1i1():
    score = get_scores(modularity, False, True, True)
    assert math.isclose(score, 0.6, abs_tol=0.2)  # 0.7510


def test_m1c0i0():
    score = get_scores(modularity, True, False, False)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9997


def test_m1c0i1():
    score = get_scores(modularity, True, False, True)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9998


def test_m1c1i0():
    score = get_scores(modularity, True, True, False)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9999


def test_m1c1i1():
    score = get_scores(modularity, True, True, True)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9999
