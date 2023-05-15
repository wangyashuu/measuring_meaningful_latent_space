import math

from .utils import get_scores
from .dcii import dcii


def test_m0c0i0():
    scores = get_scores(dcii, False, False, False)
    d_score, c_score, i_score = (
        scores["disentanglement"],
        scores["completeness"],
        scores["informativeness"],
    )
    assert math.isclose(d_score, 0.0, abs_tol=0.1)
    assert math.isclose(c_score, 0.0, abs_tol=0.1)
    assert math.isclose(i_score, 0.5, abs_tol=0.1)


def test_m0c0i1():
    scores = get_scores(dcii, False, False, True)
    d_score, c_score, i_score = (
        scores["disentanglement"],
        scores["completeness"],
        scores["informativeness"],
    )
    assert math.isclose(d_score, 0.0, abs_tol=0.1)
    assert math.isclose(c_score, 0.0, abs_tol=0.1)
    assert math.isclose(i_score, 1.0, abs_tol=0.1)


def test_m0c1i0():
    scores = get_scores(dcii, False, True, False)
    d_score, c_score, i_score = (
        scores["disentanglement"],
        scores["completeness"],
        scores["informativeness"],
    )
    assert math.isclose(d_score, 0.5, abs_tol=0.1)
    assert math.isclose(c_score, 1.0, abs_tol=0.1)
    assert math.isclose(i_score, 0.5, abs_tol=0.1)  # 0.4546


def test_m0c1i1():
    scores = get_scores(dcii, False, True, True)
    d_score, c_score, i_score = (
        scores["disentanglement"],
        scores["completeness"],
        scores["informativeness"],
    )
    assert math.isclose(d_score, 0.5, abs_tol=0.1)  # 0.4316
    assert math.isclose(c_score, 1.0, abs_tol=0.1)  # 1.0000
    assert math.isclose(i_score, 1.0, abs_tol=0.1)  # 0.9946


def test_m1c0i0():
    scores = get_scores(dcii, True, False, False)
    d_score, c_score, i_score = (
        scores["disentanglement"],
        scores["completeness"],
        scores["informativeness"],
    )
    assert math.isclose(d_score, 1.0, abs_tol=0.1)  # 0.9979
    assert math.isclose(c_score, 0.5, abs_tol=0.2)  # 0.6142
    assert math.isclose(i_score, 0.5, abs_tol=0.1)  # 0.4575


def test_m1c0i1():
    scores = get_scores(dcii, True, False, True)
    d_score, c_score, i_score = (
        scores["disentanglement"],
        scores["completeness"],
        scores["informativeness"],
    )
    assert math.isclose(d_score, 1.0, abs_tol=0.1)
    assert math.isclose(c_score, 0.5, abs_tol=0.1)
    assert math.isclose(i_score, 1.0, abs_tol=0.1)


def test_m1c1i0():
    scores = get_scores(dcii, True, True, False)
    d_score, c_score, i_score = (
        scores["disentanglement"],
        scores["completeness"],
        scores["informativeness"],
    )
    assert math.isclose(d_score, 1.0, abs_tol=0.1)
    assert math.isclose(c_score, 1.0, abs_tol=0.1)
    assert math.isclose(i_score, 0.5, abs_tol=0.1)


def test_m1c1i1():
    scores = get_scores(dcii, True, True, True)
    d_score, c_score, i_score = (
        scores["disentanglement"],
        scores["completeness"],
        scores["informativeness"],
    )
    assert math.isclose(d_score, 1.0, abs_tol=0.1)
    assert math.isclose(c_score, 1.0, abs_tol=0.1)
    assert math.isclose(i_score, 1.0, abs_tol=0.1)
