import math

from .utils import (
    run_metric,
    m0c0i0,
    m0c0i1,
    m0c1i0,
    m0c1i1,
    m1c0i0,
    m1c0i1,
    m1c1i0,
    m1c1i1,
    m0c1_PCA,
)

from .dcimig import dcimig


def test_m0c0i0():
    score = run_metric(dcimig, m0c0i0)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.0001


def test_m0c0i1():
    score = run_metric(dcimig, m0c0i1)
    # use bins mi, the score is unstable (0-.9) else (0-.25)
    assert math.isclose(score, 0, abs_tol=0.2) # 0.0106


def test_m0c1i0():
    score = run_metric(dcimig, m0c1i0)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.0008


def test_m0c1i1():
    score = run_metric(dcimig, m0c1i1)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0


def test_m1c0i0():
    score = run_metric(dcimig, m1c0i0)
    assert not math.isclose(score, 1, abs_tol=0.2)  # 0.5820


def test_m1c0i1():
    score = run_metric(dcimig, m1c0i1)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9954


def test_m1c1i0():
    score = run_metric(dcimig, m1c1i0)
    assert not math.isclose(score, 1, abs_tol=0.2)  # 0.6097


def test_m1c1i1():
    score = run_metric(dcimig, m1c1i1)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9919
