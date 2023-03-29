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

from .modularity import modularity


def test_m0c0i0():
    score = run_metric(modularity, m0c0i0)
    assert not math.isclose(score, 1, abs_tol=0.1)  # vary 0.0912 0.5 1.0 0.8848


def test_m0c0i1():
    score = run_metric(modularity, m0c0i1)
    # use bins mi, the score is unstable (0-.9)
    assert math.isclose(score, 0, abs_tol=0.2) # 0.0224


def test_m0c1i0():
    score = run_metric(modularity, m0c1i0)
    assert math.isclose(score, 0.5, abs_tol=0.2)  # 0.6669


def test_m0c1i1():
    score = run_metric(modularity, m0c1i1)
    assert math.isclose(score, 0.5, abs_tol=0.2)  # 0.6667


def test_m1c0i0():
    score = run_metric(modularity, m1c0i0)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9999


def test_m1c0i1():
    score = run_metric(modularity, m1c0i1)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9999


def test_m1c1i0():
    score = run_metric(modularity, m1c1i0)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9999


def test_m1c1i1():
    score = run_metric(modularity, m1c1i1)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9999
