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
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.0912


def test_m0c0i1():
    pass
    # score = run_metric(mig, m0c0i1)
    # assert math.isclose(score, 0, abs_tol=0.2)


def test_m0c1i0():
    score = run_metric(modularity, m0c1i0)
    # TODO: mark analysis
    assert math.isclose(score, 0.5, abs_tol=0.2)  # 0.6660


def test_m0c1i1():
    score = run_metric(modularity, m0c1i1, n_factors=4)
    # TODO: mark analysis
    assert math.isclose(score, 0.5, abs_tol=0.2)  # 0.6660


def test_m1c0i0():
    score = run_metric(modularity, m1c0i0)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9989


def test_m1c0i1():
    score = run_metric(modularity, m1c0i1)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9990


def test_m1c1i0():
    score = run_metric(modularity, m1c1i0)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9989


def test_m1c1i1():
    score = run_metric(modularity, m1c1i1)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9990


def test_m0c1_PCA():
    score = run_metric(modularity, m0c1_PCA)
    assert math.isclose(score, 0.994, abs_tol=0.2)  # #TODO: 0  vs 0.994
