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
)

from .mig import mig


def test_m0c0i0():
    score = run_metric(mig, m0c0i0)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.0002


def test_m0c0i1():
    score = run_metric(mig, m0c0i1)
    # use bins mi, the score is unstable (0-.9)
    assert math.isclose(score, 0, abs_tol=0.2) # 0.0193


def test_m0c1i0():
    score = run_metric(mig, m0c1i0)
    assert not math.isclose(score, 1, abs_tol=0.2)  # 0.5770


def test_m0c1i1():
    score = run_metric(mig, m0c1i1)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9964


def test_m1c0i0():
    score = run_metric(mig, m1c0i0)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.0014


def test_m1c0i1():
    score = run_metric(mig, m1c0i1)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.0001


def test_m1c1i0():
    score = run_metric(mig, m1c1i0)
    assert not math.isclose(score, 1, abs_tol=0.2)  # 0.5783


def test_m1c1i1():
    score = run_metric(mig, m1c1i1)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9985
