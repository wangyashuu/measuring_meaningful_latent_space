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

from .mig_sup import mig_sup


def test_m0c0i0():
    score = run_metric(mig_sup, m0c0i0)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0.0005


def test_m0c0i1():
    pass
    # score = run_metric(mig_sup, m0c0i1)
    # assert math.isclose(score, 0, abs_tol=0.2)


def test_m0c1i0():
    score = run_metric(mig_sup, m0c1i0)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0


def test_m0c1i1():
    score = run_metric(mig_sup, m0c1i1)
    assert math.isclose(score, 0, abs_tol=0.2)  # 0


def test_m1c0i0():
    score = run_metric(mig_sup, m1c0i0)
    # Mark: it does not measure informativeness
    # since it normalized by entropy of latents.
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9666


def test_m1c0i1():
    score = run_metric(mig_sup, m1c0i1)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.8441


def test_m1c1i0():
    score = run_metric(mig_sup, m1c1i0)
    # mark, it does not measure informativeness
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9661


def test_m1c1i1():
    score = run_metric(mig_sup, m1c1i1)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.8408
