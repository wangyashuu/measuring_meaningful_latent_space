import math

from .utils import run_metric, m0c0, m0c1, m1c0, m1c1, m0c1_duplicated_factors
from .mig_sup import mig_sup


def test_m0c0():
    score = run_metric(mig_sup, m0c0)
    assert math.isclose(score, 0, abs_tol=0.05)


def test_m0c1():
    score = run_metric(mig_sup, m0c1)
    assert not math.isclose(score, 1, abs_tol=0.2)


def test_m0c1_duplicated_factors():
    score = run_metric(mig_sup, m0c1_duplicated_factors)
    assert math.isclose(score, 0)


def test_m1c0():
    score = run_metric(mig_sup, m1c0)
    assert math.isclose(score, 1, abs_tol=0.05)


def test_m1c1():
    score = run_metric(mig_sup, m1c1)
    assert math.isclose(score, 1, abs_tol=0.05)