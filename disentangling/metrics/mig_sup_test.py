import math

from .utils import get_scores
from .mig_sup import mig_sup


def test_m0c0i0():
    score = get_scores(mig_sup, False, False, False)
    assert math.isclose(score, 0.0, abs_tol=0.2)  # 0.0293


def test_m0c0i1():
    score = get_scores(mig_sup, False, False, True)
    # use bins mi, the score is unstable (0-.9)
    assert math.isclose(score, 0.0, abs_tol=0.2)  # 0.0506


def test_m0c1i0():
    score = get_scores(mig_sup, False, True, False)
    assert math.isclose(score, 0.5, abs_tol=0.2)  # TODO: 0.3333 vs 0.4965


def test_m0c1i1():
    score = get_scores(mig_sup, False, True, True)
    assert math.isclose(score, 0.5, abs_tol=0.2)  # 0.5058


def test_m1c0i0():
    score = get_scores(mig_sup, True, False, False)
    # Mark: it does not measure informativeness
    # since it normalized by entropy of latents.
    assert math.isclose(score, 1, abs_tol=0.2)  # TODO: 0.5681 vs 0.9903


def test_m1c0i1():
    score = get_scores(mig_sup, True, False, True)
    assert math.isclose(score, 1, abs_tol=0.2)  #  0.9739


def test_m1c1i0():
    score = get_scores(mig_sup, True, True, False)
    # TODO: should not measure informativeness, mark, it does not measure informativeness
    assert math.isclose(score, 1, abs_tol=0.2)  # TODO: 0.6681 vs 0.9900


def test_m1c1i1():
    score = get_scores(mig_sup, True, True, True)
    assert math.isclose(score, 1, abs_tol=0.2)  # 0.9955
