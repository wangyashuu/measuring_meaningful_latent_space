import math

from .utils import get_scores
from .dci import dci


def test_m0c0i0():
    scores = get_scores(dci, False, False, False)
    assert math.isclose(scores["d"], 0, abs_tol=0.2)  # 0.0001
    assert math.isclose(scores["c"], 0, abs_tol=0.2)  # 0.0006
    assert math.isclose(scores["i"], 0, abs_tol=0.2)  # 0.4795


def test_m0c0i1():
    scores = get_scores(dci, False, False, True)
    assert math.isclose(scores["d"], 0, abs_tol=0.3)  # 0.0026
    assert math.isclose(scores["c"], 0, abs_tol=0.3)  # 0.0033
    assert math.isclose(scores["i"], 1, abs_tol=0.2)  # 1.0


def test_m0c1i0():
    scores = get_scores(dci, False, True, False)
    assert math.isclose(scores["d"], 0.5, abs_tol=0.1)  # 0.5695
    assert math.isclose(scores["c"], 1.0, abs_tol=0.1)  # 0.9914
    assert math.isclose(scores["i"], 0.6, abs_tol=0.1)  # 0.6435


def test_m0c1i1():
    scores = get_scores(dci, False, True, True)
    assert math.isclose(scores["d"], 0.5, abs_tol=0.5)  # 0.5793
    assert math.isclose(scores["c"], 1.0, abs_tol=0.2)  # 0.9999
    assert math.isclose(scores["i"], 1.0, abs_tol=0.2)  # 1.0


def test_m1c0i0():
    scores = get_scores(dci, True, False, False)
    assert math.isclose(scores["d"], 1.0, abs_tol=0.1)  # 0.9973
    assert math.isclose(scores["c"], 0.5, abs_tol=0.2)  # 0.6828
    assert math.isclose(scores["i"], 0.6, abs_tol=0.1)  # 0.5929


def test_m1c0i1():
    scores = get_scores(dci, True, False, True)
    assert math.isclose(scores["d"], 1.0, abs_tol=0.1)  # 0.9999
    assert math.isclose(scores["c"], 0.5, abs_tol=0.2)  # 0.6846
    assert math.isclose(scores["i"], 1.0, abs_tol=0.1)  # 1.0


def test_m1c1i0():
    scores = get_scores(dci, True, True, False)
    assert math.isclose(scores["d"], 0.1, abs_tol=0.3)  # 0.9961
    assert math.isclose(scores["c"], 0.1, abs_tol=0.3)  # 0.9961
    assert math.isclose(scores["i"], 0.5, abs_tol=0.2)  # 0.6444


def test_m1c1i1():
    scores = get_scores(dci, True, True, True)
    assert math.isclose(scores["d"], 1.0, abs_tol=0.2)  # 0.9999
    assert math.isclose(scores["c"], 1.0, abs_tol=0.2)  # 0.9999
    assert math.isclose(scores["i"], 1.0, abs_tol=0.2)  # 1.0
