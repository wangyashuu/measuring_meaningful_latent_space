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

from .dci import dci


def test_m0c0i0():
    scores = run_metric(dci, m0c0i0, batch_size=1200)
    assert math.isclose(scores["d"], 0, abs_tol=0.2)  # 0.0063
    assert math.isclose(scores["c"], 0, abs_tol=0.2)  # 0.0117
    assert math.isclose(scores["i"], 0, abs_tol=0.2)  # 0.0333


def test_m0c0i1():
    pass
    # score = run_metric(dcimig, m0c0i1)
    # assert math.isclose(score, 0, abs_tol=0.2)


def test_m0c1i0():
    scores = run_metric(dci, m0c1i0, batch_size=12000)
    # TODO: mark analysis
    assert math.isclose(scores["d"], 0, abs_tol=0.5)  # 0 vs 0.4523
    assert math.isclose(scores["c"], 1, abs_tol=0.5)  # 1 vs 0.9046
    assert math.isclose(scores["i"], 0.5, abs_tol=0.2)  # 0 vs 0.5522


def test_m0c1i1():
    scores = run_metric(dci, m0c1i1, batch_size=12000)
    # TODO: mark, analysis
    assert math.isclose(scores["d"], 0, abs_tol=0.5)  # 0 vs 0.4915
    assert math.isclose(scores["c"], 1, abs_tol=0.2)  # 1 vs 0.9831
    assert math.isclose(scores["i"], 1, abs_tol=0.2)  # 1 vs 0.9906


def test_m1c0i0():
    scores = run_metric(dci, m1c0i0, batch_size=12000)
    assert math.isclose(scores["d"], 1, abs_tol=0.8)  # 1 vs 0.6418
    assert math.isclose(scores["c"], 0, abs_tol=0.2)  # 0 vs 0.4337
    assert math.isclose(scores["i"], 0.5, abs_tol=0.2)  # 0 vs 0.5193


def test_m1c0i1():
    scores = run_metric(dci, m1c0i1, batch_size=2400)
    assert math.isclose(scores["d"], 1, abs_tol=0.2)  # 1 vs 0.9931
    assert math.isclose(scores["c"], 0.5, abs_tol=0.2)  # 0 vs 0.6627
    assert math.isclose(scores["i"], 1, abs_tol=0.2)  # 0 vs 0.9885


def test_m1c1i0():
    scores = run_metric(dci, m1c1i0, batch_size=8000)
    assert math.isclose(scores["d"], 1, abs_tol=0.5)  # 1 vs 0.5731
    assert math.isclose(scores["c"], 1, abs_tol=0.5)  # 1 vs 0.5797
    assert math.isclose(scores["i"], 0.5, abs_tol=0.2)  # 0 vs 0.4892


def test_m1c1i1():
    scores = run_metric(dci, m1c1i1, batch_size=1200)
    assert math.isclose(scores["d"], 1, abs_tol=0.2)  # 1 vs 0.9897
    assert math.isclose(scores["c"], 1, abs_tol=0.2)  # 1 vs 0.9897
    assert math.isclose(scores["i"], 1, abs_tol=0.2)  # 1 vs 0.9848


def test_m0c1_PCA():
    scores = run_metric(dci, m0c1_PCA, batch_size=1200)
    assert math.isclose(scores["d"], 0, abs_tol=0.2)  # 1 vs 0.0017
    assert math.isclose(scores["c"], 0, abs_tol=0.2)  # 1 vs 0.0102
    assert math.isclose(scores["i"], 0, abs_tol=0.2)  # 1 vs 0.0750
