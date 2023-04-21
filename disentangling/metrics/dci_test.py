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

from .dci import dci


def test_m0c0i0():
    scores = run_metric(dci, m0c0i0)
    assert math.isclose(scores["d"], 0, abs_tol=0.2)  # 0.0063
    assert math.isclose(scores["c"], 0, abs_tol=0.2)  # 0.0117
    assert math.isclose(scores["i"], 0, abs_tol=0.2)  # 0.0333


def test_m0c0i1():
    scores = run_metric(dci, m0c0i1)
    assert math.isclose(scores["d"], 0, abs_tol=0.3)  # 0 vs 0.2227
    assert math.isclose(scores["c"], 0, abs_tol=0.3)  # 0 vs 0.2228
    assert math.isclose(scores["i"], 1, abs_tol=0.2)  # 1 vs 0.9551


def test_m0c1i0():
    scores = run_metric(dci, m0c1i0)
    # TODO: mark analysis
    assert math.isclose(scores["d"], 0.5, abs_tol=0.1)  # 0.4552 0.2037
    assert math.isclose(scores["c"], 1.0, abs_tol=0.1)  # 0.9904 0.4089
    assert math.isclose(scores["i"], 0.5, abs_tol=0.1)  # 0.5554
    # 30000 d: 0.3867 c: 0.7772
    # 30000 d: 0.2559 c: 0.5644


def test_m0c1i1():
    scores = run_metric(dci, m0c1i1)
    assert math.isclose(scores["d"], 0, abs_tol=0.5)  # 0.4996
    assert math.isclose(scores["c"], 1, abs_tol=0.2)  # 0.9993
    assert math.isclose(scores["i"], 1, abs_tol=0.2)  # 0.9994


def test_m1c0i0():
    scores = run_metric(dci, m1c0i0, batch_size=100000)
    assert math.isclose(scores["d"], 0.6, abs_tol=0.3)  # 0.4849 0.5748 0.8925
    assert math.isclose(scores["c"], 0.2, abs_tol=0.3)  # 0.2557 0.2998 0.4655
    assert math.isclose(scores["i"], 0.5, abs_tol=0.3)  # 0.5645


def test_m1c0i1():
    scores = run_metric(dci, m1c0i1)
    assert math.isclose(scores["d"], 1, abs_tol=0.2)    # 0.9987
    assert math.isclose(scores["c"], 0.5, abs_tol=0.2)  # 0.5004
    assert math.isclose(scores["i"], 1, abs_tol=0.2)    # 0.9987


def test_m1c1i0():
    scores = run_metric(dci, m1c1i0)
    assert math.isclose(scores["d"], 0.7, abs_tol=0.3)
    # 0.6587 0.8925 0.9104 0.5184
    assert math.isclose(scores["c"], 0.7, abs_tol=0.3)
    # 0.6644 0.4655 0.9104 0.5622
    assert math.isclose(scores["i"], 0.5, abs_tol=0.2)  # 0 vs 0.5564


def test_m1c1i1():
    scores = run_metric(dci, m1c1i1)
    assert math.isclose(scores["d"], 1, abs_tol=0.2)  # 1 vs 0.9897
    assert math.isclose(scores["c"], 1, abs_tol=0.2)  # 1 vs 0.9897
    assert math.isclose(scores["i"], 1, abs_tol=0.2)  # 1 vs 0.9848
