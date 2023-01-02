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

from .dcii import dcii_d, dcii_c, dcii_i


def test_m0c0i0():
    d_score = run_metric(dcii_d, m0c0i0)
    assert math.isclose(d_score, 0, abs_tol=0.2)  # 0.0015
    c_score = run_metric(dcii_c, m0c0i0)
    assert math.isclose(c_score, 0, abs_tol=0.2)  # 0.0013
    i_score = run_metric(dcii_i, m0c0i0)
    assert math.isclose(i_score, 0, abs_tol=0.2)  # 0.0316


# def test_m0c0i1():
#     pass
#     # score = run_metric(dcimig, m0c0i1)
#     # assert math.isclose(score, 0, abs_tol=0.2)


def test_m0c1i0():
    d_score = run_metric(dcii_d, m0c1i0)
    assert math.isclose(d_score, 0, abs_tol=0.2)  # 0 vs 0.1562
    c_score = run_metric(dcii_c, m0c1i0)
    assert math.isclose(c_score, 1, abs_tol=0.2)  # 1 vs 0.9327
    i_score = run_metric(dcii_i, m0c1i0)
    assert math.isclose(i_score, 0.6, abs_tol=0.2)  # 0.5 vs 0.7139


def test_m0c1i1():
    d_score = run_metric(dcii_d, m0c1i1)
    assert math.isclose(d_score, 0, abs_tol=0.2)  # 0 vs 0.1568
    c_score = run_metric(dcii_c, m0c1i1)
    assert math.isclose(c_score, 1, abs_tol=0.2)  # 1 vs 0.9417
    i_score = run_metric(dcii_i, m0c1i1)
    assert math.isclose(i_score, 1, abs_tol=0.2)  # 0.9999


def test_m1c0i0():
    d_score = run_metric(dcii_d, m1c0i0)
    assert math.isclose(d_score, 1, abs_tol=0.2)  # 1 vs 0.8337
    c_score = run_metric(dcii_c, m1c0i0)
    assert math.isclose(c_score, 0, abs_tol=0.2)  # 0  vs 0.1768
    i_score = run_metric(dcii_i, m1c0i0)
    assert math.isclose(i_score, 0.6, abs_tol=0.2)  # 0.5 vs 0.7149


def test_m1c0i1():
    d_score = run_metric(dcii_d, m1c0i1)
    assert math.isclose(d_score, 1, abs_tol=0.2)  # 1 vs 0.8407
    c_score = run_metric(dcii_c, m1c0i1)
    assert math.isclose(c_score, 0, abs_tol=0.2)  # 0 vs 0.1800
    i_score = run_metric(dcii_i, m1c0i1)
    assert math.isclose(i_score, 1, abs_tol=0.2)  # 0.9999


def test_m1c1i0():
    d_score = run_metric(dcii_d, m1c1i0)
    assert math.isclose(d_score, 1, abs_tol=0.2)  # 1 vs 0.8322
    c_score = run_metric(dcii_c, m1c1i0)
    assert math.isclose(c_score, 1, abs_tol=0.2)  # 1 vs 0.8338
    i_score = run_metric(dcii_i, m1c1i0)
    assert math.isclose(i_score, 0.6, abs_tol=0.2)  # 0.5 vs 0.6827


def test_m1c1i1():
    d_score = run_metric(dcii_d, m1c1i1)
    assert math.isclose(d_score, 1, abs_tol=0.2)  # 1 vs 0.8358
    c_score = run_metric(dcii_c, m1c1i1)
    assert math.isclose(c_score, 1, abs_tol=0.2)  # 1 vs 0.8445
    i_score = run_metric(dcii_i, m1c1i1)
    assert math.isclose(i_score, 1, abs_tol=0.2)  # 0.9999


# def test_m0c1_PCA():
#     scores = run_metric(dci, m0c1_PCA, batch_size=1200)
#     assert math.isclose(scores["d"], 0, abs_tol=0.2)  # 1 vs 0.0133
#     assert math.isclose(scores["c"], 0, abs_tol=0.2)  # 1 vs 0.0220
#     assert math.isclose(scores["i"], 0, abs_tol=0.2)  # 1 vs 0.0833
