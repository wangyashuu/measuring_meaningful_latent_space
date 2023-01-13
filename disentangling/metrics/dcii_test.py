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


def test_m0c0i1():
    d_score = run_metric(dcii_d, m0c0i1, n_factors=2)
    assert math.isclose(d_score, 0, abs_tol=0.1)  # 0.0311
    c_score = run_metric(dcii_c, m0c0i1, n_factors=2)
    assert math.isclose(c_score, 0, abs_tol=0.1)  # 0.0326
    i_score = run_metric(dcii_i, m0c0i1, n_factors=2)
    assert math.isclose(i_score, 1, abs_tol=0.1)  # 0.9885


def test_m0c1i0():
    d_score = run_metric(dcii_d, m0c1i0)
    assert math.isclose(d_score, 0.3, abs_tol=0.1)  # 0 vs 0.1562 vs 0.3270
    c_score = run_metric(dcii_c, m0c1i0)
    assert math.isclose(c_score, 0.7, abs_tol=0.1)  # 1 vs 0.9327 vs 0.7470
    i_score = run_metric(dcii_i, m0c1i0)
    assert not math.isclose(i_score, 1, abs_tol=0.1)  # 0.5 vs 0.7139 vs 0.6181


def test_m0c1i1():
    d_score = run_metric(dcii_d, m0c1i1)
    assert not math.isclose(d_score, 1, abs_tol=0.1)  # 0 vs 0.1568 vs 0.4226
    c_score = run_metric(dcii_c, m0c1i1)
    assert math.isclose(c_score, 1, abs_tol=0.2)  # 1 vs 0.9417 vs 0.9990
    i_score = run_metric(dcii_i, m0c1i1)
    assert math.isclose(i_score, 1, abs_tol=0.2)  # 0.9999 vs 0.9946


def test_m1c0i0():
    # d_score = run_metric(dcii_d, m1c0i0)
    # assert math.isclose(d_score, 0.7, abs_tol=0.1)  # 1 vs 0.8337 vs 0.7244
    # c_score = run_metric(dcii_c, m1c0i0)
    # assert not math.isclose(c_score, 1, abs_tol=0.1)  # 0  vs 0.1768 vs 0.4548
    i_score = run_metric(dcii_i, m1c0i0)
    assert not math.isclose(i_score, 1, abs_tol=0.1)  # 0.5 vs 0.7149 vs 0.8321


def test_m1c0i1():
    d_score = run_metric(dcii_d, m1c0i1)
    assert math.isclose(d_score, 1, abs_tol=0.1)  # 1 vs 0.8407 vs 0.9986
    c_score = run_metric(dcii_c, m1c0i1)
    assert math.isclose(c_score, 0.6, abs_tol=0.1)  # 0 vs 0.1800 vs 0.6220
    i_score = run_metric(dcii_i, m1c0i1)
    assert math.isclose(i_score, 1, abs_tol=0.1)  # 1 vs 0.9999 vs 0.9973


def test_m1c1i0():
    # d_score = run_metric(dcii_d, m1c1i0)
    # assert math.isclose(d_score, 0.7, abs_tol=0.1)  # 1 vs 0.8322 vs 0.7664
    # c_score = run_metric(dcii_c, m1c1i0)
    # assert math.isclose(c_score, 0.7, abs_tol=0.1)  # 1 vs 0.8338 vs 0.7441
    i_score = run_metric(dcii_i, m1c1i0)
    assert math.isclose(i_score, 0.8, abs_tol=0.1)  # 0.5 vs 0.6827 vs 0.8188


def test_m1c1i1():
    d_score = run_metric(dcii_d, m1c1i1)
    assert math.isclose(d_score, 1, abs_tol=0.1)  # 1 vs 0.8358 vs 0.9989
    c_score = run_metric(dcii_c, m1c1i1)
    assert math.isclose(c_score, 1, abs_tol=0.1)  # 1 vs 0.8445 vs 0.9987
    i_score = run_metric(dcii_i, m1c1i1)
    # when mi_improved = false, i_score = 0.87
    # since discrete_x = false, discrete_y = true
    assert math.isclose(i_score, 0.9, abs_tol=0.1)  # 1 vs 0.9999 vs 0.9940


# def test_m0c1_PCA():
#     scores = run_metric(dci, m0c1_PCA, batch_size=1200)
#     assert math.isclose(scores["d"], 0, abs_tol=0.2)  # 1 vs 0.0133
#     assert math.isclose(scores["c"], 0, abs_tol=0.2)  # 1 vs 0.0220
#     assert math.isclose(scores["i"], 0, abs_tol=0.2)  # 1 vs 0.0833
