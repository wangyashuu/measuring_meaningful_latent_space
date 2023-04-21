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

from .dcii import dcii_d, dcii_c, dcii_i


def test_m0c0i0():
    d_score = run_metric(dcii_d, m0c0i0)
    c_score = run_metric(dcii_c, m0c0i0)
    i_score = run_metric(dcii_i, m0c0i0)
    assert math.isclose(d_score, 0, abs_tol=0.2)  # 0.0008
    assert math.isclose(c_score, 0, abs_tol=0.2)  # 0.0002
    assert math.isclose(i_score, 0, abs_tol=0.2)  # 0.0000


def test_m0c0i1():
    d_score = run_metric(dcii_d, m0c0i1)
    c_score = run_metric(dcii_c, m0c0i1)
    i_score = run_metric(dcii_i, m0c0i1)
    assert math.isclose(d_score, 0, abs_tol=0.1)  # 0.0311
    assert math.isclose(c_score, 0, abs_tol=0.1)  # 0.0326
    assert math.isclose(i_score, 1, abs_tol=0.1)  # 0.9885


def test_m0c1i0():
    d_score = run_metric(dcii_d, m0c1i0)
    c_score = run_metric(dcii_c, m0c1i0)
    i_score = run_metric(dcii_i, m0c1i0)
    assert math.isclose(d_score, 0.3, abs_tol=0.2)  # 0 vs 0.3270
    assert math.isclose(c_score, 0.7, abs_tol=0.2)  # 1 vs 0.7470
    assert math.isclose(i_score, 0.5, abs_tol=0.2)  # 0.5 vs 0.6181


def test_m0c1i1():
    d_score = run_metric(dcii_d, m0c1i1)
    c_score = run_metric(dcii_c, m0c1i1)
    i_score = run_metric(dcii_i, m0c1i1)
    assert math.isclose(d_score, 0.5, abs_tol=0.2)  # 0 vs 0.4226
    assert math.isclose(c_score, 1, abs_tol=0.2)  # 1 vs 0.9990
    assert math.isclose(i_score, 1, abs_tol=0.2)  # 1 vs 0.9946


def test_m1c0i0():
    d_score = run_metric(dcii_d, m1c0i0)
    c_score = run_metric(dcii_c, m1c0i0)
    i_score = run_metric(dcii_i, m1c0i0)
    assert math.isclose(d_score, 0.7, abs_tol=0.2)  # 1 vs 0.7802
    assert math.isclose(c_score, 0.3, abs_tol=0.2)  # 0  vs 0.3202
    assert math.isclose(i_score, 0.5, abs_tol=0.2)  # 0.5 vs 0.5521


def test_m1c0i1():
    d_score = run_metric(dcii_d, m1c0i1)
    c_score = run_metric(dcii_c, m1c0i1)
    i_score = run_metric(dcii_i, m1c0i1)
    assert math.isclose(d_score, 1, abs_tol=0.1)  # 1 vs 0.9986
    assert math.isclose(c_score, 0.5, abs_tol=0.1)  # 0 vs 0.4226
    assert math.isclose(i_score, 1, abs_tol=0.1)  # 1 vs  0.9973


def test_m1c1i0():
    d_score = run_metric(dcii_d, m1c1i0)
    c_score = run_metric(dcii_c, m1c1i0)
    i_score = run_metric(dcii_i, m1c1i0)
    assert math.isclose(d_score, 0.7, abs_tol=0.2)  # 1 vs 0.7664
    assert math.isclose(c_score, 0.7, abs_tol=0.2)  # 1 vs 0.7441
    assert math.isclose(i_score, 0.5, abs_tol=0.2)  # 0.5 vs 0.5831


def test_m1c1i1():
    d_score = run_metric(dcii_d, m1c1i1)
    c_score = run_metric(dcii_c, m1c1i1)
    i_score = run_metric(dcii_i, m1c1i1)
    assert math.isclose(d_score, 1, abs_tol=0.1)  # 1 vs 0.9989
    assert math.isclose(c_score, 1, abs_tol=0.1)  # 1 vs 0.9987
    assert math.isclose(i_score, 0.9, abs_tol=0.1)  # 1 vs 0.9940
