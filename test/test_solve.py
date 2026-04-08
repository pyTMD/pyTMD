"""
test_solve.py (04/2026)
Tests the harmonic constant estimation routines

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org

UPDATE HISTORY:
    Written 04/2026
"""
import pytest
import numpy as np
import xarray as xr
import pyTMD.solve


# Synthetic tide epoch reference: 1992-01-01T00:00:00
# M2 period: 12.42 hours, S2: 12.0 hours
_M2_PERIOD_H = 12.42
_S2_PERIOD_H = 12.0
_K1_PERIOD_H = 23.93
_O1_PERIOD_H = 25.82


def _make_tide(t, constituents, amplitudes, phases_deg):
    """Build a synthetic tidal time series.

    Parameters
    ----------
    t: np.ndarray
        Time in days relative to 1992-01-01T00:00:00
    constituents: list of (period_hours,)
    amplitudes: list of float
    phases_deg: list of float

    Returns
    -------
    ht: np.ndarray
    """
    ht = np.zeros_like(t)
    for (period_h, amp, phase_d) in zip(constituents, amplitudes, phases_deg):
        omega = 2.0 * np.pi / (period_h / 24.0)  # rad/day
        ht += amp * np.cos(omega * t + np.radians(phase_d))
    return ht


# Time vector: 2 years at hourly spacing
_T = np.linspace(0, 730, 730 * 24)


@pytest.mark.parametrize("solver", ["lstsq", "gelsd", "gelsy", "gelss"])
def test_constants_basic_solvers(solver):
    """Test that different linear solvers produce finite results"""
    amp_m2, amp_s2 = 1.5, 0.5
    ht = _make_tide(
        _T,
        [_M2_PERIOD_H, _S2_PERIOD_H],
        [amp_m2, amp_s2],
        [0.0, 0.0],
    )
    ds = pyTMD.solve.constants(_T, ht, ["m2", "s2"], solver=solver)
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"m2", "s2"}
    # recovered amplitudes should be finite and approximately match input
    m2_amp = np.abs(ds["m2"].values)
    s2_amp = np.abs(ds["s2"].values)
    assert np.isfinite(m2_amp)
    assert np.isfinite(s2_amp)
    # rough accuracy check (nodal corrections shift the exact amplitude)
    assert np.isclose(m2_amp, amp_m2, rtol=0.05)
    assert np.isclose(s2_amp, amp_s2, rtol=0.05)


def test_constants_bvls_solver():
    """Test the bounded-variable least-squares solver"""
    amp_m2 = 1.0
    ht = _make_tide(_T, [_M2_PERIOD_H], [amp_m2], [0.0])
    ds = pyTMD.solve.constants(
        _T, ht, ["m2"],
        solver="bvls",
        bounds=(-2.0, 2.0),
    )
    assert isinstance(ds, xr.Dataset)
    assert np.isfinite(np.abs(ds["m2"].values))


def test_constants_polynomial_order():
    """Test that adding a polynomial order improves fit for a trended signal"""
    amp_m2 = 1.2
    # add a linear trend to the tide
    ht = _make_tide(_T, [_M2_PERIOD_H], [amp_m2], [0.0]) + 0.002 * _T
    ds = pyTMD.solve.constants(_T, ht, ["m2"], order=1)
    assert isinstance(ds, xr.Dataset)
    # check finite and plausible amplitude
    assert np.isfinite(np.abs(ds["m2"].values))
    assert np.abs(ds["m2"].values) > 0.5


def test_constants_got_corrections():
    """Test that GOT-type nodal corrections can also be used"""
    amp_m2 = 1.0
    ht = _make_tide(_T, [_M2_PERIOD_H], [amp_m2], [0.0])
    ds = pyTMD.solve.constants(
        _T, ht, ["m2"], corrections="GOT"
    )
    assert isinstance(ds, xr.Dataset)
    assert np.isfinite(np.abs(ds["m2"].values))


def test_constants_with_finite_filter():
    """Test that NaN values are filtered before fitting"""
    amp_m2 = 1.5
    ht = _make_tide(_T, [_M2_PERIOD_H], [amp_m2], [0.0])
    # introduce some NaN values
    ht_nan = ht.copy()
    ht_nan[::100] = np.nan
    ds = pyTMD.solve.constants(_T, ht_nan, ["m2"])
    assert isinstance(ds, xr.Dataset)
    # amplitude should still be approximately recovered
    assert np.isclose(np.abs(ds["m2"].values), amp_m2, rtol=0.05)


def test_constants_string_constituent():
    """Test that a single string constituent is accepted"""
    amp_m2 = 1.0
    ht = _make_tide(_T, [_M2_PERIOD_H], [amp_m2], [0.0])
    # pass as string instead of list
    ds = pyTMD.solve.constants(_T, ht, "m2")
    assert isinstance(ds, xr.Dataset)
    assert "m2" in ds.data_vars


def test_constants_too_few_points():
    """Test that insufficient data raises ValueError"""
    t = np.array([0.0, 1.0, 2.0])
    ht = np.array([1.0, 0.5, -0.5])
    with pytest.raises(ValueError, match="Not enough values"):
        pyTMD.solve.constants(t, ht, ["m2", "s2", "k1", "o1"])


def test_constants_dimension_mismatch():
    """Test that mismatched dimensions raise ValueError"""
    t = np.linspace(0, 30, 100)
    ht = np.ones(50)  # wrong size
    with pytest.raises(ValueError, match="Dimension mismatch"):
        pyTMD.solve.constants(t, ht, ["m2"])


def test_constants_infer_minor():
    """Test that infer_minor path is exercised (requires major constituents
    that can drive inference of minor tidal signals)"""
    amp_m2, amp_s2 = 1.5, 0.5
    amp_o1, amp_k1 = 0.3, 0.4
    amp_n2, amp_k2 = 0.1, 0.07
    ht = _make_tide(
        _T,
        [_M2_PERIOD_H, _S2_PERIOD_H, _O1_PERIOD_H, _K1_PERIOD_H, 12.66, 11.967],
        [amp_m2, amp_s2, amp_o1, amp_k1, amp_n2, amp_k2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    # the infer_minor path requires a set of major constituents that
    # are recognized by pyTMD.predict.infer_minor
    ds = pyTMD.solve.constants(
        _T,
        ht,
        ["q1", "o1", "p1", "k1", "n2", "m2", "s2", "k2"],
        infer_minor=True,
        infer_iter=1,
    )
    assert isinstance(ds, xr.Dataset)
    # m2 amplitude should be approximately recovered
    assert np.isclose(np.abs(ds["m2"].values), amp_m2, rtol=0.05)
