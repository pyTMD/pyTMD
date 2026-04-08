"""
test_ellipse.py (04/2026)
Tests the tidal ellipse parameter functions

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org

UPDATE HISTORY:
    Written 04/2026
"""
import pytest
import numpy as np
import pyTMD.ellipse


def test_ellipse_scalar():
    """Test ellipse() with scalar complex amplitudes"""
    # simple circular current (u = A*e^{i*phi}, v = A*e^{i*(phi+90)})
    u = 1.0 + 0.0j
    v = 0.0 + 1.0j
    major, minor, incl, phase = pyTMD.ellipse.ellipse(u, v)
    # for circular motion, major == minor, semi-axes should be 1
    assert np.isclose(major[0], 1.0)
    assert np.isclose(np.abs(minor[0]), 1.0)


def test_ellipse_rectilinear():
    """Test ellipse() for rectilinear (degenerate) motion along x-axis"""
    # purely east-west motion: u = A, v = 0
    u = np.array([2.0 + 0j])
    v = np.array([0.0 + 0j])
    major, minor, incl, phase = pyTMD.ellipse.ellipse(u, v)
    # for rectilinear x-motion: major = 1, minor = 0 (or equal-magnitude)
    # wp = u/2, wm = conj(u)/2 => ap = am = 1, major = 2, minor = 0
    assert np.isclose(major[0], 2.0)
    assert np.isclose(minor[0], 0.0, atol=1e-10)


def test_ellipse_array():
    """Test ellipse() with array inputs"""
    n = 10
    # random complex amplitudes
    rng = np.random.default_rng(42)
    u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    major, minor, incl, phase = pyTMD.ellipse.ellipse(u, v)
    # shapes should match input
    assert major.shape == (n,)
    assert minor.shape == (n,)
    assert incl.shape == (n,)
    assert phase.shape == (n,)
    # inclination should be in [0, 180)
    assert np.all(incl >= 0.0)
    assert np.all(incl < 180.0)
    # phase should be in [0, 360)
    assert np.all(phase >= 0.0)
    assert np.all(phase < 360.0)


def test_ellipse_inverse_roundtrip():
    """Test that inverse(ellipse(u, v)) recovers u and v"""
    rng = np.random.default_rng(7)
    u_orig = rng.standard_normal(20) + 1j * rng.standard_normal(20)
    v_orig = rng.standard_normal(20) + 1j * rng.standard_normal(20)
    major, minor, incl, phase = pyTMD.ellipse.ellipse(u_orig, v_orig)
    u_rec, v_rec = pyTMD.ellipse.inverse(major, minor, incl, phase)
    assert np.allclose(u_orig, u_rec)
    assert np.allclose(v_orig, v_rec)


def test_inverse_scalar():
    """Test inverse() with scalar inputs"""
    major = np.array([1.5])
    minor = np.array([0.5])
    incl = np.array([30.0])
    phase = np.array([45.0])
    u, v = pyTMD.ellipse.inverse(major, minor, incl, phase)
    # should return arrays
    assert u.shape == (1,)
    assert v.shape == (1,)
    # magnitudes should be finite
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))


def test_xy_full_ellipse():
    """Test _xy() without phase returns a full ellipse curve"""
    major = 2.0
    minor = 1.0
    incl = 0.0
    x, y = pyTMD.ellipse._xy(major, minor, incl, N=360)
    # should have 360 points
    assert len(x) == 360
    assert len(y) == 360
    # maximum should be close to semi-major/semi-minor
    assert np.isclose(np.max(np.abs(x)), major, rtol=0.01)
    assert np.isclose(np.max(np.abs(y)), minor, rtol=0.01)


def test_xy_with_phase():
    """Test _xy() with a specific phase value"""
    major = 1.0
    minor = 0.5
    incl = 45.0
    phase = 0.0
    x, y = pyTMD.ellipse._xy(major, minor, incl, phase=phase)
    # should return single-point arrays
    assert len(x) == 1
    assert len(y) == 1
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))


def test_xy_with_center():
    """Test _xy() with a non-zero center"""
    major = 1.0
    minor = 0.5
    incl = 0.0
    center = (3.0, 4.0)
    x0, y0 = pyTMD.ellipse._xy(major, minor, incl, xy=(0.0, 0.0))
    x1, y1 = pyTMD.ellipse._xy(major, minor, incl, xy=center)
    # shifted ellipse should be offset by center
    assert np.allclose(x1, x0 + center[0])
    assert np.allclose(y1, y0 + center[1])


def test_ellipse_positive_minor():
    """Test that minor axis sign is consistent with ellipse rotation sense

    In pyTMD's complex amplitude convention, a positive rotating (counterclockwise)
    current has the form u=A, v=-iA, which yields minor > 0.
    """
    # positive rotating (counterclockwise): from inverse with wm=0
    # u = wp + conj(wm) = wp, v = -1j*(wp - conj(wm)) = -1j*wp
    # so for wp = A (real): u = A, v = -i*A
    u = np.array([1.0 + 0j])
    v = np.array([0.0 - 1.0j])
    major, minor, incl, phase = pyTMD.ellipse.ellipse(u, v)
    # counterclockwise rotation gives positive minor
    assert minor[0] > 0

    # negative rotating (clockwise): wp=0, wm = A
    # u = conj(wm) = A, v = -1j*(0 - A) = i*A
    u = np.array([1.0 + 0j])
    v = np.array([0.0 + 1.0j])
    major, minor, incl, phase = pyTMD.ellipse.ellipse(u, v)
    # clockwise rotation gives negative minor
    assert minor[0] < 0
