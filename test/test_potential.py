"""
test_potential.py (04/2026)
Tests prediction routines for gravity tides, tide-generating forces,
Earth orientation parameters, and length of day.

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Written 04/2026
"""
import pytest
import numpy as np
import xarray as xr
import pyTMD.predict
import pyTMD.predict.potential as _potential
import pyTMD.predict.polar_motion as _polar_motion


# ---------------------------------------------------------------------------
# Reference station and planetary positions (from IERS documentation)
# ---------------------------------------------------------------------------
_XYZ = xr.Dataset(
    data_vars=dict(X=4075578.385, Y=931852.890, Z=4801570.154)
)
_SXYZ = xr.Dataset(
    data_vars=dict(
        X=137859926952.015,
        Y=54228127881.435,
        Z=23509422341.696,
    )
)
_LXYZ = xr.Dataset(
    data_vars=dict(
        X=-179996231.920342,
        Y=-312468450.131567,
        Z=-169288918.592160,
    )
)
# factors for sun and moon (as used in IERS test suite)
_F2_SOLAR = 0.163271964478954
_F2_LUNAR = 0.321989090026845

# Time: days since tide epoch (1992-01-01T00:00:00)
_T = np.array([9862.5])  # ~2019-01-01


# ---------------------------------------------------------------------------
# Tests for generating_force()
# ---------------------------------------------------------------------------
class TestGeneratingForce:
    """Tests for pyTMD.predict.potential.generating_force"""

    def test_returns_dataset(self):
        """generating_force should return an xr.Dataset with X, Y, Z"""
        F = _potential.generating_force(_T, _XYZ, _SXYZ, _LXYZ)
        assert isinstance(F, xr.Dataset)
        assert set(F.data_vars) == {"X", "Y", "Z"}

    def test_units_attribute(self):
        """Output dataset should carry units attributes"""
        F = _potential.generating_force(_T, _XYZ, _SXYZ, _LXYZ)
        for var in ["X", "Y", "Z"]:
            assert "units" in F[var].attrs

    def test_finite_values(self):
        """All force components should be finite"""
        F = _potential.generating_force(_T, _XYZ, _SXYZ, _LXYZ)
        for var in ["X", "Y", "Z"]:
            assert np.all(np.isfinite(F[var].values))

    def test_nonzero_values(self):
        """At least some force components should be non-zero"""
        F = _potential.generating_force(_T, _XYZ, _SXYZ, _LXYZ)
        total = sum(float(F[v]) ** 2 for v in ["X", "Y", "Z"])
        assert total > 0.0

    def test_lmax_parameter(self):
        """Different lmax values should produce finite results"""
        for lmax in [2, 3, 4]:
            F = _potential.generating_force(_T, _XYZ, _SXYZ, _LXYZ, lmax=lmax)
            for var in ["X", "Y", "Z"]:
                assert np.all(np.isfinite(F[var].values))


# ---------------------------------------------------------------------------
# Tests for gravity_tide()
# ---------------------------------------------------------------------------
class TestGravityTide:
    """Tests for pyTMD.predict.potential.gravity_tide"""

    def test_returns_dataset(self):
        """gravity_tide should return an xr.Dataset with X, Y, Z"""
        G = _potential.gravity_tide(_T, _XYZ, _SXYZ, _LXYZ)
        assert isinstance(G, xr.Dataset)
        assert set(G.data_vars) == {"X", "Y", "Z"}

    def test_units_attribute(self):
        """Output dataset should carry units attributes"""
        G = _potential.gravity_tide(_T, _XYZ, _SXYZ, _LXYZ)
        for var in ["X", "Y", "Z"]:
            assert "units" in G[var].attrs

    def test_finite_values(self):
        """All gravity tide components should be finite"""
        G = _potential.gravity_tide(_T, _XYZ, _SXYZ, _LXYZ)
        for var in ["X", "Y", "Z"]:
            assert np.all(np.isfinite(G[var].values))

    def test_lmax_parameter(self):
        """Different lmax values should produce finite results"""
        for lmax in [2, 3]:
            G = _potential.gravity_tide(_T, _XYZ, _SXYZ, _LXYZ, lmax=lmax)
            for var in ["X", "Y", "Z"]:
                assert np.all(np.isfinite(G[var].values))

    def test_with_deltat(self):
        """gravity_tide should accept a non-zero deltat parameter"""
        G = _potential.gravity_tide(_T, _XYZ, _SXYZ, _LXYZ, deltat=0.001)
        for var in ["X", "Y", "Z"]:
            assert np.all(np.isfinite(G[var].values))

    def test_multiple_times(self):
        """gravity_tide should handle a vector of times"""
        t_vec = np.linspace(9800.0, 9900.0, 10)
        G = _potential.gravity_tide(t_vec, _XYZ, _SXYZ, _LXYZ)
        for var in ["X", "Y", "Z"]:
            assert np.all(np.isfinite(G[var].values))


# ---------------------------------------------------------------------------
# Tests for _out_of_phase functions (gravity-tide-specific versions in potential.py)
# ---------------------------------------------------------------------------
class TestOutOfPhase:
    """Tests for the out-of-phase correction helpers in potential.py"""

    def test_out_of_phase_diurnal_finite(self):
        """_out_of_phase_diurnal should return finite corrections"""
        G = _potential._out_of_phase_diurnal(_XYZ, _SXYZ, _F2_SOLAR)
        for var in ["X", "Y", "Z"]:
            assert np.all(np.isfinite(G[var].values))

    def test_out_of_phase_semidiurnal_finite(self):
        """_out_of_phase_semidiurnal should return finite corrections"""
        G = _potential._out_of_phase_semidiurnal(_XYZ, _SXYZ, _F2_SOLAR)
        for var in ["X", "Y", "Z"]:
            assert np.all(np.isfinite(G[var].values))

    def test_out_of_phase_wrapper(self):
        """_out_of_phase wrapper sums diurnal and semidiurnal contributions"""
        G_wrap = _potential._out_of_phase(_XYZ, _SXYZ, _LXYZ, _F2_SOLAR, _F2_LUNAR)
        G_d = _potential._out_of_phase_diurnal(_XYZ, _SXYZ, _F2_SOLAR)
        G_d += _potential._out_of_phase_diurnal(_XYZ, _LXYZ, _F2_LUNAR)
        G_sd = _potential._out_of_phase_semidiurnal(_XYZ, _SXYZ, _F2_SOLAR)
        G_sd += _potential._out_of_phase_semidiurnal(_XYZ, _LXYZ, _F2_LUNAR)
        G_expected = G_d + G_sd
        for var in ["X", "Y", "Z"]:
            assert np.allclose(G_wrap[var].values, G_expected[var].values)


# ---------------------------------------------------------------------------
# Tests for _frequency_dependence functions
# ---------------------------------------------------------------------------
class TestFrequencyDependence:
    """Tests for frequency-dependence correction helpers in potential.py"""

    def test_frequency_dependence_diurnal_finite(self):
        """_frequency_dependence_diurnal should return finite corrections"""
        MJD = _T + 48622.0
        G = _potential._frequency_dependence_diurnal(_XYZ, MJD)
        for var in ["X", "Y", "Z"]:
            assert np.all(np.isfinite(G[var].values))

    def test_frequency_dependence_long_period_finite(self):
        """_frequency_dependence_long_period should return finite corrections"""
        MJD = _T + 48622.0
        G = _potential._frequency_dependence_long_period(_XYZ, MJD)
        for var in ["X", "Y", "Z"]:
            assert np.all(np.isfinite(G[var].values))

    def test_frequency_dependence_wrapper(self):
        """_frequency_dependence wrapper sums diurnal and long-period"""
        MJD = _T + 48622.0
        G_wrap = _potential._frequency_dependence(_XYZ, MJD)
        G_d = _potential._frequency_dependence_diurnal(_XYZ, MJD)
        G_lp = _potential._frequency_dependence_long_period(_XYZ, MJD)
        G_expected = G_d + G_lp
        for var in ["X", "Y", "Z"]:
            assert np.allclose(G_wrap[var].values, G_expected[var].values)


# ---------------------------------------------------------------------------
# Tests for earth_orientation() from polar_motion.py
# ---------------------------------------------------------------------------
class TestEarthOrientation:
    """Tests for pyTMD.predict.polar_motion.earth_orientation"""

    def test_returns_dataset(self):
        """earth_orientation should return an xr.Dataset"""
        ds = _polar_motion.earth_orientation(_T)
        assert isinstance(ds, xr.Dataset)
        assert "dX" in ds.data_vars
        assert "dY" in ds.data_vars
        assert "dUT" in ds.data_vars

    def test_units_attributes(self):
        """Output variables should have units attributes"""
        ds = _polar_motion.earth_orientation(_T)
        assert ds["dX"].attrs.get("units") == "arcseconds"
        assert ds["dY"].attrs.get("units") == "arcseconds"
        assert ds["dUT"].attrs.get("units") == "seconds"

    def test_finite_values(self):
        """All EOP corrections should be finite"""
        ds = _polar_motion.earth_orientation(_T)
        for var in ["dX", "dY", "dUT"]:
            assert np.all(np.isfinite(ds[var].values))

    def test_small_magnitude(self):
        """EOP corrections should be small (sub-arcsecond scale)"""
        ds = _polar_motion.earth_orientation(_T)
        assert np.all(np.abs(ds["dX"].values) < 1.0)  # < 1 arcsecond
        assert np.all(np.abs(ds["dY"].values) < 1.0)
        assert np.all(np.abs(ds["dUT"].values) < 0.01)  # < 10 milliseconds

    def test_vector_times(self):
        """earth_orientation should work with a vector of times"""
        t_vec = np.linspace(9000.0, 9100.0, 50)
        ds = _polar_motion.earth_orientation(t_vec)
        for var in ["dX", "dY", "dUT"]:
            assert np.all(np.isfinite(ds[var].values))
            assert ds[var].shape[0] == 50

    def test_with_deltat(self):
        """earth_orientation should accept a non-zero deltat"""
        ds = _polar_motion.earth_orientation(_T, deltat=0.001)
        for var in ["dX", "dY", "dUT"]:
            assert np.all(np.isfinite(ds[var].values))


# ---------------------------------------------------------------------------
# Tests for length_of_day() from polar_motion.py
# ---------------------------------------------------------------------------
class TestLengthOfDay:
    """Tests for pyTMD.predict.polar_motion.length_of_day"""

    def test_returns_dataset(self):
        """length_of_day should return an xr.Dataset"""
        ds = _polar_motion.length_of_day(_T)
        assert isinstance(ds, xr.Dataset)
        assert "dUT" in ds.data_vars
        assert "dLOD" in ds.data_vars
        assert "period" in ds.data_vars

    def test_units_attributes(self):
        """Output variables should have units attributes"""
        ds = _polar_motion.length_of_day(_T)
        assert ds["dUT"].attrs.get("units") == "seconds"
        assert ds["dLOD"].attrs.get("units") == "seconds per day"
        assert ds["period"].attrs.get("units") == "days"

    def test_finite_values(self):
        """All LOD variables should be finite"""
        ds = _polar_motion.length_of_day(_T)
        for var in ["dUT", "dLOD"]:
            assert np.all(np.isfinite(ds[var].values))

    def test_positive_periods(self):
        """All constituent periods should be positive"""
        ds = _polar_motion.length_of_day(_T)
        assert np.all(ds["period"].values > 0.0)

    def test_vector_times(self):
        """length_of_day should handle a vector of times"""
        t_vec = np.linspace(9000.0, 9100.0, 20)
        ds = _polar_motion.length_of_day(t_vec)
        for var in ["dUT", "dLOD"]:
            assert np.all(np.isfinite(ds[var].values))

    def test_with_deltat(self):
        """length_of_day should accept a non-zero deltat"""
        ds = _polar_motion.length_of_day(_T, deltat=0.001)
        for var in ["dUT", "dLOD"]:
            assert np.all(np.isfinite(ds[var].values))
