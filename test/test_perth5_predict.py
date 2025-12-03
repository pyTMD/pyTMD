#!/usr/bin/env python
u"""
test_perth5_predict.py (12/2025)
Tests predictions against outputs from the NASA PERTH5 program

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 12/2025: added RE14 long-period test
    Updated 11/2025: using new xarray interface for tidal model data
    Written 09/2024
"""
import pytest
import pyTMD.io
import numpy as np
import timescale.time

# parametrize chunking
@pytest.mark.parametrize("CHUNKS", ["auto"])
def test_verify_GOT55(CHUNKS):
    # model parameters for GOT5.5
    m = pyTMD.io.model().from_database('GOT5.5')
    # open dataset and keep units as centimeters
    ds = m.open_dataset(group='z', chunks=CHUNKS, use_default_units=False)
    # test point from perth5 validation data
    MJD = np.array([45335.00000, 45335.04166667])
    lat = 59.195
    lon = -7.688
    # convert time from MJD to timescale object
    ts = timescale.time.Timescale(MJD=MJD)
    # transform coordinates to model grid
    X, Y = ds.tmd.transform(lon, lat, crs=4326)
    # extract amplitude and phase from tide model
    local = ds.tmd.interp(X, Y)
    # check constituent values
    assert np.isclose(local['q1'], 2.04 + 2.38j, atol=1e-2)
    assert np.isclose(local['o1'], 7.72 - 1.23j, atol=1e-2)
    assert np.isclose(local['p1'], -2.01 - 1.64j, atol=1e-2)
    assert np.isclose(local['s1'], 0.48 - 0.49j, atol=1e-2)
    assert np.isclose(local['k1'], -7.95 - 5.25j, atol=1e-2)
    assert np.isclose(local['n2'], -18.64 - 3.01j, atol=1e-2)
    assert np.isclose(local['m2'], -90.52 + 20.60j, atol=1e-2)
    assert np.isclose(local['s2'], -24.58 + 25.66j, atol=1e-2)
    assert np.isclose(local['k2'], -7.38 + 7.10j, atol=1e-2)
    assert np.isclose(local['m4'], 0.26 - 0.14j, atol=1e-2)
    assert np.isclose(local['ms4'], -0.06 + 0.14j, atol=1e-2)
    assert np.isclose(local['2n2'], -2.24 - 1.32j, atol=1e-2)
    assert np.isclose(local['mu2'], -2.65 - 2.41j, atol=1e-2)
    assert np.isclose(local['j1'], -0.33 - 0.15j, atol=1e-2)
    assert np.isclose(local['sigma1'], 0.03 + 0.70j, atol=1e-2)
    assert np.isclose(local['oo1'], -0.22 + 0.18j, atol=1e-2)
    # predict tidal elevations at time
    tide = local.tmd.predict(ts.tide, deltat=ts.tt_ut1, corrections='GOT')
    # infer semi-diurnal corrections
    tide += pyTMD.predict._infer_semi_diurnal(ts.tide, local,
        deltat=ts.tt_ut1, corrections='GOT')
    # infer diurnal corrections
    tide += pyTMD.predict._infer_diurnal(ts.tide, local,
        deltat=ts.tt_ut1, corrections='GOT')
    # GOT5.5 validation data from perth5
    validation = np.array([-92.96, -131.86])
    # will verify differences between model outputs are within tolerance
    eps = 0.03
    difference = tide - validation
    assert np.all(np.abs(difference) <= eps)

# parametrize chunking
@pytest.mark.parametrize("CHUNKS", ["auto"])
def test_RE14_long_period(CHUNKS):
    # model parameters for RE14
    m = pyTMD.io.model().from_database('RE14', group='z')
    # open dataset and keep units as centimeters
    ds = m.open_dataset(group='z', chunks=CHUNKS, use_default_units=False)
    # test point from perth5 validation data
    MJD = np.array([45335.00000, 45335.04166667])
    lat = 59.195
    lon = -7.688
    # convert time from MJD to timescale object
    ts = timescale.time.Timescale(MJD=MJD)
    # transform coordinates to model grid
    X, Y = ds.tmd.transform(lon, lat, crs=4326)
    # interpolate model to points
    local = ds.tmd.interp(X, Y)
    # check constituent values
    assert np.isclose(local['ssa'], -0.987 - 0.022j, atol=1e-3)
    assert np.isclose(local['sa'], -0.154 - 0.011j, atol=1e-3)
    assert np.isclose(local['mm'], -1.035 + 0.093j, atol=1e-3)
    assert np.isclose(local['mf'], -1.813 + 0.055j, atol=1e-3)
    assert np.isclose(local['mt'], -0.348 - 0.024j, atol=1e-3)
    assert np.isclose(local['node'], -0.908 + 0.000j, atol=1e-3)
    # predict tidal elevations at time and infer minor corrections
    tide = local.tmd.predict(ts.tide, deltat=ts.tt_ut1, corrections='GOT')
    tide += local.tmd.infer(ts.tide, deltat=ts.tt_ut1, corrections='GOT')
    # long period validation data from perth5
    validation = np.array([1.06, 1.03])
    # will verify differences between model outputs are within tolerance
    eps = 0.03
    difference = tide - validation
    assert np.all(np.abs(difference) <= eps)

def test_FES2014_long_period():
    # model parameters for FES2014
    m = pyTMD.io.model().from_database('FES2014', group='z')
    # reduce to long-period constituents for test
    constituents = ['ssa','mm','msf','mf','mtm']
    m.reduce_constituents(constituents)
    # open dataset and append 18.6 year node tide
    ds = m.open_dataset(type='z', append_node=True)
    ds = ds.rename(dict(mtm='mt'))
    # convert to centimeters
    ds = ds.tmd.to_units('cm')
    # Love numbers for long-period tides (Wahr, 1981)
    k2 = 0.299
    h2 = 0.606
    # tilt factor: response with respect to the solid earth
    gamma_2 = (1.0 + k2 - h2)
    # scale node equilibrium tide to match gamma_2 used in PERTH5
    perth5_gamma_2 = 0.682
    ds['node'] *= perth5_gamma_2 / gamma_2
    # test point from perth5 validation data
    MJD = np.array([45335.00000, 45335.04166667])
    lat = 59.195
    lon = -7.688
    # convert time from MJD to timescale object
    ts = timescale.time.Timescale(MJD=MJD)
    # transform coordinates to model grid
    X, Y = ds.tmd.transform(lon, lat, crs=4326)
    # interpolate model to points
    local = ds.tmd.interp(X, Y)
    # check constituent values
    assert np.isclose(local['ssa'], -1.000 - 0.009j, atol=1e-3)
    assert np.isclose(local['mm'], -1.088 + 0.099j, atol=1e-3)
    assert np.isclose(local['msf'], -0.055 + 0.045j, atol=1e-3)
    assert np.isclose(local['mf'], -1.865 + 0.088j, atol=1e-3)
    assert np.isclose(local['mt'], -0.350 - 0.016j, atol=1e-3)
    assert np.isclose(local['node'], -0.728 - 0.000j, atol=1e-3)
    # predict tidal elevations at time and infer minor corrections
    tide = local.tmd.predict(ts.tide, deltat=ts.tt_ut1, corrections='GOT')
    tide += local.tmd.infer(ts.tide, deltat=ts.tt_ut1, corrections='GOT')
    # long period validation data from perth5
    validation = np.array([1.16, 1.13])
    # will verify differences between model outputs are within tolerance
    eps = 0.03
    difference = tide - validation
    assert np.all(np.abs(difference) <= eps)

@pytest.mark.parametrize("METHOD", ['linear', 'admittance'])
def test_infer_minor(METHOD):
    # model parameters for GOT5.5
    m = pyTMD.io.model().from_database('GOT5.5')
    # open dataset and keep units as centimeters
    ds = m.open_dataset(group='z', chunks='auto', use_default_units=False)
    # test point from perth5 validation data
    MJD = np.array([45335.00000, 45335.04166667])
    lat = 59.195
    lon = -7.688
    # convert time from MJD to timescale object
    ts = timescale.time.Timescale(MJD=MJD)
    # transform coordinates to model grid
    X, Y = ds.tmd.transform(lon, lat, crs=4326)
    # interpolate model to points
    local = ds.tmd.interp(X, Y)
    # expected results
    exp_d = {}
    exp_sd = {}
    exp_d['linear'] = np.array([1.12412, 1.19974])
    exp_sd['linear'] = np.array([-0.75903, -1.40334])
    exp_d['admittance'] = np.array([1.14819, 1.23395])
    exp_sd['admittance'] = np.array([-0.84881, -1.36020])
    # infer minor constituents using interpolation method
    d = pyTMD.predict._infer_diurnal(ts.tide, local,
        deltat=ts.tt_ut1, method=METHOD)
    sd = pyTMD.predict._infer_semi_diurnal(ts.tide, local,
        deltat=ts.tt_ut1, method=METHOD)
    # check against expected values
    assert np.allclose(d, exp_d[METHOD], atol=1e-5)
    assert np.allclose(sd, exp_sd[METHOD], atol=1e-5)
