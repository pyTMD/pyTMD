#!/usr/bin/env python
u"""
test_fes_predict.py (06/2026)
Tests that FES2014 data can be downloaded from AWS S3 bucket
Tests the read program to verify that constituents are being extracted
Tests that interpolated results are comparable to FES2014 program

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5netcdf: Pythonic interface to netCDF4 via h5py
        https://h5netcdf.org/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

UPDATE HISTORY:
    Updated 06/2026: add test to verify short-term admittance calculations
    Updated 11/2025: using new xarray interface for tidal model data
    Updated 10/2025: split directories between validation and model data
        fetch data from pyTMD developers test data repository
    Updated 09/2025: added check if running on GitHub Actions or locally
    Updated 06/2025: subset to specific constituents when reading model
    Updated 09/2024: drop support for the ascii definition file format
    Updated 07/2024: add parametrize over cropping the model fields
    Updated 04/2024: use timescale for temporal operations
    Updated 01/2024: test doodson and cartwright numbers of each constituent
    Updated 04/2023: using pathlib to define and expand paths
    Updated 12/2022: add check for read and interpolate constants
    Updated 09/2021: update check tide points to add compression flags
    Updated 05/2021: added test for check point program
    Updated 03/2021: use pytest fixture to setup and teardown model data
    Updated 02/2021: replaced numpy bool to prevent deprecation warning
    Written 08/2020
"""
import io
import json
import pytest
import inspect
import pathlib
import numpy as np
import xarray as xr
import pyTMD
import timescale

# current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
filepath = pathlib.Path(filename).absolute().parent

# PURPOSE: Tests check point program
def test_check_FES2014(directory):
    lons = np.arange(6)*60
    lats = np.zeros((6)) + 59.0
    obs = pyTMD.compute.tide_masks(lons, lats, directory=directory,
        model='FES2014', crs=4326)
    exp = np.array([True, False, False, True, False, True])
    assert np.all(obs == exp)

# parametrize over cropping the model fields
@pytest.mark.parametrize("CROP", [False, True])
# PURPOSE: Tests that interpolated results are comparable to FES program
def test_verify_FES2014(directory, CROP):
    # model parameters for FES2014
    m = pyTMD.io.model(directory).from_database('FES2014', group='z')
    # constituent files included in test
    c = ['2n2','k1','k2','m2','m4','mf','mm','msqm','mtm','n2','o1',
        'p1','q1','s1','s2']
    # reduce to constituents for test
    m.reduce_constituents(c)
    # open dataset
    ds = m.open_dataset(group='z', chunks='auto', use_default_units=False)

    # read validation dataset
    # extract time (Modified Julian Days), latitude, longitude, and tide data
    names = ('CNES','Hour','Latitude','Longitude','Short_tide','LP_tide',
        'Pure_tide','Geo_tide','Rad_tide')
    formats = ('f','i','f','f','f','f','f','f','f')
    file_contents = np.loadtxt(filepath.joinpath('fes_slev.txt.gz'),
        skiprows=1,dtype=dict(names=names,formats=formats))
    longitude = file_contents['Longitude']
    latitude = file_contents['Latitude']
    validation = file_contents['Short_tide']
    npts = len(file_contents)

    # CNES Julian Days = Days relative to 1950-01-01 (MJD:33282)
    delta_time = timescale.time._to_sec['day']*file_contents['CNES']
    ts = timescale.from_deltatime(delta_time,
        epoch=timescale.time._cnes_epoch)

    # convert to xarray DataArrays
    X = xr.DataArray(longitude, dims=('time'))
    Y = xr.DataArray(latitude, dims=('time'))
    # crop tide model dataset to bounds
    if CROP:
        # default bounds if cropping data
        xmin, xmax = np.min(X), np.max(X)
        ymin, ymax = np.min(Y), np.max(Y)
        bounds = [xmin, xmax, ymin, ymax]
        # crop dataset to buffered bounds
        ds = ds.tmd.crop(bounds, buffer=1)
    # extract amplitude and phase from tide model
    local = ds.tmd.interp(X, Y)

    # predict tidal elevations at time and infer minor corrections
    tide = local.tmd.predict(ts.tide, deltat=ts.tt_ut1,
        corrections=m.corrections)
    tide += local.tmd.infer(ts.tide, deltat=ts.tt_ut1,
        corrections=m.corrections)

    # will verify differences between model outputs are within tolerance
    eps = 5.0
    # calculate differences between fes2014 and python version
    difference = np.ma.zeros((npts))
    difference.data[:] = tide.values - validation
    difference.mask = np.isnan(tide.values)
    if not np.all(difference.mask):
        assert np.all(np.abs(difference) <= eps)

# parametrize over correction types
@pytest.mark.parametrize("corrections", ["FES", "OTIS"])
# PURPOSE: Tests that the inference methods match
def test_infer_FES2014(directory, corrections):
    # model parameters for FES2014
    m = pyTMD.io.model(directory).from_database('FES2014', group='z')
    # constituent files included in test
    c = ['2n2','k1','k2','m2','m4','mf','mm','msqm','mtm','n2','o1',
        'p1','q1','s1','s2']
    # reduce to constituents for test
    m.reduce_constituents(c)
    # open dataset
    ds = m.open_dataset(group='z', chunks='auto')

    # read validation dataset
    # extract time (Modified Julian Days), latitude, longitude, and tide data
    names = ('CNES','Hour','Latitude','Longitude','Short_tide','LP_tide',
        'Pure_tide','Geo_tide','Rad_tide')
    formats = ('f','i','f','f','f','f','f','f','f')
    file_contents = np.loadtxt(filepath.joinpath('fes_slev.txt.gz'),
        skiprows=1,dtype=dict(names=names,formats=formats))
    # convert to xarray DataArrays
    X, Y = ds.tmd.coords_as(
        file_contents['Longitude'][0],
        file_contents['Latitude'][0],
        crs=4326,
    )
    # extract amplitude and phase from tide model
    local = ds.tmd.interp(X, Y).compute()

    # interpolate local admittances
    admit = pyTMD.predict._admittance_short_period(
        local, corrections=corrections
    ).to_dataset(dim='constituent')

    # relationship between major and minor constituent complex amplitudes
    dmin = xr.Dataset()
    dmin["2q1"] = 0.263 * local["q1"] - 0.0252 * local["o1"]
    dmin["sigma1"] = 0.297 * local["q1"] - 0.0264 * local["o1"]
    dmin["rho1"] = 0.164 * local["q1"] + 0.0048 * local["o1"]
    dmin["m1b"] = 0.0140 * local["o1"] + 0.0101 * local["k1"]
    dmin["m1"] = 0.0389 * local["o1"] + 0.0282 * local["k1"]
    dmin["chi1"] = 0.0064 * local["o1"] + 0.0060 * local["k1"]
    dmin["pi1"] = 0.0030 * local["o1"] + 0.0171 * local["k1"]
    dmin["phi1"] = -0.0015 * local["o1"] + 0.0152 * local["k1"]
    dmin["theta1"] = -0.0065 * local["o1"] + 0.0155 * local["k1"]
    dmin["j1"] = -0.0389 * local["o1"] + 0.0836 * local["k1"]
    dmin["oo1"] = -0.0431 * local["o1"] + 0.0613 * local["k1"]
    dmin["2n2"] = 0.264 * local["n2"] - 0.0253 * local["m2"]
    dmin["mu2"] = 0.298 * local["n2"] - 0.0264 * local["m2"]
    dmin["nu2"] = 0.165 * local["n2"] + 0.00487 * local["m2"]
    dmin["lambda2"] = 0.0040 * local["m2"] + 0.0074 * local["s2"]
    dmin["l2"] = 0.0131 * local["m2"] + 0.0326 * local["s2"]
    dmin["l2b"] = 0.0033 * local["m2"] + 0.0082 * local["s2"]
    dmin["t2"] = 0.0585 * local["s2"]
    # additional coefficients for FES models
    if corrections in ("FES",):
        # spline coefficients for admittances
        mu2 = [0.069439968323, 0.351535557706, -0.046278307672]
        nu2 = [-0.006104695053, 0.156878802427, 0.006755704028]
        l2 = [0.077137765667, -0.051653455134, 0.027869916824]
        t2 = [0.180480173707, -0.020101177502, 0.008331518844]
        la2 = [0.016503557465, -0.013307812292, 0.007753383202]
        dmin["mu2"] = (
            mu2[0] * local["k2"] + mu2[1] * local["n2"] + mu2[2] * local["m2"]
        )
        dmin["nu2"] = (
            nu2[0] * local["k2"] + nu2[1] * local["n2"] + nu2[2] * local["m2"]
        )
        dmin["lambda2"] = (
            la2[0] * local["k2"] + la2[1] * local["n2"] + la2[2] * local["m2"]
        )
        dmin["l2b"] = (
            l2[0] * local["k2"] + l2[1] * local["n2"] + l2[2] * local["m2"]
        )
        dmin["t2"] = (
            t2[0] * local["k2"] + t2[1] * local["n2"] + t2[2] * local["m2"]
        )
        dmin["eps2"] = 0.53285 * local["2n2"] - 0.03304 * local["n2"]
        dmin["eta2"] = -0.0034925 * local["m2"] + 0.0831707 * local["k2"]

    # verify that methods provide similar answers
    for constituent in admit.tmd.constituents:
        assert np.allclose(dmin[constituent].values, admit[constituent].values)

# PURPOSE: test definition file functionality
@pytest.mark.parametrize("MODEL", ['FES2014'])
def test_definition_file(MODEL):
    # get model parameters
    model = pyTMD.io.model(verify=False).from_database(MODEL)
    # create model definition file
    fid = io.StringIO()
    d = model.to_dict(serialize=True)
    json.dump(d, fid)
    fid.seek(0)
    # use model definition file as input
    m = pyTMD.io.model().from_file(fid)
    # check that (serialized) attributes are the same
    assert m.__parameters__ == model.__parameters__

# parametrize over reading with dask
@pytest.mark.parametrize("CHUNKS", [None, "auto"])
# PURPOSE: test extend function
def test_extend_array(directory, CHUNKS):
    # model parameters for FES2014
    m = pyTMD.io.model(directory).from_database('FES2014', group='z')
    # reduce to constituents for test
    m.reduce_constituents(['m2'])
    # open dataset
    ds = m.open_dataset(group='z', chunks=CHUNKS)
    # pad in longitudinal direction
    ds = ds.tmd.pad()
    # check that longitude values are as expected
    dlon = 1.0/16.0
    lon = np.arange(-dlon, 360 + dlon, dlon)
    assert np.allclose(lon, ds.x.values)
