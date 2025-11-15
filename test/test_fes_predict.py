#!/usr/bin/env python
u"""
test_fes_predict.py (11/2025)
Tests that FES2014 data can be downloaded from AWS S3 bucket
Tests the read program to verify that constituents are being extracted
Tests that interpolated results are comparable to FES2014 program

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

UPDATE HISTORY:
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
    lons = np.zeros((10)) + 178.0
    lats = -45.0 - np.arange(10)*5.0
    obs = pyTMD.compute.tide_masks(lons, lats, DIRECTORY=directory,
        MODEL='FES2014', EPSG=4326)
    exp = np.array([True, True, True, True, True,
        True, True, True, False, False])
    assert np.all(obs == exp)

# PURPOSE: Tests that interpolated results are comparable to FES program
def test_verify_FES2014(directory):
    # model parameters for FES2014
    m = pyTMD.io.model(directory).from_database('FES2014')
    # constituent files included in test
    c = ['2n2','k1','k2','m2','m4','mf','mm','msqm','mtm','n2','o1',
        'p1','q1','s1','s2']
    # reduce to constituents for test
    m.reduce_constituents(c)
    # open dataset
    ds = m.open_dataset(type='z', use_default_units=False)

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
    for attr in ['name','format','z','u','v']:
        assert getattr(model,attr) == getattr(m,attr)
