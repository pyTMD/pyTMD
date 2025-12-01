#!/usr/bin/env python
u"""
test_perth3_read.py (11/2025)
Tests the read program to verify that constituents are being extracted
Tests that interpolated results are comparable to NASA PERTH3 program

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

UPDATE HISTORY:
    Updated 11/2025: using new xarray interface for tidal model data
    Updated 10/2025: split directories between validation and model data
        fetch data from pyTMD developers test data repository
    Updated 09/2025: added check if running on GitHub Actions or locally
    Updated 08/2025: added xarray tests to verify implementation
    Updated 06/2025: subset to specific constituents when reading model
    Updated 09/2024: drop support for the ascii definition file format
        use model class attributes for file format and corrections
    Updated 08/2024: increased tolerance for comparing with GOT4.7 tests
        as using nodal corrections from PERTH5
        use a reduced list of minor constituents to match GOT4.7 tests
    Updated 07/2024: add parametrize over cropping the model fields
    Updated 04/2024: use timescale for temporal operations
    Updated 01/2024: refactored compute functions into compute.py
    Updated 04/2023: using pathlib to define and expand paths
    Updated 12/2022: add check for read and interpolate constants
    Updated 09/2021: added test for model definition files
        update check tide points to add compression flags
    Updated 07/2021: added test for invalid tide model name
    Updated 05/2021: added test for check point program
    Updated 03/2021: use pytest fixture to setup and teardown model data
        replaced numpy bool/int to prevent deprecation warnings
    Written 08/2020
"""
import io
import gzip
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

# parametrize over cropping the model fields
@pytest.mark.parametrize("CROP", [False, True])
# PURPOSE: Tests that interpolated results are comparable to PERTH3 program
def test_verify_GOT47(directory, CROP):
    # model parameters for GOT4.7
    m = pyTMD.io.model(directory).from_database('GOT4.7')
    # perth3 test program infers m4 tidal constituent
    # constituent files included in test
    constituents = ['q1','o1','p1','k1','n2','m2','s2','k2','s1']
    m.reduce_constituents(constituents)
    # open dataset
    ds = m.open_dataset(type='z', chunks='auto', use_default_units=False)

    # read validation dataset
    with gzip.open(filepath.joinpath('perth_output_got4.7.gz'),'r') as fid:
        file_contents = fid.read().decode('ISO-8859-1').splitlines()
    # extract latitude, longitude, time (Modified Julian Days) and tide data
    npts = len(file_contents) - 2
    lat = np.zeros((npts))
    lon = np.zeros((npts))
    MJD = np.zeros((npts))
    validation = np.ma.zeros((npts))
    validation.mask = np.ones((npts),dtype=bool)
    for i in range(npts):
        line_contents = file_contents[i+2].split()
        lat[i] = np.float64(line_contents[0])
        lon[i] = np.float64(line_contents[1])
        MJD[i] = np.float64(line_contents[2])
        if (len(line_contents) == 5):
            validation.data[i] = np.float64(line_contents[3])
            validation.mask[i] = False

    # convert time from MJD to timescale object
    ts = timescale.time.Timescale(MJD=MJD)
    # interpolate delta times
    deltat = ts.tt_ut1

    # convert to xarray DataArrays
    X = xr.DataArray(lon, dims=('time'))
    Y = xr.DataArray(lat, dims=('time'))
    # crop tide model dataset to bounds
    if CROP:
        # default bounds if cropping data
        xmin, xmax = np.min(X), np.max(X)
        ymin, ymax = np.min(Y), np.max(Y)
        bounds = [xmin, xmax, ymin, ymax]
        # crop dataset to buffered bounds
        ds = ds.tmd.crop(bounds, buffer=1)
    # extract amplitude and phase from tide model
    local = ds.tmd.interp(X, Y, extrapolate=True)

    # predict tidal elevations at time and infer minor corrections
    tide = local.tmd.predict(ts.tide, deltat=deltat,
        corrections='perth3')
    tide += local.tmd.infer(ts.tide, deltat=deltat,
        corrections='perth3', minor=m.minor)

    # will verify differences between model outputs are within tolerance
    eps = 0.01
    # calculate differences between perth3 and python version
    difference = np.ma.zeros((npts))
    difference.data[:] = tide - validation
    difference.mask = np.isnan(tide) | validation.mask
    if not np.all(difference.mask):
        assert np.all(np.abs(difference) <= eps)

# PURPOSE: Tests check point program
def test_check_GOT47(directory):
    lons = np.zeros((10)) + 178.0
    lats = -45.0 - np.arange(10)*5.0
    obs = pyTMD.compute.tide_masks(lons, lats, directory=directory,
        model='GOT4.7', crs=4326)
    exp = np.array([True, True, True, True, True,
        True, True, True, False, False])
    assert np.all(obs == exp)

# PURPOSE: test the tide correction wrapper function
def test_Ross_Ice_Shelf(directory):
    # create an image around the Ross Ice Shelf
    xlimits = np.array([-750000,550000])
    ylimits = np.array([-1450000,-300000])
    spacing = np.array([50e3,-50e3])
    # x and y coordinates
    x = np.arange(xlimits[0],xlimits[1]+spacing[0],spacing[0])
    y = np.arange(ylimits[1],ylimits[0]+spacing[1],spacing[1])
    xgrid,ygrid = np.meshgrid(x,y)
    # time dimension
    delta_time = 0.0
    # calculate tide map
    tide = pyTMD.compute.tide_elevations(xgrid, ygrid, delta_time,
        directory=directory, model='GOT4.7',
        epoch=timescale.time._atlas_sdp_epoch, type='grid', standard='GPS',
        crs=3031, extrapolate=True)
    assert np.any(tide)

# PURPOSE: test definition file functionality
@pytest.mark.parametrize("MODEL", ['GOT4.7'])
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
    # model parameters for GOT4.7
    m = pyTMD.io.model(directory).from_database('GOT4.7')
    # reduce to constituents for test
    m.reduce_constituents(['m2'])
    # open dataset
    ds = m.open_dataset(type='z', chunks=CHUNKS)
    # pad in longitudinal direction
    ds = ds.tmd.pad()
    # check that longitude values are as expected
    dlon = 1.0/2.0
    lon = np.arange(-dlon, 360 + dlon, dlon)
    assert np.allclose(lon, ds.x.values)

# PURPOSE: test the catch in the correction wrapper function
def test_unlisted_model(directory):
    msg = "Unlisted tide model"
    with pytest.raises(Exception, match=msg):
        pyTMD.compute.tide_elevations(None, None, None,
            directory=directory, model='invalid')
