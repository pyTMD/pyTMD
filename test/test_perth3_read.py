#!/usr/bin/env python
u"""
test_perth3_read.py (10/2025)
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

# parameterize interpolation method
@pytest.mark.parametrize("METHOD", ['spline','linear','bilinear'])
@pytest.mark.parametrize("CROP", [False, True])
# PURPOSE: Tests that interpolated results are comparable to PERTH3 program
def test_verify_GOT47(directory, METHOD, CROP):
    # model parameters for GOT4.7
    model = pyTMD.io.model(directory,compressed=True).elevation('GOT4.7')
    # perth3 test program infers m4 tidal constituent
    # constituent files included in test
    constituents = ['q1','o1','p1','k1','n2','m2','s2','k2','s1']
    # keep units consistent with test outputs
    model.scale = 1.0

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

    # extract amplitude and phase from tide model
    amp,ph,cons = model.extract_constants(lon, lat,
        method=METHOD, crop=CROP, constituents=constituents)
    assert all(c in constituents for c in cons)
    # calculate complex phase in radians for Euler's
    cph = -1j*ph*np.pi/180.0
    # calculate constituent oscillations
    hc = amp*np.exp(cph)

    # allocate for out tides at point
    tide = np.ma.zeros((npts))
    tide.mask = np.zeros((npts),dtype=bool)
    # predict tidal elevations at time and infer minor corrections
    tide.mask[:] = np.any(hc.mask, axis=1)
    tide.data[:] = pyTMD.predict.drift(ts.tide, hc, cons,
        deltat=deltat, corrections='perth3')
    minor = pyTMD.predict.infer_minor(ts.tide, hc, cons,
        deltat=deltat, corrections='perth3', minor=model.minor)
    tide.data[:] += minor.data[:]

    # will verify differences between model outputs are within tolerance
    eps = 0.01
    # calculate differences between perth3 and python version
    difference = np.ma.zeros((npts))
    difference.data[:] = tide.data - validation
    difference.mask = (tide.mask | validation.mask)
    if not np.all(difference.mask):
        assert np.all(np.abs(difference) <= eps)

# parameterize interpolation method
@pytest.mark.parametrize("METHOD", ['spline','nearest'])
# PURPOSE: Tests that interpolated results are comparable
def test_compare_GOT47(directory, METHOD):
    # model parameters for GOT4.7
    model = pyTMD.io.model(directory,compressed=True).elevation('GOT4.7')
    # perth3 test program infers m4 tidal constituent
    # constituent files included in test
    constituents = ['q1','o1','p1','k1','n2','m2','s2','k2','s1']
    # keep units consistent with test outputs
    model.scale = 1.0

    # read validation dataset
    with gzip.open(filepath.joinpath('perth_output_got4.7.gz'),'r') as fid:
        file_contents = fid.read().decode('ISO-8859-1').splitlines()
    # extract latitude, longitude, time (Modified Julian Days) and tide data
    npts = len(file_contents) - 2
    lat = np.zeros((npts))
    lon = np.zeros((npts))
    for i in range(npts):
        line_contents = file_contents[i+2].split()
        lat[i] = np.float64(line_contents[0])
        lon[i] = np.float64(line_contents[1])

    # extract amplitude and phase from tide model
    amp1, ph1, c1 = model.extract_constants(lon, lat,
        constituents=constituents, method=METHOD)
    # calculate complex form of constituent oscillation
    hc1 = amp1*np.exp(-1j*ph1*np.pi/180.0)

    # read and interpolate constituents from tide model
    model.read_constants(constituents=constituents)
    assert (constituents == model._constituents.fields)
    amp2, ph2 = model.interpolate_constants(lon, lat, method=METHOD)
    # calculate complex form of constituent oscillation
    hc2 = amp2*np.exp(-1j*ph2*np.pi/180.0)

    # will verify differences between model outputs are within tolerance
    eps = np.finfo(np.float16).eps
    # calculate differences between methods
    for i, cons in enumerate(c1):
        # verify constituents
        assert (cons == constituents[i])
        # calculate difference in amplitude and phase
        difference = hc1[:,i] - hc2[:,i]
        assert np.all(np.abs(difference) <= eps)

    # validate iteration within constituents class
    cons = iter(c1)
    for field, hc in model._constituents:
        # verify constituents
        assert (field == next(cons))
        assert np.ma.isMaskedArray(hc)
        # validate amplitude and phase functions
        amp = model._constituents.amplitude(field)
        phase = model._constituents.phase(field)
        assert np.ma.isMaskedArray(amp)
        assert np.ma.isMaskedArray(phase)
        # calculate complex form of constituent oscillation
        hcomplex = amp*np.exp(-1j*phase*np.pi/180.0)
        # calculate difference in amplitude and phase
        difference = hc - hcomplex
        assert np.all(np.abs(difference) <= eps)

# PURPOSE: Tests that xarray-based results are comparable
def test_GOT47_xarray(directory):
    # model parameters for GOT4.7
    model = pyTMD.io.model(directory,compressed=True).elevation('GOT4.7')
    # perth3 test program infers m4 tidal constituent
    # constituent files included in test
    constituents = ['q1','o1','p1','k1','n2','m2','s2','k2','s1']
    # keep units consistent with test outputs
    model.scale = 1.0

    # read validation dataset
    with gzip.open(filepath.joinpath('perth_output_got4.7.gz'),'r') as fid:
        file_contents = fid.read().decode('ISO-8859-1').splitlines()
    # extract latitude, longitude, time (Modified Julian Days) and tide data
    npts = len(file_contents) - 2
    lat = np.zeros((npts))
    lon = np.zeros((npts))
    for i in range(npts):
        line_contents = file_contents[i+2].split()
        lat[i] = np.float64(line_contents[0])
        lon[i] = np.float64(line_contents[1])
    # convert latitude and longitude to xarray DataArrays
    x = xr.DataArray(lon, dims="i")
    y = xr.DataArray(lat, dims="i")

    # read and interpolate constituents from tide model
    model.read_constants(constituents=constituents)
    assert (constituents == model._constituents.fields)
    amp1, ph1 = model.interpolate_constants(lon, lat, method='linear')
    # calculate complex form of constituent oscillation
    hc1 = amp1*np.exp(-1j*ph1*np.pi/180.0)

    # convert data to xarray Dataset and scale
    ds = model.scale*model._constituents.to_dataset()
    # interpolate constituents to points
    hc2 = ds.interp(x=x, y=y, method='linear', kwargs={"fill_value": None})

    # will verify differences between model outputs are within tolerance
    eps = np.finfo(np.float32).eps
    # calculate differences between methods
    for i, cons in enumerate(constituents):
        # verify constituents
        assert (cons in ds.tmd.constituents)
        # calculate difference in amplitude and phase
        difference = hc1[:,i] - hc2[cons].values
        assert np.all(np.abs(difference) <= eps)

# PURPOSE: Tests check point program
def test_check_GOT47(directory):
    lons = np.zeros((10)) + 178.0
    lats = -45.0 - np.arange(10)*5.0
    obs = pyTMD.compute.tide_masks(lons, lats, DIRECTORY=directory,
        MODEL='GOT4.7', GZIP=True, EPSG=4326)
    exp = np.array([True, True, True, True, True,
        True, True, True, False, False])
    assert np.all(obs == exp)

# parameterize interpolation method
@pytest.mark.parametrize("METHOD", ['spline','nearest','bilinear'])
@pytest.mark.parametrize("EXTRAPOLATE", [True])
# PURPOSE: test the tide correction wrapper function
def test_Ross_Ice_Shelf(directory, METHOD, EXTRAPOLATE):
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
        DIRECTORY=directory, MODEL='GOT4.7', GZIP=True,
        EPOCH=timescale.time._atlas_sdp_epoch, TYPE='grid', TIME='GPS',
        EPSG=3031, METHOD=METHOD, EXTRAPOLATE=EXTRAPOLATE)
    assert np.any(tide)

# PURPOSE: test definition file functionality
@pytest.mark.parametrize("MODEL", ['GOT4.7'])
def test_definition_file(directory, MODEL):
    # get model parameters
    model = pyTMD.io.model(directory,compressed=True).elevation(MODEL)
    # create model definition file
    fid = io.StringIO()
    attrs = ['name','format','model_file','compressed','type','scale']
    d = model.to_dict(fields=attrs, serialize=True)
    json.dump(d, fid)
    fid.seek(0)
    # use model definition file as input
    m = pyTMD.io.model().from_file(fid)
    for attr in attrs:
        assert getattr(model,attr) == getattr(m,attr)

# PURPOSE: test extend function
def test_extend_array():
    dlon = 1
    lon = np.arange(-180, 180, dlon)
    valid = np.arange(-180 - dlon, 180 + 2.0*dlon, dlon)
    test = pyTMD.io.GOT._extend_array(lon, dlon)
    assert np.all(test == valid)

# PURPOSE: test the catch in the correction wrapper function
def test_unlisted_model(directory):
    msg = "Unlisted tide model"
    with pytest.raises(Exception, match=msg):
        pyTMD.compute.tide_elevations(None, None, None,
            DIRECTORY=directory, MODEL='invalid')
