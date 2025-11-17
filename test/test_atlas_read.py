#!/usr/bin/env python
u"""
test_atlas_read.py (11/2025)
Tests that ATLAS compact and netCDF4 data can be downloaded from AWS S3 bucket
Tests the read program to verify that constituents are being extracted

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
        refactored compute functions into compute.py
    Updated 04/2023: using pathlib to define and expand paths
    Updated 12/2022: add check for read and interpolate constants
    Updated 11/2022: use f-strings for formatting verbose or ascii output
    Updated 09/2021: added test for model definition files
    Updated 03/2021: use pytest fixture to setup and teardown model data
        simplified netcdf inputs to be similar to binary OTIS read program
        replaced numpy bool/int to prevent deprecation warnings
    Written 09/2020
"""
import re
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
# PURPOSE: Tests that interpolated results are comparable to OTPSnc program
def test_read_TPXO9_v2(directory, CROP):
    # model parameters for TPXO9-atlas-v2
    m = pyTMD.io.model(directory).from_database('TPXO9-atlas-v2-nc', type='z')
    # reduce to constituents for test
    m.reduce_constituents(['m2','s2','k1','o1'])
    # open dataset
    ds = m.open_dataset(type='z', chunks='auto')

    # read validation dataset (m2, s2, k1, o1)
    names = ('Lat', 'Lon', 'm2_amp', 'm2_ph', 's2_amp', 's2_ph',
        'k1_amp', 'k1_ph', 'o1_amp', 'o1_ph')
    formats = ('f','f','f','f','f','f','f','f','f','f')
    val = np.loadtxt(filepath.joinpath('extract_HC_sample_out.gz'),
        skiprows=3,dtype=dict(names=names,formats=formats))

    # convert to xarray DataArrays
    X = xr.DataArray(val['Lon'], dims=('time'))
    Y = xr.DataArray(val['Lat'], dims=('time'))
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

    # will verify differences between model outputs are within tolerance
    amp_eps = 0.05
    ph_eps = 10.0
    # calculate differences between OTPSnc and python version
    for i,cons in enumerate(local.tmd.constituents):
        # amplitude and phase
        amp = local[cons].tmd.amplitude
        ph = local[cons].tmd.phase
        # convert phase from 0:360 to -180:180
        phase = np.arctan2(np.sin(np.radians(ph)), np.cos(np.radians(ph)))
        ph = np.degrees(phase)
        # calculate differences
        amp_diff = amp.values - val[f'{cons}_amp']
        ph_diff = ph.values - val[f'{cons}_ph']
        assert np.all(np.abs(amp_diff) <= amp_eps)
        assert np.all(np.abs(ph_diff) <= ph_eps)

# parametrize over cropping the model fields
@pytest.mark.parametrize("CROP", [False, True])
# PURPOSE: Tests that interpolated results are comparable to OTPSnc program
def test_verify_TPXO9_v2(directory, CROP):
    # model parameters for TPXO9-atlas-v2
    m = pyTMD.io.model(directory).from_database('TPXO9-atlas-v2-nc', type='z')
    # reduce to constituents for test
    m.reduce_constituents(['m2','s2','k1','o1'])
    # open dataset
    ds = m.open_dataset(type='z', chunks='auto')

    # compile numerical expression operator
    rx = re.compile(r'[-+]?(?:(?:\d+\.\d+\.\d+)|(?:\d+\:\d+\:\d+)'
        r'|(?:\d*\.\d+)|(?:\d+\.?))')
    # read validation dataset (m2, s2, k1, o1)
    # Lat  Lon  mm.dd.yyyy hh:mm:ss  z(m)  Depth(m)
    with gzip.open(filepath.joinpath('predict_tide_sample_out.gz'),'r') as f:
        file_contents = f.read().decode('ISO-8859-1').splitlines()
    # number of validation data points
    nval = len(file_contents) - 6
    # allocate for validation dataset
    val = dict(latitude=np.zeros((nval)),longitude=np.zeros((nval)),
        time=np.zeros((nval)),height=np.zeros((nval)))
    for i,line in enumerate(file_contents[6:]):
        # extract numerical values
        line_contents = rx.findall(line)
        val['latitude'][i] = np.float64(line_contents[0])
        val['longitude'][i] = np.float64(line_contents[1])
        val['height'][i] = np.float64(line_contents[4])
        # extract dates
        MM,DD,YY = np.array(line_contents[2].split('.'), dtype='f8')
        hh,mm,ss = np.array(line_contents[3].split(':'), dtype='f8')
        # convert from calendar dates into days since 1992-01-01T00:00:00
        val['time'][i] = timescale.time.convert_calendar_dates(YY, MM, DD,
            hour=hh, minute=mm, second=ss, epoch=timescale.time._tide_epoch)

    # convert to xarray DataArrays
    X = xr.DataArray(val['longitude'], dims=('time'))
    Y = xr.DataArray(val['latitude'], dims=('time'))
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
    # delta time
    deltat = np.zeros_like(val['time'])

    # predict tidal elevations at time
    tide = local.tmd.predict(val['time'], deltat=deltat, corrections=m['corrections'])

    # will verify differences between model outputs are within tolerance
    eps = 0.05
    # calculate differences between OTPSnc and python version
    difference = np.ma.zeros((nval))
    difference.data[:] = tide.data - val['height']
    difference.mask = np.isnan(tide)
    if not np.all(difference.mask):
        assert np.all(np.abs(difference) <= eps)

# PURPOSE: test definition file functionality
@pytest.mark.parametrize("MODEL", ['TPXO9-atlas-v2-nc'])
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

# parametrize over reading with dask
@pytest.mark.parametrize("CHUNKS", [None, "auto"])
# PURPOSE: test extend function
def test_extend_array(directory, CHUNKS):
    # model parameters for TPXO9-atlas-v2
    m = pyTMD.io.model(directory).from_database('TPXO9-atlas-v2-nc', type='z')
    # reduce to constituents for test
    m.reduce_constituents(['m2'])
    # open dataset
    ds = m.open_dataset(type='z', chunks=CHUNKS)
    # pad in longitudinal direction
    ds = ds.tmd.pad()
    # check that longitude values are as expected
    dlon = 1.0/30.0
    lon = np.arange(0, 360 + 2.0*dlon, dlon)
    assert np.all(np.isclose(lon, ds.x.values))
