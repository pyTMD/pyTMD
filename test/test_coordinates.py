#!/usr/bin/env python
"""
test_coordinates.py (06/2026)
Verify forward and backwards coordinate conversions

UPDATE HISTORY:
    Updated 06/2026: test dataset coordinate types and dimensions
    Updated 11/2025: use pyproj.CRS.from_user_input for definitions
    Updated 09/2024: add test for Arctic regions with new projection
        using new JSON dictionary format for model projections
    Updated 07/2024: add check for if projections are geographic
    Updated 12/2023: use new crs class for coordinate reprojection
    Written 08/2020
"""

import pyproj
import pytest
import numpy as np
import pyTMD


# PURPOSE: verify coordinate conversions are close for Arctic regions
def test_arctic_projection():
    # generate random latitude and longitude coordinates
    N = 10000
    i1 = -180.0 + 360.0 * np.random.rand(N)
    i2 = 60.0 + 30.0 * np.random.rand(N)
    # get model projection (simplified polar stereographic)
    model = pyTMD.models["AOTIM-5-2018"]
    # create transformer from coordinate reference systems
    crs1 = pyproj.CRS.from_user_input(4326)
    crs2 = pyproj.CRS.from_user_input(model["projection"])
    transform = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # perform forward and inverse transformations
    o1, o2 = transform.transform(i1, i2)
    lon, lat = transform.transform(o1, o2, direction="INVERSE")
    # calculate great circle distance between inputs and outputs
    cdist = np.arccos(
        np.sin(i2 * np.pi / 180.0) * np.sin(lat * np.pi / 180.0)
        + np.cos(i2 * np.pi / 180.0)
        * np.cos(lat * np.pi / 180.0)
        * np.cos((lon - i1) * np.pi / 180.0),
        dtype=np.float32,
    )
    # test that forward and backwards conversions are within tolerance
    eps = np.finfo(np.float32).eps
    assert np.all(cdist < eps)
    # convert projected coordinates from latitude and longitude
    x = (90.0 - i2) * 111.7 * np.cos(i1 / 180.0 * np.pi)
    y = (90.0 - i2) * 111.7 * np.sin(i1 / 180.0 * np.pi)
    assert np.allclose(o1, x)
    assert np.allclose(o2, y)
    # convert latitude and longitude from projected coordinates
    ln = np.arctan2(y, x) * 180.0 / np.pi
    lt = 90.0 - np.sqrt(x**2 + y**2) / 111.7
    # adjust longitudes to be -180:180
    (ii,) = np.nonzero(ln < 0)
    ln[ii] += 360.0
    # calculate great circle distance between inputs and outputs
    cdist = np.arccos(
        np.sin(lat * np.pi / 180.0) * np.sin(lt * np.pi / 180.0)
        + np.cos(lat * np.pi / 180.0)
        * np.cos(lt * np.pi / 180.0)
        * np.cos((lon - ln) * np.pi / 180.0),
        dtype=np.float32,
    )
    # test that forward and backwards conversions are within tolerance
    eps = np.finfo(np.float32).eps
    assert np.all(cdist < eps)


# PURPOSE: verify setting trajectory dataset coordinates
def test_dataset_trajectory():
    # test trajectory type
    exp = "trajectory"
    # number of data points
    npts = 30
    x = np.random.rand(npts)
    y = np.random.rand(npts)
    t = np.random.rand(npts)
    # test with specifying type
    X, Y = pyTMD.io.dataset._coords(x, y, type=exp, target_crs=4326)
    assert np.allclose(X.values, x)
    assert np.allclose(Y.values, y)
    assert X.dims[0] == "time"
    assert Y.dims[0] == "time"
    assert X.attrs["type"] == exp
    assert Y.attrs["type"] == exp
    # test with legacy drift type
    X, Y = pyTMD.io.dataset._coords(x, y, type="drift", target_crs=4326)
    assert np.allclose(X.values, x)
    assert np.allclose(Y.values, y)
    assert X.dims[0] == "time"
    assert Y.dims[0] == "time"
    assert X.attrs["type"] == exp
    assert Y.attrs["type"] == exp
    # test without specifying type
    X, Y = pyTMD.io.dataset._coords(x, y, time=t, target_crs=4326)
    assert np.allclose(X.values, x)
    assert np.allclose(Y.values, y)
    assert X.dims[0] == "time"
    assert Y.dims[0] == "time"
    assert X.attrs["type"] == exp
    assert Y.attrs["type"] == exp


# PURPOSE: verify setting gridded dataset coordinates
def test_dataset_grid():
    # test grid type
    exp = "grid"
    # number of data points
    ny, nx, ntime = 20, 30, 10
    x = np.random.rand(nx)
    y = np.random.rand(ny)
    t = np.random.rand(ntime)
    xgrid, ygrid = np.meshgrid(x, y)
    # test with using coordinate arrays
    X, Y = pyTMD.io.dataset._coords(x, y, type=exp, target_crs=4326)
    assert np.allclose(X.values, xgrid)
    assert np.allclose(Y.values, ygrid)
    assert X.dims == ("y", "x")
    assert Y.dims == ("y", "x")
    assert X.attrs["type"] == exp
    assert Y.attrs["type"] == exp
    # test with using meshgrid directly
    X, Y = pyTMD.io.dataset._coords(xgrid, ygrid, type=exp, target_crs=4326)
    assert np.allclose(X.values, xgrid)
    assert np.allclose(Y.values, ygrid)
    assert X.dims == ("y", "x")
    assert Y.dims == ("y", "x")
    assert X.attrs["type"] == exp
    assert Y.attrs["type"] == exp
    # test without specifying type
    X, Y = pyTMD.io.dataset._coords(xgrid, ygrid, time=t, target_crs=4326)
    assert np.allclose(X.values, xgrid)
    assert np.allclose(Y.values, ygrid)
    assert X.dims == ("y", "x")
    assert Y.dims == ("y", "x")
    assert X.attrs["type"] == exp
    assert Y.attrs["type"] == exp

# PURPOSE: verify setting time series dataset coordinates
def test_dataset_time_series():
    # test time series type
    exp = "time series"
    # number of data points
    nstation, ntime = 1, 10
    x = np.random.rand(nstation)
    y = np.random.rand(nstation)
    t = np.random.rand(ntime)
    # test with specifying type
    X, Y = pyTMD.io.dataset._coords(x, y, type=exp, target_crs=4326)
    assert np.allclose(X.values, x)
    assert np.allclose(Y.values, y)
    assert X.dims == ("station",)
    assert Y.dims == ("station",)
    assert X.attrs["type"] == exp
    assert Y.attrs["type"] == exp
    # test without specifying type
    X, Y = pyTMD.io.dataset._coords(x, y, time=t, target_crs=4326)
    assert np.allclose(X.values, x)
    assert np.allclose(Y.values, y)
    assert X.dims == ("station",)
    assert Y.dims == ("station",)
    assert X.attrs["type"] == exp
    assert Y.attrs["type"] == exp
    # test with scalars
    X, Y = pyTMD.io.dataset._coords(x[0], y[0], target_crs=4326)
    assert np.allclose(X.values, x[0])
    assert np.allclose(Y.values, y[0])
    assert X.dims == ()
    assert Y.dims == ()
    assert X.attrs["type"] == exp
    assert Y.attrs["type"] == exp

def test_dataset_invalid():
    # number of data points
    npts = 30
    x = np.random.rand(npts)
    y = np.random.rand(npts)
    # test catch for unknown data type
    msg = "Must provide time parameter to determine type"
    with pytest.raises(ValueError, match=msg):
        pyTMD.io.dataset._coords(x, y, target_crs=4326)
    # test catch for unknown data type
    coord_type = "unknown"
    msg = f"Unknown coordinate data type: {coord_type}"
    with pytest.raises(ValueError, match=msg):
        pyTMD.io.dataset._coords(x, y, type=coord_type, target_crs=4326)
    # test catch for invalid data type
    coord_type = ["invalid"]
    msg = "Coordinate data type must be a string"
    with pytest.raises(ValueError, match=msg):
        pyTMD.io.dataset._coords(x, y, type=coord_type, target_crs=4326)
