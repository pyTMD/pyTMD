#!/usr/bin/env python
u"""
test_coordinates.py (11/2025)
Verify forward and backwards coordinate conversions

UPDATE HISTORY:
    Updated 11/2025: use pyproj.CRS.from_user_input for definitions
    Updated 09/2024: add test for Arctic regions with new projection
        using new JSON dictionary format for model projections
    Updated 07/2024: add check for if projections are geographic
    Updated 12/2023: use new crs class for coordinate reprojection
    Written 08/2020
"""
import pyproj
import numpy as np
import pyTMD

# PURPOSE: verify coordinate conversions are close for Arctic regions
def test_arctic_projection():
    # generate random latitude and longitude coordinates
    N = 10000
    i1 = -180.0 + 360.0*np.random.rand(N)
    i2 = 60.0 + 30.0*np.random.rand(N)
    # get model projection (simplified polar stereographic)
    model = pyTMD.models['AOTIM-5-2018']
    # create transformer from coordinate reference systems
    crs1 = pyproj.CRS.from_user_input(4326)
    crs2 = pyproj.CRS.from_user_input(model['projection'])
    transform = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # perform forward and inverse transformations
    o1, o2 = transform.transform(i1, i2)
    lon, lat = transform.transform(o1, o2, direction='INVERSE')
    # calculate great circle distance between inputs and outputs
    cdist = np.arccos(np.sin(i2*np.pi/180.0)*np.sin(lat*np.pi/180.0) +
        np.cos(i2*np.pi/180.0)*np.cos(lat*np.pi/180.0)*
        np.cos((lon-i1)*np.pi/180.0),dtype=np.float32)
    # test that forward and backwards conversions are within tolerance
    eps = np.finfo(np.float32).eps
    assert np.all(cdist < eps)
    # convert projected coordinates from latitude and longitude
    x = (90.0 - i2)*111.7*np.cos(i1/180.0*np.pi)
    y = (90.0 - i2)*111.7*np.sin(i1/180.0*np.pi)
    assert np.isclose(o1, x).all()
    assert np.isclose(o2, y).all()
    # convert latitude and longitude from projected coordinates
    ln = np.arctan2(y, x)*180.0/np.pi
    lt = 90.0 - np.sqrt(x**2 + y**2)/111.7
    # adjust longitudes to be -180:180
    ii, = np.nonzero(ln < 0)
    ln[ii] += 360.0
    # calculate great circle distance between inputs and outputs
    cdist = np.arccos(np.sin(lat*np.pi/180.0)*np.sin(lt*np.pi/180.0) +
        np.cos(lat*np.pi/180.0)*np.cos(lt*np.pi/180.0)*
        np.cos((lon-ln)*np.pi/180.0),dtype=np.float32)
    # test that forward and backwards conversions are within tolerance
    eps = np.finfo(np.float32).eps
    assert np.all(cdist < eps)
