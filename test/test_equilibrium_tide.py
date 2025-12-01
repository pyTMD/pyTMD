"""
test_equilibrium_tide.py (11/2025)
Tests the calculation of long-period equilibrium tides with respect
to the LPEQMT subroutine

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

UPDATE HISTORY:
    Updated 11/2025: use xarray interface for both equilibrium tide tests
    Updated 08/2025: added option to include mantle anelasticity
    Updated 11/2024: moved normalize_angle to math.py
    Written 10/2024
"""
import pytest
import numpy as np
import xarray as xr
import pyTMD.predict
import pyTMD.math
import timescale.time

# PURPOSE: test the estimation of long-period equilibrium tides
@pytest.mark.parametrize("TYPE", ['grid','drift'])
@pytest.mark.parametrize("include_anelasticity", [False, True])
def test_equilibrium_tide(TYPE, include_anelasticity):
    """
    Test the computation of the long-period equilibrium tides
    from the summation of fifteen tidal spectral lines from
    Cartwright-Tayler-Edden tables
    """
    # create a test dataset for data type
    if (TYPE == 'drift'):
        # number of data points
        n_time = 3000
        lon = np.zeros((n_time))
        lat = np.random.randint(-90,90,size=n_time)
        delta_time = np.random.randint(0,31557600,size=n_time)
        X = xr.DataArray(lon, dims=('time'))
        Y = xr.DataArray(lat, dims=('time'))
    elif (TYPE == 'grid'):
        # number of data points
        n_lat,n_time = (181,100)
        lon = np.array([0])
        lat = np.linspace(-90,90,n_lat)
        delta_time = np.random.randint(0,31557600,size=n_time)
        X = xr.DataArray(lon, dims=('x'))
        Y = xr.DataArray(lat, dims=('y'))
    # create xarray dataset
    ds = xr.Dataset(coords={'y':Y,'x':X})

    # convert from seconds since 2018 to tide time
    EPOCH = (2018, 1, 1, 0, 0, 0)
    t = timescale.from_deltatime(delta_time, epoch=EPOCH, standard='GPS')
    # calculate long-period equilibrium tides
    lpet = pyTMD.predict.equilibrium_tide(t.tide, ds,
        include_anelasticity=include_anelasticity)
    # calculate long-period equilibrium tides using compute function
    computed = pyTMD.compute.LPET_elevations(lon, lat, delta_time,
        crs=4326, epoch=EPOCH, type=TYPE, standard='GPS')

    # longitude of moon
    # longitude of sun
    # longitude of lunar perigee
    # longitude of ascending lunar node
    PHC = np.array([290.21,280.12,274.35,343.51])
    DPD = np.array([13.1763965,0.9856473,0.1114041,0.0529539])

    # number of input points
    nt = len(np.atleast_1d(t.tide))
    nlat = len(np.atleast_1d(lat))
    # compute 4 principal mean longitudes in radians at delta time (SHPN)
    SHPN = np.zeros((4,nt))
    for N in range(4):
        # convert time from days relative to 1992-01-01 to 1987-01-01
        ANGLE = PHC[N] + (t.tide + 1826.0)*DPD[N]
        SHPN[N,:] = np.pi*pyTMD.math.normalize_angle(ANGLE)/180.0

    # assemble long-period tide potential from 15 CTE terms greater than 1 mm
    # nodal term is included but not the constant term.
    PH = np.zeros((nt))
    Z = np.zeros((nt))
    Z += 2.79*np.cos(SHPN[3,:]) - 0.49*np.cos(SHPN[1,:] - \
        283.0*np.pi/180.0) - 3.10*np.cos(2.0*SHPN[1,:])
    PH += SHPN[0,:]
    Z += -0.67*np.cos(PH - 2.0*SHPN[1,:] + SHPN[2,:]) - \
        (3.52 - 0.46*np.cos(SHPN[3,:]))*np.cos(PH - SHPN[2,:])
    PH += SHPN[0,:]
    Z += - 6.66*np.cos(PH) - 2.76*np.cos(PH + SHPN[3,:]) - \
        0.26 * np.cos(PH + 2.*SHPN[3,:]) - 0.58 * np.cos(PH - 2.*SHPN[1,:]) - \
        0.29 * np.cos(PH - 2.*SHPN[2,:])
    PH += SHPN[0,:]
    Z += - 1.27*np.cos(PH - SHPN[2,:]) - \
        0.53*np.cos(PH - SHPN[2,:] + SHPN[3,:]) - \
        0.24*np.cos(PH - 2.0*SHPN[1,:] + SHPN[2,:])

    # Multiply by gamma_2 * normalization * P20(lat)
    k2 = 0.302
    h2 = 0.609
    gamma_2 = (1.0 + k2 - h2)
    P20 = 0.5*(3.0*np.sin(lat*np.pi/180.0)**2 - 1.0)
    # calculate long-period equilibrium tide and convert to meters
    if (nlat != nt):
        exp = gamma_2*np.sqrt((4.0+1.0)/(4.0*np.pi))*np.outer(P20,Z/100.0)
    else:
        exp = gamma_2*np.sqrt((4.0+1.0)/(4.0*np.pi))*P20*(Z/100.0)
    # compare with functional values
    eps = np.finfo(np.float16).eps
    assert np.all(np.abs(lpet - exp) < eps)
    # compare with computed values
    assert np.all(np.abs(lpet - computed) < eps)

# PURPOSE: test the estimation of long-period equilibrium tides
def test_node_tide(directory):
    """
    Test the computation of the equilibrium node tides
    """
    # model parameters for GOT4.7
    m = pyTMD.io.model(directory).from_database('GOT4.7')
    # open dataset
    ds = m.open_dataset(type='z', chunks='auto')
    # append node equilibrium tide to dataset
    ds = ds.tmd.node_equilibrium()
    ds = ds.tmd.subset('node')
    # create a test dataset for data type
    # number of data points
    n_time = 20
    delta_time = np.random.randint(0,31557600,size=n_time)
    # convert from seconds since 2018 to tide time
    EPOCH = (2018, 1, 1, 0, 0, 0)
    t = timescale.from_deltatime(delta_time,
        epoch=EPOCH, standard='GPS')
    # calculate long-period equilibrium tides
    lpet = pyTMD.predict.equilibrium_tide(t.tide, ds,
        constituents='node', corrections='GOT')
    tide = ds.tmd.predict(t.tide, corrections='GOT')
    # compare with functional values
    eps = np.finfo(np.float32).eps
    assert np.all(np.abs(lpet.T - tide.sel(x=0)) < eps)
