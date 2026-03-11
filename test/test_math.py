#!/usr/bin/env python
u"""
test_math.py (11/2024)
"""
import pytest
import numpy as np
import pyTMD.ellipse
import pyTMD.math

def test_asec2rad():
    """
    Tests the conversion of arcseconds to radians
    """
    # test angles in arcseconds
    angles = np.array([-180, -90, 0, 90, 180, 270, 360])*3600.0
    # expected values in radians
    exp = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    # test conversion to radians
    test = pyTMD.math.asec2rad(angles)
    assert np.allclose(exp, test)
    # arcseconds to radians
    atr = np.pi/648000.0
    # test conversion to radians
    test = angles * atr
    assert np.allclose(exp, test)
    # test reverse conversion
    arcseconds = pyTMD.math.rad2asec(test)
    assert np.allclose(angles, arcseconds)

def test_masec2rad():
    """
    Tests the conversion of microarcseconds to radians
    """
    # test angles in microarcseconds
    angles = np.array([-180, -90, 0, 90, 180, 270, 360])*3600.0*1e6
    # expected values in radians
    exp = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    # test conversion to radians
    test = pyTMD.math.masec2rad(angles)
    assert np.allclose(exp, test)
    # microarcseconds to radians
    atr = np.pi/648e9
    # test conversion to radians
    test = angles * atr
    assert np.allclose(exp, test)
    # test reverse conversion
    microarcseconds = pyTMD.math.rad2masec(test)
    assert np.allclose(angles, microarcseconds)

def test_normalize_angle():
    """
    Tests the normalization of angles to between 0 and 360 degrees
    """
    # test angles
    angles = np.array([-180, -90, 0, 90, 180, 270, 360, 450])
    # expected values
    exp = np.array([180, 270, 0, 90, 180, 270, 0, 90])
    # test normalization of angles
    test = pyTMD.math.normalize_angle(angles)
    assert np.all(exp == test)

def test_aliasing():
    """
    Tests the calculation of an aliasing frequency
    """
    # test frequencies
    frequency = 39000
    sampling = np.array([37500, 38000, 38500, 44000, 44500, 45000])
    # expected values
    exp = np.array([1500, 1000, 500, 5000, 5500, 6000])
    # test aliasing frequencies
    test = pyTMD.math.aliasing(frequency, sampling)
    assert np.all(exp == test)

@pytest.mark.parametrize("l", [1, 2, 3])
def test_assoc_legendre(l, x=[-1.0, -0.9, -0.8]):
    """test the calculation of unnormalized Legendre polynomials
    """
    # calculate Legendre polynomials
    nx = len(x)
    obs = np.zeros((l+1, nx))
    test = np.zeros((l+1, nx))
    for m in range(l+1):
        obs[m,:], _ = pyTMD.math.legendre(l, x, m=m)
        test[m,:] = pyTMD.math._assoc_legendre(l, m, x)
    # expected values for each spherical harmonic degree
    if (l == 1):
        expected = np.array([
            [-1.00000, -0.90000, -0.80000],
            [ 0.00000, -0.43589, -0.60000]
        ])
    elif (l == 2):
        expected = np.array([
            [1.00000, 0.71500, 0.46000],
            [0.00000, 1.17690, 1.44000],
            [0.00000, 0.57000, 1.08000]
        ])
    elif (l == 3):
        expected = np.array([
            [-1.00000, -0.47250, -0.08000],
            [0.00000, -1.99420, -1.98000],
            [0.00000, -2.56500, -4.32000],
            [0.00000, -1.24229, -3.24000]
        ])
    # check with expected values
    assert np.allclose(obs, expected, atol=1e-05)
    # check that the two methods give the same values
    assert np.allclose(obs, test)

@pytest.mark.parametrize("l", [1, 2, 3, 4])
def test_legendre(l):
    """test the calculation of unnormalized Legendre polynomials
    """
    # avoid singularities at the poles
    lat = np.arange(-89, 90, 1)
    th = np.radians(90.0 - lat)
    # test values for x
    x = np.cos(th)
    u = np.sqrt(1.0 - x**2)
    for m in range(l+1):
        obs, dobs = pyTMD.math.legendre(l, x, m=m)
        # since tides only use low-degree harmonics:
        # functions are hard coded rather than using a recursion relation
        if (l == 0) and (m == 0):
            Plm = 1.0
            dPlm = 0.0
        elif (l == 1) and (m == 0):
            Plm = x
            dPlm = -u
        elif (l == 1) and (m == 1):
            Plm = u
            dPlm = x
        elif (l == 2) and (m == 0):
            Plm = 0.5 * (3.0 * x**2 - 1.0)
            dPlm = -3.0 * u * x
        elif (l == 2) and (m == 1):
            Plm = 3.0 * x * u
            dPlm = 3.0 * (1.0 - 2.0 * u**2)
        elif (l == 2) and (m == 2):
            Plm = 3.0 * u**2
            dPlm = 6.0 * u * x
        elif (l == 3) and (m == 0):
            Plm = 0.5 * (5.0 * x**2 - 3.0) * x
            dPlm = u * (1.5 - 7.5 * x**2)
        elif (l == 3) and (m == 1):
            Plm = 1.5 * (5.0 * x**2 - 1.0) * u
            dPlm = -1.5 * x * (10.0 * u**2 - 5.0 * x**2 + 1.0)
        elif (l == 3) and (m == 2):
            Plm = 15.0 * x * u**2
            dPlm = 15.0 * u * (3.0 * x**2 - 1.0)
        elif (l == 3) and (m == 3):
            Plm = 15.0 * u**3
            dPlm = 45.0 * x * u**2
        elif (l == 4) and (m == 0):
            Plm = 0.125 * (35.0 * x**4 - 30.0 * x**2 + 3.0)
            dPlm = -2.5 * (7.0 * x**2 - 3.0) * u * x
        elif (l == 4) and (m == 1):
            Plm = 2.5 * (7.0 * x**2 - 3.0) * u * x
            dPlm = 2.5 * (28.0 * x**4 - 27.0 * x**2 + 3.0)
        elif (l == 4) and (m == 2):
            Plm = 7.5 * (7.0 * x**2 - 1.0) * u**2
            dPlm = (105 * x**2 - 105 * u**2 - 15.0) * u * x
        elif (l == 4) and (m == 3):
            Plm = 105.0 * x * u**3
            dPlm = (420.0 * x**2 - 105.0) * u**2
        elif (l == 4) and (m == 4):
            Plm = 105.0 * u**4
            dPlm = 420.0 * x * u**3
        # apply Condon-Shortley phase
        Plm *= np.power(-1.0, m)
        dPlm *= np.power(-1.0, m)
        # check with expected values
        assert np.allclose(obs, Plm, atol=1e-05)
        assert np.allclose(dobs, dPlm, atol=1e-05)

# PURPOSE: test the calculation of ellipse coordinates
def test_ellipse_xy():
    """test the calculation of ellipse coordinates
    """
    # number of points
    npts = 30
    # define ellipse parameters
    umajor = 4.0 + 2.0*np.random.rand(npts)
    uminor = 2.0 + np.random.rand(npts)
    uincl = 180.0*np.random.rand(npts)
    # center of the ellipse
    xy = (10.0 - 20.0*np.random.rand(1), 10.0 - 20.0*np.random.rand(1))
    # calculate coordinates
    x, y = pyTMD.ellipse._xy(umajor, uminor, uincl, phase=0.0, xy=xy)
    # verify that the coordinates match the ellipse equation
    phi = uincl*np.pi/180.0
    X = (x - xy[0])*np.cos(phi) + (y - xy[1])*np.sin(phi)
    Y = -(x - xy[0])*np.sin(phi) + (y - xy[1])*np.cos(phi)
    test = (uminor*X)**2 + (umajor*Y)**2
    validation = (umajor*uminor)**2
    assert np.allclose(test, validation)

