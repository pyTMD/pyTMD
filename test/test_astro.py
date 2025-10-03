"""
test_astro.py (10/2025)
Tests astronomical routines

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

UPDATE HISTORY:
    Updated 10/2025: fetch data from pyTMD developers test data repository
    Updated 04/2025: added test for schureman arguments for FES models
    Updated 03/2025: added test for comparing mean longitudes
    Updated 01/2025: added function to get JPL ephemerides file from AWS
    Updated 11/2024: moved normalize_angle and polynomial_sum to math.py
    Updated 07/2024: use normalize_angle from pyTMD astro module
    Updated 04/2024: use timescale for temporal operations
    Updated 01/2024: refactored lunisolar ephemerides functions
    Updated 12/2023: phase_angles function renamed to doodson_arguments
    Updated 04/2023: added test for using JPL ephemerides for positions
    Written 04/2023
"""
import pytest
import numpy as np
import pyTMD
import timescale

def test_mean_longitudes():
    """Test that mean longitudes match between functions
    """
    MJD = 55414.0
    # Meeus method from Astronomical Algorithms
    s1, h1, p1, N1, PP1 = pyTMD.astro.mean_longitudes(MJD, method='Meeus')
    # Meeus methods as implemented in ASTRO5
    s2, h2, p2, N2, PP2 = pyTMD.astro.mean_longitudes(MJD, method='ASTRO5')
    assert np.isclose(s1, s2).all()
    assert np.isclose(h1, h2).all()
    assert np.isclose(p1, p2).all()
    assert np.isclose(N1, N2).all()
    assert np.isclose(PP1, PP2).all()
    # converted from Delaunay arguments in IERS
    s3, h3, p3, N3, PP3 = pyTMD.astro.mean_longitudes(MJD, method='IERS')
    assert np.isclose(s1, s3).all()
    assert np.isclose(h1, h3).all()
    assert np.isclose(p1, p3).all()
    assert np.isclose(N1, N3).all()
    assert np.isclose(PP1, PP3).all()

def test_phase_angles():
    """Test that longitudes and phase angles match between functions
    """
    MJD = 55414.0
    dtr = np.pi/180.0
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - 51544.5)/36525.0
    s, h, p, N, PP = pyTMD.astro.mean_longitudes(MJD, method='ASTRO5')
    PR = dtr*pyTMD.math.polynomial_sum(np.array([0.0, 1.396971278,
        3.08889e-4, 2.1e-8, 7.0e-9]), T)
    TAU, S, H, P, ZNS, PS = pyTMD.astro.doodson_arguments(MJD)
    assert np.isclose(dtr*s + PR, S)
    assert np.isclose(dtr*h, H)
    assert np.isclose(dtr*p, P)
    assert np.isclose(2.0*np.pi - N*dtr, ZNS)

def test_fundamental_arguments():
    """Test fundamental (Delaunay) arguments with IERS outputs
    """
    T = 0.07995893223819302
    # convert to MJD from centuries relative to 2000-01-01T12:00:00
    MJD = T*36525.0 + 51544.5
    assert np.isclose(MJD, 54465)
    MJD_test = T*pyTMD.astro._century + pyTMD.astro._mjd_j2000
    assert np.isclose(MJD, MJD_test)
    L_expected = 2.291187512612069099
    LP_expected = 6.212931111003726414
    F_expected = 3.658025792050572989
    D_expected = 4.554139562402433228
    OM_expected = -0.5167379217231804489 + 2.0*np.pi
    # test outputs from function
    l, lp, F, D, Om = pyTMD.astro.delaunay_arguments(MJD)
    # assert matching
    assert np.isclose(L_expected, l)
    assert np.isclose(LP_expected, lp)
    assert np.isclose(F_expected, F)
    assert np.isclose(D_expected, D)
    assert np.isclose(OM_expected, Om)

def test_schureman_arguments():
    """Test that the phase angles match expected outputs
    """
    delta_times = np.array([-3155673600.0, 0.0])
    ts = timescale.from_deltatime(delta_times, epoch=(2000,1,1,0,0,0))
    dtr = np.pi/180.0
    # calculate mean longitudes
    s, h, p, n, ps = pyTMD.astro.mean_longitudes(
        ts.MJD + ts.tt_ut1, method='ASTRO5')
    s_expected = np.array([4.83493587, 3.69552497])
    h_expected = np.array([4.89022967, 4.88647091])
    p_expected = np.array([5.83611763, 1.45381785])
    n_expected = np.array([4.52313201, 2.18290001])
    assert np.isclose(s*dtr, s_expected).all()
    assert np.isclose(h*dtr, h_expected).all()
    assert np.isclose(p*dtr, p_expected).all()
    assert np.isclose(n*dtr, n_expected).all()
    # calculate Schureman arguments
    # convert mean longitudes to radians
    II, xi, nu, Qa, Qu, Ra, Ru, nu_prime, nu_sec = \
        pyTMD.astro.schureman_arguments(dtr*p, dtr*n)
    II_expected = np.array([0.40166890, 0.36476632])
    xi_expected = np.array([-0.20894666, 0.19203231])
    Ru_expected = np.array([-0.14529701, 0.10104533])
    Ra_expected = 1.0/np.array([0.7873131, 1.17232045])
    nu_expected = np.array([-0.22723534, 0.20721813])
    nu_p_expected = np.array([-0.15525636, 0.13805659])
    nu_s_expected = np.array([-0.15460551, 0.13225844])
    assert np.isclose(II, II_expected, atol=1e-5).all()
    assert np.isclose(xi, xi_expected, atol=1e-5).all()
    assert np.isclose(nu, nu_expected, atol=1e-5).all()
    assert np.isclose(Ra, Ra_expected, atol=1e-5).all()
    assert np.isclose(Ru, Ru_expected, atol=1e-5).all()
    assert np.isclose(nu_prime, nu_p_expected, atol=1e-5).all()
    assert np.isclose(nu_sec, nu_s_expected, atol=1e-5).all()

def test_precession_matrix():
    """Test that the precession matrix matches expected outputs
    """
    MJD = 54465.0
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - 51544.5)/36525.0
    expected = np.array([
        [ 9.99998100e-01, -1.78795448e-03, -7.76914888e-04],
        [ 1.78795449e-03,  9.99998402e-01, -6.84570121e-07],
        [ 7.76914871e-04, -7.04519640e-07,  9.99999698e-01]
    ])
    P = pyTMD.astro._precession_matrix(T)
    assert np.isclose(expected, P[:,:,0]).all()

def test_nutation_matrix():
    """Test that the nutation matrix matches expected outputs
    """
    MJD = 54465.0
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - 51544.5)/36525.0
    expected = np.array([
        [ 9.99999999e-01, -3.89161321e-05, -1.68713594e-05],
        [ 3.89155170e-05,  9.99999999e-01, -3.64564011e-05],
        [ 1.68727781e-05,  3.64557445e-05,  9.99999999e-01]
    ])
    # estimate the mean obliquity
    epsilon = pyTMD.astro.mean_obliquity(MJD)
    # estimate the nutation in longitude and obliquity
    dpsi, deps = pyTMD.astro._nutation_angles(T)
    N = pyTMD.astro._nutation_matrix(epsilon, epsilon + deps, dpsi)
    assert np.isclose(expected, N[:,:,0]).all()

def test_frame_bias_matrix():
    """Test that the frame bias matrix matches expected outputs
    """
    expected = np.array([
        [ 1.00000000e+00, -7.07827974e-08,  8.05614894e-08],
        [ 7.07827974e-08,  1.00000000e+00,  3.30604145e-08],
        [-8.05614894e-08, -3.30604145e-08,  1.00000000e+00]
    ])
    B = pyTMD.astro._frame_bias_matrix()
    assert np.isclose(expected, B).all()

def test_icrs_rotation_matrix():
    """Test that the ICRS rotation matrix matches expected outputs
    """
    MJD = 54465.0
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - 51544.5)/36525.0
    expected = np.array([
        [ 9.99998016e-01, -1.82694176e-03, -7.93705916e-04],
        [ 1.82691285e-03,  9.99998331e-01, -3.71379871e-05],
        [ 7.93772440e-04,  3.56878819e-05,  9.99999684e-01]
    ])
    M = pyTMD.astro._icrs_rotation_matrix(T, include_polar_motion=False)
    assert np.isclose(expected, M[:,:,0], atol=1e-7).all()

def test_mean_obliquity():
    """Test that the mean obliquity values matches expected outputs
    """
    MJD = 54465.0
    expected = 0.40907444424006084
    mean_obliquity = pyTMD.astro.mean_obliquity(MJD)
    assert np.isclose(expected, mean_obliquity)

def test_solar_ecef():
    """Test solar ECEF coordinates with ephemeride predictions
    """
    MJD = 55414.0
    # calculate approximate solar ephemerides
    x1, y1, z1 = pyTMD.astro.solar_ecef(MJD, ephemerides='approximate')
    r1 = np.sqrt(x1**2 + y1**2 + z1**2)
    # predict solar ephemerides
    x2, y2, z2 = pyTMD.astro.solar_ecef(MJD, ephemerides='JPL')
    r2 = np.sqrt(x2**2 + y2**2 + z2**2)
    # test distances
    assert np.isclose(np.c_[x1,y1,z1], np.c_[x2,y2,z2], atol=1e9).all()
    # test absolute distance
    assert np.isclose(r1, r2, atol=1e9).all()

def test_lunar_ecef():
    """Test lunar ECEF coordinates with ephemeride predictions
    """
    MJD = 55414.0
    # calculate approximate lunar ephemerides
    x1, y1, z1 = pyTMD.astro.lunar_ecef(MJD, ephemerides='approximate')
    r1 = np.sqrt(x1**2 + y1**2 + z1**2)
    # predict lunar ephemerides
    x2, y2, z2 = pyTMD.astro.lunar_ecef(MJD, ephemerides='JPL')
    r2 = np.sqrt(x2**2 + y2**2 + z2**2)
    # test distances
    assert np.isclose(np.c_[x1,y1,z1], np.c_[x2,y2,z2], atol=5e6).all()
    # test absolute distance
    assert np.isclose(r1, r2, atol=5e6).all()

def test_earth_rotation_angle():
    """Test that the Earth rotation angle (ERA) matches expected outputs
    """
    # create timescale from modified Julian dates
    ts = timescale.time.Timescale(MJD=55414.0)
    # expected earth rotation angle as fraction of a turn
    expected = 0.8730204642501604
    assert np.isclose(360.0*expected, ts.era).all()

def test_greenwich():
    """Test approximations of Greenwich Hour Angle in degrees
    using Meeus approximation and calculation within pyTMD
    """
    # create timescale from modified Julian dates
    ts = timescale.time.Timescale(MJD=55414.0)
    # Meeus approximation
    hour_angle = 280.46061837504 + 360.9856473662862*(ts.T*36525.0)
    GHA = pyTMD.math.normalize_angle(hour_angle)
    # compare with pyTMD calculation
    assert np.isclose(GHA, ts.gha)

def test_sidereal():
    """Test that the sidereal time matches expected outputs
    """
    # create timescale from modified Julian dates
    ts = timescale.time.Timescale(MJD=55414.0)
    # expected side real time in hours
    expected = 20.96154017401333
    assert np.isclose(expected, 24.0*ts.st).all()

def test_epochs():
    """Test that the epoch conversions match expected outputs
    """
    # Modified Julian Day (MJD)
    assert np.isclose(pyTMD.astro._jd_mjd, 2400000.5)
    # J2000 time
    mjd_j2000 = timescale.time.convert_calendar_dates(
        *timescale.time._j2000_epoch,
        epoch=timescale.time._mjd_epoch)
    assert np.isclose(mjd_j2000, pyTMD.astro._mjd_j2000)
    assert np.isclose(pyTMD.astro._mjd_j2000, 51544.5)
    assert np.isclose(pyTMD.astro._jd_j2000, 2451545.0)
