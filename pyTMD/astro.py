#!/usr/bin/env python
u"""
astro.py
Written by Tyler Sutterley (08/2025)
Astronomical and nutation routines

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    jplephem: Astronomical Ephemeris for Python
        https://pypi.org/project/jplephem/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

REFERENCES:
    Jean Meeus, Astronomical Algorithms, 2nd edition, 1998.
    Oliver Montenbruck, Practical Ephemeris Calculations, 1989.

UPDATE HISTORY:
    Updated 08/2025: convert angles with numpy radians and degrees functions
        convert arcseconds to radians with asec2rad function in math.py
        convert microarcseconds to radians with masec2rad function in math.py
    Updated 05/2025: use Barycentric Dynamical Time (TDB) for JPL ephemerides
    Updated 04/2025: added schureman arguments function for FES models
        more outputs from schureman arguments function for M1 constituent
        use flexible case for mean longitude method strings
        use numpy power function over using pow for consistency
    Updated 03/2025: changed argument for method calculating mean longitudes
        split ICRS rotation matrix from the ITRS function 
        added function to correct for aberration effects
        added function to calculate equation of time
    Updated 11/2024: moved three generic mathematical functions to math.py
    Updated 07/2024: made a wrapper function for normalizing angles
        make number of days to convert days since an epoch to MJD variables
    Updated 04/2024: use wrapper to importlib for optional dependencies
    Updated 01/2024: refactored lunisolar ephemerides functions
    Updated 12/2023: refactored phase_angles function to doodson_arguments
        added option to compute mean lunar time using equinox method
    Updated 05/2023: add wrapper function for nutation angles
        download JPL kernel file if not currently existing
    Updated 04/2023: added low resolution solar and lunar positions
        added function with more phase angles of the sun and moon
        functions to calculate solar and lunar positions with ephemerides
        add jplephem documentation to Spacecraft and Planet Kernel segments
        fix solar ephemeride function to include SSB to sun segment
        use a higher resolution estimate of the Greenwich hour angle
        use ITRS reference frame for high-resolution ephemeride calculations
    Updated 03/2023: add basic variable typing to function inputs
    Updated 10/2022: fix MEEUS solar perigee rate
    Updated 04/2022: updated docstrings to numpy documentation format
    Updated 08/2020: change time variable names to not overwrite functions
    Updated 07/2020: added function docstrings
    Updated 07/2018: added option ASTRO5 to use coefficients from Richard Ray
        for use with the Goddard Ocean Tides (GOT) model
        added longitude of solar perigee (Ps) as an additional output
    Updated 09/2017: added option MEEUS to use additional coefficients
        from Meeus Astronomical Algorithms to calculate mean longitudes
    Updated 09/2017: Rewritten in Python
    Rewritten in Matlab by Lana Erofeeva 2003
    Written by Richard Ray 12/1990
"""
from __future__ import annotations

import logging
import pathlib
import warnings
import numpy as np
import timescale.eop
import timescale.time
from pyTMD.math import (
    polynomial_sum,
    normalize_angle,
    asec2rad,
    masec2rad,
    rotate
)
from pyTMD.utilities import (
    get_data_path,
    import_dependency,
    from_jpl_ssd
)
# attempt imports
jplephem = import_dependency('jplephem')
jplephem.spk = import_dependency('jplephem.spk')

__all__ = [
    "mean_longitudes",
    "phase_angles",
    "doodson_arguments",
    "delaunay_arguments",
    "schureman_arguments",
    "mean_obliquity",
    "equation_of_time",
    "solar_ecef",
    "solar_approximate",
    "solar_ephemerides",
    "lunar_ecef",
    "lunar_approximate",
    "lunar_ephemerides",
    "gast",
    "itrs",
    "_eqeq_complement",
    "_icrs_rotation_matrix",
    "_frame_bias_matrix",
    "_nutation_angles",
    "_nutation_matrix",
    "_polar_motion_matrix",
    "_precession_matrix",
    "_correct_aberration",
    "_parse_table_5_2e",
    "_parse_table_5_3a",
    "_parse_table_5_3b",
]

# default JPL Spacecraft and Planet ephemerides kernel
_default_kernel = get_data_path(['data','de440s.bsp'])

# number of days between the Julian day epoch and MJD
_jd_mjd = 2400000.5
# number of days between MJD and the J2000 epoch
_mjd_j2000 = 51544.5
# number of days between the Julian day epoch and J2000 epoch
_jd_j2000 = _jd_mjd + _mjd_j2000
# Julian century
_century = 36525.0

# PURPOSE: compute the basic astronomical mean longitudes
def mean_longitudes(
        MJD: np.ndarray,
        **kwargs
    ):
    r"""
    Computes the basic astronomical mean longitudes: :math:`S`, :math:`H`,
    :math:`P`, :math:`N` and :math:`P_s` :cite:p:`Meeus:1991vh,Simon:1994vo`

    Note :math:`N` is not :math:`N'`, i.e. :math:`N` is decreasing with time.

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    method: str, default 'Cartwright'
        method for calculating mean longitudes

            - ``'Cartwright'``: use coefficients from David Cartwright
            - ``'Meeus'``: use coefficients from Meeus Astronomical Algorithms
            - ``'ASTRO5'``: use Meeus Astronomical coefficients from ``ASTRO5``
            - ``'IERS'``: convert from IERS Delaunay arguments

    Returns
    -------
    S: np.ndarray
        mean longitude of moon (degrees)
    H: np.ndarray
        mean longitude of sun (degrees)
    P: np.ndarray
        mean longitude of lunar perigee (degrees)
    N: np.ndarray
        mean longitude of ascending lunar node (degrees)
    Ps: np.ndarray
        longitude of solar perigee (degrees)
    """
    # set default keyword arguments
    kwargs.setdefault('method', 'Cartwright')
    # check for deprecated method
    if kwargs.get('MEEUS'):
        warnings.warn("Deprecated argument", DeprecationWarning)
        kwargs['method'] = 'Meeus'
    elif kwargs.get('ASTRO5'):
        warnings.warn("Deprecated argument", DeprecationWarning)
        kwargs['method'] = 'ASTRO5'
    # compute the mean longitudes
    if (kwargs['method'].title() == 'Meeus'):
        # convert from MJD to days relative to 2000-01-01T12:00:00
        T = MJD - _mjd_j2000
        # mean longitude of moon
        lunar_longitude = np.array([218.3164591, 13.17639647754579,
            -9.9454632e-13, 3.8086292e-20, -8.6184958e-27])
        S = polynomial_sum(lunar_longitude, T)
        # mean longitude of sun
        solar_longitude = np.array([280.46645, 0.985647360164271,
            2.2727347e-13])
        H = polynomial_sum(solar_longitude, T)
        # mean longitude of lunar perigee
        lunar_perigee = np.array([83.3532430, 0.11140352391786447,
            -7.7385418e-12, -2.5636086e-19, 2.95738836e-26])
        P = polynomial_sum(lunar_perigee, T)
        # mean longitude of ascending lunar node
        lunar_node = np.array([125.0445550, -0.052953762762491446,
            1.55628359e-12, 4.390675353e-20, -9.26940435e-27])
        N = polynomial_sum(lunar_node, T)
        # mean longitude of solar perigee (Simon et al., 1994)
        Ps = 282.94 + (1.7192 * T)/_century
    elif (kwargs['method'].upper() == 'ASTRO5'):
        # convert from MJD to centuries relative to 2000-01-01T12:00:00
        T = (MJD - _mjd_j2000)/_century
        # mean longitude of moon (p. 338)
        lunar_longitude = np.array([218.3164477, 481267.88123421, -1.5786e-3,
             1.855835e-6, -1.53388e-8])
        S = polynomial_sum(lunar_longitude, T)
        # mean longitude of sun (p. 338)
        lunar_elongation = np.array([297.8501921, 445267.1114034, -1.8819e-3,
             1.83195e-6, -8.8445e-9])
        H = polynomial_sum(lunar_longitude-lunar_elongation, T)
        # mean longitude of lunar perigee (p. 343)
        lunar_perigee = np.array([83.3532465, 4069.0137287, -1.032e-2,
            -1.249172e-5])
        P = polynomial_sum(lunar_perigee, T)
        # mean longitude of ascending lunar node (p. 144)
        lunar_node = np.array([125.04452, -1934.136261, 2.0708e-3, 2.22222e-6])
        N = polynomial_sum(lunar_node, T)
        # mean longitude of solar perigee (Simon et al., 1994)
        Ps = 282.94 + 1.7192 * T
    elif (kwargs['method'].upper() == 'IERS'):
        # compute the Delaunay arguments (IERS conventions)
        l, lp, F, D, omega = delaunay_arguments(MJD)
        # convert to Doodson arguments in degrees
        # mean longitude of moon
        S = np.degrees(F + omega)
        # mean longitude of sun
        H = np.degrees(F + omega - D)
        # longitude of lunar perigee
        P = np.degrees(F + omega - l)
        # longitude of ascending lunar node
        N = np.degrees(omega)
        # longitude of solar perigee
        Ps = np.degrees(-lp + F - D + omega)
    else:
        # Formulae for the period 1990--2010 derived by David Cartwright
        # convert from MJD to days relative to 2000-01-01T12:00:00
        # convert from Universal Time to Dynamic Time at 2000-01-01
        T = MJD - 51544.4993
        # mean longitude of moon
        S = 218.3164 + 13.17639648 * T
        # mean longitude of sun
        H = 280.4661 + 0.98564736 * T
        # mean longitude of lunar perigee
        P = 83.3535 + 0.11140353 * T
        # mean longitude of ascending lunar node
        N = 125.0445 - 0.05295377 * T
        # solar perigee at epoch 2000
        Ps = np.full_like(T, 282.8)
    # take the modulus of each
    S = normalize_angle(S)
    H = normalize_angle(H)
    P = normalize_angle(P)
    N = normalize_angle(N)
    Ps = normalize_angle(Ps)
    # return as tuple
    return (S, H, P, N, Ps)

# PURPOSE: computes the phase angles of astronomical means
def phase_angles(MJD: np.ndarray):
    # raise warning for deprecated function call
    warnings.warn(("Deprecated. Please use "
        "pyTMD.astro.doodson_arguments instead"),
        DeprecationWarning)
    # call updated function to not break current workflows
    TAU, S, H, P, ZNS, PS = doodson_arguments(MJD)
    # return as tuple
    return (S, H, P, TAU, ZNS, PS)

# PURPOSE: computes the phase angles of astronomical means
def doodson_arguments(
        MJD: np.ndarray,
        equinox: bool = False,
        apply_correction: bool = True,
    ):
    r"""
    Computes astronomical phase angles for the six Doodson
    Arguments: :math:`\tau`, :math:`S`, :math:`H`, :math:`P`, 
    :math:`N'`, and :math:`P_s` :cite:p:`Doodson:1921kt,Meeus:1991vh`

    Follows IERS conventions for the Doodson arguments :cite:p:`Petit:2010tp`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    equinox: bool, default False
        use equinox method for calculating mean lunar time
    apply_correction: bool, default True
        Apply correction for mean lunar longitude

    Returns
    -------
    TAU: np.ndarray
        mean lunar time (radians)
    S: np.ndarray
        mean longitude of the moon (radians)
    H: np.ndarray
        mean longitude of the sun (radians)
    P: np.ndarray
        mean longitude of lunar perigee (radians)
    Np: np.ndarray
        negative mean longitude of the ascending node (radians)
    Ps: np.ndarray
        mean longitude of solar perigee (radians)
    """
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - _mjd_j2000)/_century
    # hour of the day
    hour = np.mod(MJD, 1)*24.0
    # calculate Doodson phase angles
    # mean longitude of moon (degrees)
    S = polynomial_sum(np.array([218.3164477, 481267.88123421,
        -1.5786e-3, 1.855835e-6, -1.53388e-8]), T)
    # mean lunar time (degrees)
    if equinox:
        # create timescale from Modified Julian Day (MJD)
        ts = timescale.time.Timescale(MJD=MJD)
        # use Greenwich Mean Sidereal Time (GMST) from the
        # Equinox method converted to degrees
        TAU = 360.0*ts.st + 180.0 - S
    else:
        LAMBDA = polynomial_sum(np.array([280.4606184,
            36000.7700536, 3.8793e-4, -2.58e-8]), T)
        TAU = (hour*15.0) - S + LAMBDA
    # calculate correction for mean lunar longitude (degrees)
    if apply_correction:
        PR = polynomial_sum(np.array([0.0, 1.396971278,
            3.08889e-4, 2.1e-8, 7.0e-9]), T)
        S += PR
    # mean longitude of sun (degrees)
    H = polynomial_sum(np.array([280.46645, 36000.7697489,
        3.0322222e-4, 2.0e-8, -6.54e-9]), T)
    # mean longitude of lunar perigee (degrees)
    P = polynomial_sum(np.array([83.3532465, 4069.0137287,
        -1.032172222e-2, -1.24991e-5, 5.263e-8]), T)
    # negative of the mean longitude of the ascending node
    # of the moon (degrees)
    Np = polynomial_sum(np.array([234.95544499, 1934.13626197,
        -2.07561111e-3, -2.13944e-6, 1.65e-8]), T)
    # mean longitude of solar perigee (degrees)
    Ps = polynomial_sum(np.array([282.93734098, 1.71945766667,
        4.5688889e-4, -1.778e-8, -3.34e-9]), T)
    # take the modulus of each and convert to radians
    S = np.radians(normalize_angle(S))
    H = np.radians(normalize_angle(H))
    P = np.radians(normalize_angle(P))
    TAU = np.radians(normalize_angle(TAU))
    Np = np.radians(normalize_angle(Np))
    Ps = np.radians(normalize_angle(Ps))
    # return as tuple
    return (TAU, S, H, P, Np, Ps)

def delaunay_arguments(MJD: np.ndarray):
    r"""
    Computes astronomical phase angles for the five primary Delaunay
    Arguments of Nutation: :math:`l`, :math:`l'`, :math:`F`,
    :math:`D`, and :math:`N`
    :cite:p:`Meeus:1991vh,Petit:2010tp,Capitaine:2003fx`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date

    Returns
    -------
    l: np.ndarray
        mean anomaly of moon (radians)
    lp: np.ndarray
        mean anomaly of the sun (radians)
    F: np.ndarray
        mean argument of the moon (radians)
    D: np.ndarray
        mean elongation of the moon from the sun (radians)
    N: np.ndarray
        mean longitude of ascending lunar node (radians)
    """
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - _mjd_j2000)/_century
    # 360 degrees
    circle = 1296000
    # mean anomaly of the moon (arcseconds)
    l = polynomial_sum(np.array([485868.249036, 1717915923.2178,
        31.8792, 0.051635, -2.447e-04]), T)
    # mean anomaly of the sun (arcseconds)
    lp = polynomial_sum(np.array([1287104.79305,  129596581.0481,
        -0.5532, 1.36e-4, -1.149e-05]), T)
    # mean argument of the moon (arcseconds)
    # (angular distance from the ascending node)
    F = polynomial_sum(np.array([335779.526232, 1739527262.8478,
        -12.7512, -1.037e-3, 4.17e-6]), T)
    # mean elongation of the moon from the sun (arcseconds)
    D = polynomial_sum(np.array([1072260.70369, 1602961601.2090,
        -6.3706, 6.593e-3, -3.169e-05]), T)
    # mean longitude of the ascending node of the moon (arcseconds)
    N = polynomial_sum(np.array([450160.398036, -6962890.5431,
        7.4722, 7.702e-3, -5.939e-05]), T)
    # take the modulus of each and convert to radians
    l = asec2rad(normalize_angle(l, circle=circle))
    lp = asec2rad(normalize_angle(lp, circle=circle))
    F = asec2rad(normalize_angle(F, circle=circle))
    D = asec2rad(normalize_angle(D, circle=circle))
    N = asec2rad(normalize_angle(N, circle=circle))
    # return as tuple
    return (l, lp, F, D, N)

def schureman_arguments(
        P: np.ndarray,
        N: np.ndarray
    ):
    r"""
    Computes additional phase angles :math:`I`, :math:`\xi`, :math:`\nu`,
    :math:`R`, :math:`R_a`, :math:`\nu'`, and :math:`\nu''` from
    :cite:t:`Schureman:1958ty`

    See the explanation of symbols in appendix of :cite:t:`Schureman:1958ty` 

    Parameters
    ----------
    P: np.ndarray
        mean longitude of lunar perigee (radians)
    N: np.ndarray
        mean longitude of ascending lunar node (radians)

    Returns
    -------
    I: np.ndarray
        obliquity of lunar orbit with respect to Earth's equator (radians)
    xi: np.ndarray
        longitude in the moon's orbit of lunar intersection (radians)
    nu: np.ndarray
        right ascension of lunar intersection (radians)
    Qa: np.ndarray
        factor in amplitude for m1 constituent (radians)
    Qu: np.ndarray
        term in argument for m1 constituent (radians)
    Ra: np.ndarray
        factor in amplitude for l2 constituent (radians)
    Ru: np.ndarray
        term in argument for l2 constituent (radians)
    nu_p: np.ndarray
        term in argument for k1 constituent (radians)
    nu_s: np.ndarray
        term in argument for k2 constituent (radians)
    """
    # additional astronomical terms for FES models
    # inclination of the moon's orbit to Earth's equator
    # Schureman (page 156)
    I = np.arccos(0.913694997 - 0.035692561*np.cos(N))
    # longitude in the moon's orbit of lunar intersection
    at1 = np.arctan(1.01883*np.tan(N/2.0))
    at2 = np.arctan(0.64412*np.tan(N/2.0))
    xi = -at1 - at2 + N
    xi = np.arctan2(np.sin(xi), np.cos(xi))
    # right ascension of lunar intersection
    nu = at1 - at2
    # mean longitude of lunar perigee reckoned from the lunar intersection
    # Schureman (page 41)
    p = (P - xi)
    # Schureman (page 42) equation 202
    Q = np.arctan((5.0*np.cos(I) - 1.0)*np.tan(p)/(7.0*np.cos(I) + 1.0))
    # Schureman (page 41) equation 197
    Qa = np.power(2.31 + 1.435*np.cos(2.0*p), -0.5)
    # Schureman (page 42) equation 204
    Qu = p - Q
    # Schureman (page 44) equation 214
    P_R = np.sin(2.0*p)
    Q_R = np.power(np.tan(I/2.0), -2.0)/6.0 - np.cos(2.0*p)
    Ru = np.arctan(P_R/Q_R)
    # Schureman (page 44) equation 213
    # note that Ra is normally used as an inverse (1/Ra)
    term1 = 12.0*np.power(np.tan(I/2.0), 2.0)*np.cos(2.0*p)
    term2 = 36.0*np.power(np.tan(I/2.0), 4.0)
    Ra = np.power(1.0 - term1 + term2, -0.5)
    # Schureman (page 45) equation 224
    P_prime = np.sin(2.0*I)*np.sin(nu)
    Q_prime = np.sin(2.0*I)*np.cos(nu) + 0.3347
    nu_p = np.arctan(P_prime/Q_prime)
    # Schureman (page 46) equation 232
    P_sec = (np.sin(I)**2)*np.sin(2.0*nu)
    Q_sec = (np.sin(I)**2)*np.cos(2.0*nu) + 0.0727
    nu_s = 0.5*np.arctan(P_sec/Q_sec)
    # return as tuple
    return (I, xi, nu, Qa, Qu, Ra, Ru, nu_p, nu_s)

def mean_obliquity(MJD: np.ndarray):
    """Mean obliquity of the ecliptic
    :cite:p:`Capitaine:2003fx,Capitaine:2003fw`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date

    Returns
    -------
    epsilon: np.ndarray
        mean obliquity of the ecliptic (radians)
    """
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - _mjd_j2000)/_century
    # mean obliquity of the ecliptic (arcseconds)
    epsilon0 = np.array([84381.406, -46.836769, -1.831e-4,
        2.00340e-4, -5.76e-07, -4.34e-08])
    return asec2rad(polynomial_sum(epsilon0, T))

def equation_of_time(MJD: np.ndarray):
    """Approximate calculation of the difference between apparent and
    mean solar times :cite:p:`Meeus:1991vh,Urban:2013vl`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date

    Returns
    -------
    E: np.ndarray
        equation of time (radians)
    """
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - _mjd_j2000)/_century
    # mean longitude of sun (degrees)
    mean_longitude = np.array([280.46645, 36000.7697489,
        3.0322222e-4, 2.0e-8, -6.54e-9])
    H = polynomial_sum(mean_longitude, T)
    # mean anomaly of the sun (degrees)
    mean_anomaly = np.array([357.5291092, 35999.0502909,
        -0.0001536, 1.0/24490000.0])
    lp = polynomial_sum(mean_anomaly, T)
    # take the modulus of each
    H = normalize_angle(H)
    lp = normalize_angle(lp)
    # ecliptic longitude of the sun (degrees)
    lambda_sun = H + 1.915*np.sin(np.radians(lp)) + \
        0.020*np.sin(2.0*np.radians(lp))
    # calculate the equation of time (degrees)
    E = -1.915*np.sin(np.radians(lp)) - \
        0.020*np.sin(2.0*np.radians(lp)) + \
        2.466*np.sin(2.0*np.radians(lambda_sun)) - \
        0.053*np.sin(4.0*np.radians(lambda_sun))
    # convert to radians
    return np.radians(E)

# PURPOSE: compute coordinates of the sun in an ECEF frame
def solar_ecef(MJD: np.ndarray, **kwargs):
    """
    Wrapper function for calculating the positional coordinates
    of the sun in an Earth-centric, Earth-Fixed (ECEF) frame
    :cite:p:`Meeus:1991vh,Montenbruck:1989uk,Park:2021fa`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    ephemerides: str, default 'approximate'
        Method for calculating solar ephemerides

            - ``'approximate'``: low-resolution ephemerides
            - ``'JPL'``: computed ephemerides from JPL kernels
    **kwargs: dict
        Keyword options for ephemeris calculation

    Returns
    -------
    X, Y, Z: np.ndarray
        ECEF coordinates of the sun (meters)
    """
    kwargs.setdefault('ephemerides', 'approximate')
    if (kwargs['ephemerides'].lower() == 'approximate'):
        return solar_approximate(MJD, **kwargs)
    elif (kwargs['ephemerides'].upper() == 'JPL'):
        return solar_ephemerides(MJD, **kwargs)

def solar_approximate(MJD, **kwargs):
    """
    Computes approximate positional coordinates of the sun in an
    Earth-centric, Earth-Fixed (ECEF) frame
    :cite:p:`Meeus:1991vh,Montenbruck:1989uk`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date

    Returns
    -------
    X, Y, Z: np.ndarray
        ECEF coordinates of the sun (meters)
    """
    # create timescale from Modified Julian Day (MJD)
    ts = timescale.time.Timescale(MJD=MJD)
    # mean longitude of solar perigee (radians)
    Ps = np.radians(282.94 + 1.7192 * ts.T)
    # mean anomaly of the sun (radians)
    solar_anomaly = np.array([357.5256, 35999.049, -1.559e-4, -4.8e-7])
    M = np.radians(polynomial_sum(solar_anomaly, ts.T))
    # series expansion for mean anomaly in solar radius (meters)
    r_sun = 1e9*(149.619 - 2.499*np.cos(M) - 0.021*np.cos(2.0*M))
    # series expansion for ecliptic longitude of the sun (radians)
    lambda_sun = Ps + M + asec2rad(6892.0*np.sin(M) + 72.0*np.sin(2.0*M))
    # ecliptic latitude is equal to 0 within 1 arcminute
    # obliquity of the J2000 ecliptic (radians)
    epsilon_j2000 = np.radians(23.43929111)
    # convert to position vectors
    x = r_sun*np.cos(lambda_sun)
    y = r_sun*np.sin(lambda_sun)*np.cos(epsilon_j2000)
    z = r_sun*np.sin(lambda_sun)*np.sin(epsilon_j2000)
    # Greenwich hour angle (radians)
    rot_z = rotate(np.radians(ts.gha), 'z')
    # rotate to cartesian (ECEF) coordinates
    # ignoring polar motion and length-of-day variations
    X = rot_z[0,0,:]*x + rot_z[0,1,:]*y + rot_z[0,2,:]*z
    Y = rot_z[1,0,:]*x + rot_z[1,1,:]*y + rot_z[1,2,:]*z
    Z = rot_z[2,0,:]*x + rot_z[2,1,:]*y + rot_z[2,2,:]*z
    # return the ECEF coordinates
    return (X, Y, Z)

# PURPOSE: compute coordinates of the sun in an ECEF frame
def solar_ephemerides(MJD: np.ndarray, **kwargs):
    """
    Computes positional coordinates of the sun in an Earth-centric,
    Earth-Fixed (ECEF) frame using JPL ephemerides
    :cite:p:`Meeus:1991vh,Park:2021fa`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    kernel: str or pathlib.Path
        Path to JPL ephemerides kernel file
    include_aberration: bool, default False
        Correct for aberration effects

    Returns
    -------
    X, Y, Z: np.ndarray
        ECEF coordinates of the sun (meters)
    """
    # set default keyword arguments
    kwargs.setdefault('kernel', _default_kernel)
    kwargs.setdefault('include_aberration', False)
    # create timescale from Modified Julian Day (MJD)
    ts = timescale.time.Timescale(MJD=MJD)
    # difference to convert to Barycentric Dynamical Time (TDB)
    tdb2 = getattr(ts, 'tdb_tt') if hasattr(ts, 'tdb_tt') else 0.0
    # download kernel file if not currently existing
    if not pathlib.Path(kwargs['kernel']).exists():
        from_jpl_ssd(kernel=None, local=kwargs['kernel'])
    # read JPL ephemerides kernel
    SPK = jplephem.spk.SPK.open(kwargs['kernel'])
    # segments for computing position of the sun
    # segment 0 SOLAR SYSTEM BARYCENTER -> segment 10 SUN
    SSB_to_Sun = SPK[0, 10]
    xyz_10, vel_10 = SSB_to_Sun.compute_and_differentiate(ts.tt, tdb2=tdb2)
    # segment 0 SOLAR SYSTEM BARYCENTER -> segment 3 EARTH BARYCENTER
    SSB_to_EMB = SPK[0, 3]
    xyz_3, vel_3 = SSB_to_EMB.compute_and_differentiate(ts.tt, tdb2=tdb2)
    # segment 3 EARTH BARYCENTER -> segment 399 EARTH
    EMB_to_Earth = SPK[3, 399]
    xyz_399, vel_399 = EMB_to_Earth.compute_and_differentiate(ts.tt, tdb2=tdb2)
    # compute the position of the sun relative to the Earth
    # Earth_to_Sun = Earth_to_EMB + EMB_to_SSB + SSB_to_Sun
    #              = -EMB_to_Earth - SSB_to_EMB + SSB_to_Sun
    if kwargs['include_aberration']:
        # astronomical unit in kilometers
        AU = 149597870.700
        # position in astronomical units
        position = (xyz_10 - xyz_399 - xyz_3)/AU
        # velocity in astronomical units per day
        velocity = (vel_399 + vel_3 - vel_10)/AU
        # correct for aberration and convert to meters
        x, y, z = _correct_aberration(position, velocity)
    else:
        # convert positions from kilometers to meters
        x, y, z = 1e3*(xyz_10 - xyz_399 - xyz_3)
    # rotate to cartesian (ECEF) coordinates
    rot_z = itrs((ts.utc - _jd_j2000)/ts.century)
    X = rot_z[0,0,:]*x + rot_z[0,1,:]*y + rot_z[0,2,:]*z
    Y = rot_z[1,0,:]*x + rot_z[1,1,:]*y + rot_z[1,2,:]*z
    Z = rot_z[2,0,:]*x + rot_z[2,1,:]*y + rot_z[2,2,:]*z
    # return the ECEF coordinates
    return (X, Y, Z)

# PURPOSE: compute coordinates of the moon in an ECEF frame
def lunar_ecef(MJD: np.ndarray, **kwargs):
    """
    Wrapper function for calculating the positional coordinates
    of the moon in an Earth-centric, Earth-Fixed (ECEF) frame
    :cite:p:`Meeus:1991vh,Montenbruck:1989uk,Park:2021fa`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    ephemerides: str, default 'approximate'
        Method for calculating lunar ephemerides

            - ``'approximate'``: low-resolution ephemerides
            - ``'JPL'``: computed ephemerides from JPL kernels
    **kwargs: dict
        Keyword options for ephemeris calculation

    Returns
    -------
    X, Y, Z: np.ndarray
        ECEF coordinates of the moon (meters)
    """
    kwargs.setdefault('ephemerides', 'approximate')
    if (kwargs['ephemerides'].lower() == 'approximate'):
        return lunar_approximate(MJD, **kwargs)
    elif (kwargs['ephemerides'].upper() == 'JPL'):
        return lunar_ephemerides(MJD, **kwargs)

def lunar_approximate(MJD, **kwargs):
    """
    Computes approximate positional coordinates of the moon in an
    Earth-centric, Earth-Fixed (ECEF) frame
    :cite:p:`Meeus:1991vh,Montenbruck:1989uk`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date

    Returns
    -------
    X, Y, Z: np.ndarray
        ECEF coordinates of the moon (meters)
    """
    # create timescale from Modified Julian Day (MJD)
    ts = timescale.time.Timescale(MJD=MJD)
    # mean longitude of moon (p. 338)
    lunar_longitude = np.array([218.3164477, 481267.88123421, -1.5786e-3,
            1.855835e-6, -1.53388e-8])
    s = np.radians(polynomial_sum(lunar_longitude, ts.T))
    # difference between the mean longitude of sun and moon (p. 338)
    lunar_elongation = np.array([297.8501921, 445267.1114034, -1.8819e-3,
            1.83195e-6, -8.8445e-9])
    D = np.radians(polynomial_sum(lunar_elongation, ts.T))
    # mean longitude of ascending lunar node (p. 144)
    lunar_node = np.array([125.04452, -1934.136261, 2.0708e-3, 2.22222e-6])
    N = np.radians(polynomial_sum(lunar_node, ts.T))
    F = s - N
    # mean anomaly of the sun (radians)
    M = np.radians((357.5256 + 35999.049*ts.T))
    # mean anomaly of the moon (radians)
    l = np.radians((134.96292 + 477198.86753*ts.T))
    # series expansion for mean anomaly in moon radius (meters)
    r_moon = 1e3*(385000.0 - 20905.0*np.cos(l) - 3699.0*np.cos(2.0*D - l) -
        2956.0*np.cos(2.0*D) - 570.0*np.cos(2.0*l) +
        246.0*np.cos(2.0*l - 2.0*D) - 205.0*np.cos(M - 2.0*D) -
        171.0*np.cos(l + 2.0*D) - 152.0*np.cos(l + M - 2.0*D))
    # series expansion for ecliptic longitude of the moon (radians)
    lambda_moon = s + asec2rad(
        22640.0*np.sin(l) + 769.0*np.sin(2.0*l) -
        4586.0*np.sin(l - 2.0*D) + 2370.0*np.sin(2.0*D) -
        668.0*np.sin(M) - 412.0*np.sin(2.0*F) -
        212.0*np.sin(2.0*l - 2.0*D) - 206.0*np.sin(l + M - 2.0*D) +
        192.0*np.sin(l + 2.0*D) - 165.0*np.sin(M - 2.0*D) -
        148.0*np.sin(l - M) - 125.0*np.sin(D) -
        110.0*np.sin(l + M) - 55.0*np.sin(2.0*F - 2.0*D)
    )
    # series expansion for ecliptic latitude of the moon (radians)
    q = asec2rad(412.0*np.sin(2.0*F) + 541.0*np.sin(M))
    beta_moon = asec2rad(18520.0*np.sin(F + lambda_moon - s + q) -
        526.0*np.sin(F - 2*D) + 44.0*np.sin(l + F - 2.0*D) -
        31.0*np.sin(-l + F - 2.0*D) - 25.0*np.sin(-2.0*l + F) -
        23.0*np.sin(M + F - 2.0*D) + 21.0*np.sin(-l + F) +
        11.0*np.sin(-M + F - 2.0*D)
    )
    # convert to position vectors
    x = r_moon*np.cos(lambda_moon)*np.cos(beta_moon)
    y = r_moon*np.sin(lambda_moon)*np.cos(beta_moon)
    z = r_moon*np.sin(beta_moon)
    # obliquity of the J2000 ecliptic (radians)
    epsilon_j2000 = np.radians(23.43929111)
    # rotate by ecliptic
    rot_x = rotate(-epsilon_j2000, 'x')
    u = rot_x[0,0,:]*x + rot_x[0,1,:]*y + rot_x[0,2,:]*z
    v = rot_x[1,0,:]*x + rot_x[1,1,:]*y + rot_x[1,2,:]*z
    w = rot_x[2,0,:]*x + rot_x[2,1,:]*y + rot_x[2,2,:]*z
    # Greenwich hour angle (radians)
    rot_z = rotate(np.radians(ts.gha), 'z')
    # rotate to cartesian (ECEF) coordinates
    # ignoring polar motion and length-of-day variations
    X = rot_z[0,0,:]*u + rot_z[0,1,:]*v + rot_z[0,2,:]*w
    Y = rot_z[1,0,:]*u + rot_z[1,1,:]*v + rot_z[1,2,:]*w
    Z = rot_z[2,0,:]*u + rot_z[2,1,:]*v + rot_z[2,2,:]*w
    # return the ECEF coordinates
    return (X, Y, Z)

# PURPOSE: compute coordinates of the moon in an ECEF frame
def lunar_ephemerides(MJD: np.ndarray, **kwargs):
    """
    Computes positional coordinates of the moon in an Earth-centric,
    Earth-Fixed (ECEF) frame using JPL ephemerides
    :cite:p:`Meeus:1991vh,Park:2021fa`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    kernel: str or pathlib.Path
        Path to JPL ephemerides kernel file
    include_aberration: bool, default False
        Correct for aberration effects

    Returns
    -------
    X, Y, Z: np.ndarray
        ECEF coordinates of the moon (meters)
    """
    # set default keyword arguments
    kwargs.setdefault('kernel', _default_kernel)
    kwargs.setdefault('include_aberration', False)
    # download kernel file if not currently existing
    if not pathlib.Path(kwargs['kernel']).exists():
        from_jpl_ssd(kernel=None, local=kwargs['kernel'])
    # create timescale from Modified Julian Day (MJD)
    ts = timescale.time.Timescale(MJD=MJD)
    # difference to convert to Barycentric Dynamical Time (TDB)
    tdb2 = getattr(ts, 'tdb_tt') if hasattr(ts, 'tdb_tt') else 0.0
    # read JPL ephemerides kernel
    SPK = jplephem.spk.SPK.open(kwargs['kernel'])
    # segments for computing position of the moon
    # segment 0 SOLAR SYSTEM BARYCENTER -> segment 3 EARTH BARYCENTER
    SSB_to_EMB = SPK[0, 3]
    xyz_3, vel_3 = SSB_to_EMB.compute_and_differentiate(ts.tt, tdb2=tdb2)
    # segment 3 EARTH BARYCENTER -> segment 399 EARTH
    EMB_to_Earth = SPK[3, 399]
    xyz_399, vel_399 = EMB_to_Earth.compute_and_differentiate(ts.tt, tdb2=tdb2)
    # segment 3 EARTH BARYCENTER -> segment 301 MOON
    EMB_to_Moon = SPK[3, 301]
    xyz_301, vel_301 = EMB_to_Moon.compute_and_differentiate(ts.tt, tdb2=tdb2)
    # compute the position of the moon relative to the Earth
    # Earth_to_Moon = Earth_to_EMB + EMB_to_Moon
    #               = -EMB_to_Earth + EMB_to_Moon
    if kwargs['include_aberration']:
        # astronomical unit in kilometers
        AU = 149597870.700
        # position in astronomical units
        position = (xyz_301 - xyz_399)/AU
        # velocity in astronomical units per day
        velocity = (vel_3 + vel_399 - vel_301)/AU
        # correct for aberration and convert to meters
        x, y, z = _correct_aberration(position, velocity)
    else:
        # convert positions from kilometers to meters
        x, y, z = 1e3*(xyz_301 - xyz_399)
    # rotate to cartesian (ECEF) coordinates
    # use UTC time as input to itrs rotation function
    rot_z = itrs((ts.utc - _jd_j2000)/ts.century)
    X = rot_z[0,0,:]*x + rot_z[0,1,:]*y + rot_z[0,2,:]*z
    Y = rot_z[1,0,:]*x + rot_z[1,1,:]*y + rot_z[1,2,:]*z
    Z = rot_z[2,0,:]*x + rot_z[2,1,:]*y + rot_z[2,2,:]*z
    # return the ECEF coordinates
    return (X, Y, Z)

def gast(T: float | np.ndarray):
    """Greenwich Apparent Sidereal Time (GAST)
    :cite:p:`Capitaine:2003fx,Capitaine:2003fw,Petit:2010tp`

    Parameters
    ----------
    T: np.ndarray
        Centuries since 2000-01-01T12:00:00
    """
    # create timescale from centuries relative to 2000-01-01T12:00:00
    ts = timescale.time.Timescale(MJD=T*_century + _mjd_j2000)
    # convert dynamical time to modified Julian days
    MJD = ts.tt - _jd_mjd
    # estimate the mean obliquity
    epsilon = mean_obliquity(MJD)
    # estimate the nutation in longitude and obliquity
    dpsi, deps = _nutation_angles(T)
    # traditional equation of the equinoxes
    c = _eqeq_complement(T)
    eqeq = dpsi*np.cos(epsilon + deps) + c
    return np.mod(ts.st + eqeq/24.0, 1.0)

def itrs(
        T: float | np.ndarray,
        include_polar_motion: bool = True
    ):
    """
    International Terrestrial Reference System (ITRS)
    :cite:p:`Capitaine:2003fx,Capitaine:2003fw,Petit:2010tp`

    An Earth-centered Earth-fixed (ECEF) coordinate system
    combining the Earth's true equator and equinox of date,
    the Earth's rotation with respect to the stars, and the
    polar wobble of the crust with respect to the pole of rotation

    Parameters
    ----------
    T: np.ndarray
        Centuries since 2000-01-01T12:00:00
    include_polar_motion: bool, default True
        Include polar motion in the rotation matrix
    """
    # create timescale from centuries relative to 2000-01-01T12:00:00
    ts = timescale.time.Timescale(MJD=T*_century + _mjd_j2000)
    # get the rotation matrix for transforming from ICRS to ITRS
    M = _icrs_rotation_matrix(T,
        include_polar_motion=include_polar_motion
    )
    # compute Greenwich Apparent Sidereal Time
    GAST = rotate(ts.tau*gast(T), 'z')
    R = np.einsum('ijt...,jkt->ikt...', GAST, M)
    # return the combined rotation matrix
    return R

def _eqeq_complement(T: float | np.ndarray):
    """
    Compute complementary terms of the equation of the equinoxes
    :cite:p:`Capitaine:2003fx,Capitaine:2003fw,Petit:2010tp`

    These include the combined effects of precession and nutation
    :cite:p:`Kaplan:2005kj,Petit:2010tp,Urban:2013vl`

    Parameters
    ----------
    T: np.ndarray
        Centuries since 2000-01-01T12:00:00
    """
    # create timescale from centuries relative to 2000-01-01T12:00:00
    ts = timescale.time.Timescale(MJD=T*_century + _mjd_j2000)
    # get the fundamental arguments in radians
    fa = np.zeros((14, len(ts)))
    # mean anomaly of the moon (arcseconds)
    fa[0,:] = asec2rad(polynomial_sum(np.array([485868.249036, 715923.2178,
        31.8792, 0.051635, -2.447e-04]), ts.T)) + ts.tau*np.mod(1325.0*ts.T, 1.0)
    # mean anomaly of the sun (arcseconds)
    fa[1,:] = asec2rad(polynomial_sum(np.array([1287104.79305,  1292581.0481,
        -0.5532, 1.36e-4, -1.149e-05]), ts.T)) + ts.tau*np.mod(99.0*ts.T, 1.0)
    # mean argument of the moon (arcseconds)
    # (angular distance from the ascending node)
    fa[2,:] = asec2rad(polynomial_sum(np.array([335779.526232, 295262.8478,
        -12.7512, -1.037e-3, 4.17e-6]), ts.T)) + ts.tau*np.mod(1342.0*ts.T, 1.0)
    # mean elongation of the moon from the sun (arcseconds)
    fa[3,:] = asec2rad(polynomial_sum(np.array([1072260.70369, 1105601.2090,
        -6.3706, 6.593e-3, -3.169e-05]), ts.T)) + ts.tau*np.mod(1236.0*ts.T, 1.0)
    # mean longitude of the ascending node of the moon (arcseconds)
    fa[4,:] = asec2rad(polynomial_sum(np.array([450160.398036, -482890.5431,
        7.4722, 7.702e-3, -5.939e-05]), ts.T)) + ts.tau*np.mod(-5.0*ts.T, 1.0)
    # additional polynomial terms
    fa[5,:] = polynomial_sum(np.array([4.402608842, 2608.7903141574]), ts.T)
    fa[6,:] = polynomial_sum(np.array([3.176146697, 1021.3285546211]), ts.T)
    fa[7,:] = polynomial_sum(np.array([1.753470314, 628.3075849991]), ts.T)
    fa[8,:] = polynomial_sum(np.array([6.203480913, 334.0612426700]), ts.T)
    fa[9,:] = polynomial_sum(np.array([0.599546497, 52.9690962641]), ts.T)
    fa[10,:] = polynomial_sum(np.array([0.874016757, 21.3299104960]), ts.T)
    fa[11,:] = polynomial_sum(np.array([5.481293872, 7.4781598567]), ts.T)
    fa[12,:] = polynomial_sum(np.array([5.311886287, 3.8133035638]), ts.T)
    fa[13,:] = polynomial_sum(np.array([0, 0.024381750, 0.00000538691]), ts.T)
    # parse IERS Greenwich Sidereal Time (GST) table
    j0, j1 = _parse_table_5_2e()
    n0 = np.c_[j0['l'], j0['lp'], j0['F'], j0['D'], j0['Om'],
        j0['L_Me'], j0['L_Ve'], j0['L_E'], j0['L_Ma'], j0['L_J'],
        j0['L_Sa'], j0['L_U'], j0['L_Ne'], j0['p_A']]
    n1 = np.c_[j1['l'], j1['lp'], j1['F'], j1['D'], j1['Om'],
        j1['L_Me'], j1['L_Ve'], j1['L_E'], j1['L_Ma'], j1['L_J'],
        j1['L_Sa'], j1['L_U'], j1['L_Ne'], j1['p_A']]
    arg0 = np.dot(n0, np.mod(fa, ts.tau))
    arg1 = np.dot(n1, np.mod(fa, ts.tau))
    # evaluate the complementary terms and convert to radians
    complement = masec2rad(np.dot(j0['Cs'], np.sin(arg0)) +
        np.dot(j0['Cc'], np.cos(arg0)) +
        ts.T*np.dot(j1['Cs'], np.sin(arg1)) +
        ts.T*np.dot(j1['Cc'], np.cos(arg1)))
    # return the complementary terms
    return complement

def _icrs_rotation_matrix(
        T: float | np.ndarray,
        include_polar_motion: bool = True
    ):
    """
    Rotation matrix for transforming from the
    International Celestial Reference System (ICRS)
    to the International Terrestrial Reference System (ITRS)
    :cite:p:`Capitaine:2003fx,Capitaine:2003fw,Petit:2010tp`

    Parameters
    ----------
    T: np.ndarray
        Centuries since 2000-01-01T12:00:00
    include_polar_motion: bool, default True
        Include polar motion in the rotation matrix
    """
    # create timescale from centuries relative to 2000-01-01T12:00:00
    ts = timescale.time.Timescale(MJD=T*_century + _mjd_j2000)
    # difference to convert to Barycentric Dynamical Time (TDB)
    tdb2 = getattr(ts, 'tdb_tt') if hasattr(ts, 'tdb_tt') else 0.0
    # convert dynamical time to modified Julian days
    MJD = ts.tt + tdb2 - _jd_mjd
    # estimate the mean obliquity
    epsilon = mean_obliquity(MJD)
    # estimate the nutation in longitude and obliquity
    dpsi, deps = _nutation_angles(T)
    # estimate the rotation matrices
    M1 = _precession_matrix(ts.T)
    M2 = _nutation_matrix(epsilon, epsilon + deps, dpsi)
    M3 = _frame_bias_matrix()
    # calculate the combined rotation matrix for
    # M1: precession
    # M2: nutation
    # M3: frame bias
    M = np.einsum('ijt...,jkt...,kl...->ilt...', M1, M2, M3)
    # add polar motion to the combined rotation matrix
    if include_polar_motion:
        # M4: polar motion
        M4 = _polar_motion_matrix(ts.T)
        M = np.einsum('ijt...,jkt...->ikt...', M, M4)
    # return the combined rotation matrix
    return M

def _frame_bias_matrix():
    """
    Frame bias rotation matrix for converting from a dynamical
    reference system to the International Celestial Reference
    System (ICRS) :cite:p:`Petit:2010tp,Urban:2013vl`
    """
    # frame bias rotation matrix
    B = np.zeros((3,3))
    xi0  = asec2rad(-0.0166170)
    eta0 = asec2rad(-0.0068192)
    da0  = asec2rad(-0.01460)
    # off-diagonal elements of the frame bias matrix
    B[0,1] = da0
    B[0,2] = -xi0
    B[1,0] = -da0
    B[1,2] = -eta0
    B[2,0] =  xi0
    B[2,1] =  eta0
    # second-order corrections to diagonal elements
    B[0,0] = 1.0 - 0.5 * (da0**2 + xi0**2)
    B[1,1] = 1.0 - 0.5 * (da0**2 + eta0**2)
    B[2,2] = 1.0 - 0.5 * (eta0**2 + xi0**2)
    # return the rotation matrix
    return B

def _nutation_angles(T: float | np.ndarray):
    """
    Calculate nutation rotation angles using tables
    from IERS Conventions :cite:p:`Petit:2010tp`

    Parameters
    ----------
    T: np.ndarray
        Centuries since 2000-01-01T12:00:00

    Returns
    -------
    dpsi: np.ndarray
        Nutation in longitude
    deps: np.ndarray
        Obliquity of the ecliptic
    """
    # create timescale from centuries relative to 2000-01-01T12:00:00
    ts = timescale.time.Timescale(MJD=T*_century + _mjd_j2000)
    # difference to convert to Barycentric Dynamical Time (TDB)
    tdb2 = getattr(ts, 'tdb_tt') if hasattr(ts, 'tdb_tt') else 0.0
    # convert dynamical time to modified Julian days
    MJD = ts.tt + tdb2 - _jd_mjd
    # get the fundamental arguments in radians
    l, lp, F, D, Om = delaunay_arguments(MJD)
    # non-polynomial terms in the equation of the equinoxes
    # parse IERS lunisolar longitude table
    l0, l1 = _parse_table_5_3a()
    n0 = np.c_[l0['l'], l0['lp'], l0['F'], l0['D'], l0['Om']]
    n1 = np.c_[l1['l'], l1['lp'], l1['F'], l1['D'], l1['Om']]
    arg0 = np.dot(n0, np.c_[l, lp, F, D, Om].T)
    arg1 = np.dot(n1, np.c_[l, lp, F, D, Om].T)
    dpsi = np.dot(l0['As'], np.sin(arg0)) + \
        np.dot(l0['Ac'], np.cos(arg0)) + \
        ts.T*np.dot(l1['As'], np.sin(arg1)) + \
        ts.T*np.dot(l1['Ac'], np.cos(arg1))
    # parse IERS lunisolar obliquity table
    o0, o1 = _parse_table_5_3b()
    n0 = np.c_[o0['l'], o0['lp'], o0['F'], o0['D'], o0['Om']]
    n1 = np.c_[o1['l'], o1['lp'], o1['F'], o1['D'], o1['Om']]
    arg0 = np.dot(n0, np.c_[l, lp, F, D, Om].T)
    arg1 = np.dot(n1, np.c_[l, lp, F, D, Om].T)
    deps = np.dot(o0['Bs'], np.sin(arg0)) + \
        np.dot(o0['Bc'], np.cos(arg0)) + \
        ts.T*np.dot(o1['Bs'], np.sin(arg1)) + \
        ts.T*np.dot(o1['Bc'], np.cos(arg1))
    # convert to radians
    return (masec2rad(dpsi), masec2rad(deps))

def _nutation_matrix(
        mean_obliquity: float | np.ndarray,
        true_obliquity: float | np.ndarray,
        psi: float | np.ndarray
    ):
    """
    Nutation rotation matrix
    :cite:p:`Kaplan:1989cf,Petit:2010tp`

    Parameters
    ----------
    mean_obliquity: np.ndarray
        Mean obliquity of the ecliptic
    true_obliquity: np.ndarray
        True obliquity of the ecliptic
    psi: np.ndarray
        Nutation in longitude
    """
    # compute elements of nutation rotation matrix
    R = np.zeros((3,3,len(np.atleast_1d(psi))))
    R[0,0,:] = np.cos(psi)
    R[0,1,:] = -np.sin(psi)*np.cos(mean_obliquity)
    R[0,2,:] = -np.sin(psi)*np.sin(mean_obliquity)
    R[1,0,:] = np.sin(psi)*np.cos(true_obliquity)
    R[1,1,:] = np.cos(psi)*np.cos(mean_obliquity)*np.cos(true_obliquity) + \
        np.sin(mean_obliquity)*np.sin(true_obliquity)
    R[1,2,:] = np.cos(psi)*np.sin(mean_obliquity)*np.cos(true_obliquity) - \
        np.cos(mean_obliquity)*np.sin(true_obliquity)
    R[2,0,:] = np.sin(psi)*np.sin(true_obliquity)
    R[2,1,:] = np.cos(psi)*np.cos(mean_obliquity)*np.sin(true_obliquity) - \
        np.sin(mean_obliquity)*np.cos(true_obliquity)
    R[2,2,:] = np.cos(psi)*np.sin(mean_obliquity)*np.sin(true_obliquity) + \
        np.cos(mean_obliquity)*np.cos(true_obliquity)
    # return the rotation matrix
    return R

def _polar_motion_matrix(T: float | np.ndarray):
    """
    Polar motion (Earth Orientation Parameters) rotation matrix
    :cite:p:`Petit:2010tp,Urban:2013vl`

    Parameters
    ----------
    T: np.ndarray
        Centuries since 2000-01-01T12:00:00
    """
    # convert to MJD from centuries relative to 2000-01-01T12:00:00
    MJD = T*_century + _mjd_j2000
    # correct longitude origin for Terrestrial Intermediate Origin (TIO)
    # this correction is negligible for most applications
    sprime = -4.7e-5*T
    # calculate the polar motion for the given dates
    px, py = timescale.eop.iers_polar_motion(MJD)
    # calculate the rotation matrices
    M1 = rotate(asec2rad(py),'x')
    M2 = rotate(asec2rad(px),'y')
    M3 = rotate(-asec2rad(sprime),'z')
    # calculate the combined rotation matrix
    return np.einsum('ij...,jk...,kl...->il...', M1, M2, M3)

def _precession_matrix(T: float | np.ndarray):
    """
    Precession rotation matrix
    :cite:p:`Capitaine:2003fx,Capitaine:2003fw,Lieske:1977ug`

    Parameters
    ----------
    T: np.ndarray
        Centuries since 2000-01-01T12:00:00
    """
    # equatorial precession angles Lieske et al. (1977)
    # Capitaine et al. (2003), eqs. (4), (37), & (39).
    # obliquity of the ecliptic
    epsilon0 = 84381.406
    EPS = asec2rad(epsilon0)
    # lunisolar precession
    phi0 = np.array([0.0, 5038.481507, -1.0790069,
        -1.14045e-3, 1.32851e-4, -9.51e-8])
    psi = asec2rad(polynomial_sum(phi0, T))
    # inclination of moving equator on fixed ecliptic
    omega0 = np.array([epsilon0, -2.5754e-2, 5.12623e-2,
        -7.72503e-3, -4.67e-7, 3.337e-7])
    omega = asec2rad(polynomial_sum(omega0, T))
    # planetary precession
    chi0 = np.array([0.0, 10.556403, -2.3814292,
        -1.21197e-3, 1.70663e-4, -5.60e-8])
    chi = asec2rad(polynomial_sum(chi0, T))
    # compute elements of precession rotation matrix
    P = np.zeros((3,3,len(np.atleast_1d(T))))
    P[0,0,:] = np.cos(chi)*np.cos(-psi) - \
        np.sin(-psi)*np.sin(chi)*np.cos(-omega)
    P[0,1,:] = np.cos(chi)*np.sin(-psi)*np.cos(EPS) + \
        np.sin(chi)*np.cos(-omega)*np.cos(-psi)*np.cos(EPS) - \
        np.sin(EPS)*np.sin(chi)*np.sin(-omega)
    P[0,2,:] = np.cos(chi)*np.sin(-psi)*np.sin(EPS) + \
        np.sin(chi)*np.cos(-omega)*np.cos(-psi)*np.sin(EPS) + \
        np.cos(EPS)*np.sin(chi)*np.sin(-omega)
    P[1,0,:] = -np.sin(chi)*np.cos(-psi) - \
        np.sin(-psi)*np.cos(chi)*np.cos(-omega)
    P[1,1,:] = -np.sin(chi)*np.sin(-psi)*np.cos(EPS) + \
        np.cos(chi)*np.cos(-omega)*np.cos(-psi)*np.cos(EPS) - \
        np.sin(EPS)*np.cos(chi)*np.sin(-omega)
    P[1,2,:] = -np.sin(chi)*np.sin(-psi)*np.sin(EPS) + \
        np.cos(chi)*np.cos(-omega)*np.cos(-psi)*np.sin(EPS) + \
        np.cos(EPS)*np.cos(chi)*np.sin(-omega)
    P[2,0,:] = np.sin(-psi)*np.sin(-omega)
    P[2,1,:] = -np.sin(-omega)*np.cos(-psi)*np.cos(EPS) - \
        np.sin(EPS)*np.cos(-omega)
    P[2,2,:] = -np.sin(-omega)*np.cos(-psi)*np.sin(EPS) + \
        np.cos(-omega)*np.cos(EPS)
    # return the rotation matrix
    return P

def _correct_aberration(position, velocity):
    """
    Correct a relative position for aberration effects
    :cite:p:`Kaplan:1989cf`

    Parameters
    ----------
    position: np.ndarray
        Position vector in astronomical units
    velocity: np.ndarray
        Velocity vector in astronomical units per day
    """
    # number of seconds per day
    day = 86400.0
    # speed of light in meters per second
    c = 299792458.0
    # astronomical unit in meters
    AU = 149597870700.0
    # speed of light in AU/day (i.e. one light day)
    c_prime = c * day / AU
    # total distance
    distance = np.sqrt(np.sum(position*position, axis=0))
    tau = distance / c_prime
    # speed
    speed = np.sqrt(np.sum(velocity*velocity, axis=0))
    beta = speed / c_prime
    # Kaplan et al. (1989) eq. 16
    # (use divide function to avoid error if denominator is zero)
    cosD = np.divide(np.sum(position*velocity, axis=0), distance*speed)
    # calculate adjustments
    gamma = np.sqrt(1.0 - beta * beta)
    f1 = beta * cosD
    f2 = (1.0 + f1 / (1.0 + gamma)) * tau
    # correct for aberration of light travel time (eq. 17)
    u = (gamma*position + f2*velocity)/(1.0 + f1)
    # return corrected position converted to meters
    x, y, z = u * AU
    return (x, y, z)

def _parse_table_5_2e():
    """Parse table with expressions for Greenwich Sidereal Time
    provided in `Chapter 5
    <https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.2e.txt>`_
    of :cite:t:`Petit:2010tp`
    """
    table_5_2e = get_data_path(['data','tab5.2e.txt'])
    with table_5_2e.open(mode='r', encoding='utf8') as f:
        file_contents = f.readlines()
    # names and formats
    names = ('i','Cs','Cc','l','lp','F','D','Om','L_Me','L_Ve',
        'L_E','L_Ma','L_J','L_Sa','L_U','L_Ne','p_A')
    formats = ('i','f','f','i','i','i','i','i','i',
        'i','i','i','i','i','i','i','i')
    dtype = np.dtype({'names':names, 'formats':formats})
    # j = 0 terms
    n0 = 33
    j0 = np.zeros((n0), dtype=dtype)
    for i,line in enumerate(file_contents[53:53+n0]):
        j0[i] = np.array(tuple(line.split()), dtype=dtype)
    # j = 1 terms
    n1 = 1
    j1 = np.zeros((n1), dtype=dtype)
    for i,line in enumerate(file_contents[90:90+n1]):
        j1[i] = np.array(tuple(line.split()), dtype=dtype)
    # return the table
    return (j0, j1)

def _parse_table_5_3a():
    """Parse table with IAU 2000A lunisolar and planetary components
    of nutation in longitude provided in `Chapter 5
    <https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.2e.txt>`_
    of :cite:t:`Petit:2010tp`
    """
    table_5_3a = get_data_path(['data','tab5.3a.txt'])
    with table_5_3a.open(mode='r', encoding='utf8') as f:
        file_contents = f.readlines()
    # names and formats
    names = ('i','As','Ac','l','lp','F','D','Om','L_Me','L_Ve',
        'L_E','L_Ma','L_J','L_Sa','L_U','L_Ne','p_A')
    formats = ('i','f','f','i','i','i','i','i','i',
        'i','i','i','i','i','i','i','i')
    dtype = np.dtype({'names':names, 'formats':formats})
    # j = 0 terms
    n0 = 1320
    j0 = np.zeros((n0), dtype=dtype)
    for i,line in enumerate(file_contents[22:22+n0]):
        j0[i] = np.array(tuple(line.split()), dtype=dtype)
    # j = 1 terms
    n1 = 38
    j1 = np.zeros((n1), dtype=dtype)
    for i,line in enumerate(file_contents[1348:1348+n1]):
        j1[i] = np.array(tuple(line.split()), dtype=dtype)
    # return the table
    return (j0, j1)

def _parse_table_5_3b():
    """Parse table with IAU 2000A lunisolar and planetary components
    of nutation in obliquity provided in `Chapter 5
    <https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.2e.txt>`_
    of :cite:t:`Petit:2010tp`
    """
    table_5_3b = get_data_path(['data','tab5.3b.txt'])
    with table_5_3b.open(mode='r', encoding='utf8') as f:
        file_contents = f.readlines()
    # names and formats
    names = ('i','Bs','Bc','l','lp','F','D','Om','L_Me','L_Ve',
        'L_E','L_Ma','L_J','L_Sa','L_U','L_Ne','p_A')
    formats = ('i','f','f','i','i','i','i','i','i',
        'i','i','i','i','i','i','i','i')
    dtype = np.dtype({'names':names, 'formats':formats})
    # j = 0 terms
    n0 = 1037
    j0 = np.zeros((n0), dtype=dtype)
    for i,line in enumerate(file_contents[22:22+n0]):
        j0[i] = np.array(tuple(line.split()), dtype=dtype)
    # j = 1 terms
    n1 = 19
    j1 = np.zeros((n1), dtype=dtype)
    for i,line in enumerate(file_contents[1065:1065+n1]):
        j1[i] = np.array(tuple(line.split()), dtype=dtype)
    # return the table
    return (j0, j1)
