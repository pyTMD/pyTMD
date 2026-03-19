#!/usr/bin/env python
"""
astro.py
Written by Tyler Sutterley (03/2026)
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
    Updated 03/2026: added functions to compute the geocentric positions
        of the sun and moon (latitude, longitude and distance)
    Updated 10/2025: change default directory for JPL SSD kernels to cache
    Updated 09/2025: added function to compute the planetary mean longitudes
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
        fix solar ephemerides function to include SSB to sun segment
        use a higher resolution estimate of the Greenwich hour angle
        use ITRS reference frame for high-resolution ephemerides calculations
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
    rotate,
)
from pyTMD.utilities import (
    get_data_path,
    get_cache_path,
    import_dependency,
    dependency_available,
)
from pyTMD.datasets import fetch_jpl_ssd

# attempt imports
jplephem = import_dependency("jplephem")
jplephem.spk = import_dependency("jplephem.spk")
jplephem_available = dependency_available("jplephem")

__all__ = [
    "mean_longitudes",
    "planetary_longitudes",
    "doodson_arguments",
    "delaunay_arguments",
    "schureman_arguments",
    "mean_obliquity",
    "equation_of_time",
    "solar_ecef",
    "solar_approximate",
    "solar_ephemerides",
    "solar_latitude",
    "solar_longitude",
    "solar_distance",
    "lunar_ecef",
    "lunar_approximate",
    "lunar_ephemerides",
    "lunar_latitude",
    "lunar_longitude",
    "lunar_distance",
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
    "_meeus_table_47A",
    "_meeus_table_47B",
    "_parse_table_5_2e",
    "_parse_table_5_3a",
    "_parse_table_5_3b",
]

# default JPL Spacecraft and Planet ephemerides kernel
_default_kernel = get_cache_path("de440s.bsp")

# number of days between the Julian day epoch and MJD
_jd_mjd = 2400000.5
# number of days between MJD and the J2000 epoch
_mjd_j2000 = 51544.5
# number of days between the Julian day epoch and J2000 epoch
_jd_j2000 = _jd_mjd + _mjd_j2000
# Julian century
_century = 36525.0
# Julian millennia
_millennia = 10.0 * _century


# PURPOSE: compute the basic astronomical mean longitudes
def mean_longitudes(MJD: np.ndarray, **kwargs):
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
    kwargs.setdefault("method", "Cartwright")
    # check for deprecated method
    if kwargs.get("MEEUS"):
        warnings.warn("Deprecated argument", DeprecationWarning)
        kwargs["method"] = "Meeus"
    elif kwargs.get("ASTRO5"):
        warnings.warn("Deprecated argument", DeprecationWarning)
        kwargs["method"] = "ASTRO5"
    # compute the mean longitudes
    if kwargs["method"].title() == "Meeus":
        # convert from MJD to days relative to 2000-01-01T12:00:00
        T = MJD - _mjd_j2000
        # mean longitude of moon
        lunar_longitude = np.array(
            [
                218.3164591,
                13.17639647754579,
                -9.9454632e-13,
                3.8086292e-20,
                -8.6184958e-27,
            ]
        )
        S = polynomial_sum(lunar_longitude, T)
        # mean longitude of sun
        solar_longitude = np.array(
            [280.46645, 0.985647360164271, 2.2727347e-13]
        )
        H = polynomial_sum(solar_longitude, T)
        # mean longitude of lunar perigee
        lunar_perigee = np.array(
            [
                83.3532430,
                0.11140352391786447,
                -7.7385418e-12,
                -2.5636086e-19,
                2.95738836e-26,
            ]
        )
        P = polynomial_sum(lunar_perigee, T)
        # mean longitude of ascending lunar node
        lunar_node = np.array(
            [
                125.0445550,
                -0.052953762762491446,
                1.55628359e-12,
                4.390675353e-20,
                -9.26940435e-27,
            ]
        )
        N = polynomial_sum(lunar_node, T)
        # mean longitude of solar perigee (Simon et al., 1994)
        Ps = 282.94 + (1.7192 * T) / _century
    elif kwargs["method"].upper() == "ASTRO5":
        # convert from MJD to centuries relative to 2000-01-01T12:00:00
        T = (MJD - _mjd_j2000) / _century
        # mean longitude of moon (p. 338)
        lunar_longitude = np.array(
            [218.3164477, 481267.88123421, -1.5786e-3, 1.855835e-6, -1.53388e-8]
        )
        S = polynomial_sum(lunar_longitude, T)
        # mean longitude of sun (p. 338)
        lunar_elongation = np.array(
            [297.8501921, 445267.1114034, -1.8819e-3, 1.83195e-6, -8.8445e-9]
        )
        H = polynomial_sum(lunar_longitude - lunar_elongation, T)
        # mean longitude of lunar perigee (p. 343)
        lunar_perigee = np.array(
            [83.3532465, 4069.0137287, -1.032e-2, -1.249172e-5]
        )
        P = polynomial_sum(lunar_perigee, T)
        # mean longitude of ascending lunar node (p. 144)
        lunar_node = np.array([125.04452, -1934.136261, 2.0708e-3, 2.22222e-6])
        N = polynomial_sum(lunar_node, T)
        # mean longitude of solar perigee (Simon et al., 1994)
        Ps = 282.94 + 1.7192 * T
    elif kwargs["method"].upper() == "IERS":
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


# PURPOSE: compute the mean longitudes of the 5 closest planets
def planetary_longitudes(MJD: np.ndarray):
    r"""
    Computes the astronomical mean longitudes of the 5 closest planets
    :cite:p:`Bretagnon:1988wg,Meeus:1991vh,Simon:1994vo`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date

    Returns
    -------
    LMe: np.ndarray
        mean longitude of Mercury (degrees)
    LVe: np.ndarray
        mean longitude of Venus (degrees)
    LMa: np.ndarray
        mean longitude of Mars (degrees)
    LJu: np.ndarray
        mean longitude of Jupiter (degrees)
    LSa: np.ndarray
        mean longitude of Saturn (degrees)
    """
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - _mjd_j2000) / _century
    # mean longitudes of Mercury
    mercury_longitude = np.array([252.250906, 149474.0722491, 3.035e-4, 1.8e-8])
    LMe = polynomial_sum(mercury_longitude, T)
    # mean longitudes of Venus
    venus_longitude = np.array([181.979801, 58519.2130302, 3.1014e-4, 1.5e-8])
    LVe = polynomial_sum(venus_longitude, T)
    # mean longitudes of Mars
    mars_longitude = np.array([355.433, 19141.6964471, 3.1052e-4, 1.6e-8])
    LMa = polynomial_sum(mars_longitude, T)
    # mean longitudes of Jupiter
    jupiter_longitude = np.array([34.351519, 3036.3027748, 2.233e-4, 3.7e-8])
    LJu = polynomial_sum(jupiter_longitude, T)
    # mean longitudes of Saturn
    saturn_longitude = np.array([50.077444, 1223.5110686, 5.1908e-4, -3.0e-8])
    LSa = polynomial_sum(saturn_longitude, T)
    # take the modulus of each
    LMe = normalize_angle(LMe)
    LVe = normalize_angle(LVe)
    LMa = normalize_angle(LMa)
    LJu = normalize_angle(LJu)
    LSa = normalize_angle(LSa)
    # return as tuple
    return (LMe, LVe, LMa, LJu, LSa)


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
    T = (MJD - _mjd_j2000) / _century
    # hour of the day
    hour = np.mod(MJD, 1) * 24.0
    # calculate Doodson phase angles
    # mean longitude of moon (degrees)
    lunar_longitude = np.array(
        [218.3164477, 481267.88123421, -1.5786e-3, 1.855835e-6, -1.53388e-8]
    )
    S = polynomial_sum(lunar_longitude, T)
    # mean lunar time (degrees)
    if equinox:
        # create timescale from Modified Julian Day (MJD)
        ts = timescale.time.Timescale(MJD=MJD)
        # use Greenwich Mean Sidereal Time (GMST) from the
        # Equinox method converted to degrees
        TAU = 360.0 * ts.st + 180.0 - S
    else:
        lambda_coefficients = np.array(
            [280.4606184, 36000.7700536, 3.8793e-4, -2.58e-8]
        )
        LAMBDA = polynomial_sum(lambda_coefficients, T)
        TAU = (hour * 15.0) - S + LAMBDA
    # calculate correction for mean lunar longitude (degrees)
    if apply_correction:
        lunar_correction = np.array(
            [0.0, 1.396971278, 3.08889e-4, 2.1e-8, 7.0e-9]
        )
        PR = polynomial_sum(lunar_correction, T)
        S += PR
    # mean longitude of sun (degrees)
    solar_longitude = np.array(
        [280.46645, 36000.7697489, 3.0322222e-4, 2.0e-8, -6.54e-9]
    )
    H = polynomial_sum(solar_longitude, T)
    # mean longitude of lunar perigee (degrees)
    lunar_perigee = np.array(
        [83.3532465, 4069.0137287, -1.032172222e-2, -1.24991e-5, 5.263e-8]
    )
    P = polynomial_sum(lunar_perigee, T)
    # negative of the mean longitude of the ascending node
    # of the moon (degrees)
    lunar_node = np.array(
        [234.95544499, 1934.13626197, -2.07561111e-3, -2.13944e-6, 1.65e-8]
    )
    Np = polynomial_sum(lunar_node, T)
    # mean longitude of solar perigee (degrees)
    solar_perigee = np.array(
        [282.93734098, 1.71945766667, 4.5688889e-4, -1.778e-8, -3.34e-9]
    )
    Ps = polynomial_sum(solar_perigee, T)
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
    T = (MJD - _mjd_j2000) / _century
    # 360 degrees
    circle = 1296000
    # mean anomaly of the moon (arcseconds)
    lunar_anomaly = np.array(
        [485868.249036, 1717915923.2178, 31.8792, 0.051635, -2.447e-04]
    )
    l = polynomial_sum(lunar_anomaly, T)
    # mean anomaly of the sun (arcseconds)
    solar_anomaly = np.array(
        [1287104.79305, 129596581.0481, -0.5532, 1.36e-4, -1.149e-05]
    )
    lp = polynomial_sum(solar_anomaly, T)
    # mean argument of the moon (arcseconds)
    # (angular distance from the ascending node)
    lunar_argument = np.array(
        [335779.526232, 1739527262.8478, -12.7512, -1.037e-3, 4.17e-6]
    )
    F = polynomial_sum(lunar_argument, T)
    # mean elongation of the moon from the sun (arcseconds)
    lunisolar_elongation = np.array(
        [1072260.70369, 1602961601.2090, -6.3706, 6.593e-3, -3.169e-05]
    )
    D = polynomial_sum(lunisolar_elongation, T)
    # mean longitude of the ascending node of the moon (arcseconds)
    lunar_node = np.array(
        [450160.398036, -6962890.5431, 7.4722, 7.702e-3, -5.939e-05]
    )
    N = polynomial_sum(lunar_node, T)
    # take the modulus of each and convert to radians
    l = asec2rad(normalize_angle(l, circle=circle))
    lp = asec2rad(normalize_angle(lp, circle=circle))
    F = asec2rad(normalize_angle(F, circle=circle))
    D = asec2rad(normalize_angle(D, circle=circle))
    N = asec2rad(normalize_angle(N, circle=circle))
    # return as tuple
    return (l, lp, F, D, N)


def schureman_arguments(P: np.ndarray, N: np.ndarray):
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
    I = np.arccos(0.913694997 - 0.035692561 * np.cos(N))
    # longitude in the moon's orbit of lunar intersection
    at1 = np.arctan(1.01883 * np.tan(N / 2.0))
    at2 = np.arctan(0.64412 * np.tan(N / 2.0))
    xi = -at1 - at2 + N
    xi = np.arctan2(np.sin(xi), np.cos(xi))
    # right ascension of lunar intersection
    nu = at1 - at2
    # mean longitude of lunar perigee reckoned from the lunar intersection
    # Schureman (page 41)
    p = P - xi
    # Schureman (page 42) equation 202
    Q = np.arctan((5.0 * np.cos(I) - 1.0) * np.tan(p) / (7.0 * np.cos(I) + 1.0))
    # Schureman (page 41) equation 197
    Qa = np.power(2.31 + 1.435 * np.cos(2.0 * p), -0.5)
    # Schureman (page 42) equation 204
    Qu = p - Q
    # Schureman (page 44) equation 214
    P_R = np.sin(2.0 * p)
    Q_R = np.power(np.tan(I / 2.0), -2.0) / 6.0 - np.cos(2.0 * p)
    Ru = np.arctan(P_R / Q_R)
    # Schureman (page 44) equation 213
    # note that Ra is normally used as an inverse (1/Ra)
    term1 = 12.0 * np.power(np.tan(I / 2.0), 2.0) * np.cos(2.0 * p)
    term2 = 36.0 * np.power(np.tan(I / 2.0), 4.0)
    Ra = np.power(1.0 - term1 + term2, -0.5)
    # Schureman (page 45) equation 224
    P_prime = np.sin(2.0 * I) * np.sin(nu)
    Q_prime = np.sin(2.0 * I) * np.cos(nu) + 0.3347
    nu_p = np.arctan(P_prime / Q_prime)
    # Schureman (page 46) equation 232
    P_sec = (np.sin(I) ** 2) * np.sin(2.0 * nu)
    Q_sec = (np.sin(I) ** 2) * np.cos(2.0 * nu) + 0.0727
    nu_s = 0.5 * np.arctan(P_sec / Q_sec)
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
    T = (MJD - _mjd_j2000) / _century
    # mean obliquity of the ecliptic (arcseconds)
    epsilon0 = np.array(
        [84381.406, -46.836769, -1.831e-4, 2.00340e-4, -5.76e-07, -4.34e-08]
    )
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
    T = (MJD - _mjd_j2000) / _century
    # mean longitude of sun (degrees)
    mean_longitude = np.array(
        [280.46645, 36000.7697489, 3.0322222e-4, 2.0e-8, -6.54e-9]
    )
    H = polynomial_sum(mean_longitude, T)
    # mean anomaly of the sun (degrees)
    mean_anomaly = np.array(
        [357.5291092, 35999.0502909, -0.0001536, 1.0 / 24490000.0]
    )
    lp = polynomial_sum(mean_anomaly, T)
    # take the modulus of each
    H = normalize_angle(H)
    lp = normalize_angle(lp)
    # ecliptic longitude of the sun (degrees)
    lambda_sun = (
        H
        + 1.915 * np.sin(np.radians(lp))
        + 0.020 * np.sin(2.0 * np.radians(lp))
    )
    # calculate the equation of time (degrees)
    E = (
        -1.915 * np.sin(np.radians(lp))
        - 0.020 * np.sin(2.0 * np.radians(lp))
        + 2.466 * np.sin(2.0 * np.radians(lambda_sun))
        - 0.053 * np.sin(4.0 * np.radians(lambda_sun))
    )
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
    kwargs.setdefault("ephemerides", "approximate")
    if kwargs["ephemerides"].lower() == "approximate":
        return solar_approximate(MJD, **kwargs)
    elif kwargs["ephemerides"].upper() == "JPL":
        assert jplephem_available, "jplephem is required for JPL ephemerides"
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
    r_sun = 1e9 * (149.619 - 2.499 * np.cos(M) - 0.021 * np.cos(2.0 * M))
    # series expansion for ecliptic longitude of the sun (radians)
    lambda_sun = Ps + M + asec2rad(6892.0 * np.sin(M) + 72.0 * np.sin(2.0 * M))
    # ecliptic latitude is equal to 0 within 1 arcminute
    # obliquity of the J2000 ecliptic (radians)
    epsilon_j2000 = np.radians(23.43929111)
    # convert to position vectors
    x = r_sun * np.cos(lambda_sun)
    y = r_sun * np.sin(lambda_sun) * np.cos(epsilon_j2000)
    z = r_sun * np.sin(lambda_sun) * np.sin(epsilon_j2000)
    # Greenwich hour angle (radians)
    rot_z = rotate(np.radians(ts.gha), "z")
    # rotate to cartesian (ECEF) coordinates
    # ignoring polar motion and length-of-day variations
    X = rot_z[0, 0, :] * x + rot_z[0, 1, :] * y + rot_z[0, 2, :] * z
    Y = rot_z[1, 0, :] * x + rot_z[1, 1, :] * y + rot_z[1, 2, :] * z
    Z = rot_z[2, 0, :] * x + rot_z[2, 1, :] * y + rot_z[2, 2, :] * z
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
    kwargs.setdefault("kernel", _default_kernel)
    kwargs.setdefault("include_aberration", False)
    # create timescale from Modified Julian Day (MJD)
    ts = timescale.time.Timescale(MJD=MJD)
    # difference to convert to Barycentric Dynamical Time (TDB)
    tdb2 = getattr(ts, "tdb_tt") if hasattr(ts, "tdb_tt") else 0.0
    # download kernel file if not currently existing
    if not pathlib.Path(kwargs["kernel"]).exists():
        fetch_jpl_ssd(kernel=None, local=kwargs["kernel"])
    # read JPL ephemerides kernel
    SPK = jplephem.spk.SPK.open(kwargs["kernel"])
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
    if kwargs["include_aberration"]:
        # distance of 1 Astronomical Unit (kilometers)
        AU = 149597870.700
        # position in astronomical units
        position = (xyz_10 - xyz_399 - xyz_3) / AU
        # velocity in astronomical units per day
        velocity = (vel_399 + vel_3 - vel_10) / AU
        # correct for aberration and convert to meters
        x, y, z = _correct_aberration(position, velocity)
    else:
        # convert positions from kilometers to meters
        x, y, z = 1e3 * (xyz_10 - xyz_399 - xyz_3)
    # rotate to cartesian (ECEF) coordinates
    rot_z = itrs((ts.utc - _jd_j2000) / ts.century)
    X = rot_z[0, 0, :] * x + rot_z[0, 1, :] * y + rot_z[0, 2, :] * z
    Y = rot_z[1, 0, :] * x + rot_z[1, 1, :] * y + rot_z[1, 2, :] * z
    Z = rot_z[2, 0, :] * x + rot_z[2, 1, :] * y + rot_z[2, 2, :] * z
    # return the ECEF coordinates
    return (X, Y, Z)


def solar_latitude(
    MJD: np.ndarray,
    **kwargs,
):
    """
    Calculates the geocentric latitude of the sun
    :cite:p:`Meeus:1991vh,Bretagnon:1988wg`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date

    Returns
    -------
    beta: np.ndarray
        latitude of the sun (radians)
    """
    # set default keyword arguments
    kwargs.setdefault("method", "Meeus")
    if kwargs["method"].lower() == "vsop87":
        # convert from MJD to millennia relative to 2000-01-01T12:00:00
        T = (MJD - _mjd_j2000) / _millennia
        # calculate the geocentric latitude of the sun
        # using the terms in Appendix III of Meeus (1991)
        # beta0 terms
        beta0 = np.array(
            [
                [280e-8, 3.199, 84334.662],
                [102e-8, 5.422, 5507.553],
                [80e-8, 3.88, 5223.69],
                [44e-8, 3.70, 2352.87],
                [32e-8, 4.00, 1577.34],
            ]
        )
        # beta1 terms
        beta1 = np.array(
            [
                [9e-8, 3.90, 5507.55],
                [6e-8, 1.73, 5223.69],
            ]
        )
        # list of coefficients for each power
        coefficients = [beta0, beta1]
        # latitude in degrees
        beta = 0.0
        # iterate over each power and coefficients
        for p, B in enumerate(coefficients):
            for i, (a, b, c) in enumerate(B):
                beta -= np.degrees(a) * np.cos(b + c * T) * np.power(T, p)
    else:
        # latitude of the sun is equal to 0 within 1 arcminute
        beta = np.zeros_like(MJD)
    # convert to radians
    beta = np.radians(beta)
    # return the latitude of the sun
    return beta


def solar_longitude(
    MJD: np.ndarray,
    include_aberration: bool = True,
    **kwargs,
):
    """
    Calculates the apparent longitude of the sun
    :cite:p:`Kubo:1980ut,Meeus:1991vh,Bretagnon:1988wg`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    include_aberration: bool, default True
        Correct for aberration effects
    method: str, default 'Meeus'
        Method of calculating the longitude

        - ``'Kubo'``: :cite:t:`Kubo:1980ut`
        - ``'Meeus'``: :cite:t:`Meeus:1991vh`
        - ``'VSOP87'``: :cite:t:`Bretagnon:1988wg`

    Returns
    -------
    H: np.ndarray
        longitude of the sun (radians)
    """
    # set default keyword arguments
    kwargs.setdefault("method", "Meeus")
    if kwargs["method"].lower() == "meeus":
        # convert from MJD to centuries relative to 2000-01-01T12:00:00
        T = (MJD - _mjd_j2000) / _century
        # mean longitude of sun (degrees)
        solar_mean_longitude = np.array([280.46646, 36000.76983, 0.0003032])
        H0 = polynomial_sum(solar_mean_longitude, T)
        # mean anomaly of the sun (radians)
        solar_anomaly = np.array([357.5256, 35999.049, -1.559e-4, -4.8e-7])
        M = np.radians(polynomial_sum(solar_anomaly, T))
        # equation of center
        C = (
            polynomial_sum([1.914602, -0.004817, -0.000014], T) * np.sin(M)
            + polynomial_sum([0.019993, -0.000101], T) * np.sin(2.0 * M)
            + 0.000289 * np.sin(3.0 * M)
        )
        # true longitude of the sun (degrees)
        H = H0 + C
        # correct for aberration of light
        if include_aberration:
            # convert to apparent longitude
            omega = np.radians(125.04 - 1934.136 * T)
            H += -0.00569 - 0.00478 * np.sin(omega)
    elif kwargs["method"].lower() == "kubo":
        # convert from MJD to centuries relative to 2000-01-01T12:00:00
        T = (MJD - _mjd_j2000) / _century
        # coefficients for calculating the longitude of the sun
        coefficients = np.array(
            [
                [19147e-4, 267.52, 35999.05],
                [200e-4, 265.10, 71998.10],
                [20e-4, 158.0, 32964.0],
                [18e-4, 159.0, 19.0],
                [18e-4, 208.0, 445267.0],
                [15e-4, 254.0, 45038.0],
                [13e-4, 352.0, 22519.0],
                [7e-4, 45.0, 65929.0],
                [7e-4, 110.0, 3035.0],
                [7e-4, 64.0, 9038.0],
                [6e-4, 316.0, 33718.0],
                [5e-4, 118.0, 155.0],
                [5e-4, 221.0, 2281.0],
                [4e-4, 48.0, 29930.0],
                [4e-4, 161.0, 31557.0],
            ]
        )
        # calculate the longitude of the sun
        H = 280.4659 + 36000.7695 * T
        # calculate the true longitude of the sun (degrees)
        for i, (a, b, c) in enumerate(coefficients):
            if i == 0:
                a -= 48e-4 * T
            H += a * np.cos(np.radians(b + c * T))
        # correct for aberration of light
        if include_aberration:
            # convert to apparent longitude
            omega = np.radians(145.0 + 1934.0 * T)
            H += -0.0057 + 0.0048 * np.cos(np.radians(omega))
    elif kwargs["method"].lower() == "vsop87":
        # convert from MJD to millennia relative to 2000-01-01T12:00:00
        T = (MJD - _mjd_j2000) / _millennia
        # calculate the longitude of the sun
        # using the terms in Appendix III of Meeus (1991)
        # L0 terms
        L0 = np.array(
            [
                [175347046e-8, 0.0, 0.0],
                [3341656e-8, 4.6692568, 6283.0758500],
                [34894e-8, 4.62610, 12566.15170],
                [3497e-8, 2.7441, 5753.3849],
                [3418e-8, 2.8289, 3.5231],
                [3136e-8, 3.6277, 77713.7715],
                [2676e-8, 4.4181, 7860.4194],
                [2343e-8, 6.1352, 3930.2097],
                [1324e-8, 0.7425, 11506.7698],
                [1273e-8, 2.0371, 529.6910],
                [1199e-8, 1.1096, 1577.3435],
                [990e-8, 5.233, 5884.927],
                [902e-8, 2.045, 26.298],
                [857e-8, 3.508, 398.149],
                [780e-8, 1.179, 5223.694],
                [753e-8, 2.533, 5507.553],
                [505e-8, 4.583, 18849.228],
                [492e-8, 4.205, 775.523],
                [357e-8, 2.92, 0.067],
                [317e-8, 5.849, 11790.629],
                [284e-8, 1.899, 796.298],
                [271e-8, 0.315, 10977.079],
                [243e-8, 0.345, 5486.778],
                [206e-8, 4.806, 2544.314],
                [205e-8, 1.869, 5573.143],
                [202e-8, 2.458, 6069.777],
                [156e-8, 0.833, 213.299],
                [132e-8, 3.411, 2942.463],
                [126e-8, 1.083, 20.775],
                [115e-8, 0.645, 0.980],
                [103e-8, 0.636, 4694.003],
                [102e-8, 0.976, 15720.839],
                [102e-8, 4.267, 7.114],
                [99e-8, 6.21, 2146.17],
                [98e-8, 0.68, 155.42],
                [86e-8, 5.98, 161000.69],
                [85e-8, 1.30, 6275.96],
                [85e-8, 3.67, 71430.70],
                [80e-8, 1.81, 17260.15],
                [79e-8, 3.04, 12036.46],
                [75e-8, 1.76, 5088.63],
                [74e-8, 3.50, 3154.69],
                [74e-8, 4.68, 801.82],
                [70e-8, 0.83, 9437.76],
                [62e-8, 3.98, 8827.39],
                [61e-8, 1.82, 7084.90],
                [57e-8, 2.78, 6286.60],
                [56e-8, 4.39, 14143.50],
                [56e-8, 3.47, 6279.55],
                [52e-8, 0.19, 12139.55],
                [52e-8, 1.33, 1748.02],
                [51e-8, 0.28, 5856.48],
                [49e-8, 0.49, 1194.45],
                [41e-8, 5.37, 8429.24],
                [41e-8, 2.40, 19651.05],
                [39e-8, 6.17, 10447.39],
                [37e-8, 6.04, 10213.29],
                [37e-8, 2.57, 1059.38],
                [36e-8, 1.71, 2352.87],
                [36e-8, 1.78, 6812.77],
                [33e-8, 0.59, 17789.85],
                [30e-8, 0.44, 83996.85],
                [30e-8, 2.74, 1349.87],
                [25e-8, 3.16, 4690.48],
            ]
        )
        # L1 terms
        L1 = np.array(
            [
                [628331996747e-8, 0.0, 0.0],
                [206059e-8, 2.678235, 6283.075850],
                [4303e-8, 2.6351, 12566.1517],
                [425e-8, 1.590, 3.523],
                [119e-8, 5.796, 26.298],
                [109e-8, 2.966, 1577.344],
                [93e-8, 2.59, 18849.23],
                [72e-8, 1.14, 529.69],
                [68e-8, 1.87, 398.15],
                [67e-8, 4.41, 5507.55],
                [59e-8, 2.89, 5223.69],
                [56e-8, 2.17, 155.42],
                [45e-8, 0.40, 796.30],
                [36e-8, 0.47, 775.52],
                [29e-8, 2.65, 7.11],
                [21e-8, 5.34, 0.98],
                [19e-8, 1.85, 5498.78],
                [19e-8, 4.97, 213.30],
                [17e-8, 2.99, 6275.96],
                [16e-8, 0.03, 2544.31],
                [16e-8, 1.43, 2146.17],
                [15e-8, 1.21, 10977.08],
                [12e-8, 2.83, 1748.02],
                [12e-8, 3.26, 5088.63],
                [12e-8, 5.27, 1194.45],
                [12e-8, 2.08, 4694.00],
                [11e-8, 0.77, 553.57],
                [10e-8, 1.30, 6286.60],
                [10e-8, 4.24, 1349.87],
                [9e-8, 2.70, 242.73],
                [9e-8, 5.64, 951.72],
                [8e-8, 5.30, 2352.87],
                [6e-8, 2.65, 9437.76],
                [6e-8, 4.67, 4690.48],
            ]
        )
        # L2 terms
        L2 = np.array(
            [
                [52919e-8, 0.0, 0.0],
                [8270e-8, 1.0721, 6283.0758],
                [309e-8, 0.867, 12566.152],
                [27e-8, 0.05, 3.52],
                [16e-8, 5.19, 26.30],
                [16e-8, 3.68, 155.42],
                [10e-8, 0.76, 18849.23],
                [9e-8, 2.06, 77713.77],
                [7e-8, 0.83, 775.52],
                [5e-8, 4.66, 1577.34],
                [4e-8, 1.03, 7.11],
                [4e-8, 3.44, 5573.14],
                [3e-8, 5.14, 796.30],
                [3e-8, 6.05, 5507.55],
                [3e-8, 1.19, 242.73],
                [3e-8, 6.12, 529.69],
                [3e-8, 0.31, 398.15],
                [3e-8, 2.28, 553.57],
                [2e-8, 4.38, 5223.69],
                [2e-8, 3.75, 0.98],
            ]
        )
        # L3 terms
        L3 = np.array(
            [
                [289e-8, 5.844, 6283.076],
                [35e-8, 0.0, 0.0],
                [17e-8, 5.49, 12566.15],
                [3e-8, 5.20, 155.42],
                [1e-8, 4.72, 3.52],
                [1e-8, 5.30, 18849.23],
                [1e-8, 5.97, 242.73],
            ]
        )
        # L4 terms
        L4 = np.array(
            [
                [114e-8, 3.142, 0.0],
                [8e-8, 4.13, 6283.08],
                [1e-8, 3.84, 12566.15],
            ]
        )
        # L5 terms
        L5 = np.array([[1e-8, 3.14, 0.0]])
        # list of coefficients for each power
        coefficients = [L0, L1, L2, L3, L4, L5]
        # longitude in degrees
        H = 180.0
        # iterate over each power and coefficients
        for p, L in enumerate(coefficients):
            for i, (a, b, c) in enumerate(L):
                H += np.degrees(a) * np.cos(b + c * T) * np.power(T, p)
        # correct for aberration of light
        if include_aberration:
            # convert to apparent longitude
            omega = np.radians(125.04 - 1934.136 * T)
            H += -0.00569 - 0.00478 * np.sin(omega)
    # take the modulus and convert to radians
    H = np.radians(normalize_angle(H, circle=360.0))
    # return the longitude of the sun
    return H


def solar_distance(
    MJD: np.ndarray,
    AU: float = 1.495978707e11,
    **kwargs,
):
    """
    Calculates the distance from the sun to the Earth
    :cite:p:`Kubo:1980ut,Meeus:1991vh,Bretagnon:1988wg`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    AU: float, default 1.495978707e11
        distance of 1 Astronomical Unit (meters)
    method: str, default 'Meeus'
        Method of calculating the distance

        - ``'Kubo'``: :cite:t:`Kubo:1980ut`
        - ``'Meeus'``: :cite:t:`Meeus:1991vh`
        - ``'VSOP87'``: :cite:t:`Bretagnon:1988wg`

    Returns
    -------
    R: np.ndarray
        distance from the sun to the Earth (meters)
    """
    # set default keyword arguments
    kwargs.setdefault("method", "Meeus")
    if kwargs["method"].lower() == "meeus":
        # convert from MJD to centuries relative to 2000-01-01T12:00:00
        T = (MJD - _mjd_j2000) / _century
        # mean anomaly of the sun (radians)
        solar_anomaly = np.array([357.5256, 35999.049, -1.559e-4, -4.8e-7])
        M = np.radians(polynomial_sum(solar_anomaly, T))
        # eccentricity of the Earth's orbit
        earth_eccentricity = np.array([16708.634e-6, -42.037e-6, -1.267e-7])
        ee = polynomial_sum(earth_eccentricity, T)
        # equation of center
        C = (
            polynomial_sum([1.914602, -0.004817, -0.000014], T) * np.sin(M)
            + polynomial_sum([0.019993, -0.000101], T) * np.sin(2.0 * M)
            + 0.000289 * np.sin(3.0 * M)
        )
        nu = M + np.radians(C)
        solar_au = 1.000001018 * (1.0 - ee**2) / (1.0 + ee * np.cos(nu))
    elif kwargs["method"].lower() == "kubo":
        # convert from MJD to centuries relative to 2000-01-01T12:00:00
        T = (MJD - _mjd_j2000) / _century
        # coefficients for calculating the distance to the sun
        coefficients = np.array(
            [
                [16706e-6, 177.53, 35999.05],
                [139e-6, 175.0, 71998.0],
                [31e-6, 298.0, 445267.0],
                [16e-6, 68.0, 32964.0],
                [16e-6, 164.0, 45038.0],
                [5e-6, 233.0, 22519.0],
                [5e-6, 226.0, 33718.0],
            ]
        )
        # calculate the distance from the sun to the Earth
        solar_au = 1000140e-6
        for i, (a, b, c) in enumerate(coefficients):
            if i == 0:
                a -= 42e-6 * T
            solar_au += a * np.cos(np.radians(b + c * T))
    elif kwargs["method"].lower() == "vsop87":
        # convert from MJD to millennia relative to 2000-01-01T12:00:00
        T = (MJD - _mjd_j2000) / _millennia
        # calculate the distance from the sun to the Earth
        # using the terms in Appendix III of Meeus (1991)
        solar_au = 100013989e-8
        # R0 terms
        R0 = np.array(
            [
                [1670700e-8, 3.098463, 6283.075850],
                [13956e-8, 3.05525, 12566.15170],
                [3084e-8, 5.1985, 77713.7715],
                [1628e-8, 1.1739, 5753.3849],
                [1576e-8, 2.8469, 7860.4194],
                [925e-8, 5.453, 11506.770],
                [542e-8, 4.564, 3930.210],
                [472e-8, 3.661, 5884.927],
                [346e-8, 0.964, 5507.553],
                [329e-8, 5.900, 5223.694],
                [307e-8, 0.299, 5573.143],
                [243e-8, 4.273, 11790.629],
                [212e-8, 5.847, 1577.344],
                [186e-8, 5.022, 10977.079],
                [175e-8, 3.012, 18849.228],
                [110e-8, 5.055, 5486.778],
                [98e-8, 0.89, 6069.78],
                [86e-8, 5.69, 15720.84],
                [86e-8, 1.27, 161000.69],
                [65e-8, 0.27, 17260.15],
                [63e-8, 0.92, 529.69],
                [57e-8, 2.01, 83996.85],
                [56e-8, 5.24, 71430.70],
                [49e-8, 3.25, 2544.31],
                [47e-8, 2.58, 775.52],
                [45e-8, 5.54, 9437.76],
                [43e-8, 6.01, 6275.96],
                [39e-8, 5.36, 4694.00],
                [38e-8, 2.39, 8827.39],
                [37e-8, 0.83, 19651.05],
                [37e-8, 4.90, 12139.46],
                [36e-8, 1.67, 12036.46],
                [35e-8, 1.84, 2942.46],
                [33e-8, 0.24, 7084.90],
                [32e-8, 0.18, 5088.63],
                [32e-8, 1.78, 398.15],
                [28e-8, 1.21, 6286.60],
                [28e-8, 1.90, 6279.55],
                [26e-8, 4.59, 10447.39],
            ]
        )
        # R1 terms
        R1 = np.array(
            [
                [103019e-8, 1.10749, 6283.07585],
                [1721e-8, 1.0644, 12566.1517],
                [702e-8, 3.142, 0.000],
                [32e-8, 1.02, 18849.23],
                [31e-8, 2.84, 5507.55],
                [25e-8, 1.32, 5223.69],
                [18e-8, 1.42, 1577.34],
                [10e-8, 5.91, 10977.08],
                [9e-8, 1.42, 6275.96],
                [9e-8, 0.27, 5486.78],
            ]
        )
        # R2 terms
        R2 = np.array(
            [
                [4359e-8, 5.7846, 6283.0758],
                [124e-8, 5.5790, 12566.1520],
                [12e-8, 3.14, 0.00],
                [9e-8, 3.63, 77713.77],
                [6e-8, 1.87, 5573.14],
                [3e-8, 5.47, 18849.23],
            ]
        )
        # R3 terms
        R3 = np.array(
            [
                [145e-8, 4.273, 6283.076],
                [7e-8, 3.92, 12566.15],
            ]
        )
        # R4 terms
        R4 = np.array(
            [
                [4e-8, 2.56, 6283.08],
            ]
        )
        # list of coefficients for each power
        coefficients = [R0, R1, R2, R3, R4]
        # iterate over each power and coefficient
        for p, R in enumerate(coefficients):
            for i, (a, b, c) in enumerate(R):
                solar_au += a * np.cos(b + c * T) * np.power(T, p)
    # convert from AU to meters
    R = solar_au * AU
    # return the distance from the sun to the Earth
    return R


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
    kwargs.setdefault("ephemerides", "approximate")
    if kwargs["ephemerides"].lower() == "approximate":
        return lunar_approximate(MJD, **kwargs)
    elif kwargs["ephemerides"].upper() == "JPL":
        assert jplephem_available, "jplephem is required for JPL ephemerides"
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
    lunar_longitude = np.array(
        [218.3164477, 481267.88123421, -1.5786e-3, 1.855835e-6, -1.53388e-8]
    )
    s = np.radians(polynomial_sum(lunar_longitude, ts.T))
    # difference between the mean longitude of sun and moon (p. 338)
    lunar_elongation = np.array(
        [297.8501921, 445267.1114034, -1.8819e-3, 1.83195e-6, -8.8445e-9]
    )
    D = np.radians(polynomial_sum(lunar_elongation, ts.T))
    # mean longitude of ascending lunar node (p. 144)
    lunar_node = np.array([125.04452, -1934.136261, 2.0708e-3, 2.22222e-6])
    N = np.radians(polynomial_sum(lunar_node, ts.T))
    F = s - N
    # mean anomaly of the sun (radians)
    M = np.radians((357.5256 + 35999.049 * ts.T))
    # mean anomaly of the moon (radians)
    l = np.radians((134.96292 + 477198.86753 * ts.T))
    # series expansion for mean anomaly in moon radius (meters)
    r_moon = 1e3 * (
        385000.0
        - 20905.0 * np.cos(l)
        - 3699.0 * np.cos(2.0 * D - l)
        - 2956.0 * np.cos(2.0 * D)
        - 570.0 * np.cos(2.0 * l)
        + 246.0 * np.cos(2.0 * l - 2.0 * D)
        - 205.0 * np.cos(M - 2.0 * D)
        - 171.0 * np.cos(l + 2.0 * D)
        - 152.0 * np.cos(l + M - 2.0 * D)
    )
    # series expansion for ecliptic longitude of the moon (radians)
    lambda_moon = s + asec2rad(
        22640.0 * np.sin(l)
        + 769.0 * np.sin(2.0 * l)
        - 4586.0 * np.sin(l - 2.0 * D)
        + 2370.0 * np.sin(2.0 * D)
        - 668.0 * np.sin(M)
        - 412.0 * np.sin(2.0 * F)
        - 212.0 * np.sin(2.0 * l - 2.0 * D)
        - 206.0 * np.sin(l + M - 2.0 * D)
        + 192.0 * np.sin(l + 2.0 * D)
        - 165.0 * np.sin(M - 2.0 * D)
        - 148.0 * np.sin(l - M)
        - 125.0 * np.sin(D)
        - 110.0 * np.sin(l + M)
        - 55.0 * np.sin(2.0 * F - 2.0 * D)
    )
    # series expansion for ecliptic latitude of the moon (radians)
    q = asec2rad(412.0 * np.sin(2.0 * F) + 541.0 * np.sin(M))
    beta_moon = asec2rad(
        18520.0 * np.sin(F + lambda_moon - s + q)
        - 526.0 * np.sin(F - 2 * D)
        + 44.0 * np.sin(l + F - 2.0 * D)
        - 31.0 * np.sin(-l + F - 2.0 * D)
        - 25.0 * np.sin(-2.0 * l + F)
        - 23.0 * np.sin(M + F - 2.0 * D)
        + 21.0 * np.sin(-l + F)
        + 11.0 * np.sin(-M + F - 2.0 * D)
    )
    # convert to position vectors
    x = r_moon * np.cos(lambda_moon) * np.cos(beta_moon)
    y = r_moon * np.sin(lambda_moon) * np.cos(beta_moon)
    z = r_moon * np.sin(beta_moon)
    # obliquity of the J2000 ecliptic (radians)
    epsilon_j2000 = np.radians(23.43929111)
    # rotate by ecliptic
    rot_x = rotate(-epsilon_j2000, "x")
    u = rot_x[0, 0, :] * x + rot_x[0, 1, :] * y + rot_x[0, 2, :] * z
    v = rot_x[1, 0, :] * x + rot_x[1, 1, :] * y + rot_x[1, 2, :] * z
    w = rot_x[2, 0, :] * x + rot_x[2, 1, :] * y + rot_x[2, 2, :] * z
    # Greenwich hour angle (radians)
    rot_z = rotate(np.radians(ts.gha), "z")
    # rotate to cartesian (ECEF) coordinates
    # ignoring polar motion and length-of-day variations
    X = rot_z[0, 0, :] * u + rot_z[0, 1, :] * v + rot_z[0, 2, :] * w
    Y = rot_z[1, 0, :] * u + rot_z[1, 1, :] * v + rot_z[1, 2, :] * w
    Z = rot_z[2, 0, :] * u + rot_z[2, 1, :] * v + rot_z[2, 2, :] * w
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
    kwargs.setdefault("kernel", _default_kernel)
    kwargs.setdefault("include_aberration", False)
    # download kernel file if not currently existing
    if not pathlib.Path(kwargs["kernel"]).exists():
        fetch_jpl_ssd(kernel=None, local=kwargs["kernel"])
    # create timescale from Modified Julian Day (MJD)
    ts = timescale.time.Timescale(MJD=MJD)
    # difference to convert to Barycentric Dynamical Time (TDB)
    tdb2 = getattr(ts, "tdb_tt") if hasattr(ts, "tdb_tt") else 0.0
    # read JPL ephemerides kernel
    SPK = jplephem.spk.SPK.open(kwargs["kernel"])
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
    if kwargs["include_aberration"]:
        # astronomical unit in kilometers
        AU = 149597870.700
        # position in astronomical units
        position = (xyz_301 - xyz_399) / AU
        # velocity in astronomical units per day
        velocity = (vel_3 + vel_399 - vel_301) / AU
        # correct for aberration and convert to meters
        x, y, z = _correct_aberration(position, velocity)
    else:
        # convert positions from kilometers to meters
        x, y, z = 1e3 * (xyz_301 - xyz_399)
    # rotate to cartesian (ECEF) coordinates
    # use UTC time as input to itrs rotation function
    rot_z = itrs((ts.utc - _jd_j2000) / ts.century)
    X = rot_z[0, 0, :] * x + rot_z[0, 1, :] * y + rot_z[0, 2, :] * z
    Y = rot_z[1, 0, :] * x + rot_z[1, 1, :] * y + rot_z[1, 2, :] * z
    Z = rot_z[2, 0, :] * x + rot_z[2, 1, :] * y + rot_z[2, 2, :] * z
    # return the ECEF coordinates
    return (X, Y, Z)


def lunar_latitude(
    MJD: np.ndarray,
    **kwargs,
):
    """
    Calculate the apparent latitude of the moon
    :cite:p:`Kubo:1980ut,Meeus:1991vh`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    method: str, default 'Meeus'
        Method of calculating the latitude

        - ``'Kubo'``: :cite:t:`Kubo:1980ut`
        - ``'Meeus'``: :cite:t:`Meeus:1991vh`

    Returns
    -------
    beta: np.ndarray
        latitude of the moon (radians)
    """
    # set default keyword arguments
    kwargs.setdefault("method", "Meeus")
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - _mjd_j2000) / _century
    # coefficients for calculating the latitude of the moon
    if kwargs["method"].lower() == "meeus":
        # mean longitude of the moon (degrees)
        lunar_mean_longitude = np.array(
            [
                218.3164477,
                481267.88123421,
                -0.0015786,
                1.0 / 538841.0,
                -1.0 / 65194000.0,
            ]
        )
        Lp = polynomial_sum(lunar_mean_longitude, T)
        # mean elongation of the moon (degrees)
        lunar_elongation = np.array(
            [
                297.8501921,
                445267.1114034,
                -0.0018819,
                1.0 / 545868.0,
                -1.0 / 113065000.0,
            ]
        )
        D = polynomial_sum(lunar_elongation, T)
        # mean anomaly of the sun (degrees)
        solar_anomaly = np.array(
            [357.5291092, 35999.0502909, -0.0001536, 1.0 / 24490000.0]
        )
        M = polynomial_sum(solar_anomaly, T)
        # mean anomaly of the moon (degrees)
        lunar_anomaly = np.array(
            [
                134.9633964,
                477198.8675055,
                0.0087414,
                1.0 / 69699.0,
                1.0 / 14712000.0,
            ]
        )
        Mp = polynomial_sum(lunar_anomaly, T)
        # mean argument of latitude of the moon (degrees)
        # (angular distance from the ascending node)
        lunar_argument = np.array(
            [
                93.2720950,
                483202.0175233,
                -0.0036539,
                -1.0 / 3526000.0,
                1.0 / 863310000.0,
            ]
        )
        F = polynomial_sum(lunar_argument, T)
        # eccentricity of the Earth's orbit
        earth_eccentricity = np.array([1.0, -0.002516, -0.0000074])
        ee = polynomial_sum(earth_eccentricity, T)
        # additional arguments (actions of Venus and Jupiter)
        A1 = 119.75 + 131.849 * T
        A2 = 53.09 + 479264.290 * T
        A3 = 313.45 + 481266.484 * T
        # calculate latitude
        beta = 0.0
        # add additional arguments to latitude
        beta -= 2235e-6 * np.sin(np.radians(Lp))
        beta += 382e-6 * np.sin(np.radians(A3))
        beta += 175e-6 * np.sin(np.radians(A1 - F))
        beta += 175e-6 * np.sin(np.radians(A1 + F))
        beta += 127e-6 * np.sin(np.radians(Lp - Mp))
        beta -= 115e-6 * np.sin(np.radians(Lp + Mp))
        # calculate the lunar latitude
        table_47B = _meeus_table_47B()
        for i, line in enumerate(table_47B):
            d, m, mp, f, coeff = line
            delta_b = np.radians(d * D + m * M + mp * Mp + f * F)
            beta += 1e-6 * coeff * np.power(ee, np.abs(m)) * np.sin(delta_b)
    elif kwargs["method"].lower() == "kubo":
        # coefficients for calculating the latitude of the moon
        coefficients = np.array(
            [
                [51281e-4, 3.273, 483202.019],
                [2806e-4, 138.24, 960400.89],
                [2777e-4, 48.31, 6003.15],
                [1733e-4, 52.34, 407332.2],
                [554e-4, 104.0, 896537.4],
                [463e-4, 82.5, 69866.7],
                [326e-4, 239.0, 1373736.2],
                [172e-4, 273.2, 1437599.8],
                [93e-4, 187.0, 884531.0],
                [88e-4, 87.0, 471196.0],
                [82e-4, 55.0, 371333.0],
                [43e-4, 217.0, 547066.0],
                [42e-4, 14.0, 1850935.0],
                [34e-4, 230.0, 443331.0],
                [25e-4, 106.0, 860538.0],
                [22e-4, 308.0, 481268.0],
                [22e-4, 241.0, 1337737.0],
                [21e-4, 80.0, 105866.0],
                [19e-4, 141.0, 924402.0],
                [18e-4, 153.0, 820668.0],
                [18e-4, 181.0, 519201.0],
                [18e-4, 10.0, 1449606.0],
                [15e-4, 46.0, 42002.0],
                [15e-4, 121.0, 928469.0],
                [15e-4, 316.0, 996400.0],
                [14e-4, 129.0, 29996.0],
                [13e-4, 6.0, 447203.0],
                [13e-4, 65.0, 37935.0],
                [11e-4, 48.0, 1914799.0],
                [10e-4, 288.0, 1297866.0],
                [9e-4, 340.0, 1787072.0],
                [8e-4, 235.0, 972407.0],
                [7e-4, 205.0, 1309873.0],
                [6e-4, 134.0, 559072.0],
                [6e-4, 322.0, 1361730.0],
                [5e-4, 190.0, 848532.0],
                [5e-4, 149.0, 419339.0],
                [5e-4, 222.0, 948395.0],
                [4e-4, 149.0, 2328134.0],
                [4e-4, 352.0, 1024264.0],
                [3e-4, 282.0, 932536.0],
                [3e-4, 57.0, 1409735.0],
                [3e-4, 115.0, 2264270.0],
                [3e-4, 16.0, 1814936.0],
                [3e-4, 57.0, 335334.0],
            ]
        )
        # calculate the latitude of the moon
        beta = 0.0
        for i, (a, b, c) in enumerate(coefficients):
            beta += a * np.cos(np.radians(b + c * T))

    # convert to radians
    beta = np.radians(beta)
    # return the latitude of the moon
    return beta


def lunar_longitude(
    MJD: np.ndarray,
    **kwargs,
):
    """
    Calculates the geocentric longitude of the moon
    :cite:p:`Kubo:1980ut,Meeus:1991vh`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    method: str, default 'Meeus'
        Method of calculating the longitude

        - ``'Kubo'``: :cite:t:`Kubo:1980ut`
        - ``'Meeus'``: :cite:t:`Meeus:1991vh`

    Returns
    -------
    S: np.ndarray
        longitude of the moon (radians)
    """
    # set default keyword arguments
    kwargs.setdefault("method", "Meeus")
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - _mjd_j2000) / _century
    if kwargs["method"].lower() == "meeus":
        # mean longitude of the moon (degrees)
        lunar_mean_longitude = np.array(
            [
                218.3164477,
                481267.88123421,
                -0.0015786,
                1.0 / 538841.0,
                -1.0 / 65194000.0,
            ]
        )
        Lp = polynomial_sum(lunar_mean_longitude, T)
        # mean elongation of the moon (degrees)
        lunar_elongation = np.array(
            [
                297.8501921,
                445267.1114034,
                -0.0018819,
                1.0 / 545868.0,
                -1.0 / 113065000.0,
            ]
        )
        D = polynomial_sum(lunar_elongation, T)
        # mean anomaly of the sun (degrees)
        solar_anomaly = np.array(
            [357.5291092, 35999.0502909, -0.0001536, 1.0 / 24490000.0]
        )
        M = polynomial_sum(solar_anomaly, T)
        # mean anomaly of the moon (degrees)
        lunar_anomaly = np.array(
            [
                134.9633964,
                477198.8675055,
                0.0087414,
                1.0 / 69699.0,
                1.0 / 14712000.0,
            ]
        )
        Mp = polynomial_sum(lunar_anomaly, T)
        # mean argument of latitude of the moon (degrees)
        # (angular distance from the ascending node)
        lunar_argument = np.array(
            [
                93.2720950,
                483202.0175233,
                -0.0036539,
                -1.0 / 3526000.0,
                1.0 / 863310000.0,
            ]
        )
        F = polynomial_sum(lunar_argument, T)
        # eccentricity of the Earth's orbit
        earth_eccentricity = np.array([1.0, -0.002516, -0.0000074])
        ee = polynomial_sum(earth_eccentricity, T)
        # additional arguments (actions of Venus and Jupiter)
        A1 = 119.75 + 131.849 * T
        A2 = 53.09 + 479264.290 * T
        # calculate longitude
        S = np.copy(Lp)
        # add additional arguments to longitude
        S += 3958e-6 * np.sin(np.radians(A1))
        S += 1962e-6 * np.sin(np.radians(Lp - F))
        S += 318e-6 * np.sin(np.radians(A2))
        # calculate the lunar longitude
        table_47A = _meeus_table_47A()
        for i, line in enumerate(table_47A):
            d, m, mp, f, coeff, _ = line
            delta_S = np.radians(d * D + m * M + mp * Mp + f * F)
            S += 1e-6 * coeff * np.power(ee, np.abs(m)) * np.sin(delta_S)
    elif kwargs["method"].lower() == "kubo":
        # coefficients for calculating the longitude of the moon
        coefficients = np.array(
            [
                [62888e-4, 44.963, 477198.868],
                [12740e-4, 10.74, 413335.35],
                [6583e-4, 145.70, 890534.22],
                [2136e-4, 179.93, 954397.74],
                [1851e-4, 87.53, 35999.05],
                [1144e-4, 276.5, 966404.0],
                [588e-4, 124.2, 63863.5],
                [571e-4, 13.2, 377336.3],
                [533e-4, 280.7, 1367733.1],
                [458e-4, 148.2, 854535.2],
                [409e-4, 47.4, 441199.8],
                [347e-4, 27.9, 445267.1],
                [304e-4, 222.5, 513197.9],
                [154e-4, 41.0, 75870.0],
                [125e-4, 52.0, 1443603.0],
                [110e-4, 142.0, 489205.0],
                [107e-4, 246.0, 1303870.0],
                [100e-4, 315.0, 1431597.0],
                [85e-4, 111.0, 826671.0],
                [79e-4, 188.0, 449334.0],
                [68e-4, 323.0, 926533.0],
                [52e-4, 107.0, 31932.0],
                [50e-4, 205.0, 481266.0],
                [40e-4, 283.0, 1331734.0],
                [40e-4, 56.0, 1844932.0],
                [40e-4, 29.0, 133.0],
                [38e-4, 21.0, 1781068.0],
                [37e-4, 259.0, 541062.0],
                [28e-4, 145.0, 1934.0],
                [27e-4, 182.0, 918399.0],
                [26e-4, 17.0, 1379739.0],
                [24e-4, 122.0, 99863.0],
                [23e-4, 163.0, 922466.0],
                [22e-4, 151.0, 818536.0],
                [21e-4, 357.0, 990397.0],
                [21e-4, 85.0, 71998.0],
                [21e-4, 16.0, 341337.0],
                [18e-4, 274.0, 401329.0],
                [16e-4, 152.0, 1856938.0],
                [12e-4, 249.0, 1267871.0],
                [11e-4, 186.0, 1920802.0],
                [9e-4, 129.0, 858602.0],
                [8e-4, 98.0, 1403732.0],
                [7e-4, 114.0, 790672.0],
                [7e-4, 50.0, 405201.0],
                [7e-4, 186.0, 485333.0],
                [7e-4, 127.0, 27864.0],
                [6e-4, 38.0, 111869.0],
                [6e-4, 156.0, 2258267.0],
                [5e-4, 90.0, 1908795.0],
                [5e-4, 24.0, 1745069.0],
                [5e-4, 242.0, 509131.0],
                [4e-4, 223.0, 39871.0],
                [4e-4, 187.0, 12006.0],
                [3e-4, 340.0, 958465.0],
                [3e-4, 354.0, 381404.0],
                [3e-4, 337.0, 349472.0],
                [3e-4, 58.0, 1808933.0],
                [3e-4, 220.0, 549197.0],
                [3e-4, 70.0, 4067.0],
                [3e-4, 191.0, 2322131.0],
            ]
        )
        # calculate the longitude of the moon
        S = 218.3162 + 481267.8809 * T
        for i, (a, b, c) in enumerate(coefficients):
            S += a * np.cos(np.radians(b + c * T))

    # take the modulus and convert to radians
    S = np.radians(normalize_angle(S, circle=360.0))
    # return the longitude of the moon
    return S


def lunar_distance(
    MJD: np.ndarray,
    a_axis: float = 6378137.0,
    **kwargs,
):
    """
    Calculate the geocentric distance from the moon to the Earth
    :cite:p:`Kubo:1980ut,Meeus:1991vh`

    Parameters
    ----------
    MJD: np.ndarray
        Modified Julian Day (MJD) of input date
    a_axis: float, default 6378137.0
        Semi-major axis of the Earth (meters)
    method: str, default 'Meeus'
        Method of calculating the distance

        - ``'Kubo'``: :cite:t:`Kubo:1980ut`
        - ``'Meeus'``: :cite:t:`Meeus:1991vh`

    Returns
    -------
    R: np.ndarray
        distance from the moon to the Earth (meters)
    """
    # set default keyword arguments
    kwargs.setdefault("method", "Meeus")
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (MJD - _mjd_j2000) / _century
    if kwargs["method"].lower() == "meeus":
        # mean elongation of the moon (degrees)
        lunar_elongation = np.array(
            [
                297.8501921,
                445267.1114034,
                -0.0018819,
                1.0 / 545868.0,
                -1.0 / 113065000.0,
            ]
        )
        D = polynomial_sum(lunar_elongation, T)
        # mean anomaly of the sun (degrees)
        solar_anomaly = np.array(
            [357.5291092, 35999.0502909, -0.0001536, 1.0 / 24490000.0]
        )
        M = polynomial_sum(solar_anomaly, T)
        # mean anomaly of the moon (degrees)
        lunar_anomaly = np.array(
            [
                134.9633964,
                477198.8675055,
                0.0087414,
                1.0 / 69699.0,
                1.0 / 14712000.0,
            ]
        )
        Mp = polynomial_sum(lunar_anomaly, T)
        # mean argument of latitude of the moon (degrees)
        # (angular distance from the ascending node)
        lunar_argument = np.array(
            [
                93.2720950,
                483202.0175233,
                -0.0036539,
                -1.0 / 3526000.0,
                1.0 / 863310000.0,
            ]
        )
        F = polynomial_sum(lunar_argument, T)
        # eccentricity of the Earth's orbit
        earth_eccentricity = np.array([1.0, -0.002516, -0.0000074])
        ee = polynomial_sum(earth_eccentricity, T)
        # calculate the distance from the moon to the Earth
        R = 385000560.0
        table_47A = _meeus_table_47A()
        for i, line in enumerate(table_47A):
            d, m, mp, f, _, coeff = line
            delta_R = np.radians(d * D + m * M + mp * Mp + f * F)
            R += coeff * np.power(ee, np.abs(m)) * np.cos(delta_R)
    elif kwargs["method"].lower() == "kubo":
        # horizontal parallax of the moon (degrees)
        parallax = 0.950725
        # coefficients for calculating the distance to the moon
        coefficients = np.array(
            [
                [51820e-6, 134.963, 477198.868],
                [9530e-6, 100.74, 413335.35],
                [7842e-6, 235.70, 890534.22],
                [2824e-6, 269.93, 954397.74],
                [858e-6, 10.7, 1367733.1],
                [531e-6, 238.2, 854535.2],
                [400e-6, 103.2, 377336.3],
                [319e-6, 137.4, 441199.8],
                [271e-6, 118.0, 445267.0],
                [263e-6, 312.0, 513198.0],
                [197e-6, 232.0, 489205.0],
                [173e-6, 45.0, 1431597.0],
                [167e-6, 336.0, 1303870.0],
                [111e-6, 178.0, 35999.0],
                [103e-6, 201.0, 826671.0],
                [84e-6, 214.0, 63864.0],
                [83e-6, 53.0, 926533.0],
                [78e-6, 146.0, 1844932.0],
                [73e-6, 111.0, 1781068.0],
                [64e-6, 13.0, 1331734.0],
                [63e-6, 278.0, 449334.0],
                [41e-6, 295.0, 481266.0],
                [34e-6, 272.0, 918399.0],
                [33e-6, 349.0, 541062.0],
                [31e-6, 253.0, 922466.0],
                [30e-6, 131.0, 75870.0],
                [29e-6, 87.0, 990397.0],
                [26e-6, 241.0, 818536.0],
                [23e-6, 266.0, 553069.0],
                [19e-6, 339.0, 1267871.0],
                [13e-6, 188.0, 1403732.0],
                [13e-6, 106.0, 341337.0],
                [13e-6, 4.0, 401329.0],
                [12e-6, 246.0, 2258267.0],
                [11e-6, 180.0, 1908795.0],
                [11e-6, 219.0, 858602.0],
                [10e-6, 144.0, 1745069.0],
                [9e-6, 204.0, 790672.0],
                [7e-6, 281.0, 2322131.0],
                [7e-6, 148.0, 1808933.0],
                [6e-6, 276.0, 485333.0],
                [6e-6, 212.0, 99863.0],
                [5e-6, 140.0, 405201.0],
            ]
        )
        # calculate the distance from the moon to the Earth
        for i, (a, b, c) in enumerate(coefficients):
            parallax += a * np.cos(np.radians(b + c * T))
        # convert parallax to radians
        p = np.radians(parallax)
        # convert to meters
        R = a_axis / (p * (1.0 - p * p / 6.0))
    # return the distance from the moon to the Earth
    return R


def gast(T: float | np.ndarray):
    """Greenwich Apparent Sidereal Time (GAST)
    :cite:p:`Capitaine:2003fx,Capitaine:2003fw,Petit:2010tp`

    Parameters
    ----------
    T: np.ndarray
        Centuries since 2000-01-01T12:00:00
    """
    # create timescale from centuries relative to 2000-01-01T12:00:00
    ts = timescale.time.Timescale(MJD=T * _century + _mjd_j2000)
    # convert dynamical time to modified Julian days
    MJD = ts.tt - _jd_mjd
    # estimate the mean obliquity
    epsilon = mean_obliquity(MJD)
    # estimate the nutation in longitude and obliquity
    dpsi, deps = _nutation_angles(T)
    # traditional equation of the equinoxes
    c = _eqeq_complement(T)
    eqeq = dpsi * np.cos(epsilon + deps) + c
    return np.mod(ts.st + eqeq / 24.0, 1.0)


def itrs(T: float | np.ndarray, include_polar_motion: bool = True):
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
    ts = timescale.time.Timescale(MJD=T * _century + _mjd_j2000)
    # get the rotation matrix for transforming from ICRS to ITRS
    M = _icrs_rotation_matrix(T, include_polar_motion=include_polar_motion)
    # compute Greenwich Apparent Sidereal Time
    GAST = rotate(ts.tau * gast(T), "z")
    R = np.einsum("ijt...,jkt->ikt...", GAST, M)
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
    ts = timescale.time.Timescale(MJD=T * _century + _mjd_j2000)
    # get the fundamental arguments in radians
    fa = np.zeros((14, len(ts)))
    # mean anomaly of the moon (arcseconds)
    lunar_anomaly = np.array(
        [485868.249036, 715923.2178, 31.8792, 0.051635, -2.447e-04]
    )
    fa[0, :] = asec2rad(polynomial_sum(lunar_anomaly, ts.T)) + ts.tau * np.mod(
        1325.0 * ts.T, 1.0
    )
    # mean anomaly of the sun (arcseconds)
    solar_anomaly = np.array(
        [1287104.79305, 1292581.0481, -0.5532, 1.36e-4, -1.149e-05]
    )
    fa[1, :] = asec2rad(polynomial_sum(solar_anomaly, ts.T)) + ts.tau * np.mod(
        99.0 * ts.T, 1.0
    )
    # mean argument of the moon (arcseconds)
    # (angular distance from the ascending node)
    lunar_argument = np.array(
        [335779.526232, 295262.8478, -12.7512, -1.037e-3, 4.17e-6]
    )
    fa[2, :] = asec2rad(polynomial_sum(lunar_argument, ts.T)) + ts.tau * np.mod(
        1342.0 * ts.T, 1.0
    )
    # mean elongation of the moon from the sun (arcseconds)
    lunisolar_elongation = np.array(
        [1072260.70369, 1105601.2090, -6.3706, 6.593e-3, -3.169e-05]
    )
    fa[3, :] = asec2rad(
        polynomial_sum(lunisolar_elongation, ts.T)
    ) + ts.tau * np.mod(1236.0 * ts.T, 1.0)
    # mean longitude of the ascending node of the moon (arcseconds)
    lunar_node = np.array(
        [450160.398036, -482890.5431, 7.4722, 7.702e-3, -5.939e-05]
    )
    fa[4, :] = asec2rad(polynomial_sum(lunar_node, ts.T)) + ts.tau * np.mod(
        -5.0 * ts.T, 1.0
    )
    # additional polynomial terms
    fa[5, :] = polynomial_sum(np.array([4.402608842, 2608.7903141574]), ts.T)
    fa[6, :] = polynomial_sum(np.array([3.176146697, 1021.3285546211]), ts.T)
    fa[7, :] = polynomial_sum(np.array([1.753470314, 628.3075849991]), ts.T)
    fa[8, :] = polynomial_sum(np.array([6.203480913, 334.0612426700]), ts.T)
    fa[9, :] = polynomial_sum(np.array([0.599546497, 52.9690962641]), ts.T)
    fa[10, :] = polynomial_sum(np.array([0.874016757, 21.3299104960]), ts.T)
    fa[11, :] = polynomial_sum(np.array([5.481293872, 7.4781598567]), ts.T)
    fa[12, :] = polynomial_sum(np.array([5.311886287, 3.8133035638]), ts.T)
    fa[13, :] = polynomial_sum(np.array([0, 0.024381750, 0.00000538691]), ts.T)
    # parse IERS Greenwich Sidereal Time (GST) table
    j0, j1 = _parse_table_5_2e()
    n0 = np.c_[
        j0["l"],
        j0["lp"],
        j0["F"],
        j0["D"],
        j0["Om"],
        j0["L_Me"],
        j0["L_Ve"],
        j0["L_E"],
        j0["L_Ma"],
        j0["L_J"],
        j0["L_Sa"],
        j0["L_U"],
        j0["L_Ne"],
        j0["p_A"],
    ]
    n1 = np.c_[
        j1["l"],
        j1["lp"],
        j1["F"],
        j1["D"],
        j1["Om"],
        j1["L_Me"],
        j1["L_Ve"],
        j1["L_E"],
        j1["L_Ma"],
        j1["L_J"],
        j1["L_Sa"],
        j1["L_U"],
        j1["L_Ne"],
        j1["p_A"],
    ]
    arg0 = np.dot(n0, np.mod(fa, ts.tau))
    arg1 = np.dot(n1, np.mod(fa, ts.tau))
    # evaluate the complementary terms and convert to radians
    complement = masec2rad(
        np.dot(j0["Cs"], np.sin(arg0))
        + np.dot(j0["Cc"], np.cos(arg0))
        + ts.T * np.dot(j1["Cs"], np.sin(arg1))
        + ts.T * np.dot(j1["Cc"], np.cos(arg1))
    )
    # return the complementary terms
    return complement


def _icrs_rotation_matrix(
    T: float | np.ndarray, include_polar_motion: bool = True
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
    ts = timescale.time.Timescale(MJD=T * _century + _mjd_j2000)
    # difference to convert to Barycentric Dynamical Time (TDB)
    tdb2 = getattr(ts, "tdb_tt") if hasattr(ts, "tdb_tt") else 0.0
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
    M = np.einsum("ijt...,jkt...,kl...->ilt...", M1, M2, M3)
    # add polar motion to the combined rotation matrix
    if include_polar_motion:
        # M4: polar motion
        M4 = _polar_motion_matrix(ts.T)
        M = np.einsum("ijt...,jkt...->ikt...", M, M4)
    # return the combined rotation matrix
    return M


def _frame_bias_matrix():
    """
    Frame bias rotation matrix for converting from a dynamical
    reference system to the International Celestial Reference
    System (ICRS) :cite:p:`Petit:2010tp,Urban:2013vl`
    """
    # frame bias rotation matrix
    B = np.zeros((3, 3))
    xi0 = asec2rad(-0.0166170)
    eta0 = asec2rad(-0.0068192)
    da0 = asec2rad(-0.01460)
    # off-diagonal elements of the frame bias matrix
    B[0, 1] = da0
    B[0, 2] = -xi0
    B[1, 0] = -da0
    B[1, 2] = -eta0
    B[2, 0] = xi0
    B[2, 1] = eta0
    # second-order corrections to diagonal elements
    B[0, 0] = 1.0 - 0.5 * (da0**2 + xi0**2)
    B[1, 1] = 1.0 - 0.5 * (da0**2 + eta0**2)
    B[2, 2] = 1.0 - 0.5 * (eta0**2 + xi0**2)
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
    ts = timescale.time.Timescale(MJD=T * _century + _mjd_j2000)
    # difference to convert to Barycentric Dynamical Time (TDB)
    tdb2 = getattr(ts, "tdb_tt") if hasattr(ts, "tdb_tt") else 0.0
    # convert dynamical time to modified Julian days
    MJD = ts.tt + tdb2 - _jd_mjd
    # get the fundamental arguments in radians
    l, lp, F, D, Om = delaunay_arguments(MJD)
    # non-polynomial terms in the equation of the equinoxes
    # parse IERS lunisolar longitude table
    l0, l1 = _parse_table_5_3a()
    n0 = np.c_[l0["l"], l0["lp"], l0["F"], l0["D"], l0["Om"]]
    n1 = np.c_[l1["l"], l1["lp"], l1["F"], l1["D"], l1["Om"]]
    arg0 = np.dot(n0, np.c_[l, lp, F, D, Om].T)
    arg1 = np.dot(n1, np.c_[l, lp, F, D, Om].T)
    dpsi = (
        np.dot(l0["As"], np.sin(arg0))
        + np.dot(l0["Ac"], np.cos(arg0))
        + ts.T * np.dot(l1["As"], np.sin(arg1))
        + ts.T * np.dot(l1["Ac"], np.cos(arg1))
    )
    # parse IERS lunisolar obliquity table
    o0, o1 = _parse_table_5_3b()
    n0 = np.c_[o0["l"], o0["lp"], o0["F"], o0["D"], o0["Om"]]
    n1 = np.c_[o1["l"], o1["lp"], o1["F"], o1["D"], o1["Om"]]
    arg0 = np.dot(n0, np.c_[l, lp, F, D, Om].T)
    arg1 = np.dot(n1, np.c_[l, lp, F, D, Om].T)
    deps = (
        np.dot(o0["Bs"], np.sin(arg0))
        + np.dot(o0["Bc"], np.cos(arg0))
        + ts.T * np.dot(o1["Bs"], np.sin(arg1))
        + ts.T * np.dot(o1["Bc"], np.cos(arg1))
    )
    # convert to radians
    return (masec2rad(dpsi), masec2rad(deps))


def _nutation_matrix(
    mean_obliquity: float | np.ndarray,
    true_obliquity: float | np.ndarray,
    psi: float | np.ndarray,
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
    # compute trigonometric terms
    cospsi = np.cos(psi)
    sinpsi = np.sin(psi)
    cosmean = np.cos(mean_obliquity)
    sinmean = np.sin(mean_obliquity)
    costrue = np.cos(true_obliquity)
    sintrue = np.sin(true_obliquity)
    # compute elements of nutation rotation matrix
    R = np.zeros((3, 3, len(np.atleast_1d(psi))))
    R[0, 0, :] = cospsi
    R[0, 1, :] = -sinpsi * cosmean
    R[0, 2, :] = -sinpsi * sinmean
    R[1, 0, :] = sinpsi * costrue
    R[1, 1, :] = cospsi * cosmean * costrue + sinmean * sintrue
    R[1, 2, :] = cospsi * sinmean * costrue - cosmean * sintrue
    R[2, 0, :] = sinpsi * sintrue
    R[2, 1, :] = cospsi * cosmean * sintrue - sinmean * costrue
    R[2, 2, :] = cospsi * sinmean * sintrue + cosmean * costrue
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
    MJD = T * _century + _mjd_j2000
    # correct longitude origin for Terrestrial Intermediate Origin (TIO)
    # this correction is negligible for most applications
    sprime = -4.7e-5 * T
    # calculate the polar motion for the given dates
    px, py = timescale.eop.iers_polar_motion(MJD)
    # calculate the rotation matrices
    M1 = rotate(asec2rad(py), "x")
    M2 = rotate(asec2rad(px), "y")
    M3 = rotate(-asec2rad(sprime), "z")
    # calculate the combined rotation matrix
    return np.einsum("ij...,jk...,kl...->il...", M1, M2, M3)


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
    # lunisolar precession
    psi0 = np.array(
        [0.0, 5038.481507, -1.0790069, -1.14045e-3, 1.32851e-4, -9.51e-8]
    )
    psi = asec2rad(polynomial_sum(psi0, T))
    # inclination of moving equator on fixed ecliptic
    omega0 = np.array(
        [epsilon0, -2.5754e-2, 5.12623e-2, -7.72503e-3, -4.67e-7, 3.337e-7]
    )
    omega = asec2rad(polynomial_sum(omega0, T))
    # planetary precession
    chi0 = np.array(
        [0.0, 10.556403, -2.3814292, -1.21197e-3, 1.70663e-4, -5.60e-8]
    )
    chi = asec2rad(polynomial_sum(chi0, T))
    # compute trigonometric terms
    coschi = np.cos(chi)
    sinchi = np.sin(chi)
    cospsi = np.cos(-psi)
    sinpsi = np.sin(-psi)
    cosomega = np.cos(-omega)
    sinomega = np.sin(-omega)
    coseps = np.cos(asec2rad(epsilon0))
    sineps = np.sin(asec2rad(epsilon0))
    # compute elements of precession rotation matrix
    P = np.zeros((3, 3, len(np.atleast_1d(T))))
    P[0, 0, :] = coschi * cospsi - sinpsi * sinchi * cosomega
    P[0, 1, :] = (
        coschi * sinpsi * coseps
        + sinchi * cosomega * cospsi * coseps
        - sineps * sinchi * sinomega
    )
    P[0, 2, :] = (
        coschi * sinpsi * sineps
        + sinchi * cosomega * cospsi * sineps
        + coseps * sinchi * sinomega
    )
    P[1, 0, :] = -sinchi * cospsi - sinpsi * coschi * cosomega
    P[1, 1, :] = (
        -sinchi * sinpsi * coseps
        + coschi * cosomega * cospsi * coseps
        - sineps * coschi * sinomega
    )
    P[1, 2, :] = (
        -sinchi * sinpsi * sineps
        + coschi * cosomega * cospsi * sineps
        + coseps * coschi * sinomega
    )
    P[2, 0, :] = sinpsi * sinomega
    P[2, 1, :] = -sinomega * cospsi * coseps - sineps * cosomega
    P[2, 2, :] = -sinomega * cospsi * sineps + cosomega * coseps
    # return the rotation matrix
    return P


def _correct_aberration(position, velocity):
    """
    Correct a relative position for aberration effects
    :cite:p:`Kaplan:1989cf`

    Parameters
    ----------
    position: np.ndarray
        Position vector (astronomical units)
    velocity: np.ndarray
        Velocity vector (astronomical units per day)
    """
    # number of seconds per day
    day = 86400.0
    # speed of light (meters per second)
    c = 299792458.0
    # distance of 1 Astronomical Unit (meters)
    AU = 149597870700.0
    # speed of light in AU/day (i.e. one light day)
    c_prime = c * day / AU
    # total distance
    distance = np.sqrt(np.sum(position * position, axis=0))
    tau = distance / c_prime
    # speed
    speed = np.sqrt(np.sum(velocity * velocity, axis=0))
    beta = speed / c_prime
    # Kaplan et al. (1989) eq. 16
    # (use divide function to avoid error if denominator is zero)
    cosD = np.divide(np.sum(position * velocity, axis=0), distance * speed)
    # calculate adjustments
    gamma = np.sqrt(1.0 - beta * beta)
    f1 = beta * cosD
    f2 = (1.0 + f1 / (1.0 + gamma)) * tau
    # correct for aberration of light travel time (eq. 17)
    u = (gamma * position + f2 * velocity) / (1.0 + f1)
    # return corrected position converted to meters
    x, y, z = u * AU
    return (x, y, z)


def _meeus_table_47A():
    """
    Coefficients for the periodic terms in lunar longitude
    and distance from Table 47.A of :cite:t:`Meeus:1991vh`
    """
    # table 47.A from Meeus (1991)
    # column 1: mean elongation of the moon
    # column 2: suns mean anomaly
    # column 3: moons mean anomaly
    # column 4: moons argument of latitude
    # column 5: coefficient of the sine term for lunar longitude
    #     units: microdegrees (1e-6 degrees)
    # column 6: coefficient of the cosine term for lunar distance
    #     units: meters (1e-3 kilometers)
    table_47A = np.array(
        [
            [0, 0, 1, 0, 6288774.0, -20905355.0],
            [2, 0, -1, 0, 1274027.0, -3699111.0],
            [2, 0, 0, 0, 658314.0, -2955968.0],
            [0, 0, 2, 0, 213618.0, -569925.0],
            [0, 1, 0, 0, -185116.0, 48888.0],
            [0, 0, 0, 2, -114332.0, -3149.0],
            [2, 0, -2, 0, 58793.0, 246158.0],
            [2, -1, -1, 0, 57066.0, -152138.0],
            [2, 0, 1, 0, 53322.0, -170733.0],
            [2, -1, 0, 0, 45758.0, -204586.0],
            [0, 1, -1, 0, -40923.0, -129620.0],
            [1, 0, 0, 0, -34720.0, 108743.0],
            [0, 1, 1, 0, -30383.0, 104755.0],
            [2, 0, 0, -2, 15327.0, 10321.0],
            [0, 0, 1, 2, -12528.0, 0.0],
            [0, 0, 1, -2, 10980.0, 79661.0],
            [4, 0, -1, 0, 10675.0, -34782.0],
            [0, 0, 3, 0, 10034.0, -23210.0],
            [4, 0, -2, 0, 8548.0, -21636.0],
            [2, 1, -1, 0, -7888.0, 24208.0],
            [2, 1, 0, 0, -6766.0, 30824.0],
            [1, 0, -1, 0, -5163.0, -8379.0],
            [1, 1, 0, 0, 4987.0, -16675.0],
            [2, -1, 1, 0, 4036.0, -12831.0],
            [2, 0, 2, 0, 3994.0, -10445.0],
            [4, 0, 0, 0, 3861.0, -11650.0],
            [2, 0, -3, 0, 3665.0, 14403.0],
            [0, 1, -2, 0, -2689.0, -7003.0],
            [2, 0, -1, 2, -2602.0, 0.0],
            [2, -1, -2, 0, 2390.0, 10056.0],
            [1, 0, 1, 0, -2348.0, 6322.0],
            [2, -2, 0, 0, 2236.0, -9884.0],
            [0, 1, 2, 0, -2120.0, 5751.0],
            [0, 2, 0, 0, -2069.0, 0.0],
            [2, -2, -1, 0, 2048.0, -4950.0],
            [2, 0, 1, -2, -1773.0, 4130.0],
            [2, 0, 0, 2, -1595.0, 0.0],
            [4, -1, -1, 0, 1215.0, -3958.0],
            [0, 0, 2, 2, -1110.0, 0.0],
            [3, 0, -1, 0, -892.0, 3258.0],
            [2, 1, 1, 0, -810.0, 2616.0],
            [4, -1, -2, 0, 759.0, -1897.0],
            [0, 2, -1, 0, -713.0, -2117.0],
            [2, 2, -1, 0, -700.0, 2354.0],
            [2, 1, -2, 0, 691.0, 0.0],
            [2, -1, 0, -2, 596.0, 0.0],
            [4, 0, 1, 0, 549.0, -1423.0],
            [0, 0, 4, 0, 537.0, -1117.0],
            [4, -1, 0, 0, 520.0, -1571.0],
            [1, 0, -2, 0, -487.0, -1739.0],
            [2, 1, 0, -2, -399.0, 0.0],
            [0, 0, 2, -2, -381.0, -4421.0],
            [1, 1, 1, 0, 351.0, 0.0],
            [3, 0, -2, 0, -340.0, 0.0],
            [4, 0, -3, 0, 330.0, 0.0],
            [2, -1, 2, 0, 327.0, 0.0],
            [0, 2, 1, 0, -323.0, 1165.0],
            [1, 1, -1, 0, 299.0, 0.0],
            [2, 0, 3, 0, 394.0, 0.0],
            [2, 0, -1, -2, 0.0, 8752.0],
        ]
    )
    # return the table of coefficients
    return table_47A


def _meeus_table_47B():
    """
    Coefficients for the sine and cosine terms in lunar latitude
    from Table 47.B of :cite:t:`Meeus:1991vh`
    """
    # table 47.B from Meeus (1991)
    # column 1: mean elongation of the moon
    # column 2: suns mean anomaly
    # column 3: moons mean anomaly
    # column 4: moons argument of latitude
    # column 5: coefficient of the sine term for lunar latitude
    #     units: microdegrees (1e-6 degrees)
    table_47B = np.array(
        [
            [0, 0, 0, 1, 5128122.0],
            [0, 0, 1, 1, 280602.0],
            [0, 0, 1, -1, 277693.0],
            [2, 0, 0, -1, 173237.0],
            [2, 0, -1, 1, 55413.0],
            [2, 0, -1, -1, 46271.0],
            [2, 0, 0, 1, 32573.0],
            [0, 0, 2, 1, 17198.0],
            [2, 0, 1, -1, 9266.0],
            [0, 0, 2, -1, 8822.0],
            [2, -1, 0, -1, 8216.0],
            [2, 0, -2, -1, 4324.0],
            [2, 0, 1, -1, 4200.0],
            [2, 1, 0, -1, -3359.0],
            [2, -1, -1, 1, 2463.0],
            [2, -1, 0, 1, 2211.0],
            [2, -1, -1, -1, 2065.0],
            [0, 1, -1, -1, -1870.0],
            [4, 0, -1, -1, 1828.0],
            [0, 1, 0, 1, -1794.0],
            [0, 0, 0, 3, -1749.0],
            [0, 1, -1, 1, -1565.0],
            [1, 0, 0, 1, -1491.0],
            [0, 1, 1, 1, -1475.0],
            [0, 1, 1, -1, -1410.0],
            [0, 1, 0, -1, -1344.0],
            [1, 0, 0, -1, -1335.0],
            [0, 0, 3, 1, 1107.0],
            [4, 0, 0, -1, 1021.0],
            [4, 0, -1, 1, 833.0],
            [0, 0, 1, -3, 777.0],
            [4, 0, -2, 1, 671.0],
            [2, 0, 0, -3, 607.0],
            [2, 0, 2, -1, 596.0],
            [2, -1, 1, -1, 491.0],
            [2, 0, -2, 1, -451.0],
            [0, 0, 3, -1, 439.0],
            [2, 0, 2, 1, 422.0],
            [2, 0, -3, -1, 421.0],
            [2, 1, -1, -1, -366.0],
            [2, 1, 0, 1, -351.0],
            [4, 0, 0, 1, 331.0],
            [2, -1, 1, 1, 315.0],
            [2, -2, 0, -1, 302.0],
            [0, 0, 1, 3, -283.0],
            [2, 1, 1, -1, -229.0],
            [1, 1, 0, -1, 223.0],
            [1, 1, 0, 1, 223.0],
            [0, 1, -2, -1, -220.0],
            [2, 1, -1, 1, -220.0],
            [1, 0, 1, 1, -185.0],
            [2, -1, -2, -1, 181.0],
            [0, 1, 2, 1, -177.0],
            [4, 0, -2, -1, 176.0],
            [4, -1, -1, -1, 166.0],
            [1, 0, 1, -1, -164.0],
            [4, 0, 1, -1, 132.0],
            [1, 0, -1, -1, -119.0],
            [4, -1, 0, -1, 115.0],
            [2, -2, 0, 1, 107.0],
        ]
    )
    # return the table of coefficients
    return table_47B


def _parse_table_5_2e():
    """Parse table with expressions for Greenwich Sidereal Time
    provided in `Chapter 5
    <https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.2e.txt>`_
    of :cite:t:`Petit:2010tp`
    """
    table_5_2e = get_data_path(["data", "tab5.2e.txt"])
    with table_5_2e.open(mode="r", encoding="utf8") as f:
        file_contents = f.readlines()
    # names and formats
    names = (
        "i",
        "Cs",
        "Cc",
        "l",
        "lp",
        "F",
        "D",
        "Om",
        "L_Me",
        "L_Ve",
        "L_E",
        "L_Ma",
        "L_J",
        "L_Sa",
        "L_U",
        "L_Ne",
        "p_A",
    )
    formats = (
        "i",
        "f",
        "f",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
    )
    dtype = np.dtype({"names": names, "formats": formats})
    # j = 0 terms
    n0 = 33
    j0 = np.zeros((n0), dtype=dtype)
    for i, line in enumerate(file_contents[53 : 53 + n0]):
        j0[i] = np.array(tuple(line.split()), dtype=dtype)
    # j = 1 terms
    n1 = 1
    j1 = np.zeros((n1), dtype=dtype)
    for i, line in enumerate(file_contents[90 : 90 + n1]):
        j1[i] = np.array(tuple(line.split()), dtype=dtype)
    # return the table
    return (j0, j1)


def _parse_table_5_3a():
    """Parse table with IAU 2000A lunisolar and planetary components
    of nutation in longitude provided in `Chapter 5
    <https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.2e.txt>`_
    of :cite:t:`Petit:2010tp`
    """
    table_5_3a = get_data_path(["data", "tab5.3a.txt"])
    with table_5_3a.open(mode="r", encoding="utf8") as f:
        file_contents = f.readlines()
    # names and formats
    names = (
        "i",
        "As",
        "Ac",
        "l",
        "lp",
        "F",
        "D",
        "Om",
        "L_Me",
        "L_Ve",
        "L_E",
        "L_Ma",
        "L_J",
        "L_Sa",
        "L_U",
        "L_Ne",
        "p_A",
    )
    formats = (
        "i",
        "f",
        "f",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
    )
    dtype = np.dtype({"names": names, "formats": formats})
    # j = 0 terms
    n0 = 1320
    j0 = np.zeros((n0), dtype=dtype)
    for i, line in enumerate(file_contents[22 : 22 + n0]):
        j0[i] = np.array(tuple(line.split()), dtype=dtype)
    # j = 1 terms
    n1 = 38
    j1 = np.zeros((n1), dtype=dtype)
    for i, line in enumerate(file_contents[1348 : 1348 + n1]):
        j1[i] = np.array(tuple(line.split()), dtype=dtype)
    # return the table
    return (j0, j1)


def _parse_table_5_3b():
    """Parse table with IAU 2000A lunisolar and planetary components
    of nutation in obliquity provided in `Chapter 5
    <https://iers-conventions.obspm.fr/content/chapter5/additional_info/tab5.2e.txt>`_
    of :cite:t:`Petit:2010tp`
    """
    table_5_3b = get_data_path(["data", "tab5.3b.txt"])
    with table_5_3b.open(mode="r", encoding="utf8") as f:
        file_contents = f.readlines()
    # names and formats
    names = (
        "i",
        "Bs",
        "Bc",
        "l",
        "lp",
        "F",
        "D",
        "Om",
        "L_Me",
        "L_Ve",
        "L_E",
        "L_Ma",
        "L_J",
        "L_Sa",
        "L_U",
        "L_Ne",
        "p_A",
    )
    formats = (
        "i",
        "f",
        "f",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
        "i",
    )
    dtype = np.dtype({"names": names, "formats": formats})
    # j = 0 terms
    n0 = 1037
    j0 = np.zeros((n0), dtype=dtype)
    for i, line in enumerate(file_contents[22 : 22 + n0]):
        j0[i] = np.array(tuple(line.split()), dtype=dtype)
    # j = 1 terms
    n1 = 19
    j1 = np.zeros((n1), dtype=dtype)
    for i, line in enumerate(file_contents[1065 : 1065 + n1]):
        j1[i] = np.array(tuple(line.split()), dtype=dtype)
    # return the table
    return (j0, j1)
