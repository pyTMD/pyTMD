#!/usr/bin/env python
"""
response.py
Written by Tyler Sutterley (11/2024)
Routines for estimating tidal constituents using the response method

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://scipy.org

PROGRAM DEPENDENCIES:
    astro.py: computes the basic astronomical mean longitudes
    constituents.py: calculates constituent parameters and nodal arguments
    math.py: special mathematical functions
    spatial.py: utilities for spatial operations

UPDATE HISTORY:
    Written 11/2024
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.optimize
import pyTMD.constituents
import pyTMD.astro
import pyTMD.math
import pyTMD.spatial

__all__ = ["response", "_gravitational", "_radiational", "_kappa"]


def response(
    t: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    ht: np.ndarray,
    constituents: str | list | np.ndarray,
    corrections: str = "OTIS",
    **kwargs: dict,
):
    """
    Estimate tidal constituents using the response method
    :cite:p:`Munk:1966go,Groves:1975ky,Zetler:1975uv,Cartwright:1990ck`

    Parameters
    ----------
    t: np.ndarray
        Time (days relative to January 1, 1992)
    lon: np.ndarray
        longitude coordinates
    lat: np.ndarray
        latitude coordinates
    ht: np.ndarray
        input time series (elevations)
    constituents: str, list or np.ndarray
        tidal constituent ID(s)
    corrections: str, default 'OTIS'
        use nodal corrections from OTIS/ATLAS or GOT/FES models
    **kwargs: dict
        keyword arguments for ephemeris calculations

    Returns
    -------
    amp: np.ndarray
        amplitude of each harmonic constant (meters)
    phase: np.ndarray
        phase of each harmonic constant (degrees)
    """
    # default keyword arguments
    kwargs.setdefault("a_axis", 6378137.0)
    kwargs.setdefault("flat", 1.0 / 298.257223563)
    kwargs.setdefault("ephemerides", "approximate")
    # distances between the Earth and the sun/moon (meters)
    kwargs.setdefault("AU", 1.495978707e11)
    kwargs.setdefault("LD", 3.84399e8)
    # mass ratios between earth and sun/moon
    kwargs.setdefault("mass_ratio_solar", 332946.0482)
    kwargs.setdefault("mass_ratio_lunar", 0.0123000371)
    # check if input constituents is a string
    if isinstance(constituents, str):
        constituents = [constituents]
    # verify shape of variables
    t = np.ravel(t)
    lon = np.ravel(lon)
    lat = np.ravel(lat)
    ht = np.ravel(ht)
    # reduce variables to finite values
    if not np.isfinite(ht).all():
        (valid,) = np.nonzero(np.isfinite(t) & np.isfinite(ht))
        t = t[valid]
        lon = lon[valid]
        lat = lat[valid]
        ht = ht[valid]
    # check that there are enough values for a time series fit
    nt = len(t)
    nc = len(constituents)
    # assert dimensions
    assert len(t) == len(lat), "coordinates must have the same dimensions"
    assert len(lon) == len(lat), "coordinates must have the same dimensions"
    # check that the number of time values matches the number of height values
    if nt != len(np.atleast_1d(ht)):
        raise ValueError("Dimension mismatch between input variables")
    # convert lon, lat to ECEF coordinates
    X, Y, Z = pyTMD.spatial.to_cartesian(
        lon, lat, a_axis=kwargs["a_axis"], flat=kwargs["flat"]
    )
    XYZ = np.c_[X, Y, Z]
    # geocentric latitude (radians)
    theta = np.pi / 2.0 - np.arctan(XYZ[:, 2] / np.hypot(XYZ[:, 0], XYZ[:, 1]))
    # longitude (radians)
    phi = np.arctan2(XYZ[:, 1], XYZ[:, 0])
    # solar ephemerides (convert time to Modified Julian Days)
    SXYZ = pyTMD.astro.solar_ecef(t + 48622.0, **kwargs)
    # lunar ephemerides (convert time to Modified Julian Days)
    LXYZ = pyTMD.astro.lunar_ecef(t + 48622.0, **kwargs)
    # spherical harmonics for degree and order
    Ylm, _ = pyTMD.math.sph_harm(2, theta, phi, m=0, phase=0.0)
    # gravitational potentials
    US = _gravitational(
        XYZ,
        SXYZ,
        l=2,
        mass_ratio=kwargs["mass_ratio_solar"],
        distance=kwargs["AU"],
        **kwargs,
    )
    UL = _gravitational(
        XYZ,
        LXYZ,
        l=2,
        mass_ratio=kwargs["mass_ratio_lunar"],
        distance=kwargs["LD"],
        **kwargs,
    )
    # radiational function
    RS = _radiational(
        XYZ,
        SXYZ,
        l=2,
        distance=kwargs["AU"],
        **kwargs,
    )


def _gravitational(XYZ: np.ndarray, LSXYZ: np.ndarray, l: int, **kwargs):
    """
    Estimate gravitational tides using the response method
    :cite:p:`Munk:1966go`

    Parameters
    ----------
    XYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the observation points (meters)
    LSXYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the sun or moon (meters)
    l: int
        spherical harmonic degree

    Returns
    -------
    grav: np.ndarray
        gravitational potential of the sun or moon
    """
    # default keyword arguments
    a_axis = kwargs.get("a_axis", 6378137.0)
    # average distance between the Earth and the planetary body (meters)
    kwargs.setdefault("distance", 1.495978707e11)
    # mass ratio between the Earth and the planetary body
    kwargs.setdefault("mass_ratio", 332946.0482)
    # radius of the point on the Earth
    radius = pyTMD.math.radius(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2])
    # lunisolar radius from ephemerides
    lunisolar_radius = pyTMD.math.radius(LSXYZ[:, 0], LSXYZ[:, 1], LSXYZ[:, 2])
    # cosine of angles between vectors of the point and the sun/moon
    lunisolar_scalar = pyTMD.math.scalar_product(
        XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], LSXYZ[:, 0], LSXYZ[:, 1], LSXYZ[:, 2]
    ) / (radius * lunisolar_radius)
    # associated Legendre functions of zenith angle for degree l
    Pl, _ = pyTMD.math.legendre(l, lunisolar_scalar)
    # k values from Munk and Cartwright (1966)
    k = a_axis * kwargs["mass_ratio"] * (a_axis / lunisolar_radius) ** (l + 1)
    # gravitational potential of the sun and moon
    grav = k * Pl * (kwargs["distance"] / lunisolar_radius) ** (l + 1)
    # return the gravitational potential
    return grav


def _radiational(XYZ: np.ndarray, SXYZ: np.ndarray, l: int, **kwargs):
    """
    Estimate radiational tides using the response method
    :cite:p:`Munk:1966go`

    Parameters
    ----------
    XYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the observation points (meters)
    SXYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the sun (meters)
    l: int
        spherical harmonic degree

    Returns
    -------
    solar_rad: np.ndarray
        radiation function of the sun
    """
    # default keyword arguments
    a_axis = kwargs.get("a_axis", 6378137.0)
    S = kwargs.get("solar_constant", 1380.0)
    AU = kwargs.get("distance", 1.495978707e11)
    # Earth equatorial radius in AU (~1/23455)
    # defined as parallax in Munk and Cartwright (1966)
    xi = a_axis / AU
    # kappa value (Munk and Cartwright, 1966)
    kappa = _kappa(l, xi)
    # radius of the point on the Earth
    radius = pyTMD.math.radius(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2])
    # lunisolar radius from ephemerides
    solar_radius = pyTMD.math.radius(SXYZ[:, 0], SXYZ[:, 1], SXYZ[:, 2])
    # cosine of angles between vectors of the point and the sun/moon
    solar_scalar = pyTMD.math.scalar_product(
        XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], SXYZ[:, 0], SXYZ[:, 1], SXYZ[:, 2]
    ) / (radius * solar_radius)
    solar_zenith = np.arccos(solar_scalar)
    # create radiation function
    solar_rad = S * (AU / solar_radius) * kappa * solar_scalar**l
    # radiation is only valid during the day
    solar_rad = np.where(solar_zenith < (np.pi / 2.0), solar_rad, 0.0)
    return solar_rad


def _kappa(l: int, xi: float) -> float:
    """
    Kappa values as a function of degree :cite:p:`Munk:1966go`

    Parameters
    ----------
    l: int
        spherical harmonic degree
    xi: float
        parallax

    Returns
    -------
    kappa: float
        kappa value for the given degree
    """
    kappa = np.zeros((4))
    kappa[0] = 1.0 / 4.0 + xi / 6.0
    kappa[1] = 1.0 / 2.0 + 3.0 * xi / 8.0
    kappa[2] = 5.0 / 16.0 + xi / 3.0
    return kappa[l]
