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
    theta = np.pi / 2.0 - np.arctan(
        XYZ[:, 2] / np.sqrt(XYZ[:, 0] ** 2.0 + XYZ[:, 1] ** 2.0)
    )
    # longitude (radians)
    phi = np.arctan2(XYZ[:, 1], XYZ[:, 0])
    # solar ephemerides (convert time to Modified Julian Days)
    SXYZ = pyTMD.astro.solar_ecef(t + 48622.0, **kwargs)
    # lunar ephemerides (convert time to Modified Julian Days)
    LXYZ = pyTMD.astro.lunar_ecef(t + 48622.0, **kwargs)
    # spherical harmonics for degree and order
    Ylm, _ = pyTMD.math.sph_harm(2, theta, phi, m=0, phase=0.0)
    # gravitational potentials
    US, UL = _gravitational(XYZ, SXYZ, LXYZ, l=2, **kwargs)
    # radiational function
    RS = _radiational(XYZ, SXYZ, l=2, **kwargs)


def _gravitational(
    XYZ: np.ndarray, SXYZ: np.ndarray, LXYZ: np.ndarray, l: int, **kwargs
):
    """
    Estimate gravitational tides using the response method
    :cite:p:`Munk:1966go`

    Parameters
    ----------
    XYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the observation points (meters)
    SXYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the sun (meters)
    LXYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the moon (meters)
    l: int
        spherical harmonic degree

    Returns
    -------
    solar_grav: np.ndarray
        gravitatinal potential of the sun
    lunar_grav: np.ndarray
        gravitatinal potential of the moon
    """
    # default keyword arguments
    kwargs.setdefault("a_axis", 6378137.0)
    kwargs.setdefault("flat", 1.0 / 298.257223563)
    kwargs.setdefault("omega", 7.2921151467e-5)
    kwargs.setdefault("AU", 1.495978707e11)
    kwargs.setdefault("LD", 3.84399e8)
    # mass ratios between earth and sun/moon
    kwargs.setdefault("mass_ratio_solar", 332946.0482)
    kwargs.setdefault("mass_ratio_lunar", 0.0123000371)
    # average radius of the earth
    rad_e = kwargs["a_axis"] * (1.0 - kwargs["flat"]) ** (1.0 / 3.0)
    # solar and lunar radii from ephemerides
    solar_radius = np.sqrt(SXYZ[:, 0] ** 2 + SXYZ[:, 1] ** 2 + SXYZ[:, 2] ** 2)
    lunar_radius = np.sqrt(LXYZ[:, 0] ** 2 + LXYZ[:, 1] ** 2 + LXYZ[:, 2] ** 2)
    # geocentric latitude (radians)
    theta = np.arctan(XYZ[:, 2] / np.sqrt(XYZ[:, 0] ** 2.0 + XYZ[:, 1] ** 2.0))
    # longitude (radians)
    phi = np.arctan2(XYZ[:, 1], XYZ[:, 0])
    # convert solar and lunar ephemerides from ECEF to zenith angle
    solar_zenith = pyTMD.spatial.to_zenith(
        SXYZ[:, 0],
        SXYZ[:, 1],
        SXYZ[:, 2],
        np.degrees(phi),
        np.degrees(theta),
        **kwargs,
    )
    lunar_zenith = pyTMD.spatial.to_zenith(
        LXYZ[:, 0],
        LXYZ[:, 1],
        LXYZ[:, 2],
        np.degrees(phi),
        np.degrees(theta),
        **kwargs,
    )
    # associated Legendre functions of zenith angle for degree l
    solar_legendre, _ = pyTMD.math.legendre(l, np.cos(np.radians(solar_zenith)))
    lunar_legendre, _ = pyTMD.math.legendre(l, np.cos(np.radians(lunar_zenith)))
    # k values from Munk and Cartwright (1966)
    solar_k = (
        rad_e * kwargs["mass_ratio_solar"] * (rad_e / solar_radius) ** (l + 1)
    )
    lunar_k = (
        rad_e * kwargs["mass_ratio_lunar"] * (rad_e / lunar_radius) ** (l + 1)
    )
    # gravitational potential of the sun and moon
    solar_grav = (
        solar_k * solar_legendre * (kwargs["AU"] / solar_radius) ** (l + 1)
    )
    lunar_grav = (
        lunar_k * lunar_legendre * (kwargs["LD"] / lunar_radius) ** (l + 1)
    )
    return (solar_grav, lunar_grav)


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
    kwargs.setdefault("a_axis", 6378137.0)
    kwargs.setdefault("solar_constant", 1380.0)
    kwargs.setdefault("AU", 1.495978707e11)
    # Earth equatorial radius in AU (~1/23455)
    # defined as parallax in Munk and Cartwright (1966)
    xi = kwargs["a_axis"] / kwargs["AU"]
    # kappa value (Munk and Cartwright, 1966)
    kappa = _kappa(l, xi)
    # solar radius from ephemerides
    solar_radius = np.sqrt(SXYZ[:, 0] ** 2 + SXYZ[:, 1] ** 2 + SXYZ[:, 2] ** 2)
    # geocentric latitude (radians)
    theta = np.arctan(XYZ[:, 2] / np.sqrt(XYZ[:, 0] ** 2.0 + XYZ[:, 1] ** 2.0))
    # longitude (radians)
    phi = np.arctan2(XYZ[:, 1], XYZ[:, 0])
    # convert solar ephemerides from ECEF to zenith angle
    solar_zenith = pyTMD.spatial.to_zenith(
        SXYZ[:, 0],
        SXYZ[:, 1],
        SXYZ[:, 2],
        np.degrees(phi),
        np.degrees(theta),
        **kwargs,
    )
    # create radiation function
    solar_rad = (
        kwargs["solar_constant"]
        * (kwargs["AU"] / solar_radius)
        * kappa
        * np.cos(np.radians(solar_zenith)) ** l
    )
    # radiation is only valid during the day
    solar_rad = np.where(solar_zenith < 0.5 * np.pi, solar_rad, 0.0)
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
