#!/usr/bin/env python
"""
predict.py
Written by Tyler Sutterley (03/2026)
Prediction routines for gravity tides and tide-generating forces

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

PROGRAM DEPENDENCIES:
    astro.py: computes the basic astronomical mean longitudes
    math.py: Special functions of mathematical physics

UPDATE HISTORY:
    Written 03/2026: split up prediction functions into separate files
"""

from __future__ import annotations

import logging
import numpy as np
import xarray as xr
import pyTMD.astro
import pyTMD.math
import pyTMD.spatial

__all__ = [
    "generating_force",
    "gravity_tide",
    "_out_of_phase",
    "_out_of_phase_diurnal",
    "_out_of_phase_semidiurnal",
    "_frequency_dependence",
    "_frequency_dependence_diurnal",
    "_frequency_dependence_long_period",
    "_frequency_dependence_semidiurnal",
    "_free_to_zero",
]

# number of days between the Julian day epoch and MJD
_jd_mjd = 2400000.5
# number of days between MJD and the tide epoch (1992-01-01T00:00:00)
_mjd_tide = 48622.0
# number of days between MJD and the J2000 epoch
_mjd_j2000 = 51544.5
# Julian century
_century = 36525.0

# get ellipsoidal parameters
_iers = pyTMD.spatial.datum(ellipsoid="IERS", units="MKS")


def generating_force(
    t: np.ndarray,
    XYZ: xr.Dataset,
    SXYZ: xr.Dataset,
    LXYZ: xr.Dataset,
    **kwargs,
):
    r"""
    Compute the tide-generating force due to the gravitational
    attraction of the moon and sun :cite:p:`Tamura:1982wx,Tamura:1987tp`

    Parameters
    ----------
    t: np.ndarray
        Days relative to 1992-01-01T00:00:00
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    SXYZ: xr.Dataset
        Dataset with Earth-centered Earth-fixed coordinates of the sun
    LXYZ: xr.Dataset
        Dataset with Earth-centered Earth-fixed coordinates of the moon
    lmax: int, default 4
        Maximum degree of spherical harmonic expansion
    GM: float, default 3.986004418e14
        Geocentric gravitational constant (m\ :sup:`3` s\ :sup:`-2`)
    mass_ratio_solar: float, default 332946.0482
        Mass ratio between Earth and Sun
    mass_ratio_lunar: float, default 0.0123000371
        Mass ratio between Earth and Moon

    Returns
    -------
    F: xr.Dataset
        Tide-generating force (m s\ :sup:`-2`)
    """
    # set default keyword arguments
    # maximum degree of spherical harmonic expansion
    kwargs.setdefault("lmax", 4)
    # gravitational constants (m^3 s^-2)
    kwargs.setdefault("GM", 3.986004418e14)
    # mass ratios between earth and sun/moon
    kwargs.setdefault("mass_ratio_solar", 332946.0482)
    kwargs.setdefault("mass_ratio_lunar", 0.0123000371)

    # convert dates to Modified Julian Days
    MJD = t + _mjd_tide
    # Earth's radius at the given latitude (meters)
    radius = pyTMD.math.radius(XYZ.X, XYZ.Y, XYZ.Z)
    # distance between the Earth and the sun/moon
    solar_radius = pyTMD.math.radius(SXYZ.X, SXYZ.Y, SXYZ.Z)
    lunar_radius = pyTMD.math.radius(LXYZ.X, LXYZ.Y, LXYZ.Z)
    # cosine of angles between vectors of the point and the sun/moon
    solar_scalar = pyTMD.math.scalar_product(
        XYZ.X, XYZ.Y, XYZ.Z, SXYZ.X, SXYZ.Y, SXYZ.Z
    ) / (radius * solar_radius)
    lunar_scalar = pyTMD.math.scalar_product(
        XYZ.X, XYZ.Y, XYZ.Z, LXYZ.X, LXYZ.Y, LXYZ.Z
    ) / (radius * lunar_radius)
    # unit vectors for dimensions
    unit_vector = XYZ / radius
    solar_unit_vector = SXYZ / solar_radius
    lunar_unit_vector = LXYZ / lunar_radius

    # factors for sun and moon using IAU estimates of mass ratios
    GM_solar = kwargs["mass_ratio_solar"] * kwargs["GM"]
    GM_lunar = kwargs["mass_ratio_lunar"] * kwargs["GM"]
    # gravitational parameters
    K_solar = GM_solar * radius / np.power(solar_radius, 2)
    K_lunar = GM_lunar * radius / np.power(lunar_radius, 2)

    # initialize output tide-generating forces
    F_solar = xr.Dataset()
    F_lunar = xr.Dataset()
    for var in ["X", "Y", "Z"]:
        F_solar[var] = xr.zeros_like(solar_scalar)
        F_lunar[var] = xr.zeros_like(lunar_scalar)

    # compute tide-generating forces
    # for each spherical harmonic degree
    for l in range(2, kwargs["lmax"] + 1):
        # update gravitational parameters for degree
        K_solar *= radius / solar_radius
        K_lunar *= radius / lunar_radius
        # legendre polynomial for degree
        Pl_solar, dPl_solar = pyTMD.math.legendre(l, solar_scalar)
        Pl_lunar, dPl_lunar = pyTMD.math.legendre(l, lunar_scalar)
        # divide differential by u
        # ignore divide by zero and invalid value warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            dPl_solar /= np.sqrt(1 - solar_scalar**2)
            dPl_lunar /= np.sqrt(1 - lunar_scalar**2)
        # add solar and lunar terms for degree
        F_solar += (K_solar / radius) * (
            l * Pl_solar * unit_vector
            + dPl_solar * solar_scalar * unit_vector
            - dPl_solar * solar_unit_vector
        )
        F_lunar += (K_lunar / radius) * (
            l * Pl_lunar * unit_vector
            + dPl_lunar * lunar_scalar * unit_vector
            - dPl_lunar * lunar_unit_vector
        )

    # sum solar and lunar components
    F = F_solar + F_lunar
    # add units attributes to output dataset
    for var in F.data_vars:
        F[var].attrs["units"] = "m/s^2"

    # return the tide generating force
    return F


def gravity_tide(
    t: np.ndarray,
    XYZ: xr.Dataset,
    SXYZ: xr.Dataset,
    LXYZ: xr.Dataset,
    deltat: float = 0.0,
    a_axis: float = _iers.a_axis,
    tide_system: str = "tide_free",
    **kwargs,
):
    r"""
    Compute the estimated gravity tides due to the gravitational
    attraction of the moon and sun :cite:p:`Tamura:1987tp,Hartmann:1995jp`

    Parameters
    ----------
    t: np.ndarray
        Days relative to 1992-01-01T00:00:00
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    SXYZ: xr.Dataset
        Dataset with Earth-centered Earth-fixed coordinates of the sun
    LXYZ: xr.Dataset
        Dataset with Earth-centered Earth-fixed coordinates of the moon
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    a_axis: float, default 6378136.3
        Semi-major axis of the Earth (meters)
    tide_system: str, default 'tide_free'
        Permanent tide system for the output solid Earth tide

            - ``'tide_free'``: no permanent direct and indirect tidal potentials
            - ``'zero_tide'``: direct permanent tidal potentials (no indirect)
    lmax: int, default 3
        Maximum degree of spherical harmonic expansion
    h2: float, default 0.6078
        Degree-2 Love number of vertical displacement
    k2: float, default 0.3019
        Degree-2 Love number of gravitational potential
    h3: float, default 0.292
        Degree-3 Love number of vertical displacement
    k3: float, default 0.093
        Degree-3 Love number of gravitational potential
    h4: float, default 0.18
        Degree-4 Love number of vertical displacement
    k4: float, default 0.0
        Degree-4 Love number of gravitational potential
    GM: float, default 3.986004418e14
        Geocentric gravitational constant (m\ :sup:`3` s\ :sup:`-2`)
    mass_ratio_solar: float, default 332946.0482
        Mass ratio between Earth and Sun
    mass_ratio_lunar: float, default 0.0123000371
        Mass ratio between Earth and Moon

    Returns
    -------
    G: xr.Dataset
        Gravity tides (m s\ :sup:`-2`)
    """
    # set default keyword arguments
    # maximum degree of spherical harmonic expansion
    kwargs.setdefault("lmax", 4)
    # nominal Love numbers for degrees 2 through 4
    kwargs.setdefault("h2", 0.609)
    kwargs.setdefault("k2", 0.3019)
    kwargs.setdefault("h3", 0.292)
    kwargs.setdefault("k3", 0.093)
    kwargs.setdefault("h4", 0.18)
    kwargs.setdefault("k4", 0.0)
    # Earth parameters
    kwargs.setdefault("flat", _iers.flat)
    kwargs.setdefault("J2", _iers.J2)
    # gravitational constants (m^3 s^-2)
    kwargs.setdefault("GM", 3.986004418e14)
    # mass ratios between earth and sun/moon
    kwargs.setdefault("mass_ratio_solar", 332946.0482)
    kwargs.setdefault("mass_ratio_lunar", 0.0123000371)

    # convert dates to Modified Julian Days in Ephemeris time
    MJD = t + _mjd_tide + deltat
    # Earth's radius at the given latitude (meters)
    radius = pyTMD.math.radius(XYZ.X, XYZ.Y, XYZ.Z)
    # average radius of the Earth with same volume as ellipsoid
    rad_e = a_axis * (1.0 - kwargs["flat"]) ** (1.0 / 3.0)
    # sine of geocentric latitude
    sinphi = XYZ["Z"] / radius
    # distance between the Earth and the sun/moon
    solar_radius = pyTMD.math.radius(SXYZ.X, SXYZ.Y, SXYZ.Z)
    lunar_radius = pyTMD.math.radius(LXYZ.X, LXYZ.Y, LXYZ.Z)
    # cosine of angles between vectors of the point and the sun/moon
    solar_scalar = pyTMD.math.scalar_product(
        XYZ.X, XYZ.Y, XYZ.Z, SXYZ.X, SXYZ.Y, SXYZ.Z
    ) / (radius * solar_radius)
    lunar_scalar = pyTMD.math.scalar_product(
        XYZ.X, XYZ.Y, XYZ.Z, LXYZ.X, LXYZ.Y, LXYZ.Z
    ) / (radius * lunar_radius)
    # unit vectors for dimensions
    unit_vector = XYZ / radius

    # factors for sun and moon using IAU estimates of mass ratios
    GM_solar = kwargs["mass_ratio_solar"] * kwargs["GM"]
    GM_lunar = kwargs["mass_ratio_lunar"] * kwargs["GM"]
    # gravitational parameters
    K_solar = GM_solar * radius / np.power(solar_radius, 2)
    K_lunar = GM_lunar * radius / np.power(lunar_radius, 2)
    # factors for degree 2
    F2_solar = K_solar * (radius / solar_radius)
    F2_lunar = K_lunar * (radius / lunar_radius)

    # initialize output gravity tide estimates
    G_solar = xr.Dataset()
    G_lunar = xr.Dataset()
    for var in ["X", "Y", "Z"]:
        G_solar[var] = xr.zeros_like(solar_scalar)
        G_lunar[var] = xr.zeros_like(lunar_scalar)

    # compute estimated gravity tides
    # for each spherical harmonic degree
    for l in range(2, kwargs["lmax"] + 1):
        # get the degree-dependent Love numbers for the gravity tide
        hl = kwargs.get(f"h{l}", 0)
        kl = kwargs.get(f"k{l}", 0)
        # gravimetric factor from Farrell
        dl = 1.0 + 2.0 * hl / l - (l + 1.0) * kl / l
        # include latitudinal dependence for degree 2
        if l == 2:
            dl += -0.005 * np.sqrt(3 / 4) * (7.0 * sinphi**2 - 1.0)
        # update gravitational parameters for degree
        K_solar *= radius / solar_radius
        K_lunar *= radius / lunar_radius
        # legendre polynomial for degree
        Pl_solar = pyTMD.math._assoc_legendre(l, 0, solar_scalar)
        Pl_lunar = pyTMD.math._assoc_legendre(l, 0, lunar_scalar)
        # add solar and lunar terms for degree
        G_solar -= (K_solar * dl / radius) * (l * Pl_solar * unit_vector)
        G_lunar -= (K_lunar * dl / radius) * (l * Pl_lunar * unit_vector)

    # sum solar and lunar components
    G = G_solar + G_lunar
    # add units attributes to output dataset
    for var in G.data_vars:
        G[var].attrs["units"] = "m/s^2"

    # return the estimated gravity tides
    return G

