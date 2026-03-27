#!/usr/bin/env python
"""
potential.py
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
]

# number of days between MJD and the tide epoch (1992-01-01T00:00:00)
_mjd_tide = 48622.0

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
    lmax: int, default 3
        Maximum degree of spherical harmonic expansion
    h2: float, default 0.6078
        Degree-2 Love number of vertical displacement
    k2: float, default 0.30102
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
    kwargs.setdefault("lmax", 3)
    # nominal Love numbers for degrees 2 through 4
    kwargs.setdefault("h2", 0.6078)
    kwargs.setdefault("k2", 0.30102)
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
    K_earth = kwargs["GM"] / radius**2
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
        gl = 1.0 + 2.0 * hl / l - (l + 1.0) * kl / l
        # include latitudinal dependence for degree 2
        if l == 2:
            gl += -0.005 * np.sqrt(3.0 / 4.0) * (7.0 * sinphi**2 - 1.0)
        # update gravitational parameters for degree
        K_solar *= radius / solar_radius
        K_lunar *= radius / lunar_radius
        # legendre polynomial for degree
        Pl_solar = pyTMD.math._assoc_legendre(l, 0, solar_scalar)
        Pl_lunar = pyTMD.math._assoc_legendre(l, 0, lunar_scalar)
        # add solar and lunar terms for degree
        G_solar -= (K_solar * gl / radius) * (l * Pl_solar * unit_vector)
        G_lunar -= (K_lunar * gl / radius) * (l * Pl_lunar * unit_vector)

    # sum solar and lunar components
    G = G_solar + G_lunar
    # corrections for out-of-phase portions of the Love numbers
    G += _out_of_phase(XYZ, SXYZ, LXYZ, F2_solar, F2_lunar)
    # corrections for the frequency dependence
    G += _frequency_dependence(XYZ, MJD, deltat=deltat) * K_earth
    # add units attributes to output dataset
    for var in G.data_vars:
        G[var].attrs["units"] = "m/s^2"

    # return the estimated gravity tides
    return G


def _out_of_phase(
    XYZ: xr.Dataset,
    SXYZ: xr.Dataset,
    LXYZ: xr.Dataset,
    F2_solar: np.ndarray,
    F2_lunar: np.ndarray,
):
    """
    Wrapper function to compute the out-of-phase corrections induced
    by mantle anelasticity :cite:p:`Petit:2010tp`

    Parameters
    ----------
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    SXYZ: xr.Dataset
        Dataset with Earth-centered Earth-fixed coordinates of the sun
    LXYZ: xr.Dataset
        Dataset with Earth-centered Earth-fixed coordinates of the moon
    F2_solar: np.ndarray
        Factors for the sun
    F2_lunar: np.ndarray
        Factors for the moon

    Returns
    -------
    G: xr.Dataset
        Gravity tide corrections
    """
    # compute diurnal and semi-diurnal corrections separately
    # for both the sun and moon
    G = _out_of_phase_diurnal(XYZ, SXYZ, F2_solar)
    G += _out_of_phase_diurnal(XYZ, LXYZ, F2_lunar)
    G += _out_of_phase_semidiurnal(XYZ, SXYZ, F2_solar)
    G += _out_of_phase_semidiurnal(XYZ, LXYZ, F2_lunar)
    # return the out-of-phase corrections
    return G


def _out_of_phase_diurnal(
    XYZ: xr.Dataset,
    LSXYZ: xr.Dataset,
    F2: np.ndarray,
    dh2: float = -0.0025,
    dk2: float = -0.00144,
):
    """
    Computes the out-of-phase corrections induced by mantle
    anelasticity in the diurnal band :cite:p:`Petit:2010tp`

    Parameters
    ----------
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    LSXYZ: xr.Dataset
        Dataset with Earth-centered Earth-fixed coordinates of the sun or moon
    F2: np.ndarray
        Factors for the sun or moon
    dh2: float, default -0.0025
        Love number correction for the diurnal band
    dk2: float, default -0.00144
        Love number correction for the diurnal band

    Returns
    -------
    G: xr.Dataset
        Gravity tide corrections
    """
    # degree and order for diurnal tides
    l, m = (2, 1)
    # differential in the gravimetric factors from Farrell
    dg2 = 2.0 * dh2 / l - (l + 1.0) * dk2 / l
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    # sine and cosine of (geocentric) latitude
    sinphi = XYZ["Z"] / radius
    cosphi = np.sqrt(XYZ["X"] ** 2 + XYZ["Y"] ** 2) / radius
    # double angle formulas of cosine/sine latitude
    sin2phi = 2.0 * sinphi * cosphi
    # sine and cosine of longitude
    sinla = XYZ["Y"] / cosphi / radius
    cosla = XYZ["X"] / cosphi / radius
    # compute the normalized position vector of the Sun/Moon
    lunisolar_radius = pyTMD.math.radius(LSXYZ["X"], LSXYZ["Y"], LSXYZ["Z"])
    # sine and cosine of Solar/Lunar declinations
    lunisolar_sinphi = LSXYZ["Z"] / lunisolar_radius
    lunisolar_cosphi = (
        np.sqrt(LSXYZ["X"] ** 2 + LSXYZ["Y"] ** 2) / lunisolar_radius
    )
    # double angle formulas of sine Solar/Lunar declinations
    lunisolar_sin2phi = 2.0 * lunisolar_cosphi * lunisolar_sinphi
    # sine and cosine of Solar/Lunar hour angles
    lunisolar_sinla = LSXYZ["Y"] / lunisolar_cosphi / lunisolar_radius
    lunisolar_cosla = LSXYZ["X"] / lunisolar_cosphi / lunisolar_radius
    # calculate offsets
    GR = (-0.75 * dg2 * F2 * sin2phi * lunisolar_sin2phi) * (
        sinla * lunisolar_cosla - cosla * lunisolar_sinla
    )
    # rotate to cartesian coordinates
    GX = GR * cosla * cosphi
    GY = GR * sinla * cosphi
    GZ = GR * sinphi
    # compute as additive correction
    G = xr.Dataset()
    G["X"] = -l * GX / radius
    G["Y"] = -l * GY / radius
    G["Z"] = -l * GZ / radius
    # return the corrections
    return G


def _out_of_phase_semidiurnal(
    XYZ: xr.Dataset,
    LSXYZ: xr.Dataset,
    F2: np.ndarray,
    dh2: float = -0.0022,
    dk2: float = -0.0013,
):
    """
    Computes the out-of-phase corrections induced by mantle
    anelasticity in the semi-diurnal band :cite:p:`Petit:2010tp`

    Parameters
    ----------
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    LSXYZ: xr.Dataset
        Dataset with Earth-centered Earth-fixed coordinates of the sun or moon
    F2: np.ndarray
        Factors for the sun or moon
    dh2: float, default -0.0022
        Love number correction for the semi-diurnal band
    dk2: float, default -0.0013
        Love number correction for the semi-diurnal band

    Returns
    -------
    G: xr.Dataset
        Gravity tide corrections
    """
    # degree and order for semi-diurnal tides
    l, m = (2, 2)
    # differential in the gravimetric factors from Farrell
    dg2 = 2.0 * dh2 / l - (l + 1.0) * dk2 / l
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    # sine and cosine of (geocentric) latitude
    sinphi = XYZ["Z"] / radius
    cosphi = np.sqrt(XYZ["X"] ** 2 + XYZ["Y"] ** 2) / radius
    # sine and cosine of longitude
    sinla = XYZ["Y"] / cosphi / radius
    cosla = XYZ["X"] / cosphi / radius
    # double angle formulas of cosine/sine longitude
    cos2la = cosla**2 - sinla**2
    sin2la = 2.0 * cosla * sinla
    # compute the normalized position vector of the Sun/Moon
    lunisolar_radius = pyTMD.math.radius(LSXYZ["X"], LSXYZ["Y"], LSXYZ["Z"])
    # cosine of Solar/Lunar declinations
    lunisolar_cosphi = (
        np.sqrt(LSXYZ["X"] ** 2 + LSXYZ["Y"] ** 2) / lunisolar_radius
    )
    # sine and cosine of Solar/Lunar hour angles
    lunisolar_sinla = LSXYZ["Y"] / lunisolar_cosphi / lunisolar_radius
    lunisolar_cosla = LSXYZ["X"] / lunisolar_cosphi / lunisolar_radius
    # double angle formulas of cosine/sine Solar/Lunar hour angles
    lunisolar_cos2la = lunisolar_cosla**2 - lunisolar_sinla**2
    lunisolar_sin2la = 2.0 * lunisolar_cosla * lunisolar_sinla
    # calculate offsets
    GR = (-0.75 * dg2 * F2 * cosphi**2 * lunisolar_cosphi**2) * (
        sin2la * lunisolar_cos2la - cos2la * lunisolar_sin2la
    )
    # rotate to cartesian coordinates
    GX = GR * cosla * cosphi
    GY = GR * sinla * cosphi
    GZ = GR * sinphi
    # compute as additive correction
    G = xr.Dataset()
    G["X"] = -l * GX / radius
    G["Y"] = -l * GY / radius
    G["Z"] = -l * GZ / radius
    # return the corrections
    return G


def _frequency_dependence(
    XYZ: xr.Dataset,
    MJD: np.ndarray,
    deltat: float | np.ndarray = 0.0,
):
    """
    Wrapper function to compute the frequency dependent in-phase and
    out-of-phase corrections :cite:p:`Petit:2010tp`

    Parameters
    ----------
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    MJD: np.ndarray
        Modified Julian Day (MJD)
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)

    Returns
    -------
    G: xr.Dataset
        Gravity tide corrections
    """
    # compute corrections for each species separately
    G = _frequency_dependence_diurnal(XYZ, MJD, deltat=deltat)
    G += _frequency_dependence_long_period(XYZ, MJD, deltat=deltat)
    # return the frequency dependent corrections
    return G


def _frequency_dependence_diurnal(
    XYZ: xr.Dataset,
    MJD: np.ndarray,
    deltat: float | np.ndarray = 0.0,
):
    """
    Computes the frequency dependent in-phase and out-of-phase corrections
    of the diurnal band :cite:p:`Petit:2010tp`

    Parameters
    ----------
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    MJD: np.ndarray
        Modified Julian Day (MJD)
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)

    Returns
    -------
    G: xr.Dataset
        Gravity tide corrections
    """
    # Corrections for Frequency Dependence of Diurnal Tides
    # based on tables 6.5a and 7.3a from IERS conventions
    columns = [
        "tau",
        "s",
        "h",
        "p",
        "np",
        "ps",
        "k",
        "dG_ip",
        "dG_op",
    ]
    # note: dG_ip and dG_op are scaled by 1e6
    table = xr.DataArray(
        np.array(
            [
                [1, -3, 2, 0, 0, 0, -1, -2.79, -0.21],
                [1, -2, 0, 1, -1, 0, -1, -2.64, -0.29],
                [1, -2, 0, 1, 0, 0, -1, -13.95, -1.56],
                [1, -1, 0, 0, -1, 0, -1, -5.32, -1.96],
                [1, -1, 0, 0, 0, 0, -1, -27.92, -10.42],
                [1, 0, 0, -1, 0, 0, -1, -3.09, 0.47],
                [1, 0, 0, 1, 0, 0, -1, -8.95, 1.31],
                [1, 1, -3, 0, 0, 1, -1, 24.65, -1.38],
                [1, 1, -2, 0, -1, 0, -1, -6.61, 0.34],
                [1, 1, -2, 0, 0, 0, -1, 599.77, -31.09],
                [1, 1, -1, 0, 0, -1, -1, -8.12, 0.38],
                [1, 1, -1, 0, 0, 1, -1, -22.93, 1.06],
                [1, 1, 0, 0, -1, 0, -1, 126.19, -6.54],
                [1, 1, 0, 0, 0, 0, -1, -6786.96, 361.70],
                [1, 1, 0, 0, 1, 0, -1, -984.51, 53.74],
                [1, 1, 0, 0, 2, 0, -1, 22.70, -1.27],
                [1, 1, 1, 0, 0, -1, -1, 310.08, 5.83],
                [1, 1, 1, 0, 0, 1, -1, -4.71, -0.09],
                [1, 1, 1, 0, 1, -1, -1, 4.15, 0.01],
                [1, 1, 2, -2, 0, 0, -1, 3.32, -0.10],
                [1, 1, 2, 0, 0, 0, -1, 77.34, -2.38],
                [1, 1, 2, 0, 1, 0, -1, -2.84, 0.09],
                [1, 2, -2, 1, 0, 0, -1, 8.86, -0.17],
                [1, 2, 0, -1, 0, 0, -1, 41.92, -0.70],
                [1, 2, 0, -1, 1, 0, -1, 8.29, -0.14],
            ]
        ),
        dims=["constituent", "argument"],
        coords=dict(argument=columns),
    )
    coef = table.to_dataset(dim="argument")
    # get phase angles (Doodson arguments)
    TAU, S, H, P, ZNS, PS = pyTMD.astro.doodson_arguments(MJD + deltat)
    # variable for multiples of 90 degrees (Ray technical note 2017)
    # full expansion of Equilibrium Tide includes some negative cosine
    # terms and some sine terms (Pugh and Woodworth, 2014)
    K = np.pi / 2.0 + np.zeros_like(TAU)
    # dataset of arguments
    arguments = xr.Dataset(
        data_vars=dict(
            tau=(["time"], TAU),
            s=(["time"], S),
            h=(["time"], H),
            p=(["time"], P),
            np=(["time"], ZNS),
            ps=(["time"], PS),
            k=(["time"], K),
        ),
        coords=dict(time=np.atleast_1d(MJD)),
    )
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    # geocentric latitude and colatitude (radians)
    phi = np.arctan2(XYZ.Z, np.sqrt(XYZ.X**2.0 + XYZ.Y**2.0))
    theta = np.pi / 2.0 - phi
    # calculate longitude (radians)
    la = np.arctan2(XYZ.Y, XYZ.X)
    # compute phase angle of tide potential (Greenwich)
    phase = (
        arguments.tau * coef["tau"]
        + arguments.s * coef["s"]
        + arguments.h * coef["h"]
        + arguments.p * coef["p"]
        + arguments.np * coef["np"]
        + arguments.ps * coef["ps"]
        + arguments.k * coef["k"]
    )
    # rotate spherical harmonic functions by phase angles
    l, m = (2, 1)
    Ylm, _ = pyTMD.math.sph_harm(l, theta, la, m=m, phase=phase)
    # calculate offsets in local coordinates
    GR = (coef["dG_ip"] * Ylm.real - coef["dG_op"] * Ylm.imag).sum(
        dim="constituent", skipna=False
    )
    # rotate to cartesian coordinates
    GX = GR * np.cos(la) * np.cos(phi)
    GY = GR * np.sin(la) * np.cos(phi)
    GZ = GR * np.sin(phi)
    # compute as additive correction
    G = xr.Dataset()
    G["X"] = -1e-6 * l * GX / radius
    G["Y"] = -1e-6 * l * GY / radius
    G["Z"] = -1e-6 * l * GZ / radius
    # return the corrections
    return G


def _frequency_dependence_long_period(
    XYZ: xr.Dataset,
    MJD: np.ndarray,
    deltat: float | np.ndarray = 0.0,
):
    """
    Computes the frequency dependent in-phase and out-of-phase corrections
    induced by mantle anelasticity in the long-period band
    :cite:p:`Petit:2010tp`

    Parameters
    ----------
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    MJD: np.ndarray
        Modified Julian Day (MJD)
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)

    Returns
    -------
    G: xr.Dataset
        Gravity tide corrections
    """
    # Corrections for Frequency Dependence of Long-Period Tides
    # based on tables 6.5b and 7.3b from IERS conventions
    columns = [
        "tau",
        "s",
        "h",
        "p",
        "np",
        "ps",
        "k",
        "dG_ip",
        "dG_op",
    ]
    # note: dG_ip and dG_op are scaled by 1e6
    table = xr.DataArray(
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 141.08, -33.01],
                [0, 0, 0, 0, 2, 0, 0, -1.25, 0.29],
                [0, 0, 1, 0, 0, -1, 0, -16.25, 3.75],
                [0, 0, 2, 0, 0, 0, 0, -92.65, 21.29],
                [0, 0, 2, 0, 1, 0, 0, 2.28, -0.52],
                [0, 0, 3, 0, 0, -1, 0, -5.10, 1.17],
                [0, 1, -2, 1, -1, 0, 0, 1.13, -0.26],
                [0, 1, -2, 1, 0, 0, 0, -15.67, 3.56],
                [0, 1, 0, -1, -1, 0, 0, 5.27, -1.20],
                [0, 1, 0, -1, 0, 0, 0, -80.31, 18.20],
                [0, 1, 0, -1, 1, 0, 0, 5.21, -1.18],
                [0, 1, 0, 1, 0, 0, 0, 4.28, -0.97],
                [0, 1, 0, 1, 1, 0, 0, 1.74, -0.40],
                [0, 1, 2, -1, 0, 0, 0, 1.09, -0.25],
                [0, 2, -2, 0, 0, 0, 0, -12.20, 2.75],
                [0, 2, 0, -2, 0, 0, 0, -5.97, 1.34],
                [0, 2, 0, 0, 0, 0, 0, -137.69, 31.01],
                [0, 2, 0, 0, 1, 0, 0, -57.07, 12.85],
                [0, 2, 0, 0, 2, 0, 0, -5.34, 1.20],
                [0, 3, -2, -1, 0, 0, 0, -1.82, 0.41],
                [0, 3, -2, 1, 0, 0, 0, -4.76, 1.07],
                [0, 3, -2, 1, 1, 0, 0, -1.97, 0.44],
                [0, 3, 0, -1, 0, 0, 0, -24.91, 5.59],
                [0, 3, 0, -1, 1, 0, 0, -10.32, 2.32],
                [0, 4, -2, 0, 0, 0, 0, -3.84, 0.86],
                [0, 4, -2, 0, 1, 0, 0, -1.59, 0.36],
                [0, 4, 0, -2, 0, 0, 0, -3.17, 0.71],
                [0, 4, 0, -2, 1, 0, 0, -1.31, 0.29],
            ]
        ),
        dims=["constituent", "argument"],
        coords=dict(argument=columns),
    )
    coef = table.to_dataset(dim="argument")
    # get phase angles (Doodson arguments)
    TAU, S, H, P, ZNS, PS = pyTMD.astro.doodson_arguments(MJD + deltat)
    # variable for multiples of 90 degrees (Ray technical note 2017)
    # full expansion of Equilibrium Tide includes some negative cosine
    # terms and some sine terms (Pugh and Woodworth, 2014)
    K = np.pi / 2.0 + np.zeros_like(TAU)
    # dataset of arguments
    arguments = xr.Dataset(
        data_vars=dict(
            tau=(["time"], TAU),
            s=(["time"], S),
            h=(["time"], H),
            p=(["time"], P),
            np=(["time"], ZNS),
            ps=(["time"], PS),
            k=(["time"], K),
        ),
        coords=dict(time=np.atleast_1d(MJD)),
    )
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    # geocentric latitude and colatitude (radians)
    phi = np.arctan2(XYZ.Z, np.sqrt(XYZ.X**2.0 + XYZ.Y**2.0))
    theta = np.pi / 2.0 - phi
    # calculate longitude (radians)
    la = np.arctan2(XYZ.Y, XYZ.X)
    # compute phase angle of tide potential (Greenwich)
    phase = (
        arguments.tau * coef["tau"]
        + arguments.s * coef["s"]
        + arguments.h * coef["h"]
        + arguments.p * coef["p"]
        + arguments.np * coef["np"]
        + arguments.ps * coef["ps"]
        + arguments.k * coef["k"]
    )
    # rotate spherical harmonic functions by phase angles
    l, m = (2, 0)
    Ylm, _ = pyTMD.math.sph_harm(l, theta, la, m=m, phase=phase)
    # calculate offsets in local coordinates
    GR = (coef["dG_ip"] * Ylm.real - coef["dG_op"] * Ylm.imag).sum(
        dim="constituent", skipna=False
    )
    # rotate to cartesian coordinates
    GX = GR * np.cos(la) * np.cos(phi)
    GY = GR * np.sin(la) * np.cos(phi)
    GZ = GR * np.sin(phi)
    # compute as additive correction
    G = xr.Dataset()
    G["X"] = -1e-6 * l * GX / radius
    G["Y"] = -1e-6 * l * GY / radius
    G["Z"] = -1e-6 * l * GZ / radius
    # return the corrections
    return G
