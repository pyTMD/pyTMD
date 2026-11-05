#!/usr/bin/env python
"""
response.py
Written by Tyler Sutterley (11/2026)
Routines for estimating tidal constituents using the response method

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://scipy.org
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

PROGRAM DEPENDENCIES:
    astro.py: computes the basic astronomical mean longitudes
    constituents.py: calculates constituent parameters and nodal arguments
    math.py: special mathematical functions
    spatial.py: utilities for spatial operations
    constants.py: harmonic tidal analysis

UPDATE HISTORY:
    Updated 11/2026: added function for building the design matrix
        added new arguments for the output latitude and longitude
        added new arguments for setting the lags
    Written 11/2024
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import xarray as xr
import pyTMD.astro
import pyTMD.math
import pyTMD.spatial
from pyTMD.solve.constants import constants

__all__ = [
    "response",
    "_build_design_matrix",
    "_gravitational",
    "_radiational",
    "_kappa",
]

# number of days between MJD and the tide epoch (1992-01-01T00:00:00)
_mjd_tide = 48622.0

# default constituents used when deriving harmonic constants
_default_constituents = [
    "k1",
    "k2",
    "m2",
    "m4",
    "o1",
    "n2",
    "q1",
    "s1",
    "s2",
]


def response(
    t: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ht: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    constituents: str | list | np.ndarray,
    deltat: float | np.ndarray = 0.0,
    corrections: str = "OTIS",
    lags: list | np.ndarray = (-2, -1, 0, 1, 2),
    lag_interval: float = 2.0,
    lmax: int = 3,
    radiational: bool = False,
    order: int = 0,
    solver: str = "lstsq",
    span: float = 731.0,
    spacing: float = 1.0 / 24.0,
    **kwargs,
):
    r"""
    Estimate tidal constituents using the response method
    :cite:p:`Munk:1966go,Groves:1975ky,Zetler:1975uv,Cartwright:1990ck`

    The tide :math:`\zeta(t)` is modeled as a weighted sum of lagged
    reference input functions :math:`c_l^m(t)` for each degree :math:`l`:

    .. math::
        \zeta(t) = \sum_{l,m}\sum_{s}
            u_{lm}(s)\,\mathrm{Re}\left[c_l^m(t - s\Delta\tau)\right]
            + v_{lm}(s)\,\mathrm{Im}\left[c_l^m(t - s\Delta\tau)\right]

    Parameters
    ----------
    t: np.ndarray
        Time (days relative to January 1, 1992)
    xs: np.ndarray
        Longitude coordinates (degrees east)
    ys: np.ndarray
        Latitude coordinates (degrees north)
    ht: np.ndarray
        Input time series (elevations)
    x: np.ndarray
        Output longitude coordinates (degrees east)
    y: np.ndarray
        Output latitude coordinates (degrees north)
    constituents: str, list or np.ndarray
        Tidal constituent ID(s) for output harmonic constants
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    corrections: str, default 'OTIS'
        Use nodal corrections from OTIS/ATLAS or GOT/FES models
    lags: list or np.ndarray, default (-2, -1, 0, 1, 2)
        Lag indices :math:`s` for the response weights
    lag_interval: float, default 2.0
        Lag interval :math:`\Delta\tau` (days)
    lmax: int, default 3
        Maximum degree of the gravitational inputs
    radiational: bool, default False
        Include radiational inputs (degrees 1 and 2)
    order: int, default 0
        Degree of the polynomial to add to the fit
    solver: str, default 'lstsq'
        Least squares solver to use

        - ``'lstsq'``: least squares solution
        - ``'gelsy'``: complete orthogonal factorization
        - ``'gelss'``: singular value decomposition (SVD)
        - ``'gelsd'``: SVD with divide-and-conquer method
    span: float, default 731.0
        Length of the synthetic time series used to derive
        harmonic constants from the response weights (days)
    spacing: float, default 1/24
        Time spacing of the synthetic time series (days)
    **kwargs: dict
        Keyword arguments for ephemeris calculations

    Returns
    -------
    ds: xr.Dataset
        ``Dataset`` of complex harmonic constants
    """
    # check if input constituents is a string
    if isinstance(constituents, str):
        constituents = [constituents]
    # apply tt_ut1 to convert to Terrestrial Time
    tt = t + deltat
    # verify shape of variables
    tt = np.ravel(tt).astype(np.float64)
    ht = np.ravel(ht)
    # broadcast coordinates to the time dimension
    xs = np.broadcast_to(np.ravel(xs), tt.shape).astype(np.float64)
    ys = np.broadcast_to(np.ravel(ys), tt.shape).astype(np.float64)
    # check that the number of time values matches the number of height values
    if len(tt) != len(ht):
        raise ValueError("Dimension mismatch between input variables")
    # reduce variables to finite values
    if not np.isfinite(ht).all():
        (valid,) = np.nonzero(
            np.isfinite(tt)
            & np.isfinite(ht)
            & np.isfinite(xs)
            & np.isfinite(ys)
        )
        tt = tt[valid]
        ht = ht[valid]
        xs = xs[valid]
        ys = ys[valid]

    # create design matrix
    M = []
    # build polynomial functions for design matrix
    tmean = np.mean(tt)
    for o in range(order + 1):
        # add polynomial term
        M.append(np.power(tt - tmean, o))
    # for each lag
    for lag in lags:
        # calculate functionals at times and locations
        D = _build_design_matrix(
            tt,
            xs,
            ys,
            lag,
            lag_interval=lag_interval,
            lmax=lmax,
            radiational=radiational,
            **kwargs,
        )
        # add to complete design matrix
        M.extend(D)
    # take the transpose of the design matrix
    M = np.transpose(M)
    # check that there are enough values for a fit
    nt, nparams = M.shape
    if nt <= nparams:
        raise ValueError("Not enough values for fit")

    # use a least-squares fit to solve for the response weights
    if solver == "lstsq":
        p, res, rnk, s = np.linalg.lstsq(M, ht, rcond=None)
    elif solver in ("gelsd", "gelsy", "gelss"):
        p, res, rnk, s = scipy.linalg.lstsq(M, ht, lapack_driver=solver)
    else:
        raise ValueError(f"Unknown solver {solver}")
    # response weights (without the polynomial terms)
    w = p[order + 1 :]

    # create a synthetic time series of the tidal response at the
    # output coordinates for deriving harmonic constants
    ts = tmean + np.arange(-span / 2.0, span / 2.0, spacing)
    S = []
    # for each lag
    for lag in lags:
        # calculate functionals at synthetic times and output locations
        D = _build_design_matrix(
            ts,
            x,
            y,
            lag,
            lag_interval=lag_interval,
            lmax=lmax,
            radiational=radiational,
            **kwargs,
        )
        S.extend(D)
    # take the transpose of the synthetic design matrix
    S = np.transpose(S)
    # compute the time series from the synthetic
    hs = np.dot(S, w)

    # include a set of default constituents along with the output list
    # this should hopefully reduce the chances of signal leakage
    fit_constituents = sorted(set(constituents) | set(_default_constituents))
    # harmonic analysis of the synthetic response
    hc = constants(
        ts,
        hs,
        fit_constituents,
        corrections=corrections,
    )

    # reduce to the output constituents
    ds = hc[list(constituents)]
    ds.attrs["method"] = "response"
    ds.attrs["longitude"] = x
    ds.attrs["latitude"] = y
    return ds


def _build_design_matrix(
    t: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    lag: int,
    lag_interval: float = 2.0,
    lmax: int = 3,
    radiational: bool = False,
    **kwargs,
):
    r"""
    Calculate the (lagged) reference input functions for the
    response method :cite:p:`Munk:1966go`

    .. math::
        c_l^m(t) = \epsilon_m\frac{4\pi}{2l+1}\sum_{b}K_{l,b}(t)
            Y_l^{m*}(\theta_b, \lambda_b)\,Y_l^m(\theta, \lambda)

    where :math:`\epsilon_m` is 1 for :math:`m=0` and 2 otherwise,
    and the summation is over the Moon and Sun.

    Parameters
    ----------
    t: np.ndarray
        Time (days relative to January 1, 1992)
    lon: np.ndarray
        Longitude coordinates (degrees east)
    lat: np.ndarray
        Latitude coordinates (degrees north)
    lag: int
        Lag index :math:`s` for the response weights
    lag_interval: float, default 2.0
        Lag interval :math:`\Delta\tau` (days)
    lmax: int, default 3
        Maximum degree of the gravitational inputs
    radiational: bool, default False
        Include radiational inputs (degrees 1 and 2)
    **kwargs: dict
        Keyword arguments for ephemeris calculations

    Returns
    -------
    M: np.ndarray
        Input functions for design matrix
    """
    # default keyword arguments
    kwargs.setdefault("a_axis", 6378137.0)
    kwargs.setdefault("flat", 1.0 / 298.257223563)
    kwargs.setdefault("ephemerides", "approximate")
    # mean distances between the Earth and the sun/moon (meters)
    kwargs.setdefault("AU", 1.495978707e11)
    # mass ratios between the sun/moon and the Earth
    kwargs.setdefault("mass_ratio_solar", 332946.0482)
    kwargs.setdefault("mass_ratio_lunar", 0.0123000371)
    # verify degree of the gravitational inputs
    if lmax < 2:
        raise ValueError("Maximum degree must be at least 2")
    # verify shapes of variables
    t = np.atleast_1d(t).astype(np.float64)
    lon = np.broadcast_to(np.atleast_1d(lon), t.shape)
    lat = np.broadcast_to(np.atleast_1d(lat), t.shape)
    # convert lon, lat to ECEF coordinates
    X, Y, Z = pyTMD.spatial.to_cartesian(
        lon,
        lat,
        a_axis=kwargs["a_axis"],
        flat=kwargs["flat"],
    )
    XYZ = np.c_[X, Y, Z]
    # output input functions
    M = []
    # lagged time (convert to Modified Julian Days)
    MJD = t - lag * lag_interval + _mjd_tide
    # solar and lunar ephemerides
    SX, SY, SZ = pyTMD.astro.solar_ecef(MJD, **kwargs)
    LX, LY, LZ = pyTMD.astro.lunar_ecef(MJD, **kwargs)
    SXYZ = np.c_[SX, SY, SZ]
    LXYZ = np.c_[LX, LY, LZ]
    # for each spherical harmonic degree
    for l in range(lmax + 1):
        # for each spherical harmonic degree and order
        for m in range(0, l + 1):
            # gravitational effects
            if l == 2:
                # summation of gravitational effects for the sun and moon
                cS = _gravitational(
                    XYZ,
                    SXYZ,
                    l=l,
                    m=m,
                    mass_ratio=kwargs["mass_ratio_solar"],
                    a_axis=kwargs["a_axis"],
                )
                cL = _gravitational(
                    XYZ,
                    LXYZ,
                    l=l,
                    m=m,
                    mass_ratio=kwargs["mass_ratio_lunar"],
                    a_axis=kwargs["a_axis"],
                )
                M.append(cS.real + cL.real)
                if m > 0:
                    M.append(cS.imag + cL.imag)
            elif (l > 2) and (lag == 0):
                # gravitational effects for the moon (lag 0 only)
                cL = _gravitational(
                    XYZ,
                    LXYZ,
                    l=l,
                    m=m,
                    mass_ratio=kwargs["mass_ratio_lunar"],
                    a_axis=kwargs["a_axis"],
                )
                M.append(cL.real)
                if m > 0:
                    M.append(cL.imag)
            # radiational effects
            if radiational and (l <= 2):
                cR = _radiational(
                    XYZ,
                    SXYZ,
                    l=l,
                    m=m,
                    a_axis=kwargs["a_axis"],
                    distance=kwargs["AU"],
                )
                M.append(cR.real)
                if m > 0:
                    M.append(cR.imag)
    # return the design matrix functions
    return M


def _legendre_addition(
    l: int,
    m: int,
    th1: np.ndarray,
    lmda1: np.ndarray,
    th2: np.ndarray,
    lmda2: np.ndarray,
):
    r"""
    Decompose the Legendre polynomial of the angle between two points
    into orders using the addition theorem of spherical harmonics

    Parameters
    ----------
    l: int
        Spherical harmonic degree
    m: int
        Spherical harmonic order
    th1: np.ndarray
        Colatitude of the observation point (radians)
    lmda1: np.ndarray
        Longitude of the observation point (radians)
    th2: np.ndarray
        Colatitude of the body (radians)
    lmda2: np.ndarray
        Longitude of the body (radians)

    Returns
    -------
    P: np.ndarray
        complex contributions for each order (time, order)
    """
    Ylms1, _ = pyTMD.math.sph_harm(l, th1, lmda1, m=m)
    Ylms2, _ = pyTMD.math.sph_harm(l, th2, lmda2, m=m)
    epsilon = 2.0 - pyTMD.math._kronecker_delta(m, 0)
    P = 4.0 * np.pi / (2.0 * l + 1.0) * epsilon * Ylms1 * np.conj(Ylms2)
    return P


def _gravitational(
    XYZ: np.ndarray,
    LSXYZ: np.ndarray,
    l: int,
    m: int,
    **kwargs,
):
    r"""
    Calculate the gravitational input functions for the response method
    for a given degree :cite:p:`Munk:1966go`

    The inputs are the tide generating potential of the sun or moon
    converted to equilibrium elevations

    .. math::
        \frac{V_l}{g} = a\frac{M_b}{M_e}\left(\frac{a}{R_b}\right)^{l+1}
            \left(\frac{r}{a}\right)^{l} P_l(\cos\psi)

    Parameters
    ----------
    XYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the observation points (meters)
    LSXYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the sun or moon (meters)
    l: int
        spherical harmonic degree
    m: int
        spherical harmonic order
    a_axis: float, default 6378137.0
        Semi-major axis of the Earth (meters)
    mass_ratio: float, default 332946.0482
        mass ratio between the planetary body and the Earth

    Returns
    -------
    potential: np.ndarray
        gravitational inputs (meters)
    """
    # default keyword arguments
    a_axis = kwargs.get("a_axis", 6378137.0)
    mass_ratio = kwargs.get("mass_ratio", 332946.0482)
    # radius of the earth at coordinates
    radius = pyTMD.math.radius(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2])
    # colatitude and longitude of coordinates
    th = np.arccos(XYZ[:, 2] / radius)
    lmda = np.arctan2(XYZ[:, 1], XYZ[:, 0])
    # distance between the Earth and the Sun/Moon
    lunisolar_radius = pyTMD.math.radius(LSXYZ[:, 0], LSXYZ[:, 1], LSXYZ[:, 2])
    # Solar/Lunar declinations
    lunisolar_theta = np.arccos(LSXYZ[:, 2] / lunisolar_radius)
    # Solar/Lunar hour angles
    lunisolar_lambda = np.arctan2(LSXYZ[:, 1], LSXYZ[:, 0])
    # scaling factors for the potential (meters)
    K = (
        a_axis
        * mass_ratio
        * np.power(a_axis / lunisolar_radius, l + 1)
        * np.power(radius / a_axis, l)
    )
    # decompose the potential
    P = _legendre_addition(l, m, th, lmda, lunisolar_theta, lunisolar_lambda)
    # return the gravitational inputs
    return K * P


def _radiational(
    XYZ: np.ndarray,
    SXYZ: np.ndarray,
    l: int,
    m: int,
    **kwargs,
):
    r"""
    Calculate the radiational input functions for the response method
    for a given degree :cite:p:`Munk:1966go`

    The radiation function is the (normalized) solar flux on a
    horizontal surface, which is zero during the night

    .. math::
        \frac{R}{S} = \left(\frac{\bar{R}}{R_s}\right)^2
            \sum_{l} k_l(\xi) P_l(\cos\alpha)

    Parameters
    ----------
    XYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the observation points (meters)
    SXYZ: np.ndarray
        Earth-centered Earth-fixed coordinates of the sun (meters)
    l: int
        spherical harmonic degree (0 to 2)
    a_axis: float, default 6378137.0
        Semi-major axis of the Earth (meters)
    distance: float, default 1.495978707e11
        Mean distance between the Earth and the sun (meters)

    Returns
    -------
    radiation: np.ndarray
        radiational inputs
    """
    # default keyword arguments
    a_axis = kwargs.get("a_axis", 6378137.0)
    AU = kwargs.get("distance", 1.495978707e11)
    # radius of the earth at coordinates
    radius = pyTMD.math.radius(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2])
    # colatitude and longitude of coordinates
    th = np.arccos(XYZ[:, 2] / radius)
    lmda = np.arctan2(XYZ[:, 1], XYZ[:, 0])
    # distance between the Earth and the Sun
    solar_radius = pyTMD.math.radius(SXYZ[:, 0], SXYZ[:, 1], SXYZ[:, 2])
    # solar declination
    solar_theta = np.arccos(SXYZ[:, 2] / solar_radius)
    # solar hour angle
    solar_lambda = np.arctan2(SXYZ[:, 1], SXYZ[:, 0])
    # solar parallax (~1/23455)
    xi = a_axis / solar_radius
    # kappa value (Munk and Cartwright, 1966)
    kappa = _kappa(l, xi)
    # decompose the radiation potential
    P = _legendre_addition(l, m, th, lmda, solar_theta, solar_lambda)
    # return the radiation inputs
    return kappa * P * np.power(AU / solar_radius, 2)


def _kappa(
    l: int,
    xi: float | np.ndarray,
):
    """
    Coefficients of the Legendre expansion of the radiation function
    as a function of degree :cite:p:`Munk:1966go`

    Parameters
    ----------
    l: int
        spherical harmonic degree (0 to 2)
    xi: float or np.ndarray
        solar parallax

    Returns
    -------
    kappa: float or np.ndarray
        kappa value for the given degree
    """
    if l == 0:
        return 1.0 / 4.0 + xi / 6.0
    elif l == 1:
        return 1.0 / 2.0 + 3.0 * xi / 8.0
    elif l == 2:
        return 5.0 / 16.0 + xi / 3.0
    else:
        raise ValueError(f"Unsupported degree {l} for radiational inputs")
