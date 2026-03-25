#!/usr/bin/env python
"""
ocean_load_tide.py
Written by Tyler Sutterley (03/2026)
Prediction routines for ocean, load and equilibrium  tides

REFERENCES:
    G. D. Egbert and S. Erofeeva, "Efficient Inverse Modeling of Barotropic
        Ocean Tides", Journal of Atmospheric and Oceanic Technology, (2002).
    R. Ray and S. Erofeeva, "Long-period tidal variations in the length of day",
        Journal of Geophysical Research: Solid Earth, 119, (2014).

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
    constituents.py: calculates constituent parameters and nodal arguments
    interpolate.py: interpolation routines for spatial data

UPDATE HISTORY:
    Updated 03/2026: simplify structure by spliting up IERS corrections
        and adding wrapper functions where appropriate
        set the maximum degree and order for the HW1995 catalog to 6
        clean up the ephemerides method of calculating solid earth tides
        clean up all of the ephemerides corrections for solid earth tides
        calculate tide-generating forces following Tamura (1982 and 1987)
        split up prediction functions into separate files
    Updated 02/2026: added attributes for constituents to output DataArrays
        do not infer minor constituents with frequencies equal to any major
        revert (again) load pole tides to a newer IERS convention definition
        but allow for both sign definitions based on the convention variable
        add function to calculate variations in Earth orientation parameters
    Updated 12/2025: added tidal LOD calculation from Ray and Erofeeva (2014)
    Updated 11/2025: update all prediction functions to use xarray Datasets
    Updated 09/2025: make permanent tide amplitude an input parameter
        can choose different tide potential catalogs for body tides
        generalize the calculation of body tides for degrees 3+
    Updated 08/2025: add simplified solid earth tide prediction function
        add correction of anelastic effects for long-period body tides
        use sign convention from IERS for complex body tide Love numbers
        include mantle anelastic effects when inferring long-period tides
        allow definition of nominal Love numbers for degree-2 constituents
        added option to include mantle anelastic effects for LPET predict
        switch time decimal in pole tides to nominal years of 365.25 days
        convert angles with numpy radians and degrees functions
        convert arcseconds to radians with asec2rad function in math.py
        return numpy arrays if cannot infer minor constituents
        use a vectorized linear interpolator for inferring from major tides
    Updated 07/2025: revert free-to-mean conversion to April 2023 version
        revert load pole tide to IERS 1996 convention definitions
        mask mean pole values prior to valid epoch of convention
    Updated 05/2025: pass keyword arguments to nodal corrections functions
    Updated 03/2025: changed argument for method calculating mean longitudes
    Updated 02/2025: verify dimensions of harmonic constants
    Updated 11/2024: use Love numbers for long-period tides when inferring
        move body tide Love/Shida numbers to arguments module
    Updated 10/2024: use PREM as the default Earth model for Love numbers
        more descriptive error message if cannot infer minor constituents
        updated calculation of long-period equilibrium tides
        added option to use Munk-Cartwright admittance interpolation for minor
    Updated 09/2024: verify order of minor constituents to infer
        fix to use case insensitive assertions of string argument values
        split infer minor function into short and long period calculations
        add two new functions to infer semi-diurnal and diurnal tides separately
    Updated 08/2024: minor nodal angle corrections in radians to match arguments
        include inference of eps2 and eta2 when predicting from GOT models
        add keyword argument to allow inferring specific minor constituents
        use nodal arguments for all non-OTIS model type cases
        add load pole tide function that exports in cartesian coordinates
        add ocean pole tide function that exports in cartesian coordinates
    Updated 07/2024: use normalize_angle from pyTMD astro module
        make number of days to convert tide time to MJD a variable
    Updated 02/2024: changed class name for ellipsoid parameters to datum
    Updated 01/2024: moved minor arguments calculation into new function
        moved constituent parameters function from predict to arguments
    Updated 12/2023: phase_angles function renamed to doodson_arguments
    Updated 09/2023: moved constituent parameters function within this module
    Updated 08/2023: changed ESR netCDF4 format to TMD3 format
    Updated 04/2023: using renamed astro mean_longitudes function
        using renamed arguments function for nodal corrections
        adding prediction routine for solid earth tides
        output solid earth tide corrections as combined XYZ components
    Updated 03/2023: add basic variable typing to function inputs
    Updated 12/2022: merged prediction functions into a single module
    Updated 05/2022: added ESR netCDF4 formats to list of model types
    Updated 04/2022: updated docstrings to numpy documentation format
    Updated 02/2021: replaced numpy bool to prevent deprecation warning
    Updated 09/2020: append output mask over each constituent
    Updated 08/2020: change time variable names to not overwrite functions
    Updated 07/2020: added function docstrings
    Updated 11/2019: output as numpy masked arrays instead of nan-filled arrays
    Updated 09/2019: added netcdf option to CORRECTIONS option
    Updated 08/2018: added correction option ATLAS for localized OTIS solutions
    Updated 07/2018: added option to use GSFC GOT nodal corrections
    Updated 09/2017: Rewritten in Python
"""

from __future__ import annotations

import logging
import numpy as np
import xarray as xr
import pyTMD.astro
import pyTMD.constituents
import pyTMD.math
import pyTMD.interpolate

__all__ = [
    "time_series",
    "infer_minor",
    "_infer_short_period",
    "_infer_semi_diurnal",
    "_infer_diurnal",
    "_infer_long_period",
    "equilibrium_tide",
]

# number of days between the Julian day epoch and MJD
_jd_mjd = 2400000.5
# number of days between MJD and the tide epoch (1992-01-01T00:00:00)
_mjd_tide = 48622.0
# number of days between MJD and the J2000 epoch
_mjd_j2000 = 51544.5
# Julian century
_century = 36525.0


def time_series(t: float | np.ndarray, ds: xr.Dataset, **kwargs):
    """
    Predict tides from ``Dataset`` at times

    Parameters
    ----------
    t: float or np.ndarray
        Days relative to 1992-01-01T00:00:00
    ds: xarray.Dataset
        Dataset containing tidal harmonic constants
    kwargs: keyword arguments
        Additional keyword arguments

    Returns
    -------
    darr: xarray.DataArray
        Predicted tidal time series
    """
    # set default keyword arguments
    kwargs.setdefault("corrections", "OTIS")
    # convert time to Modified Julian Days (MJD)
    MJD = t + _mjd_tide
    # list of constituents
    constituents = ds.tmd.constituents
    # load the nodal corrections
    pu, pf, G = pyTMD.constituents.arguments(MJD, constituents, **kwargs)
    # calculate constituent phase angles
    if kwargs["corrections"] in ("OTIS", "ATLAS", "TMD3"):
        theta = np.zeros_like(pu)
        for i, c in enumerate(constituents):
            # load parameters for constituent
            amp, ph, omega, alpha, species = (
                pyTMD.constituents._constituent_parameters(c)
            )
            # phase angle from frequency and phase-0
            theta[:, i] = omega * t * 86400.0 + ph + pu[:, i]
    else:
        # phase angle from arguments
        theta = np.radians(G) + pu
    # dataset of arguments
    arguments = xr.Dataset(
        data_vars=dict(
            u=(["time", "constituent"], pu),
            f=(["time", "constituent"], pf),
            G=(["time", "constituent"], G),
            theta=(["time", "constituent"], np.exp(1j * theta)),
        ),
        coords=dict(time=np.atleast_1d(MJD), constituent=constituents),
    )
    # convert Dataset to DataArray of complex tidal harmonics
    darr = ds.tmd.to_dataarray(constituents=constituents)
    # sum over tidal constituents
    tpred = (
        darr.real * arguments.f * arguments.theta.real
        - darr.imag * arguments.f * arguments.theta.imag
    ).sum(dim="constituent", skipna=False)
    # check if chunks are present
    if tpred.chunks is not None:
        tpred = tpred.chunk(-1).compute()
    # copy units attribute
    tpred.attrs["units"] = ds[constituents[0]].attrs.get("units", None)
    tpred.attrs["constituents"] = constituents
    # return the predicted tides
    return tpred


# PURPOSE: infer the minor corrections from the major constituents
def infer_minor(t: float | np.ndarray, ds: xr.Dataset, **kwargs):
    """
    Infer the tidal values for minor constituents using their
    relation with major constituents
    :cite:p:`Doodson:1941td,Schureman:1958ty,Foreman:1989dt,Egbert:2002ge`

    Parameters
    ----------
    t: float or np.ndarray
        Days relative to 1992-01-01T00:00:00
    ds: xarray.Dataset
        Dataset containing major tidal harmonic constants
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    corrections: str, default 'OTIS'
        Use nodal corrections from OTIS/ATLAS or GOT/FES models
    minor: list or None, default None
        Tidal constituent IDs of minor constituents for inference
    infer_long_period, bool, default True
        Try to infer long period tides from constituents
    raise_exception: bool, default False
        Raise a ``ValueError`` if major constituents are not found

    Returns
    -------
    tinfer: xr.DataArray
        Tidal time series for minor constituents
    """
    # set default keyword arguments
    kwargs.setdefault("deltat", 0.0)
    kwargs.setdefault("corrections", "OTIS")
    kwargs.setdefault("infer_long_period", True)
    kwargs.setdefault("raise_exception", False)
    # list of minor constituents
    kwargs.setdefault("minor", None)
    # infer the minor tidal constituents
    tinfer = 0.0
    constituents = []
    species = []
    # infer short-period tides for minor constituents
    if kwargs["corrections"] in ("GOT",):
        species.extend(["semi_diurnal", "diurnal"])
    else:
        species.append("short_period")
    # infer long-period tides for minor constituents
    if kwargs["infer_long_period"]:
        species.append("long_period")
    # infer minor constituents for each species
    for s in species:
        result = _infer[s](t, ds, **kwargs)
        tinfer += result
        if hasattr(result, "constituents"):
            constituents.extend(result.constituents)
    # check if chunks are present
    if hasattr(tinfer, "chunks") and tinfer.chunks is not None:
        tinfer = tinfer.chunk(-1).compute()
    # update attributes for inferred constituents
    if hasattr(tinfer, "constituents"):
        tinfer.attrs["constituents"] = constituents
    # return the inferred values
    return tinfer


# PURPOSE: infer short-period minor constituents
def _infer_short_period(t: float | np.ndarray, ds: xr.Dataset, **kwargs):
    """
    Infer the tidal values for short-period minor constituents
    using their relation with major constituents
    :cite:p:`Egbert:2002ge,Ray:1999vm`

    Parameters
    ----------
    t: float or np.ndarray
        Days relative to 1992-01-01T00:00:00
    ds: xarray.Dataset
        Dataset containing major tidal harmonic constants
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    corrections: str, default 'OTIS'
        Use nodal corrections from OTIS/ATLAS or GOT/FES models
    minor: list or None, default None
        Tidal constituent IDs of minor constituents for inference
    raise_exception: bool, default False
        Raise a ``ValueError`` if major constituents are not found

    Returns
    -------
    tinfer: xr.DataArray
        Tidal time series for minor constituents
    """
    # set default keyword arguments
    kwargs.setdefault("deltat", 0.0)
    kwargs.setdefault("corrections", "OTIS")
    kwargs.setdefault("raise_exception", False)
    # list of minor constituents
    kwargs.setdefault("minor", None)
    # major constituents used for inferring minor tides
    cindex = ["q1", "o1", "p1", "k1", "n2", "m2", "s2", "k2", "2n2"]
    # check that major constituents are in the dataset for inference
    nz = sum([(c in ds.tmd.constituents) for c in cindex])
    # raise exception or log error
    msg = "Not enough constituents to infer short-period tides"
    if (nz < 6) and kwargs["raise_exception"]:
        raise Exception(msg)
    elif nz < 6:
        logging.debug(msg)
        return 0.0

    # complete list of minor constituents
    minor_constituents = [
        "2q1",
        "sigma1",
        "rho1",
        "m1b",
        "m1",
        "chi1",
        "pi1",
        "phi1",
        "theta1",
        "j1",
        "oo1",
        "2n2",
        "mu2",
        "nu2",
        "lambda2",
        "l2",
        "l2b",
        "t2",
        "eps2",
        "eta2",
    ]
    # possibly reduced list of minor constituents
    minor = kwargs["minor"] or minor_constituents
    # only add minor constituents that are not on the list of major values
    constituents = [
        m
        for i, m in enumerate(minor_constituents)
        if (m not in ds.tmd.constituents) and (m in minor)
    ]
    # if there are no constituents to infer
    msg = "No short-period tidal constituents to infer"
    if not any(constituents):
        logging.debug(msg)
        return 0.0

    # relationship between major and minor constituent amplitude and phase
    dmin = xr.Dataset()
    dmin["2q1"] = 0.263 * ds["q1"] - 0.0252 * ds["o1"]
    dmin["sigma1"] = 0.297 * ds["q1"] - 0.0264 * ds["o1"]
    dmin["rho1"] = 0.164 * ds["q1"] + 0.0048 * ds["o1"]
    dmin["m1b"] = 0.0140 * ds["o1"] + 0.0101 * ds["k1"]
    dmin["m1"] = 0.0389 * ds["o1"] + 0.0282 * ds["k1"]
    dmin["chi1"] = 0.0064 * ds["o1"] + 0.0060 * ds["k1"]
    dmin["pi1"] = 0.0030 * ds["o1"] + 0.0171 * ds["k1"]
    dmin["phi1"] = -0.0015 * ds["o1"] + 0.0152 * ds["k1"]
    dmin["theta1"] = -0.0065 * ds["o1"] + 0.0155 * ds["k1"]
    dmin["j1"] = -0.0389 * ds["o1"] + 0.0836 * ds["k1"]
    dmin["oo1"] = -0.0431 * ds["o1"] + 0.0613 * ds["k1"]
    dmin["2n2"] = 0.264 * ds["n2"] - 0.0253 * ds["m2"]
    dmin["mu2"] = 0.298 * ds["n2"] - 0.0264 * ds["m2"]
    dmin["nu2"] = 0.165 * ds["n2"] + 0.00487 * ds["m2"]
    dmin["lambda2"] = 0.0040 * ds["m2"] + 0.0074 * ds["s2"]
    dmin["l2"] = 0.0131 * ds["m2"] + 0.0326 * ds["s2"]
    dmin["l2b"] = 0.0033 * ds["m2"] + 0.0082 * ds["s2"]
    dmin["t2"] = 0.0585 * ds["s2"]
    dmin["eps2"] = xr.zeros_like(ds["m2"])
    dmin["eta2"] = xr.zeros_like(ds["m2"])
    # additional coefficients for FES models
    if kwargs["corrections"] in ("FES",):
        # spline coefficients for admittances
        mu2 = [0.069439968323, 0.351535557706, -0.046278307672]
        nu2 = [-0.006104695053, 0.156878802427, 0.006755704028]
        l2 = [0.077137765667, -0.051653455134, 0.027869916824]
        t2 = [0.180480173707, -0.020101177502, 0.008331518844]
        lda2 = [0.016503557465, -0.013307812292, 0.007753383202]
        dmin["mu2"] = mu2[0] * ds["k2"] + mu2[1] * ds["n2"] + mu2[2] * ds["m2"]
        dmin["nu2"] = nu2[0] * ds["k2"] + nu2[1] * ds["n2"] + nu2[2] * ds["m2"]
        dmin["lambda2"] = (
            lda2[0] * ds["k2"] + lda2[1] * ds["n2"] + lda2[2] * ds["m2"]
        )
        dmin["l2b"] = l2[0] * ds["k2"] + l2[1] * ds["n2"] + l2[2] * ds["m2"]
        dmin["t2"] = t2[0] * ds["k2"] + t2[1] * ds["n2"] + t2[2] * ds["m2"]
        dmin["eps2"] = 0.53285 * ds["2n2"] - 0.03304 * ds["n2"]
        dmin["eta2"] = -0.0034925 * ds["m2"] + 0.0831707 * ds["k2"]

    # convert time to Modified Julian Days (MJD)
    MJD = t + _mjd_tide
    # load the nodal corrections for minor constituents
    pu, pf, G = pyTMD.constituents.minor_arguments(
        MJD, deltat=kwargs["deltat"], corrections=kwargs["corrections"]
    )
    # phase angle from arguments
    theta = np.radians(G) + pu
    # dataset of minor arguments
    arguments = xr.Dataset(
        data_vars=dict(
            u=(["time", "constituent"], pu),
            f=(["time", "constituent"], pf),
            G=(["time", "constituent"], G),
            theta=(["time", "constituent"], np.exp(1j * theta)),
        ),
        coords=dict(time=np.atleast_1d(MJD), constituent=minor_constituents),
    )
    # convert Dataset to DataArray of complex tidal harmonics
    # (reduce list to only those constituents to infer)
    darr = dmin.tmd.to_dataarray(constituents=constituents)
    # select argument for constituents
    arg = arguments.sel(constituent=constituents)
    # sum over tidal constituents
    tinfer = (
        darr.real * arg.f * arg.theta.real - darr.imag * arg.f * arg.theta.imag
    ).sum(dim="constituent", skipna=False)
    # copy units attribute
    tinfer.attrs["units"] = ds["q1"].attrs.get("units", None)
    tinfer.attrs["constituents"] = constituents
    # return the inferred values
    return tinfer


# PURPOSE: infer semi-diurnal minor constituents
def _infer_semi_diurnal(t: float | np.ndarray, ds: xr.Dataset, **kwargs):
    """
    Infer the tidal values for semi-diurnal minor constituents
    using their relation with major constituents
    :cite:p:`Munk:1966go,Ray:1999vm,Cartwright:1971iz`

    Parameters
    ----------
    t: float or np.ndarray
        Days relative to 1992-01-01T00:00:00
    ds: xarray.Dataset
        Dataset containing major tidal harmonic constants
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    minor: list or None, default None
        Tidal constituent IDs of minor constituents for inference
    method: str, default 'linear'
        Method for interpolating between major constituents

            * ``'linear'``: linear interpolation
            * ``'admittance'``: Munk-Cartwright interpolation
    raise_exception: bool, default False
        Raise a ``ValueError`` if major constituents are not found

    Returns
    -------
    tinfer: xr.DataArray
        Tidal time series for minor constituents
    """
    # set default keyword arguments
    kwargs.setdefault("deltat", 0.0)
    kwargs.setdefault("corrections", "GOT")
    kwargs.setdefault("method", "linear")
    kwargs.setdefault("raise_exception", False)
    # list of minor constituents
    kwargs.setdefault("minor", None)
    # validate interpolation method
    assert kwargs["method"].lower() in ("linear", "admittance")
    # major constituents used for inferring semi-diurnal minor tides
    # pivot waves listed in Table 6.7 of the 2010 IERS Conventions
    cindex = ["n2", "m2", "s2"]
    # check that major constituents are in the dataset for inference
    nz = sum([(c in ds.tmd.constituents) for c in cindex])
    # raise exception or log error
    msg = "Not enough constituents to infer semi-diurnal tides"
    if (nz < 3) and kwargs["raise_exception"]:
        raise Exception(msg)
    elif nz < 3:
        logging.debug(msg)
        return 0.0

    # angular frequencies for major constituents
    omajor = pyTMD.constituents.frequency(cindex, **kwargs)
    # Cartwright and Edden potential amplitudes for major constituents
    amajor = np.zeros((3))
    amajor[0] = 0.121006  # n2
    amajor[1] = 0.631931  # m2
    amajor[2] = 0.294019  # s2
    # "normalize" tide values
    dnorm = xr.Dataset()
    for i, c in enumerate(cindex):
        dnorm[c] = ds[c] / amajor[i]
    # major constituents as a dataarray
    z = dnorm.tmd.to_dataarray()

    # complete list of minor constituents
    minor_constituents = [
        "eps2",
        "2n2",
        "mu2",
        "nu2",
        "gamma2",
        "alpha2",
        "beta2",
        "delta2",
        "lambda2",
        "l2",
        "t2",
        "r2",
        "k2",
        "eta2",
    ]
    # possibly reduced list of minor constituents
    minor = kwargs["minor"] or minor_constituents
    # angular frequencies for inferred constituents
    omega = pyTMD.constituents.frequency(minor_constituents, **kwargs)
    # only add minor constituents that are not on the list of major values
    # and with frequencies not equal to any major constituent
    constituents = [
        m
        for i, m in enumerate(minor_constituents)
        if (m not in ds.tmd.constituents)
        and (m in minor)
        and (np.all(omega[i] != omajor))
    ]
    # if there are no constituents to infer
    msg = "No semi-diurnal tidal constituents to infer"
    if not any(constituents):
        logging.debug(msg)
        return 0.0

    # Cartwright and Edden potential amplitudes for inferred constituents
    amin = np.zeros((14))
    amin[0] = 0.004669  # eps2
    amin[1] = 0.016011  # 2n2
    amin[2] = 0.019316  # mu2
    amin[3] = 0.022983  # nu2
    amin[4] = 0.001902  # gamma2
    amin[5] = 0.002178  # alpha2
    amin[6] = 0.001921  # beta2
    amin[7] = 0.000714  # delta2
    amin[8] = 0.004662  # lambda2
    amin[9] = 0.017862  # l2
    amin[10] = 0.017180  # t2
    amin[11] = 0.002463  # r2
    amin[12] = 0.079924  # k2
    amin[13] = 0.004467  # eta

    # convert time to Modified Julian Days (MJD)
    MJD = t + _mjd_tide
    # load the nodal corrections for minor constituents
    pu, pf, G = pyTMD.constituents.arguments(
        MJD,
        minor_constituents,
        deltat=kwargs["deltat"],
        corrections=kwargs["corrections"],
    )
    # phase angle from arguments
    theta = np.radians(G) + pu
    # dataset of minor arguments
    coords = dict(time=np.atleast_1d(MJD), constituent=minor_constituents)
    arguments = xr.Dataset(
        data_vars=dict(
            u=(["time", "constituent"], pu),
            f=(["time", "constituent"], pf),
            G=(["time", "constituent"], G),
            theta=(["time", "constituent"], np.exp(1j * theta)),
            amplitude=(["constituent"], amin),
            omega=(["constituent"], omega),
        ),
        coords=coords,
    )

    # reduce to selected constituents
    arg = arguments.sel(constituent=constituents)
    # interpolate from major constituents
    if kwargs["method"].lower() == "linear":
        # linearly interpolate using constituent frequencies
        zmin = pyTMD.interpolate.interp1d(arg.omega.values, omajor, z)
        coords = z.coords.assign(dict(constituent=arg.constituent))
        zmin = xr.DataArray(zmin, dims=z.dims, coords=coords)
    elif kwargs["method"].lower() == "admittance":
        # admittance interpolation using Munk-Cartwright approach
        # coefficients for Munk-Cartwright admittance interpolation
        Ainv = xr.DataArray(
            [
                [3.3133, -4.2538, 1.9405],
                [-3.3133, 4.2538, -0.9405],
                [1.5018, -3.2579, 1.7561],
            ],
            coords=[np.arange(3), cindex],
            dims=["coefficient", "constituent"],
        )
        coef = Ainv.dot(z)
        # convert frequency to radians per 48 hours
        # following Munk and Cartwright (1966)
        f = np.exp(2.0 * 86400.0 * arg.omega * 1j)
        # calculate interpolated values for constituent
        zmin = coef[0] + coef[1] * f.real + coef[2] * f.imag
    # rescale tide values
    darr = arg.amplitude * zmin
    # sum over tidal constituents
    tinfer = (
        darr.real * arg.f * arg.theta.real - darr.imag * arg.f * arg.theta.imag
    ).sum(dim="constituent", skipna=False)
    # copy units attribute
    tinfer.attrs["units"] = ds["n2"].attrs.get("units", None)
    tinfer.attrs["constituents"] = constituents
    # return the inferred values
    return tinfer


# PURPOSE: infer diurnal minor constituents
def _infer_diurnal(t: float | np.ndarray, ds: xr.Dataset, **kwargs):
    """
    Infer the tidal values for diurnal minor constituents
    using their relation with major constituents taking into
    account resonance due to free core nutation
    :cite:p:`Munk:1966go,Ray:2017jx,Wahr:1981if,Cartwright:1973em`

    Parameters
    ----------
    t: float or np.ndarray
        Days relative to 1992-01-01T00:00:00
    ds: xarray.Dataset
        Dataset containing major tidal harmonic constants
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    minor: list or None, default None
        Tidal constituent IDs of minor constituents for inference
    method: str, default 'linear'
        Method for interpolating between major constituents

            * ``'linear'``: linear interpolation
            * ``'admittance'``: Munk-Cartwright interpolation
    raise_exception: bool, default False
        Raise a ``ValueError`` if major constituents are not found

    Returns
    -------
    tinfer: xr.DataArray
        Tidal time series for minor constituents
    """
    # set default keyword arguments
    kwargs.setdefault("deltat", 0.0)
    kwargs.setdefault("corrections", "GOT")
    kwargs.setdefault("method", "linear")
    kwargs.setdefault("raise_exception", False)
    # list of minor constituents
    kwargs.setdefault("minor", None)
    # validate interpolation method
    assert kwargs["method"].lower() in ("linear", "admittance")
    # major constituents used for inferring diurnal minor tides
    # pivot waves listed in Table 6.7 of the 2010 IERS Conventions
    cindex = ["q1", "o1", "k1"]
    # check that major constituents are in the dataset for inference
    nz = sum([(c in ds.tmd.constituents) for c in cindex])
    # raise exception or log error
    msg = "Not enough constituents to infer diurnal tides"
    if (nz < 3) and kwargs["raise_exception"]:
        raise Exception(msg)
    elif nz < 3:
        logging.debug(msg)
        return 0.0

    # angular frequencies for major constituents
    omajor = pyTMD.constituents.frequency(cindex, **kwargs)
    # Cartwright and Edden potential amplitudes for major constituents
    amajor = np.zeros((3))
    amajor[0] = 0.050184  # q1
    amajor[1] = 0.262163  # o1
    amajor[2] = 0.368731  # k1
    # "normalize" tide values
    dnorm = xr.Dataset()
    for i, c in enumerate(cindex):
        # Love numbers of degree 2 for constituent
        h2, k2, l2 = pyTMD.constituents._love_numbers(omajor[i])
        # tilt factor: response with respect to the solid earth
        gamma_2 = 1.0 + k2 - h2
        dnorm[c] = ds[c] / (amajor[i] * gamma_2)
    # major constituents as a dataarray
    z = dnorm.tmd.to_dataarray()

    # raise exception or log error
    msg = "Not enough constituents to infer diurnal tides"
    if (nz < 3) and kwargs["raise_exception"]:
        raise Exception(msg)
    elif nz < 3:
        logging.debug(msg)
        return 0.0

    # complete list of minor constituents
    minor_constituents = [
        "2q1",
        "sigma1",
        "rho1",
        "tau1",
        "beta1",
        "m1a",
        "m1b",
        "chi1",
        "pi1",
        "p1",
        "psi1",
        "phi1",
        "theta1",
        "j1",
        "so1",
        "oo1",
        "ups1",
    ]
    # possibly reduced list of minor constituents
    minor = kwargs["minor"] or minor_constituents
    # angular frequencies for inferred constituents
    omega = pyTMD.constituents.frequency(minor_constituents, **kwargs)
    # only add minor constituents that are not on the list of major values
    # and with frequencies not equal to any major constituent
    constituents = [
        m
        for i, m in enumerate(minor_constituents)
        if (m not in ds.tmd.constituents)
        and (m in minor)
        and (np.all(omega[i] != omajor))
    ]
    # if there are no constituents to infer
    msg = "No diurnal tidal constituents to infer"
    if not any(constituents):
        logging.debug(msg)
        return 0.0

    # Cartwright and Edden potential amplitudes for inferred constituents
    amin = np.zeros((17))
    amin[0] = 0.006638  # 2q1
    amin[1] = 0.008023  # sigma1
    amin[2] = 0.009540  # rho1
    amin[3] = 0.003430  # tau1
    amin[4] = 0.001941  # beta1
    amin[5] = 0.020604  # m1a
    amin[6] = 0.007420  # m1b
    amin[7] = 0.003925  # chi1
    amin[8] = 0.007125  # pi1
    amin[9] = 0.122008  # p1
    amin[10] = 0.002929  # psi1
    amin[11] = 0.005247  # phi1
    amin[12] = 0.003966  # theta1
    amin[13] = 0.020618  # j1
    amin[14] = 0.003417  # so1
    amin[15] = 0.011293  # oo1
    amin[16] = 0.002157  # ups1

    # convert time to Modified Julian Days (MJD)
    MJD = t + _mjd_tide
    # load the nodal corrections for minor constituents
    pu, pf, G = pyTMD.constituents.arguments(
        MJD,
        minor_constituents,
        deltat=kwargs["deltat"],
        corrections=kwargs["corrections"],
    )
    # phase angle from arguments
    theta = np.radians(G) + pu
    # compute tilt factors for minor constituents
    nc = len(minor_constituents)
    gamma_2 = np.zeros((nc))
    for i, c in enumerate(minor_constituents):
        # Love numbers of degree 2 for constituent
        h2, k2, l2 = pyTMD.constituents._love_numbers(omega[i])
        # tilt factor: response with respect to the solid earth
        gamma_2[i] = 1.0 + k2 - h2

    # dataset of minor arguments
    coords = dict(time=np.atleast_1d(MJD), constituent=minor_constituents)
    arguments = xr.Dataset(
        data_vars=dict(
            u=(["time", "constituent"], pu),
            f=(["time", "constituent"], pf),
            G=(["time", "constituent"], G),
            theta=(["time", "constituent"], np.exp(1j * theta)),
            amplitude=(["constituent"], amin),
            omega=(["constituent"], omega),
            gamma_2=(["constituent"], gamma_2),
        ),
        coords=coords,
    )

    # reduce to selected constituents
    arg = arguments.sel(constituent=constituents)
    # interpolate from major constituents
    if kwargs["method"].lower() == "linear":
        # linearly interpolate using constituent frequencies
        zmin = pyTMD.interpolate.interp1d(arg.omega.values, omajor, z)
        coords = z.coords.assign(dict(constituent=arg.constituent))
        zmin = xr.DataArray(zmin, dims=z.dims, coords=coords)
    elif kwargs["method"].lower() == "admittance":
        # admittance interpolation using Munk-Cartwright approach
        # coefficients for Munk-Cartwright admittance interpolation
        Ainv = xr.DataArray(
            [
                [3.1214, -3.8494, 1.728],
                [-3.1727, 3.9559, -0.7832],
                [1.438, -3.0297, 1.5917],
            ],
            coords=[np.arange(3), cindex],
            dims=["coefficient", "constituent"],
        )
        coef = Ainv.dot(z)
        # convert frequency to radians per 48 hours
        # following Munk and Cartwright (1966)
        f = np.exp(2.0 * 86400.0 * arg.omega * 1j)
        # calculate interpolated values for constituent
        zmin = coef[0] + coef[1] * f.real + coef[2] * f.imag
    # rescale tide values
    darr = arg.amplitude * arg.gamma_2 * zmin
    # sum over tidal constituents
    tinfer = (
        darr.real * arg.f * arg.theta.real - darr.imag * arg.f * arg.theta.imag
    ).sum(dim="constituent", skipna=False)
    # copy units attribute
    tinfer.attrs["units"] = ds["q1"].attrs.get("units", None)
    tinfer.attrs["constituents"] = constituents
    # return the inferred values
    return tinfer


# PURPOSE: infer long-period minor constituents
def _infer_long_period(t: float | np.ndarray, ds: xr.Dataset, **kwargs):
    """
    Infer the tidal values for long-period minor constituents
    using their relation with major constituents with option to
    take into account variations due to mantle anelasticity
    :cite:p:`Ray:1999vm,Ray:2014fu,Cartwright:1973em,Mathews:2002cr`

    Parameters
    ----------
    t: float or np.ndarray
        Days relative to 1992-01-01T00:00:00
    ds: xarray.Dataset
        Dataset containing major tidal harmonic constants
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    minor: list or None, default None
        Tidal constituent IDs of minor constituents for inference
    include_anelasticity: bool, default False
        Compute Love numbers taking into account mantle anelasticity
    raise_exception: bool, default False
        Raise a ``ValueError`` if major constituents are not found

    Returns
    -------
    tinfer: xr.DataArray
        Tidal time series for minor constituents
    """
    # set default keyword arguments
    kwargs.setdefault("deltat", 0.0)
    kwargs.setdefault("corrections", "OTIS")
    kwargs.setdefault("include_anelasticity", False)
    kwargs.setdefault("raise_exception", False)
    # list of minor constituents
    kwargs.setdefault("minor", None)
    # major constituents used for inferring long period minor tides
    # pivot waves listed in Table 6.7 of the 2010 IERS Conventions
    cindex = ["node", "mm", "mf"]
    # check that major constituents are in the dataset for inference
    nz = sum([(c in ds.tmd.constituents) for c in cindex])
    # raise exception or log error
    msg = "Not enough constituents to infer long-period tides"
    if (nz < 3) and kwargs["raise_exception"]:
        raise Exception(msg)
    elif nz < 3:
        logging.debug(msg)
        return 0.0

    # angular frequencies for major constituents
    omajor = pyTMD.constituents.frequency(cindex, **kwargs)
    # Cartwright and Edden potential amplitudes for major constituents
    amajor = np.zeros((3))
    amajor[0] = 0.027929  # node
    amajor[1] = 0.035184  # mm
    amajor[2] = 0.066607  # mf
    # "normalize" tide values
    dnorm = xr.Dataset()
    for i, c in enumerate(cindex):
        # complex Love numbers of degree 2 for long-period band
        if kwargs["include_anelasticity"]:
            # include variations largely due to mantle anelasticity
            h2, k2, l2 = pyTMD.constituents._complex_love_numbers(omajor[i])
        else:
            # Love numbers for long-period tides (Wahr, 1981)
            h2, k2, l2 = pyTMD.constituents._love_numbers(
                omajor[i], astype=np.complex128
            )
        # tilt factor: response with respect to the solid earth
        # use real components from Mathews et al. (2002)
        gamma_2 = 1.0 + k2.real - h2.real
        dnorm[c] = ds[c] / (amajor[i] * gamma_2)
    # major constituents as a dataarray
    z = dnorm.tmd.to_dataarray()

    # complete list of minor constituents
    minor_constituents = [
        "sa",
        "ssa",
        "sta",
        "msm",
        "msf",
        "mst",
        "mt",
        "msqm",
        "mq",
    ]
    # possibly reduced list of minor constituents
    minor = kwargs["minor"] or minor_constituents
    # angular frequencies for inferred constituents
    omega = pyTMD.constituents.frequency(minor_constituents, **kwargs)
    # only add minor constituents that are not on the list of major values
    # and with frequencies not equal to any major constituent
    constituents = [
        m
        for i, m in enumerate(minor_constituents)
        if (m not in ds.tmd.constituents)
        and (m in minor)
        and (np.all(omega[i] != omajor))
    ]
    # if there are no constituents to infer
    msg = "No long-period tidal constituents to infer"
    if not any(constituents):
        logging.debug(msg)
        return 0.0

    # Cartwright and Edden potential amplitudes for inferred constituents
    amin = np.zeros((9))
    amin[0] = 0.004922  # sa
    amin[1] = 0.030988  # ssa
    amin[2] = 0.001809  # sta
    amin[3] = 0.006728  # msm
    amin[4] = 0.005837  # msf
    amin[5] = 0.002422  # mst
    amin[6] = 0.012753  # mt
    amin[7] = 0.002037  # msqm
    amin[8] = 0.001687  # mq

    # convert time to Modified Julian Days (MJD)
    MJD = t + _mjd_tide
    # load the nodal corrections for minor constituents
    pu, pf, G = pyTMD.constituents.arguments(
        MJD,
        minor_constituents,
        deltat=kwargs["deltat"],
        corrections=kwargs["corrections"],
    )
    # phase angle from arguments
    theta = np.radians(G) + pu

    # compute tilt factors for minor constituents
    nc = len(minor_constituents)
    gamma_2 = np.zeros((nc))
    for i, c in enumerate(minor_constituents):
        # complex Love numbers of degree 2 for long-period band
        if kwargs["include_anelasticity"]:
            # include variations largely due to mantle anelasticity
            h2, k2, l2 = pyTMD.constituents._complex_love_numbers(omega[i])
        else:
            # Love numbers for long-period tides (Wahr, 1981)
            h2, k2, l2 = pyTMD.constituents._love_numbers(
                omega[i], astype=np.complex128
            )
        # tilt factor: response with respect to the solid earth
        # use real components from Mathews et al. (2002)
        gamma_2[i] = 1.0 + k2.real - h2.real

    # dataset of minor arguments
    coords = dict(time=np.atleast_1d(MJD), constituent=minor_constituents)
    arguments = xr.Dataset(
        data_vars=dict(
            u=(["time", "constituent"], pu),
            f=(["time", "constituent"], pf),
            G=(["time", "constituent"], G),
            theta=(["time", "constituent"], np.exp(1j * theta)),
            amplitude=(["constituent"], amin),
            omega=(["constituent"], omega),
            gamma_2=(["constituent"], gamma_2),
        ),
        coords=coords,
    )

    # reduce to selected constituents
    arg = arguments.sel(constituent=constituents)
    # linearly interpolate using constituent frequencies
    zmin = pyTMD.interpolate.interp1d(arg.omega.values, omajor, z)
    coords = z.coords.assign(dict(constituent=arg.constituent))
    zmin = xr.DataArray(zmin, dims=z.dims, coords=coords)
    # rescale tide values
    darr = arg.amplitude * arg.gamma_2 * zmin
    # sum over tidal constituents
    tinfer = (
        darr.real * arg.f * arg.theta.real - darr.imag * arg.f * arg.theta.imag
    ).sum(dim="constituent", skipna=False)
    # copy units attribute
    tinfer.attrs["units"] = ds["node"].attrs.get("units", None)
    tinfer.attrs["constituents"] = constituents
    # return the inferred values
    return tinfer


# dictionary of functions for inferring minor tidal constituents
_infer = {}
_infer["short_period"] = _infer_short_period
_infer["semi_diurnal"] = _infer_semi_diurnal
_infer["diurnal"] = _infer_diurnal
_infer["long_period"] = _infer_long_period


# PURPOSE: estimate long-period equilibrium tides
def equilibrium_tide(t: np.ndarray, ds: xr.Dataset, **kwargs):
    """
    Compute the long-period equilibrium tides the summation of fifteen
    tidal spectral lines from Cartwright-Tayler-Edden tables
    :cite:p:`Cartwright:1971iz,Cartwright:1973em`

    Parameters
    ----------
    t: np.ndarray
        Days relative to 1992-01-01T00:00:00
    ds: xarray.Dataset
        Dataset with spatial coordinates
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    corrections: str, default 'OTIS'
        Use nodal corrections from OTIS/ATLAS or GOT/FES models
    include_anelasticity: bool, default False
        Compute Love numbers taking into account mantle anelasticity
    constituents: list
        Long-period tidal constituent IDs

    Returns
    -------
    tpred: xr.DataArray
        Predicted tidal time series (meters)
    """
    # set default keyword arguments
    cindex = [
        "node",
        "sa",
        "ssa",
        "msm",
        "065.445",
        "mm",
        "065.465",
        "msf",
        "075.355",
        "mf",
        "mf+",
        "075.575",
        "mst",
        "mt",
        "085.465",
    ]
    kwargs.setdefault("constituents", cindex)
    kwargs.setdefault("deltat", 0.0)
    kwargs.setdefault("include_anelasticity", False)
    kwargs.setdefault("corrections", "OTIS")

    # number of constituents
    nc = len(cindex)
    # set function for astronomical longitudes
    # use ASTRO5 routines if not using an OTIS type model
    if kwargs["corrections"] in ("OTIS", "ATLAS", "TMD3", "netcdf"):
        method = "Cartwright"
    else:
        method = "ASTRO5"
    # convert time to Modified Julian Days (MJD)
    MJD = t + _mjd_tide
    # compute principal mean longitudes
    s, h, p, n, pp = pyTMD.astro.mean_longitudes(
        MJD + kwargs["deltat"], method=method
    )
    # initial time conversions
    hour = 24.0 * np.mod(MJD, 1)
    # convert from hours solar time into mean lunar time in degrees
    tau = 15.0 * hour - s + h
    # variable for multiples of 90 degrees (Ray technical note 2017)
    # full expansion of Equilibrium Tide includes some negative cosine
    # terms and some sine terms (Pugh and Woodworth, 2014)
    k = 90.0 + np.zeros_like(MJD)
    # convert to negative mean longitude of the ascending node (N')
    Np = pyTMD.math.normalize_angle(360.0 - n)
    # determine equilibrium arguments
    fargs = np.c_[tau, s, h, p, Np, pp, k]

    # Cartwright and Edden potential amplitudes (centimeters)
    # assemble long-period tide potential from 15 CTE terms greater than 1 mm
    amajor = xr.Dataset()
    # group 0,0
    # nodal term is included but not the constant term.
    amajor["node"] = 2.7929  # node
    amajor["sa"] = -0.4922  # sa
    amajor["ssa"] = -3.0988  # ssa
    # group 0,1
    amajor["msm"] = -0.6728  # msm
    amajor["065.445"] = 0.231
    amajor["mm"] = -3.5184  # mm
    amajor["065.465"] = 0.228
    # group 0,2
    amajor["msf"] = -0.5837  # msf
    amajor["075.355"] = -0.288
    amajor["mf"] = -6.6607  # mf
    amajor["mf+"] = -2.763  # mf+
    amajor["075.575"] = -0.258
    # group 0,3
    amajor["mst"] = -0.2422  # mst
    amajor["mt"] = -1.2753  # mt
    amajor["085.465"] = -0.528

    # set constituents to be iterable and lower case
    if isinstance(kwargs["constituents"], str):
        constituents = [kwargs["constituents"].lower()]
    else:
        constituents = [c.lower() for c in kwargs["constituents"]]

    # Doodson coefficients for 15 long-period terms
    coef = np.zeros((7, nc))
    # group 0,0
    coef[:, 0] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # node
    coef[:, 1] = [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0]  # sa
    coef[:, 2] = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]  # ssa
    # group 0,1
    coef[:, 3] = [0.0, 1.0, -2.0, 1.0, 0.0, 0.0, 0.0]  # msm
    coef[:, 4] = [0.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0]
    coef[:, 5] = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]  # mm
    coef[:, 6] = [0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0]
    # group 0,2
    coef[:, 7] = [0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0]  # msf
    coef[:, 8] = [0.0, 2.0, 0.0, -2.0, 0.0, 0.0, 0.0]
    coef[:, 9] = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # mf
    coef[:, 10] = [0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # mf+
    coef[:, 11] = [0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    # group 0,3
    coef[:, 12] = [0.0, 3.0, -2.0, 1.0, 0.0, 0.0, 0.0]  # mst
    coef[:, 13] = [0.0, 3.0, 0.0, -1.0, 0.0, 0.0, 0.0]  # mt
    coef[:, 14] = [0.0, 3.0, 0.0, -1.0, 1.0, 0.0, 0.0]

    # spherical harmonic degree and order
    l = 2
    m = 0
    # colatitude in radians
    theta = np.radians(90.0 - ds.y)
    # degree dependent normalization (4-pi)
    dfactor = np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi))
    # 2nd degree Legendre polynomials
    Plm = pyTMD.math._assoc_legendre(l, m, np.cos(theta))
    P20 = dfactor * Plm.real

    # calculate tilt factors for each constituent
    gamma_2 = np.zeros((nc))
    for i, c in enumerate(cindex):
        # calculate angular frequencies of constituents
        omega = pyTMD.constituents._frequency(coef[:, i])
        # complex Love numbers of degree 2 for long-period band
        if kwargs["include_anelasticity"]:
            # include variations largely due to mantle anelasticity
            h2, k2, l2 = pyTMD.constituents._complex_love_numbers(omega)
        else:
            # Love numbers for long-period tides (Wahr, 1981)
            h2, k2, l2 = pyTMD.constituents._love_numbers(
                omega, astype=np.complex128
            )
        # tilt factor: response with respect to the solid earth
        # use real components from Mathews et al. (2002)
        gamma_2[i] = 1.0 + k2.real - h2.real

    # determine equilibrium arguments
    G = np.radians(np.dot(fargs, coef))
    # dataset of arguments
    arguments = xr.Dataset(
        data_vars=dict(
            G=(["time", "constituent"], G), gamma_2=(["constituent"], gamma_2)
        ),
        coords=dict(time=np.atleast_1d(MJD), constituent=cindex),
    )
    # reduce to selected constituents
    arg = arguments.sel(constituent=constituents)
    # convert dataset to dataarray of complex tidal elevations
    darr = amajor.tmd.to_dataarray(constituents=constituents)
    # sum equilibrium tide elevations
    tpred = (P20 * darr * arg.gamma_2 * np.cos(arg.G)).sum(
        dim="constituent", skipna=False
    )
    # add units attribute
    tpred.attrs["units"] = "centimeters"
    tpred.attrs["constituents"] = constituents
    # return the long-period equilibrium tides
    return tpred.tmd.to_units("meters")
