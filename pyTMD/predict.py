#!/usr/bin/env python
"""
predict.py
Written by Tyler Sutterley (03/2026)
Prediction routines for ocean, load, equilibrium and solid earth tides

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
    spatial.py: utilities for working with geospatial data

UPDATE HISTORY:
    Updated 03/2026: simplify structure by spliting up IERS corrections
        and adding wrapper functions where appropriate
        set the maximum degree and order for the HW1995 catalog to 6
        clean up the ephemerides method of calculating solid earth tides
        calculate tidal generating forces following Tamura (1982 and 1987)
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
import pyTMD.spatial
import timescale.eop

__all__ = [
    "time_series",
    "infer_minor",
    "_infer_short_period",
    "_infer_semi_diurnal",
    "_infer_diurnal",
    "_infer_long_period",
    "equilibrium_tide",
    "load_pole_tide",
    "ocean_pole_tide",
    "solid_earth_tide",
    "_out_of_phase",
    "_out_of_phase_diurnal",
    "_out_of_phase_semidiurnal",
    "_latitude_dependence",
    "_latitude_dependence_diurnal",
    "_latitude_dependence_semidiurnal",
    "_frequency_dependence",
    "_frequency_dependence_diurnal",
    "_frequency_dependence_long_period",
    "_free_to_mean",
    "body_tide",
    "generating_force",
    "earth_orientation",
    "length_of_day",
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


# PURPOSE: estimate load pole tides in Cartesian coordinates
def load_pole_tide(
    t: np.ndarray,
    XYZ: xr.Dataset,
    deltat: float = 0.0,
    gamma_0: float = 9.80665,
    omega: float = 7.2921151467e-5,
    h2: float = 0.6207,
    l2: float = 0.0836,
    convention: str = "2018",
):
    r"""
    Estimate load pole tide displacements in Cartesian coordinates
    :cite:p:`Petit:2010tp`

    Parameters
    ----------
    t: np.ndarray
        Days relative to 1992-01-01T00:00:00
    XYZ: xarray.Dataset
        Dataset with cartesian coordinates
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    gamma_0: float, default 9.80665
        Normal gravity (m s\ :sup:`-2`)
    omega: float, default 7.2921151467e-5
        Earth's rotation rate (radians/second)
    h2: float, default 0.6207
        Degree-2 Love number of vertical displacement
    l2: float, default 0.0836
        Degree-2 Love (Shida) number of horizontal displacement
    convention: str, default '2018'
        IERS Mean or Secular Pole Convention

            - ``'2003'``
            - ``'2010'``
            - ``'2015'``
            - ``'2018'``

    Returns
    -------
    dxt: xr.Dataset
        Load pole tide displacements (meters)
    """
    # convert time to nominal years (Terrestrial Time)
    time_decimal = 1992.0 + np.atleast_1d(t + deltat) / 365.25
    # convert time to Modified Julian Days (MJD)
    MJD = t + deltat + _mjd_tide

    # radius of the Earth
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    # geocentric latitude (radians)
    latitude = np.arctan(XYZ["Z"] / np.sqrt(XYZ["X"] ** 2.0 + XYZ["Y"] ** 2.0))
    # geocentric colatitude (radians)
    theta = np.pi / 2.0 - latitude
    # calculate longitude (radians)
    phi = np.arctan2(XYZ["Y"], XYZ["X"])

    # calculate angular coordinates of mean/secular pole at time
    mpx, mpy, fl = timescale.eop.iers_mean_pole(
        time_decimal, convention=convention
    )
    sign_convention = -1.0 if convention in ("1996", "2003") else 1.0
    # read and interpolate IERS daily polar motion values
    px, py = timescale.eop.iers_polar_motion(MJD, k=3, s=0)
    # calculate differentials from mean/secular pole positions
    # using the latest definition from IERS Conventions (2010)
    # convert angles from arcseconds to radians
    mx = pyTMD.math.asec2rad(px - mpx)
    my = -pyTMD.math.asec2rad(py - mpy)
    # dataset of polar motion differentials
    pm = xr.Dataset(
        data_vars=dict(
            X=(["time"], mx),
            Y=(["time"], my),
        ),
        coords=dict(time=np.atleast_1d(MJD)),
    )

    # conversion factors in latitude, longitude, and radial directions
    dfactor = xr.Dataset()
    dfactor["N"] = -l2 * (omega**2 * radius**2) / (gamma_0)
    dfactor["E"] = l2 * (omega**2 * radius**2) / (gamma_0)
    dfactor["R"] = -h2 * (omega**2 * radius**2) / (2.0 * gamma_0)

    # calculate pole tide displacements (meters)
    S = xr.Dataset()
    # pole tide displacements in latitude, longitude, and radial directions
    S["N"] = (
        dfactor["N"]
        * np.cos(2.0 * theta)
        * (pm.X * np.cos(phi) + sign_convention * pm.Y * np.sin(phi))
    )
    S["E"] = (
        dfactor["E"]
        * np.cos(theta)
        * (pm.X * np.sin(phi) - sign_convention * pm.Y * np.cos(phi))
    )
    S["R"] = (
        dfactor["R"]
        * np.sin(2.0 * theta)
        * (pm.X * np.cos(phi) + sign_convention * pm.Y * np.sin(phi))
    )

    # rotation matrix for converting to/from cartesian coordinates
    R = xr.Dataset()
    R[0, 0] = np.cos(phi) * np.cos(theta)
    R[0, 1] = -np.sin(phi)
    R[0, 2] = np.cos(phi) * np.sin(theta)
    R[1, 0] = np.sin(phi) * np.cos(theta)
    R[1, 1] = np.cos(phi)
    R[1, 2] = np.sin(phi) * np.sin(theta)
    R[2, 0] = -np.sin(theta)
    R[2, 1] = xr.zeros_like(theta)
    R[2, 2] = np.cos(theta)
    # rotate displacements to ECEF coordinates
    dxt = xr.Dataset()
    dxt["X"] = R[0, 0] * S["N"] + R[0, 1] * S["E"] + R[0, 2] * S["R"]
    dxt["Y"] = R[1, 0] * S["N"] + R[1, 1] * S["E"] + R[1, 2] * S["R"]
    dxt["Z"] = R[2, 0] * S["N"] + R[2, 1] * S["E"] + R[2, 2] * S["R"]
    # add units attributes to output dataset
    for var in dxt.data_vars:
        dxt[var].attrs["units"] = "meters"
    # return the pole tide displacements
    # in Cartesian coordinates
    return dxt


# PURPOSE: estimate ocean pole tides in Cartesian coordinates
def ocean_pole_tide(
    t: np.ndarray,
    UXYZ: np.ndarray,
    deltat: float = 0.0,
    gamma_0: float = 9.780325,
    a_axis: float = 6378136.3,
    GM: float = 3.986004418e14,
    omega: float = 7.2921151467e-5,
    rho_w: float = 1025.0,
    g2: complex = 0.6870 + 0.0036j,
    convention: str = "2018",
):
    r"""
    Estimate ocean pole tide displacements in Cartesian coordinates
    :cite:p:`Desai:2002ev,Desai:2015jr,Petit:2010tp`

    Parameters
    ----------
    t: np.ndarray
        Days relative to 1992-01-01T00:00:00
    UXYZ: np.ndarray
        Ocean pole tide values from Desai (2002)
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    a_axis: float, default 6378136.3
        Semi-major axis of the Earth (meters)
    gamma_0: float, default 9.780325
        Normal gravity (m s\ :sup:`-2`)
    GM: float, default 3.986004418e14
        Geocentric gravitational constant (m\ :sup:`3` s\ :sup:`-2`)
    omega: float, default 7.2921151467e-5
        Earth's rotation rate (radians/second)
    rho_w: float, default 1025.0
        Density of sea water  (kg m\ :sup:`-3`)
    g2: complex, default 0.6870 + 0.0036j
        Degree-2 Love number tilt factor (1 + k2 - h2)
    convention: str, default '2018'
        IERS Mean or Secular Pole Convention

            - ``'2003'``
            - ``'2010'``
            - ``'2015'``
            - ``'2018'``

    Returns
    -------
    dxt: xr.Dataset
        Ocean pole tide displacements (meters)
    """
    # convert time to nominal years (Terrestrial Time)
    time_decimal = 1992.0 + np.atleast_1d(t + deltat) / 365.25
    # convert time to Modified Julian Days (MJD)
    MJD = t + deltat + _mjd_tide

    # calculate angular coordinates of mean/secular pole at time
    mpx, mpy, fl = timescale.eop.iers_mean_pole(
        time_decimal, convention=convention
    )
    # read and interpolate IERS daily polar motion values
    px, py = timescale.eop.iers_polar_motion(MJD, k=3, s=0)
    # calculate differentials from mean/secular pole positions
    # using the latest definition from IERS Conventions (2010)
    # convert angles from arcseconds to radians
    mx = pyTMD.math.asec2rad(px - mpx)
    my = -pyTMD.math.asec2rad(py - mpy)
    # dataset of polar motion differentials
    pm = xr.Dataset(
        data_vars=dict(
            X=(["time"], mx),
            Y=(["time"], my),
        ),
        coords=dict(time=np.atleast_1d(MJD)),
    )

    # universal gravitational constant (N*m^2 kg^-2)
    G = 6.67430e-11
    # pole tide displacement factors
    Hp = np.sqrt(8.0 * np.pi / 15.0) * (omega**2 * a_axis**4) / GM
    K = 4.0 * np.pi * G * rho_w * Hp * a_axis / (3.0 * gamma_0)
    # calculate ocean pole tide displacements (meters)
    dxt = K * np.real(
        UXYZ.real * (pm.X * g2.real + pm.Y * g2.imag)
        + UXYZ.imag * (pm.Y * g2.real - pm.X * g2.imag)
    )
    # add units attributes to output dataset
    for var in dxt.data_vars:
        dxt[var].attrs["units"] = "meters"
    # return the ocean pole tide displacements
    # in Cartesian coordinates
    return dxt


# get ellipsoidal parameters
_iers = pyTMD.spatial.datum(ellipsoid="IERS", units="MKS")


# PURPOSE: estimate solid Earth tides due to gravitational attraction
def solid_earth_tide(
    t: np.ndarray,
    XYZ: xr.Dataset,
    SXYZ: xr.Dataset,
    LXYZ: xr.Dataset,
    deltat: float = 0.0,
    a_axis: float = _iers.a_axis,
    tide_system: str = "tide_free",
    **kwargs,
):
    """
    Compute the solid Earth tides in Cartesian coordinates
    due to the gravitational attraction of the moon and sun
    :cite:p:`Mathews:1991kv,Mathews:1997js,Ries:1992ip,Wahr:1981ea`

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
            - ``'mean_tide'``: permanent tidal potentials (direct and indirect)
    lmax: int, default 3
        Maximum degree of spherical harmonic expansion
    h2: float, default 0.6078
        Degree-2 Love number of vertical displacement
    l2: float, default 0.0847
        Degree-2 Love (Shida) number of horizontal displacement
    h3: float, default 0.292
        Degree-3 Love number of vertical displacement
    l3: float, default 0.015
        Degree-3 Love (Shida) number of horizontal displacement
    h4: float, default 0.18
        Degree-4 Love number of vertical displacement
    l4: float, default 0.014
        Degree-4 Love (Shida) number of horizontal displacement
    mass_ratio_solar: float, default 332946.0482
        Mass ratio between the Earth and the Sun
    mass_ratio_lunar: float, default 0.0123000371
        Mass ratio between the Earth and the Moon

    Returns
    -------
    dxt: xr.Dataset
        Solid Earth tide displacements (meters)
    """
    # set default keyword arguments
    # maximum degree of spherical harmonic expansion
    kwargs.setdefault("lmax", 3)
    # nominal Love and Shida numbers for degrees 2 through 4
    kwargs.setdefault("h2", 0.6078)
    kwargs.setdefault("l2", 0.0847)
    kwargs.setdefault("h3", 0.292)
    kwargs.setdefault("l3", 0.015)
    kwargs.setdefault("h4", 0.18)
    kwargs.setdefault("l4", 0.014)
    # mass ratios between earth and sun/moon
    kwargs.setdefault("mass_ratio_solar", 332946.0482)
    kwargs.setdefault("mass_ratio_lunar", 0.0123000371)
    # validate output tide system
    assert tide_system.lower() in ("tide_free", "mean_tide")
    # convert time to Modified Julian Days (MJD)
    MJD = t + _mjd_tide
    # radius of the point on the Earth's surface
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    sinphi = XYZ["Z"] / radius
    # distance between the Earth and the sun/moon
    solar_radius = pyTMD.math.radius(SXYZ["X"], SXYZ["Y"], SXYZ["Z"])
    lunar_radius = pyTMD.math.radius(LXYZ["X"], LXYZ["Y"], LXYZ["Z"])
    # cosine of angles between vectors of the point and the sun/moon
    solar_scalar = pyTMD.math.scalar_product(
        XYZ["X"], XYZ["Y"], XYZ["Z"], SXYZ["X"], SXYZ["Y"], SXYZ["Z"]
    ) / (radius * solar_radius)
    lunar_scalar = pyTMD.math.scalar_product(
        XYZ["X"], XYZ["Y"], XYZ["Z"], LXYZ["X"], LXYZ["Y"], LXYZ["Z"]
    ) / (radius * lunar_radius)
    # unit vectors for dimensions
    unit_vector = XYZ / radius
    solar_unit_vector = SXYZ / solar_radius
    lunar_unit_vector = LXYZ / lunar_radius
    # compute new h2 and l2 (Mathews et al., 1997)
    # from equations 5 and 6
    h2 = kwargs["h2"] - 0.0006 * (1.5 * sinphi**2 - 0.5)
    l2 = kwargs["l2"] + 0.0002 * (1.5 * sinphi**2 - 0.5)
    # factors for sun and moon using IAU estimates of mass ratios
    K_solar = kwargs["mass_ratio_solar"] * a_axis**3 / solar_radius**2
    K_lunar = kwargs["mass_ratio_lunar"] * a_axis**3 / lunar_radius**2
    # factors for degree 2
    F2_solar = K_solar * (a_axis / solar_radius)
    F2_lunar = K_lunar * (a_axis / lunar_radius)
    # allocate for output displacements
    dxt = xr.Dataset()
    for d in ("X", "Y", "Z"):
        dxt[d] = xr.zeros_like(solar_scalar)
    # compute total displacement (Mathews et al. 1997)
    # from the tide-generating potentials
    # for each spherical harmonic degree
    for l in range(2, kwargs["lmax"] + 1):
        # use calculated love numbers or nominal
        if l == 2:
            hl = np.copy(h2)
            ll = np.copy(l2)
        else:
            # use nominal Love numbers for all other degrees
            hl = kwargs.get(f"h{l}", 0)
            ll = kwargs.get(f"l{l}", 0)
        # update gravitational parameters for degree
        K_solar *= a_axis / solar_radius
        K_lunar *= a_axis / lunar_radius
        # legendre polynomial for degree
        Pl_solar, dPl_solar = pyTMD.math.legendre(l, solar_scalar)
        Pl_lunar, dPl_lunar = pyTMD.math.legendre(l, lunar_scalar)
        # divide differential by u
        # ignore divide by zero and invalid value warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            dPl_solar /= np.sqrt(1 - solar_scalar**2)
            dPl_lunar /= np.sqrt(1 - lunar_scalar**2)
        # calculate solar and lunar terms for degree
        S = K_solar * (
            hl * Pl_solar * unit_vector
            + ll * dPl_solar * solar_scalar * unit_vector
            - ll * dPl_solar * solar_unit_vector
        )
        L = K_lunar * (
            hl * Pl_lunar * unit_vector
            + ll * dPl_lunar * lunar_scalar * unit_vector
            - ll * dPl_lunar * lunar_unit_vector
        )
        # add displacements for degree
        dxt += S + L
    # corrections for out-of-phase portions of the Love and Shida numbers
    dxt += _out_of_phase(XYZ, SXYZ, LXYZ, F2_solar, F2_lunar)
    # corrections for the latitudinal dependence (diurnal and semi-diurnal)
    dxt += _latitude_dependence(XYZ, SXYZ, LXYZ, F2_solar, F2_lunar)
    # corrections for the frequency dependence (diurnal and long-period)
    dxt += _frequency_dependence(XYZ, MJD, deltat=deltat)
    # convert the permanent tide system if specified
    if tide_system.lower() == "mean_tide":
        dxt += _free_to_mean(XYZ, h2, l2)
    # add units attributes to output dataset
    for var in dxt.data_vars:
        dxt[var].attrs["units"] = "meters"
    # return the solid earth tide
    return dxt


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
    D: xr.Dataset
        Solid Earth tide corrections
    """
    # compute diurnal and semi-diurnal corrections separately
    # for both the sun and moon
    D = _out_of_phase_diurnal(XYZ, SXYZ, F2_solar)
    D += _out_of_phase_diurnal(XYZ, LXYZ, F2_lunar)
    D += _out_of_phase_semidiurnal(XYZ, SXYZ, F2_solar)
    D += _out_of_phase_semidiurnal(XYZ, LXYZ, F2_lunar)
    # return the out-of-phase corrections
    return D


def _out_of_phase_diurnal(
    XYZ: xr.Dataset,
    LSXYZ: xr.Dataset,
    F2: np.ndarray,
    dh2: float = -0.0025,
    dl2: float = -0.0007,
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
    dl2: float, default -0.0007
        Shida number correction for the diurnal band

    Returns
    -------
    D: xr.Dataset
        Solid Earth tide corrections
    """
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    sinphi = XYZ["Z"] / radius
    cosphi = np.sqrt(XYZ["X"] ** 2 + XYZ["Y"] ** 2) / radius
    cos2phi = cosphi**2 - sinphi**2
    sinla = XYZ["Y"] / cosphi / radius
    cosla = XYZ["X"] / cosphi / radius
    # compute the normalized position vector of the Sun/Moon
    lunisolar_radius = pyTMD.math.radius(LSXYZ["X"], LSXYZ["Y"], LSXYZ["Z"])
    # calculate offsets
    DR = (
        -3.0
        * dh2
        * sinphi
        * cosphi
        * F2
        * LSXYZ["Z"]
        * (LSXYZ["X"] * sinla - LSXYZ["Y"] * cosla)
        / lunisolar_radius**2
    )
    DN = (
        -3.0
        * dl2
        * cos2phi
        * F2
        * LSXYZ["Z"]
        * (LSXYZ["X"] * sinla - LSXYZ["Y"] * cosla)
        / lunisolar_radius**2
    )
    DE = (
        -3.0
        * dl2
        * sinphi
        * F2
        * LSXYZ["Z"]
        * (LSXYZ["X"] * cosla + LSXYZ["Y"] * sinla)
        / lunisolar_radius**2
    )
    # output corrections
    D = xr.Dataset()
    # compute corrections in cartesian coordinates
    D["X"] = DR * cosla * cosphi - DE * sinla - DN * cosla * sinphi
    D["Y"] = DR * sinla * cosphi + DE * cosla - DN * sinla * sinphi
    D["Z"] = DR * sinphi + DN * cosphi
    # return the corrections
    return D


def _out_of_phase_semidiurnal(
    XYZ: xr.Dataset,
    LSXYZ: xr.Dataset,
    F2: np.ndarray,
    dh2: float = -0.0022,
    dl2: float = -0.0007,
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
    dl2: float, default -0.0007
        Shida number correction for the semi-diurnal band

    Returns
    -------
    D: xr.Dataset
        Solid Earth tide corrections
    """
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    sinphi = XYZ["Z"] / radius
    cosphi = np.sqrt(XYZ["X"] ** 2 + XYZ["Y"] ** 2) / radius
    sinla = XYZ["Y"] / cosphi / radius
    cosla = XYZ["X"] / cosphi / radius
    cos2la = cosla**2 - sinla**2
    sin2la = 2.0 * cosla * sinla
    # compute the normalized position vector of the Sun/Moon
    lunisolar_radius = pyTMD.math.radius(LSXYZ["X"], LSXYZ["Y"], LSXYZ["Z"])
    # calculate offsets
    DR = (
        (-3.0 / 4.0)
        * dh2
        * cosphi**2
        * F2
        * (
            (LSXYZ["X"] ** 2 - LSXYZ["Y"] ** 2) * sin2la
            - 2.0 * LSXYZ["X"] * LSXYZ["Y"] * cos2la
        )
        / lunisolar_radius**2
    )
    DN = (
        (3.0 / 2.0)
        * dl2
        * sinphi
        * cosphi
        * F2
        * (
            (LSXYZ["X"] ** 2 - LSXYZ["Y"] ** 2) * sin2la
            - 2.0 * LSXYZ["X"] * LSXYZ["Y"] * cos2la
        )
        / lunisolar_radius**2
    )
    DE = (
        (-3.0 / 2.0)
        * dl2
        * cosphi
        * F2
        * (
            (LSXYZ["X"] ** 2 - LSXYZ["Y"] ** 2) * cos2la
            + 2.0 * LSXYZ["X"] * LSXYZ["Y"] * sin2la
        )
        / lunisolar_radius**2
    )
    # output corrections
    D = xr.Dataset()
    # compute corrections in cartesian coordinates
    D["X"] = DR * cosla * cosphi - DE * sinla - DN * cosla * sinphi
    D["Y"] = DR * sinla * cosphi + DE * cosla - DN * sinla * sinphi
    D["Z"] = DR * sinphi + DN * cosphi
    # return the corrections
    return D


def _latitude_dependence(
    XYZ: xr.Dataset,
    SXYZ: xr.Dataset,
    LXYZ: xr.Dataset,
    F2_solar: np.ndarray,
    F2_lunar: np.ndarray,
):
    r"""
    Wrapper function to compute the latitudinal dependent corrections
    given by L\ :sup:`1` for both the diurnal and semi-diurnal bands
    :cite:p:`Petit:2010tp`

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
    D: xr.Dataset
        Solid Earth tide corrections
    """
    # compute diurnal and semi-diurnal corrections separately
    # for both the sun and moon
    D = _latitude_dependence_diurnal(XYZ, SXYZ, F2_solar)
    D += _latitude_dependence_diurnal(XYZ, LXYZ, F2_lunar)
    D += _latitude_dependence_semidiurnal(XYZ, SXYZ, F2_solar)
    D += _latitude_dependence_semidiurnal(XYZ, LXYZ, F2_lunar)
    # return the latitudinal dependent corrections
    return D


def _latitude_dependence_diurnal(
    XYZ: xr.Dataset,
    LSXYZ: xr.Dataset,
    F2: np.ndarray,
    L1: float = 0.0012,
):
    r"""
    Computes the corrections induced by the latitudinal
    dependence of the diurnal band :cite:p:`Petit:2010tp`

    Parameters
    ----------
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    LSXYZ: xr.Dataset
        Dataset with Earth-centered Earth-fixed coordinates of the sun or moon
    F2: np.ndarray
        Factors for the sun or moon
    L1: float, default 0.0012
        Love/Shida number correction for the diurnal band

    Returns
    -------
    D: xr.Dataset
        Solid Earth tide corrections
    """
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    sinphi = XYZ["Z"] / radius
    cosphi = np.sqrt(XYZ["X"] ** 2 + XYZ["Y"] ** 2) / radius
    sinla = XYZ["Y"] / cosphi / radius
    cosla = XYZ["X"] / cosphi / radius
    # compute the normalized position vector of the Sun/Moon
    lunisolar_radius = pyTMD.math.radius(LSXYZ["X"], LSXYZ["Y"], LSXYZ["Z"])
    # calculate offsets for the diurnal band
    DN = (
        (-3.0 * L1)
        * sinphi**2
        * F2
        * LSXYZ["Z"]
        * (LSXYZ["X"] * cosla + LSXYZ["Y"] * sinla)
        / lunisolar_radius**2
    )
    DE = (
        (3.0 * L1)
        * sinphi
        * (cosphi**2 - sinphi**2)
        * F2
        * LSXYZ["Z"]
        * (LSXYZ["X"] * sinla - LSXYZ["Y"] * cosla)
        / lunisolar_radius**2
    )
    # output corrections
    D = xr.Dataset()
    # compute corrections in cartesian coordinates
    D["X"] = -DE * sinla - DN * cosla * sinphi
    D["Y"] = DE * cosla - DN * sinla * sinphi
    D["Z"] = DN * cosphi
    # return the corrections
    return D


def _latitude_dependence_semidiurnal(
    XYZ: xr.Dataset,
    LSXYZ: xr.Dataset,
    F2: np.ndarray,
    L1: float = 0.0024,
):
    r"""
    Computes the corrections induced by the latitudinal
    dependence of the semi-diurnal band :cite:p:`Petit:2010tp`

    Parameters
    ----------
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    LSXYZ: xr.Dataset
        Dataset with Earth-centered Earth-fixed coordinates of the sun or moon
    F2: np.ndarray
        Factors for the sun or moon
    L1: float, default 0.0024
        Love/Shida number correction for the semi-diurnal band

    Returns
    -------
    D: xr.Dataset
        Solid Earth tide corrections
    """
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    sinphi = XYZ["Z"] / radius
    cosphi = np.sqrt(XYZ["X"] ** 2 + XYZ["Y"] ** 2) / radius
    sinla = XYZ["Y"] / cosphi / radius
    cosla = XYZ["X"] / cosphi / radius
    cos2la = cosla**2 - sinla**2
    sin2la = 2.0 * cosla * sinla
    # compute the normalized position vector of the Sun/Moon
    lunisolar_radius = pyTMD.math.radius(LSXYZ["X"], LSXYZ["Y"], LSXYZ["Z"])
    # calculate offsets for the semi-diurnal band
    DN = (
        (-3.0 * L1 / 2.0)
        * sinphi
        * cosphi
        * F2
        * (
            (LSXYZ["X"] ** 2 - LSXYZ["Y"] ** 2) * cos2la
            + 2.0 * LSXYZ["X"] * LSXYZ["Y"] * sin2la
        )
        / lunisolar_radius**2
    )
    DE = (
        (-3.0 * L1 / 2.0)
        * sinphi**2
        * cosphi
        * F2
        * (
            (LSXYZ["X"] ** 2 - LSXYZ["Y"] ** 2) * sin2la
            - 2.0 * LSXYZ["X"] * LSXYZ["Y"] * cos2la
        )
        / lunisolar_radius**2
    )
    # output corrections
    D = xr.Dataset()
    # compute corrections in cartesian coordinates
    D["X"] = -DE * sinla - DN * cosla * sinphi
    D["Y"] = DE * cosla - DN * sinla * sinphi
    D["Z"] = DN * cosphi
    # return the corrections
    return D


def _frequency_dependence(
    XYZ: xr.Dataset, MJD: np.ndarray, deltat: float | np.ndarray = 0.0
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
    D: xr.Dataset
        Solid Earth tide corrections
    """
    # Love/Shida number corrections (diurnal and long-period)
    # compute diurnal and long-period corrections separately
    D = _frequency_dependence_diurnal(XYZ, MJD, deltat=deltat)
    D += _frequency_dependence_long_period(XYZ, MJD, deltat=deltat)
    # return the frequency dependent corrections
    return D


def _frequency_dependence_diurnal(
    XYZ: xr.Dataset, MJD: np.ndarray, deltat: float | np.ndarray = 0.0
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
    D: xr.Dataset
        Solid Earth tide corrections
    """
    # Corrections to Diurnal Tides for Frequency Dependence
    # of Love and Shida Number Parameters
    # reduced version of table 7.3a from IERS conventions
    columns = [
        "tau",
        "s",
        "h",
        "p",
        "np",
        "ps",
        "dR_ip",
        "dR_op",
        "dT_ip",
        "dT_op",
    ]
    table = xr.DataArray(
        np.array(
            [
                [1.0, -3.0, 0.0, 2.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
                [1.0, -3.0, 2.0, 0.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
                [1.0, -2.0, 0.0, 1.0, -1.0, 0.0, -0.02, 0.0, 0.0, 0.0],
                [1.0, -2.0, 0.0, 1.0, 0.0, 0.0, -0.08, 0.0, -0.01, 0.01],
                [1.0, -2.0, 2.0, -1.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0, -1.0, 0.0, -0.10, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -0.51, 0.0, -0.02, 0.03],
                [1.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                [1.0, 0.0, -2.0, 1.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                [1.0, 1.0, -3.0, 0.0, 0.0, 1.0, -0.06, 0.0, 0.0, 0.0],
                [1.0, 1.0, -2.0, 0.0, -1.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                [1.0, 1.0, -2.0, 0.0, 0.0, 0.0, -1.23, -0.07, 0.06, 0.01],
                [1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.02, 0.0, 0.0, 0.0],
                [1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.04, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, -1.0, 0.0, -0.22, 0.01, 0.01, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 12.00, -0.80, -0.67, -0.03],
                [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.73, -0.12, -0.10, 0.0],
                [1.0, 1.0, 0.0, 0.0, 2.0, 0.0, -0.04, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -0.50, -0.01, 0.03, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.01, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, -1.0, -0.01, 0.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, -2.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
                [1.0, 1.0, 2.0, 0.0, 0.0, 0.0, -0.11, 0.01, 0.01, 0.0],
                [1.0, 2.0, -2.0, 1.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, -1.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0],
                [1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        dims=["constituent", "argument"],
        coords=dict(argument=columns),
    )
    coef = table.to_dataset(dim="argument")
    # get phase angles (Doodson arguments)
    TAU, S, H, P, ZNS, PS = pyTMD.astro.doodson_arguments(MJD + deltat)
    # dataset of arguments
    arguments = xr.Dataset(
        data_vars=dict(
            tau=(["time"], TAU),
            s=(["time"], S),
            h=(["time"], H),
            p=(["time"], P),
            np=(["time"], ZNS),
            ps=(["time"], PS),
        ),
        coords=dict(time=np.atleast_1d(MJD)),
    )
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    sinphi = XYZ["Z"] / radius
    cosphi = np.sqrt(XYZ["X"] ** 2 + XYZ["Y"] ** 2) / radius
    sinla = XYZ["Y"] / cosphi / radius
    cosla = XYZ["X"] / cosphi / radius
    # compute longitude
    zla = np.arctan2(XYZ["Y"], XYZ["X"])
    # compute phase angle of tide potential (Greenwich)
    thetaf = (
        arguments.tau * coef["tau"]
        + arguments.s * coef["s"]
        + arguments.h * coef["h"]
        + arguments.p * coef["p"]
        + arguments.np * coef["np"]
        + arguments.ps * coef["ps"]
    )
    # calculate complex phase (local hour angle)
    cphase = np.exp(1j * thetaf + 1j * zla)
    # calculate offsets in local coordinates
    dr = (
        2.0
        * sinphi
        * cosphi
        * (coef["dR_ip"] * cphase.imag + coef["dR_op"] * cphase.real)
    )
    dn = (cosphi**2 - sinphi**2) * (
        coef["dT_ip"] * cphase.imag + coef["dT_op"] * cphase.real
    )
    de = sinphi * (coef["dT_ip"] * cphase.real - coef["dT_op"] * cphase.imag)
    # compute corrections (Mathews et al. 1997)
    # rotate to cartesian coordinates
    DX = (dr * cosla * cosphi - de * sinla - dn * cosla * sinphi).sum(
        dim="constituent", skipna=False
    )
    DY = (dr * sinla * cosphi + de * cosla - dn * sinla * sinphi).sum(
        dim="constituent", skipna=False
    )
    DZ = (dr * sinphi + dn * cosphi).sum(dim="constituent", skipna=False)
    # convert from millimeters to meters
    D = xr.Dataset()
    D["X"] = 1e-3 * DX
    D["Y"] = 1e-3 * DY
    D["Z"] = 1e-3 * DZ
    # return the corrections
    return D


def _frequency_dependence_long_period(
    XYZ: xr.Dataset, MJD: np.ndarray, deltat: float | np.ndarray = 0.0
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
    D: xr.Dataset
        Solid Earth tide corrections
    """
    # Corrections to Long-Period Tides for Frequency Dependence
    # of Love and Shida Number Parameters
    # reduced version of table 7.3b from IERS conventions
    columns = [
        "tau",
        "s",
        "h",
        "p",
        "np",
        "ps",
        "dR_ip",
        "dR_op",
        "dT_ip",
        "dT_op",
    ]
    table = xr.DataArray(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.47, 0.23, 0.16, 0.07],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, -0.20, -0.12, -0.11, -0.05],
                [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, -0.11, -0.08, -0.09, -0.04],
                [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -0.13, -0.11, -0.15, -0.07],
                [0.0, 2.0, 0.0, 0.0, 1.0, 0.0, -0.05, -0.05, -0.06, -0.03],
            ]
        ),
        dims=["constituent", "argument"],
        coords=dict(argument=columns),
    )
    coef = table.to_dataset(dim="argument")
    # get phase angles (Doodson arguments)
    TAU, S, H, P, ZNS, PS = pyTMD.astro.doodson_arguments(MJD + deltat)
    # dataset of arguments
    arguments = xr.Dataset(
        data_vars=dict(
            tau=(["time"], TAU),
            s=(["time"], S),
            h=(["time"], H),
            p=(["time"], P),
            np=(["time"], ZNS),
            ps=(["time"], PS),
        ),
        coords=dict(time=np.atleast_1d(MJD)),
    )
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    sinphi = XYZ["Z"] / radius
    cosphi = np.sqrt(XYZ["X"] ** 2 + XYZ["Y"] ** 2) / radius
    sinla = XYZ["Y"] / cosphi / radius
    cosla = XYZ["X"] / cosphi / radius
    # compute phase angle of tide potential (Greenwich)
    thetaf = (
        arguments.tau * coef["tau"]
        + arguments.s * coef["s"]
        + arguments.h * coef["h"]
        + arguments.p * coef["p"]
        + arguments.np * coef["np"]
        + arguments.ps * coef["ps"]
    )
    # calculate complex phase (zonal harmonics have no longitude dependence)
    cphase = np.exp(1j * thetaf)
    # calculate offsets in local coordinates
    dr = (1.5 * sinphi**2 - 0.5) * (
        coef["dT_ip"] * cphase.imag + coef["dR_ip"] * cphase.real
    )
    dn = (2.0 * cosphi * sinphi) * (
        coef["dT_op"] * cphase.imag + coef["dR_op"] * cphase.real
    )
    de = 0.0
    # compute corrections (Mathews et al. 1997)
    # rotate to cartesian coordinates
    DX = (dr * cosla * cosphi - de * sinla - dn * cosla * sinphi).sum(
        dim="constituent", skipna=False
    )
    DY = (dr * sinla * cosphi + de * cosla - dn * sinla * sinphi).sum(
        dim="constituent", skipna=False
    )
    DZ = (dr * sinphi + dn * cosphi).sum(dim="constituent", skipna=False)
    # convert from millimeters to meters
    D = xr.Dataset()
    D["X"] = 1e-3 * DX
    D["Y"] = 1e-3 * DY
    D["Z"] = 1e-3 * DZ
    # return the corrections
    return D


def _free_to_mean(
    XYZ: xr.Dataset,
    h2: float | np.ndarray,
    l2: float | np.ndarray,
    H0: float = -0.31460,
):
    """
    Calculate offsets for converting the permanent tide from
    a tide-free to a mean-tide state :cite:p:`Mathews:1997js`

    Parameters
    ----------
    XYZ: xr.Dataset
        Dataset with cartesian coordinates
    h2: float or np.ndarray
        Degree-2 Love number of vertical displacement
    l2: float or np.ndarray
        Degree-2 Love (Shida) number of horizontal displacement
    H0: float, default -0.31460
        Mean amplitude of the permanent tide (meters)

    Returns
    -------
    D: xr.Dataset
        free-to-mean tide offset
    """
    # compute the normalized position vector of coordinates
    radius = pyTMD.math.radius(XYZ["X"], XYZ["Y"], XYZ["Z"])
    sinphi = XYZ["Z"] / radius
    cosphi = np.sqrt(XYZ["X"] ** 2 + XYZ["Y"] ** 2) / radius
    sinla = XYZ["Y"] / cosphi / radius
    cosla = XYZ["X"] / cosphi / radius
    # in Mathews et al. (1997): dR0=-0.1196 m with h2=0.6026
    dR0 = np.sqrt(5.0 / (4.0 * np.pi)) * h2 * H0
    # in Mathews et al. (1997): dN0=-0.0247 m with l2=0.0831
    dN0 = np.sqrt(45.0 / (16.0 * np.pi)) * l2 * H0
    # use double angle formula for sin(2*phi)
    dr = dR0 * (3.0 / 2.0 * sinphi**2 - 1.0 / 2.0)
    dn = 2.0 * dN0 * cosphi * sinphi
    # compute corrections (Mathews et al. 1997)
    D = xr.Dataset()
    # compute as an additive correction (Mathews et al. 1997)
    D["X"] = -dr * cosla * cosphi + dn * cosla * sinphi
    D["Y"] = -dr * sinla * cosphi + dn * sinla * sinphi
    D["Z"] = -dr * sinphi - dn * cosphi
    # return the corrections
    return D


# tide potential tables
_tide_potential_table = {}
# Cartwright and Tayler (1971) table with 3rd-degree values
# Cartwright and Edden (1973) table with updated values
_tide_potential_table["CTE1973"] = pyTMD.constituents._cte1973_table
# Hartmann and Wenzel (1995) tidal potential catalog
_tide_potential_table["HW1995"] = pyTMD.constituents._hw1995_table
# Tamura (1987) tidal potential catalog
_tide_potential_table["T1987"] = pyTMD.constituents._t1987_table
# Woodworth (1990) tables with updated and 3rd-degree values
_tide_potential_table["W1990"] = pyTMD.constituents._w1990_table


# PURPOSE: estimate solid Earth tides due to gravitational attraction
# using a simplified approach based on Cartwright and Tayler (1971)
def body_tide(
    t: np.ndarray,
    ds: xr.Dataset,
    deltat: float | np.ndarray = 0.0,
    method: str = "ASTRO5",
    tide_system: str = "tide_free",
    catalog: str = "CTE1973",
    **kwargs,
):
    """
    Compute the solid Earth tides due to the gravitational
    attraction of the moon and sun using the approach of
    :cite:t:`Cartwright:1971iz` adjusting the degree-2 Love numbers
    for a near-diurnal frequency dependence :cite:p:`Mathews:1995go`

    Parameters
    ----------
    t: np.ndarray
        Days relative to 1992-01-01T00:00:00
    ds: xarray.Dataset
        Dataset with spatial coordinates
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    method: str, default 'ASTRO5'
        Method for computing the mean longitudes

            - ``'Cartwright'``
            - ``'Meeus'``
            - ``'ASTRO5'``
            - ``'IERS'``
    tide_system: str, default 'tide_free'
        Permanent tide system for the output solid Earth tide

            - ``'tide_free'``: no permanent direct and indirect tidal potentials
            - ``'mean_tide'``: permanent tidal potentials (direct and indirect)
    catalog: str, default 'CTE1973'
        Name of the tide potential catalog

            - ``'CTE1973'``: :cite:t:`Cartwright:1973em`
            - ``'HW1995'``: :cite:t:`Hartmann:1995jp`
            - ``'T1987'``: :cite:t:`Tamura:1987tp`
            - ``'W1990'``: Woodworth updates to ``'CTE1973'``
    lmax: int, default 6
        Maximum degree of spherical harmonic expansion

        Will be based on the maximum degree available in the catalog
    include_planets: bool, default False
        Include tide potentials from planetary bodies
    h2: float or None, default None
        Degree-2 Love number of vertical displacement
    l2: float or None, default None
        Degree-2 Love (Shida) number of horizontal displacement
    h3: float, default 0.291
        Degree-3 Love number of vertical displacement
    l3: float, default 0.015
        Degree-3 Love (Shida) number of horizontal displacement
    h4: float, default 0.18
        Degree-4 Love number of vertical displacement
    l4: float, default 0.014
        Degree-4 Love (Shida) number of horizontal displacement

    Returns
    -------
    zeta: xr.Dataset
        Solid Earth tide (meters)
    """
    # set default keyword arguments
    # maximum degree of spherical harmonic expansion
    kwargs.setdefault("lmax", 6)
    # include contributions from planets
    kwargs.setdefault("include_planets", False)
    # nominal Love and Shida numbers for degrees 2, 3, and 4
    kwargs.setdefault("h2", None)
    kwargs.setdefault("l2", None)
    kwargs.setdefault("h3", 0.291)
    kwargs.setdefault("l3", 0.015)
    kwargs.setdefault("h4", 0.18)
    kwargs.setdefault("l4", 0.014)
    # check if user has provided degree-2 Love numbers
    user_degree_2 = (kwargs["h2"] is not None) and (kwargs["l2"] is not None)
    # validate method and output tide system
    assert method.lower() in ("cartwright", "meeus", "astro5", "iers")
    assert tide_system.lower() in ("tide_free", "mean_tide")
    assert catalog in _tide_potential_table.keys()

    # convert dates to Modified Julian Days
    MJD = t + _mjd_tide

    # compute principal mean longitudes
    # convert dates into Ephemeris Time
    s, h, p, n, pp = pyTMD.astro.mean_longitudes(MJD + deltat, method=method)
    # initial time conversions
    hour = 24.0 * np.mod(MJD, 1)
    # convert from hours solar time into mean lunar time in degrees
    tau = 15.0 * hour - s + h
    # variable for multiples of 90 degrees (Ray technical note 2017)
    # full expansion of Equilibrium Tide includes some negative cosine
    # terms and some sine terms (Pugh and Woodworth, 2014)
    k = 90.0 + np.zeros_like(MJD)

    # astronomical and planetary mean longitudes
    if kwargs["include_planets"]:
        # calculate planetary mean longitudes
        # me: Mercury, ve: Venus, ma: Mars, ju: Jupiter, sa: Saturn
        me, ve, ma, ju, sa = pyTMD.astro.planetary_longitudes(MJD)
        fargs = np.c_[tau, s, h, p, n, pp, k, me, ve, ma, ju, sa]
        nargs = 12
    else:
        fargs = np.c_[tau, s, h, p, n, pp, k]
        nargs = 7
    # allocate array for Doodson coefficients
    coef = np.zeros((nargs))

    # longitudes and colatitudes in radians
    phi = np.radians(ds.x)
    th = np.radians(90.0 - ds.y)

    # allocate for output body tide estimates (meters)
    # latitudinal, longitudinal and radial components
    zeta = xr.Dataset()

    # check if tide catalog includes planetary contributions
    if catalog == "HW1995":
        # catalog includes planetary contributions
        # and harmonics up to degree and order 6
        include_planets = True
        lmax = np.minimum(6, kwargs["lmax"])
    elif catalog == "T1987":
        # catalog includes planetary contributions
        # and harmonics up to degree and order 4
        include_planets = True
        lmax = np.minimum(4, kwargs["lmax"])
    else:
        # older catalogs without planetary contributions
        # and harmonics up to degree and order 3
        include_planets = False
        lmax = np.minimum(3, kwargs["lmax"])

    # parse tide potential table for constituents
    table = _tide_potential_table[catalog]
    CTE = pyTMD.constituents._parse_tide_potential_table(
        table,
        skiprows=1,
        columns=1,
        include_degree=True,
        include_planets=include_planets,
    )

    # precompute spherical harmonic functions and derivatives
    # will need to be rotated by constituent phase
    Ylm = xr.Dataset()
    dYlm = xr.Dataset()
    # for each degree and order
    for l in range(2, lmax + 1):
        for m in range(l + 1):
            Ylm[l, m], dYlm[l, m] = pyTMD.math.sph_harm(l, th, phi, m=m)

    # initialize phase array
    phase = xr.DataArray(np.zeros_like(MJD), dims=dict(time=np.atleast_1d(MJD)))
    # initialize output body tides
    for key in ["R", "N", "E"]:
        zeta[key] = xr.zeros_like(th * phase)
    # for each line in the table
    for i, line in enumerate(CTE):
        # spherical harmonic degree
        l = line["l"]
        # skip if degree is above the specified expansion limit
        if l > lmax:
            continue
        # spherical harmonic dependence (order)
        TAU = line["tau"]
        # update Doodson coefficients for constituent
        coef[0] = TAU
        coef[1] = line["s"]
        coef[2] = line["h"]
        coef[3] = line["p"]
        # convert N for ascending lunar node (from N')
        coef[4] = -1.0 * line["n"]
        coef[5] = line["pp"]
        # use cosines for (l + tau) even
        # and sines for (l + tau) odd
        coef[6] = -1.0 * np.mod(l + TAU, 2)
        # include planetary contributions
        if kwargs["include_planets"]:
            # coefficients including planetary terms
            coef[7] = line["lme"]
            coef[8] = line["lve"]
            coef[9] = line["lma"]
            coef[10] = line["lju"]
            coef[11] = line["lsa"]
        # calculate angular frequency of constituent
        omega = pyTMD.constituents._frequency(
            coef, method=method, include_planets=kwargs["include_planets"]
        )
        # skip the permanent tide if using a mean-tide system
        if (omega == 0) and (tide_system.lower() == "mean_tide"):
            continue
        # determine constituent phase using equilibrium arguments
        G = pyTMD.math.normalize_angle(np.dot(fargs, coef))
        # convert phase angles to radians
        phase[:] = np.radians(G)
        # rotate spherical harmonic functions by phase angles
        S = Ylm[l, TAU] * np.exp(1j * phase)
        dS = dYlm[l, TAU] * np.exp(1j * phase)
        # add components for degree and order to output body tides
        if (l == 2) and user_degree_2:
            # user-defined Love numbers for all constituents
            hl = np.complex128(kwargs["h2"])
            ll = np.complex128(kwargs["l2"])
        elif (l == 2) and (method == "IERS"):
            # IERS: including both in-phase and out-of-phase components
            # 1) using resonance formula for tides in the diurnal band
            # 2) adjusting some long-period tides for anelastic effects
            hl, kl, ll = pyTMD.constituents._complex_love_numbers(
                omega, method=method
            )
            # 3) including complex latitudinal dependence
            hl -= (0.615e-3 + 0.122e-4j) * (1.0 - 1.5 * np.sin(th) ** 2)
            ll += (0.19334e-3 - 0.3819e-5j) * (1.0 - 1.5 * np.sin(th) ** 2)
        elif l == 2:
            # use resonance formula for tides in the diurnal band
            hl, kl, ll = pyTMD.constituents._love_numbers(
                omega, method=method, astype=np.complex128
            )
            # include latitudinal dependence
            hl -= 0.0006 * (1.0 - 1.5 * np.sin(th) ** 2)
            ll += 0.0002 * (1.0 - 1.5 * np.sin(th) ** 2)
        else:
            # use nominal Love numbers for all other degrees
            hl = np.complex128(kwargs.get(f"h{l}", 0))
            ll = np.complex128(kwargs.get(f"l{l}", 0))
        # convert potentials for constituent and add to the total
        # (latitudinal, longitudinal and radial components)
        zeta["N"] += line["Hs1"] * (ll.real * dS.real - ll.imag * dS.imag)
        zeta["E"] -= line["Hs1"] * TAU * (ll.real * S.imag - ll.imag * S.real)
        zeta["R"] += line["Hs1"] * (hl.real * S.real - hl.imag * S.imag)

    # add units attributes to output dataset
    for var in zeta.data_vars:
        zeta[var].attrs["units"] = "meters"
    # return the body tides
    return zeta


def generating_force(
    t: np.ndarray,
    XYZ: xr.Dataset,
    deltat: float | np.ndarray = 0.0,
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
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)
    method: str, default 'Meeus'
        Method for calculating solar and lunar positions

        - ``'Kubo'``: :cite:t:`Kubo:1980ut`
        - ``'Meeus'``: :cite:t:`Meeus:1991vh`
    lmax: int, default 4
        Maximum degree of spherical harmonic expansion
    AU: float, default 149597870700.0
        Distance of 1 Astronomical Unit (meters)
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
    # method for calculating solar and lunar positions
    kwargs.setdefault("method", "Meeus")
    # distance of 1 Astronomical Unit (meters)
    kwargs.setdefault("AU", 149597870700.0)
    # gravitational constants (m^3 s^-2)
    kwargs.setdefault("GM", 3.986004418e14)
    # mass ratios between earth and sun/moon
    kwargs.setdefault("mass_ratio_solar", 332946.0482)
    kwargs.setdefault("mass_ratio_lunar", 0.0123000371)

    # Julian centuries from J2000.0 (UT1)
    T = xr.DataArray(
        (t + _mjd_tide - _mjd_j2000) / _century,
        dims=("time"),
        coords=dict(time=(t + _mjd_tide)),
    )
    # convert dates to Modified Julian Days in Ephemeris time
    MJD = (T * _century) + _mjd_j2000 + deltat

    # geocentric colatitude and longitude (radians)
    theta = np.pi / 2.0 - np.arctan(XYZ.Z / np.sqrt(XYZ.X**2.0 + XYZ.Y**2.0))
    phi = np.arctan2(XYZ.Y, XYZ.X)
    # Earth's radius at the given latitude (meters)
    rad_e = pyTMD.math.radius(XYZ.X, XYZ.Y, XYZ.Z)

    # obliquity of the ecliptic
    epsilon = pyTMD.astro.mean_obliquity(MJD)
    # correct obliquity for nutation
    if kwargs["method"].lower() in ("kubo", "meeus"):
        # simple correction for principal nutation (radians)
        omega = np.radians(1934.136 * T + 235.0)
        epsilon += np.radians(0.00256 * np.cos(omega))
    else:
        # calculate nutation rotation angles using IERS tables
        dpsi, deps = pyTMD.astro._nutation_angles(T)
        epsilon += deps

    # get phase angles and distances for the sun
    solar_latitude = xr.DataArray(
        pyTMD.astro.solar_latitude(MJD, **kwargs), dims=T.dims, coords=T.coords
    )
    solar_longitude = xr.DataArray(
        pyTMD.astro.solar_longitude(MJD, **kwargs), dims=T.dims, coords=T.coords
    )
    solar_radius = xr.DataArray(
        pyTMD.astro.solar_distance(MJD, **kwargs), dims=T.dims, coords=T.coords
    )
    # get phase angles and distances for the moon
    lunar_latitude = xr.DataArray(
        pyTMD.astro.lunar_latitude(MJD, **kwargs), dims=T.dims, coords=T.coords
    )
    lunar_longitude = xr.DataArray(
        pyTMD.astro.lunar_longitude(MJD, **kwargs), dims=T.dims, coords=T.coords
    )
    lunar_radius = xr.DataArray(
        pyTMD.astro.lunar_distance(MJD, **kwargs), dims=T.dims, coords=T.coords
    )

    # calculate the right ascensions and declinations of the sun
    # Meeus equations 25.6 and 25.7
    solar_right_ascension = np.arctan2(
        np.cos(epsilon) * np.sin(solar_longitude) * np.cos(solar_latitude)
        - np.sin(epsilon) * np.sin(solar_latitude),
        np.cos(solar_longitude) * np.cos(solar_latitude),
    )
    solar_declination = np.arcsin(
        np.sin(epsilon) * np.sin(solar_longitude) * np.cos(solar_latitude)
        + np.cos(epsilon) * np.sin(solar_latitude)
    )
    # calculate the right ascensions and declinations of the moon
    lunar_right_ascension = np.arctan2(
        np.cos(epsilon) * np.sin(lunar_longitude) * np.cos(lunar_latitude)
        - np.sin(epsilon) * np.sin(lunar_latitude),
        np.cos(lunar_longitude) * np.cos(lunar_latitude),
    )
    lunar_declination = np.arcsin(
        np.sin(epsilon) * np.sin(lunar_longitude) * np.cos(lunar_latitude)
        + np.cos(epsilon) * np.sin(lunar_latitude)
    )
    # local hour angles in radians
    gast = 2.0 * np.pi * pyTMD.astro.gast(T)
    local_hour_angle = gast + phi
    solar_hour_angle = local_hour_angle - solar_right_ascension
    lunar_hour_angle = local_hour_angle - lunar_right_ascension

    # cosine of geocentric zenith distances
    solar_zenith = np.sin(theta) * np.sin(solar_declination) + (
        np.cos(theta) * np.cos(solar_declination) * np.cos(solar_hour_angle)
    )
    lunar_zenith = np.sin(theta) * np.sin(lunar_declination) + (
        np.cos(theta) * np.cos(lunar_declination) * np.cos(lunar_hour_angle)
    )
    # differential of zenith angle with respect to theta
    solar_dtheta = np.cos(theta) * np.sin(solar_declination) - (
        np.sin(theta) * np.cos(solar_declination) * np.cos(solar_hour_angle)
    )
    lunar_dtheta = np.cos(theta) * np.sin(lunar_declination) - (
        np.sin(theta) * np.cos(lunar_declination) * np.cos(lunar_hour_angle)
    )
    # differential of zenith angle with respect to phi
    solar_dphi = np.cos(solar_declination) * np.sin(solar_hour_angle)
    lunar_dphi = np.cos(lunar_declination) * np.sin(lunar_hour_angle)
    # factors for sun and moon using IAU estimates of mass ratios
    GM_solar = kwargs["mass_ratio_solar"] * kwargs["GM"]
    GM_lunar = kwargs["mass_ratio_lunar"] * kwargs["GM"]
    # gravitational parameters
    K_solar = GM_solar * rad_e / np.power(solar_radius, 2)
    K_lunar = GM_lunar * rad_e / np.power(lunar_radius, 2)

    # components of attraction for the sun and moon
    F_solar = xr.Dataset()
    F_lunar = xr.Dataset()
    # initialize output generating forces
    for key in ["R", "N", "E"]:
        F_solar[key] = xr.zeros_like(solar_zenith)
        F_lunar[key] = xr.zeros_like(lunar_zenith)

    # for each spherical harmonic degree
    for l in range(2, kwargs["lmax"] + 1):
        # update gravitational parameter for degree
        K_solar *= rad_e / solar_radius
        K_lunar *= rad_e / lunar_radius
        # legendre polynomial for degree
        Pl_solar, dPl_solar = pyTMD.math.legendre(l, solar_zenith)
        Pl_lunar, dPl_lunar = pyTMD.math.legendre(l, lunar_zenith)
        # divide differential by u
        # ignore divide by zero and invalid value warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            dPl_solar /= np.sqrt(1 - solar_zenith**2)
            dPl_lunar /= np.sqrt(1 - lunar_zenith**2)
        # radial (up)
        F_solar["R"] += (K_solar * l / rad_e) * Pl_solar
        F_lunar["R"] += (K_lunar * l / rad_e) * Pl_lunar
        # north
        F_solar["N"] -= (K_solar / rad_e) * dPl_solar * solar_dtheta
        F_lunar["N"] -= (K_lunar / rad_e) * dPl_lunar * lunar_dtheta
        # east
        F_solar["E"] += (K_solar / rad_e) * dPl_solar * solar_dphi
        F_lunar["E"] += (K_lunar / rad_e) * dPl_lunar * lunar_dphi

    # sum solar and lunar components
    # add units attributes to output dataset
    F = xr.Dataset()
    for key in ["R", "N", "E"]:
        F[key] = F_solar[key] + F_lunar[key]
        F[key].attrs["units"] = "m/s^2"

    # return the tide generating force
    return F


def earth_orientation(t: np.ndarray, deltat: float | np.ndarray = 0.0):
    """
    Compute the variations in earth rotation caused by diurnal
    and semidiurnal tides :cite:p:`Herring:1994ku,Ray:1994dk`

    Parameters
    ----------
    t: np.ndarray
        Days relative to 1992-01-01T00:00:00
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)

    Returns
    -------
    ds: xr.Dataset
        Dataset containing:

        - ``dX``: anomaly in polar motion in X (arcseconds)
        - ``dY``: anomaly in polar motion in Y (arcseconds)
        - ``dUT``: anomaly in UT1-TAI (seconds)
    """
    # convert dates to Modified Julian Days
    MJD = t + _mjd_tide
    # convert to centuries relative to 2000-01-01T12:00:00
    T = (MJD + deltat - _mjd_j2000) / _century
    # 360 degrees in arcseconds
    circle = 1296000
    # compute the Delaunay arguments (IERS conventions)
    l, lp, F, D, omega = pyTMD.astro.delaunay_arguments(MJD + deltat)
    # convert from radians to arcseconds
    l = pyTMD.math.rad2asec(l)
    lp = pyTMD.math.rad2asec(lp)
    F = pyTMD.math.rad2asec(F)
    D = pyTMD.math.rad2asec(D)
    omega = pyTMD.math.rad2asec(omega)
    # angle of Greenwich Mean Standard Time (fractions of day)
    GMST = np.array([24110.54841, 8640184.812866, 9.3104e-2, -6.2e-6])
    gmst = (1.0 / 86400.0) * pyTMD.math.normalize_angle(
        pyTMD.math.polynomial_sum(GMST, T), circle=86400.0
    )
    # Greenwich Hour Angle (GHA) in arcseconds
    gha = circle * (gmst + _century * T + 0.5)
    # IERS conventions: gamma = GHA + 180 degrees
    gamma = gha + circle / 2.0
    # variable for multiples of 90 degrees (Ray technical note 2017)
    K = circle / 4.0 + np.zeros_like((MJD))
    # data array of arguments
    arguments = xr.DataArray(
        np.c_[l, lp, F, D, omega, gamma, K],
        dims=["time", "argument"],
        coords=dict(
            time=np.atleast_1d(MJD),
            argument=["l", "lp", "F", "D", "omega", "gamma", "k"],
        ),
    )
    # major constituents in Ray (1994) and latest IERS conventions
    constituents = [
        "2q1",
        "sigma1",
        "q1",
        "rho1",
        "o1",
        "tau1",
        "m1",
        "chi1",
        "pi1",
        "p1",
        "s1",
        "k1",
        "psi1",
        "phi1",
        "theta1",
        "j1",
        "so1",
        "oo1",
        "ups1",
        "2n2",
        "mu2",
        "n2",
        "nu2",
        "m2",
        "lambda2",
        "l2",
        "t2",
        "s2",
        "r2",
        "k2",
    ]
    # table of coefficients [l, lp, F, D, omega, gamma, K]
    delaunay_table = np.zeros((7, 30))
    delaunay_table[:, 0] = [-2, 0, -2, 0, -2, 1, -1]  # 2q1
    delaunay_table[:, 1] = [0, 0, -2, -2, -2, 1, -1]  # sigma1
    delaunay_table[:, 2] = [-1, 0, -2, 0, -2, 1, -1]  # q1
    delaunay_table[:, 3] = [1, 0, -2, -2, -2, 1, -1]  # rho1
    delaunay_table[:, 4] = [0, 0, -2, 0, -2, 1, -1]  # o1
    delaunay_table[:, 5] = [0, 0, 0, -2, 0, 1, 1]  # tau1
    delaunay_table[:, 6] = [-1, 0, 0, 0, 0, 1, 1]  # m1
    delaunay_table[:, 7] = [1, 0, 0, -2, 0, 1, 1]  # chi1
    delaunay_table[:, 8] = [0, -1, -2, 2, -2, 1, -1]  # pi1
    delaunay_table[:, 9] = [0, 0, -2, 2, -2, 1, -1]  # p1
    delaunay_table[:, 10] = [0, -1, 0, 0, 0, 1, 2]  # s1
    delaunay_table[:, 11] = [0, 0, 0, 0, 0, 1, 1]  # k1
    delaunay_table[:, 12] = [0, 1, 0, 0, 0, 1, 1]  # psi1
    delaunay_table[:, 13] = [0, 0, 2, -2, 2, 1, 1]  # phi1
    delaunay_table[:, 14] = [-1, 0, 0, 2, 0, 1, 1]  # theta1
    delaunay_table[:, 15] = [1, 0, 0, 0, 0, 1, 1]  # j1
    delaunay_table[:, 16] = [0, 0, 0, 2, 0, 1, 1]  # so1
    delaunay_table[:, 17] = [0, 0, 2, 0, 2, 1, 1]  # oo1
    delaunay_table[:, 18] = [1, 0, 2, 0, 2, 1, 1]  # ups1
    delaunay_table[:, 19] = [-2, 0, -2, 0, -2, 2, 0]  # 2n2
    delaunay_table[:, 20] = [0, 0, -2, -2, -2, 0, 0]  # mu2
    delaunay_table[:, 21] = [-1, 0, -2, 0, -2, 2, 0]  # n2
    delaunay_table[:, 22] = [1, 0, -2, -2, -2, 2, 0]  # nu2
    delaunay_table[:, 23] = [0, 0, -2, 0, -2, 2, 0]  # m2
    delaunay_table[:, 24] = [-1, 0, -2, 2, -2, 2, 2]  # lambda2
    delaunay_table[:, 25] = [1, 0, -2, 0, -2, 2, 2]  # l2
    delaunay_table[:, 26] = [0, -1, -2, 2, -2, 2, 0]  # t2
    delaunay_table[:, 27] = [0, 0, -2, 2, -2, 2, 0]  # s2
    delaunay_table[:, 28] = [0, 1, -2, 2, -2, 2, 2]  # r2
    delaunay_table[:, 29] = [0, 0, 0, 0, 0, 2, 0]  # k2
    # convert to data array of coefficients
    delaunay_table = xr.DataArray(
        delaunay_table,
        dims=["argument", "constituent"],
        coords=dict(
            argument=["l", "lp", "F", "D", "omega", "gamma", "k"],
            constituent=constituents,
        ),
    )
    # EOP corrections table [dX, dY, dUT]
    dEOP = np.zeros((3, 30), dtype=np.complex128)
    dEOP[:, 0] = [0.0003 - 0.0034j, -0.0034 - 0.0003j, 0.0103 + 0.0031j]
    dEOP[:, 1] = [0.0005 - 0.0042j, -0.0041 - 0.0005j, 0.0119 + 0.0039j]
    dEOP[:, 2] = [0.0062 - 0.0263j, -0.0263 - 0.0062j, 0.0512 + 0.0250j]
    dEOP[:, 3] = [0.0013 - 0.0050j, -0.0050 - 0.0013j, 0.0097 + 0.0047j]
    dEOP[:, 4] = [0.0488 - 0.1329j, -0.1329 - 0.0488j, 0.1602 + 0.1207j]
    dEOP[:, 5] = [-0.0007 + 0.0017j, 0.0017 + 0.0007j, -0.0019 - 0.0007j]
    dEOP[:, 6] = [-0.0045 + 0.0096j, 0.0096 + 0.0045j, -0.0086 - 0.0075j]
    dEOP[:, 7] = [-0.0009 + 0.0018j, 0.0018 + 0.0009j, -0.0016 - 0.0014j]
    dEOP[:, 8] = [0.0015 - 0.0030j, -0.0030 - 0.0015j, 0.0031 + 0.0019j]
    dEOP[:, 9] = [0.0261 - 0.0512j, -0.0512 - 0.0261j, 0.0551 + 0.0310j]
    dEOP[:, 10] = [0.0006 + 0.0012j, -0.0012 + 0.0006j, 0.0007 + 0.0013j]
    dEOP[:, 11] = [-0.0775 + 0.1517j, -0.1517 - 0.0775j, 0.1762 + 0.0855j]
    dEOP[:, 12] = [-0.0006 + 0.0012j, 0.0012 + 0.0006j, -0.0014 - 0.0006j]
    dEOP[:, 13] = [-0.0011 + 0.0021j, 0.0021 + 0.0011j, -0.0027 - 0.0011j]
    dEOP[:, 14] = [-0.0007 + 0.0014j, 0.0014 + 0.0007j, -0.0029 - 0.0004j]
    dEOP[:, 15] = [-0.0035 + 0.0073j, 0.0073 + 0.0035j, -0.0019 - 0.0161j]
    dEOP[:, 16] = [-0.0004 + 0.0011j, 0.0011 + 0.0004j, -0.0041 + 0.0001j]
    dEOP[:, 17] = [-0.0011 + 0.0034j, 0.0034 + 0.0011j, -0.0144 + 0.0004j]
    dEOP[:, 18] = [0.0000 + 0.0006j, 0.0006 + 0.0000j, -0.0040 + 0.0002j]
    dEOP[:, 19] = [-0.0016 - 0.0061j, 0.0034 + 0.0031j, -0.0018 - 0.0064j]
    dEOP[:, 20] = [-0.0020 - 0.0076j, 0.0042 + 0.0034j, -0.0022 - 0.0074j]
    dEOP[:, 21] = [-0.0129 - 0.0569j, 0.0329 + 0.0111j, -0.0156 - 0.0379j]
    dEOP[:, 22] = [-0.0024 - 0.0110j, 0.0064 + 0.0019j, -0.0030 - 0.0070j]
    dEOP[:, 23] = [-0.0270 - 0.3302j, 0.1959 + 0.0376j, -0.0725 - 0.1619j]
    dEOP[:, 24] = [0.0003 - 0.0025j, 0.0015 + 0.0004j, -0.0003 - 0.0011j]
    dEOP[:, 25] = [0.0014 - 0.0094j, 0.0056 + 0.0019j, -0.0012 - 0.0042j]
    dEOP[:, 26] = [0.0035 - 0.0085j, 0.0051 + 0.0033j, -0.0002 - 0.0044j]
    dEOP[:, 27] = [0.0636 - 0.1441j, 0.0866 + 0.0592j, -0.0016 - 0.0755j]
    dEOP[:, 28] = [0.0006 - 0.0012j, 0.0007 + 0.0005j, -0.0000 - 0.0006j]
    dEOP[:, 29] = [0.0191 - 0.0385j, 0.0231 + 0.0177j, -0.0004 - 0.0210j]
    # convert to data array of EOP corrections
    dEOP = xr.DataArray(
        dEOP,
        dims=["EOP", "constituent"],
        coords=dict(
            EOP=["dX", "dY", "dUT"],
            constituent=constituents,
        ),
    )
    # calculate phase of arguments (arcseconds)
    G = arguments.dot(delaunay_table)
    # convert from arcseconds to complex phase in radians
    phase = np.exp(1j * pyTMD.math.asec2rad(G))
    # calculate EOP corrections
    corrections = (dEOP.real * phase.real + dEOP.imag * phase.imag).sum(
        dim="constituent"
    )
    # create output dataset from data arrays
    ds = xr.Dataset()
    ds["dX"] = 1e-3 * corrections.sel(EOP="dX")
    ds["dX"].attrs["units"] = "arcseconds"
    ds["dX"].attrs["long_name"] = "anomaly in polar motion in X"
    ds["dY"] = 1e-3 * corrections.sel(EOP="dY")
    ds["dY"].attrs["units"] = "arcseconds"
    ds["dY"].attrs["long_name"] = "anomaly in polar motion in Y"
    ds["dUT"] = 1e-4 * corrections.sel(EOP="dUT")
    ds["dUT"].attrs["units"] = "seconds"
    ds["dUT"].attrs["long_name"] = "anomaly in UT1-TAI"
    # return the variations in earth rotation
    return ds


# PURPOSE: estimate variations in length of day
def length_of_day(t: np.ndarray, deltat: float | np.ndarray = 0.0):
    """
    Compute the variations in earth rotation caused by long-period (zonal)
    tides :cite:p:`Ray:2014fu`

    Parameters
    ----------
    t: np.ndarray
        Days relative to 1992-01-01T00:00:00
    deltat: float or np.ndarray, default 0.0
        Time correction for converting to Ephemeris Time (days)

    Returns
    -------
    ds: xr.Dataset
        Dataset containing:

        - ``dUT``: anomaly in UT1-TAI (seconds)
        - ``dLOD``: excess LOD (seconds per day)
        - ``period``: period of constituent (days)
    """
    # convert dates to Modified Julian Days
    MJD = t + _mjd_tide
    # compute astronomical arguments
    s, h, p, n, pp = pyTMD.astro.mean_longitudes(MJD + deltat, method="ASTRO5")
    # initial time conversions
    hour = 24.0 * np.mod(MJD, 1)
    # convert from hours solar time into mean lunar time in degrees
    tau = 15.0 * hour - s + h
    # variable for multiples of 90 degrees (Ray technical note 2017)
    k = 90.0 + 0.0 * MJD
    # dataset of arguments
    # note the sign change to go from N to N'
    arguments = xr.Dataset(
        data_vars=dict(
            tau=(["time"], tau),
            s=(["time"], s),
            h=(["time"], h),
            p=(["time"], p),
            n=(["time"], -n),
            pp=(["time"], pp),
            k=(["time"], k),
        ),
        coords=dict(time=np.atleast_1d(MJD)),
    ).to_dataarray(dim="argument")
    # parse rotation rate table from Ray and Erofeeva (2014)
    ZROT = pyTMD.constituents._parse_rotation_rate_table()
    # Doodson coefficients
    coefficients = xr.DataArray(
        np.array(
            [
                ZROT["tau"],
                ZROT["s"],
                ZROT["h"],
                ZROT["p"],
                ZROT["n"],
                ZROT["pp"],
                ZROT["k"],
            ]
        ),
        dims=["argument", "constituent"],
        coords=dict(argument=["tau", "s", "h", "p", "n", "pp", "k"]),
    )
    # equilibrium phase converted to radians
    G = np.radians(arguments.dot(coefficients))
    # create output dataset
    ds = xr.Dataset(coords=dict(time=np.atleast_1d(MJD)))
    # compute delta UT1-TAI (seconds)
    dUT = ZROT["UTc"] * np.cos(G) + ZROT["UTs"] * np.sin(G)
    ds["dUT"] = 1e-6 * dUT
    ds["dUT"].attrs["units"] = "seconds"
    ds["dUT"].attrs["long_name"] = "anomaly in UT1-TAI"
    # compute delta LOD (seconds per day)
    dLOD = ZROT["dLODc"] * np.cos(G) + ZROT["dLODs"] * np.sin(G)
    ds["dLOD"] = 1e-6 * dLOD
    ds["dLOD"].attrs["units"] = "seconds per day"
    ds["dLOD"].attrs["long_name"] = "excess length of day"
    # period of constituent (days)
    ds["period"] = ("constituent", ZROT["period"])
    ds["period"].attrs["units"] = "days"
    # return the variations in earth rotation
    return ds
