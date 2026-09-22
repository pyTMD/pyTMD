#!/usr/bin/env python
"""
potential.py
Written by Tyler Sutterley (09/2026)
Spherical harmonic expansions and summations

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

PROGRAM DEPENDENCIES:
    constituents.py: calculates constituent parameters and nodal arguments
    earth.py: calculates Earth parameters and Body Tide Love numbers
    math.py: Special functions of mathematical physics
    spatial.py: utilities for working with geospatial data

UPDATE HISTORY:
    Updated 09/2026: added function to estimate sea water densities
    Written 09/2026
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import pyTMD.constituents
import pyTMD.earth
import pyTMD.math
import pyTMD.spatial

__all__ = [
    "ocean_harmonics",
    "crustal_loading",
    "_seawater_density",
]

# tables of load Love/Shida numbers
_lln_table = {}
_lln_table["han-wahr"] = pyTMD.earth._han_wahr_lln_table
_lln_table["gegout"] = pyTMD.earth._gegout_lln_table
_lln_table["wang-prem"] = pyTMD.earth._wang_prem_lln_table
# earth and physical parameters for WGS84 ellipsoid
# using meters-kilogram-seconds standard
_wgs84 = pyTMD.earth.datum(ellipsoid="WGS84", units="MKS")


# PURPOSE: convert tidal constituents into spherical harmonics
def ocean_harmonics(
    ds: xr.Dataset,
    lmax: int = 696,
    a_axis: float = _wgs84.a_axis,
    flat: float = _wgs84.flat,
    GM: float = _wgs84.GM,
    rho_w: float | xr.DataArray = 1025.0,
    lln: str = "han-wahr",
    reference: str = "CE",
    **kwargs,
):
    r"""
    Converts ocean tide model constituents into spherical
    harmonic coefficients :cite:p:`Petit:2010tp,Ray:1989wf,Wahr:1998hy`

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing tidal harmonic constants
    lmax: int, default 696
        Upper bound of spherical harmonic degrees
    a_axis: float, default 6378136.3
        Semi-major axis of the Earth (meters)
    flat: float, default 1.0/298.257223563
        Ellipsoidal flattening
    GM: float, default 3.986004418e14
        Geocentric gravitational constant (m\ :sup:`3` s\ :sup:`-2`)
    rho_w: float or xarray.DataArray, default 1025.0
        Density of sea water (kg m\ :sup:`-3`)

        Can be spatially uniform or a grid matching the tidal dataset
    lln: str, default 'han-wahr'
        name of the Load Love number dataset to use

            - ``'han-wahr'``: :cite:t:`Han:1995go`
            - ``'gegout'``: :cite:t:`Gegout:2010gc`
            - ``'wang-prem'``: :cite:t:`Wang:2012gc`
    reference: str, default 'CE'
        Reference frame of degree 1 load Love numbers

            - ``'CF'``: Center of Surface Figure
            - ``'CL'``: Center of Surface Lateral Figure
            - ``'CH'``: Center of Surface Height Figure
            - ``'CM'``: Center of Mass of Earth System
            - ``'CE'``: Center of Mass of Solid Earth

    Returns
    -------
    Ylms: xr.Dataset
        Fully-normalized spherical harmonic coefficients

            - ``alm``: tidal constituent (in-phase) components
            - ``blm``: tidal constituent (out-of-phase) components
    """
    # verify units of input data are in meters
    ds = ds.tmd.to_units("meters")
    # verify coordinate reference system (global and geographic)
    if not ds.tmd.is_global:
        raise ValueError("Unsupported coordinate reference system")
    # verify load Love number table name
    if lln not in _lln_table.keys():
        raise ValueError(f"Unknown load Love number dataset {lln}")
    # verify maximum degree and order
    lmax = int(lmax)
    if lmax < 2:
        raise ValueError("Degree of truncation should be at least 2")
    # list of tidal constituents
    constituents = ds.tmd.constituents
    nc = len(constituents)
    # angular frequencies for constituents
    omega = pyTMD.constituents.frequency(constituents, **kwargs)

    # Earth parameters and physical constants
    # universal gravitational constant [N*m^2/kg^2]
    G = 6.67430e-11
    # average radius of the Earth with same volume as ellipsoid [m]
    rad_e = a_axis * np.power(1.0 - flat, 1.0 / 3.0)
    # average density of the Earth [kg / m^3]
    rho_e = 0.75 * GM / (G * np.pi * rad_e**3)

    # convert from geodetic latitude to geocentric latitude
    geolat = pyTMD.spatial.geocentric_latitude(ds.y, flat=flat)
    # calculate colatitude and longitude (radians)
    theta = np.radians(90.0 - geolat)
    lmda = np.radians(ds.x)
    # convert longitudes to range 0:360 (if previously -180:180)
    lmda = lmda.where(lmda >= 0, lmda + 2.0 * np.pi, drop=False)
    # multiply sin(th) with differentials of theta and lambda
    # to calculate the integration factor at each latitude
    dlam = np.abs(lmda[1] - lmda[0])
    dth = np.abs(theta[1] - theta[0])
    int_fact = np.sin(theta) * dlam * dth

    # spherical harmonic degree and order
    l = np.arange(lmax + 1)
    m = np.arange(lmax + 1)
    # calculate polynomials using Martin Mohlenkamp's relation
    Plm, dPlm = pyTMD.math.legendreP(lmax, np.cos(theta))
    # read load Love numbers from table
    hl, kl, ll = pyTMD.earth.load_love_numbers(
        _lln_table[lln], reference=reference, lmax=lmax
    )
    # convert to xarray data arrays
    Plm = xr.DataArray(
        Plm,
        dims=("l", "m", "y"),
        coords={"l": l, "m": m, "y": ds.y},
    )
    kl = xr.DataArray(
        kl,
        dims=("l",),
        coords={"l": l},
    )
    # allocate for frequency-dependent load Love numbers adjustments
    dk = xr.DataArray(
        np.zeros((lmax + 1, lmax + 1)),
        dims=("l", "m"),
        coords={"l": l, "m": m},
    )

    # calculate cos/sin of lambda arrays using Euler's formula
    m_lmda = np.exp(1j * Plm.m.dot(lmda))
    # integration coefficients for converting to spherical harmonics
    int_coeff = int_fact * Plm

    # allocate for output spherical harmonics
    # alm: in-phase (real) parts of the tidal constituents
    # blm: out-of-phase (imaginary) parts of the tidal constituents
    alm = np.zeros((lmax + 1, lmax + 1, nc), dtype=np.complex128)
    blm = np.zeros((lmax + 1, lmax + 1, nc), dtype=np.complex128)
    # for each constituent
    for i, c in enumerate(constituents):
        # get constituent and replace nans with 0
        data = ds[c].fillna(0.0)
        # multiply gridded data with sin/cos of m#lambda
        # sum through all lambdas in the dot product
        # multiply heights by sea water density
        # sea water density can be a scalar value for uniform
        # or a map of the column averages
        d_real = m_lmda.dot(rho_w * data.real)
        d_imag = m_lmda.dot(rho_w * data.imag)
        # adjust load Love numbers for frequency dependence
        dh, dk[2, 1], dl = pyTMD.earth.adjust_load_love_numbers(omega[i])
        # degree dependent factors for converting from water equivalent
        # taking into account frequency dependence of load Love numbers
        # modified from Wahr et al., (2018)
        dfactor = (
            3.0
            * (1.0 + kl + dk)
            / (1.0 + 2.0 * Plm.l)
            / (4.0 * np.pi * rad_e * rho_e)
        )
        # integrate over all latitudes
        # fully-normalize output spherical harmonics
        alm[:, :, i] = dfactor * int_coeff.dot(d_real, dim="y")
        blm[:, :, i] = dfactor * int_coeff.dot(d_imag, dim="y")
    # convert to xarray dataset
    Ylms = xr.Dataset(
        data_vars=dict(
            alm=(["l", "m", "constituent"], alm),
            blm=(["l", "m", "constituent"], blm),
        ),
        coords={"l": l, "m": m, "constituent": constituents},
    )
    # add attributes for dimensions
    Ylms.l.attrs["long_name"] = "spherical harmonic degree"
    Ylms.m.attrs["long_name"] = "spherical harmonic order"
    Ylms.l.attrs["standard_name"] = "degree"
    Ylms.m.attrs["standard_name"] = "order"
    Ylms.l.attrs["units"] = "wavenumber"
    Ylms.m.attrs["units"] = "wavenumber"
    # add attributes for spherical harmonics
    Ylms.alm.attrs["long_name"] = "complex spherical harmonics (in-phase)"
    Ylms.blm.attrs["long_name"] = "complex spherical harmonics (out-of-phase)"
    Ylms.alm.attrs["description"] = (
        "spherical harmonic coefficients containing the "
        "real (in-phase) part of the tidal constituents"
    )
    Ylms.blm.attrs["description"] = (
        "spherical harmonic coefficients containing the "
        "imaginary (out-of-phase) part of the tidal constituents"
    )
    # copy attributes from original dataset
    Ylms.attrs.update(ds.attrs)
    Ylms.attrs["product_type"] = "gravity_field"
    Ylms.attrs["normalization"] = "fully-normalized"
    # add attributes for degree of truncation
    Ylms.attrs["max_degree"] = lmax
    Ylms.attrs["max_order"] = lmax
    # add attributes for earth model and love numbers
    Ylms.attrs["earth_love_numbers"] = lln
    Ylms.attrs["reference_frame"] = reference
    # add attributes for earth and model parameters
    Ylms.attrs["earth_radius"] = f"{rad_e:0.3f} m"
    Ylms.attrs["earth_density"] = f"{rho_e:0.3f} kg/m^3"
    Ylms.attrs["earth_inverse_flattening"] = f"{1.0 / flat:0.3f}"
    Ylms.attrs["earth_gravity_constant"] = f"{GM:0.3f} m^3/s^2"
    # add attribute for seawater density (uniform or gridded)
    if isinstance(rho_w, float):
        Ylms.attrs["seawater_density"] = f"{rho_w:0.3f} kg/m^3"
    else:
        Ylms.attrs["seawater_density"] = "gridded"
    # check if chunks were present in original dataset
    if hasattr(ds, "chunks") and ds.chunks is not None:
        Ylms = Ylms.chunk("auto")
    # return the spherical harmonic dataset
    return Ylms


# PURPOSE: convert spherical harmonics into load tide maps
def crustal_loading(
    ds: xr.Dataset,
    Ylms: xr.Dataset,
    lmax: int | None = None,
    a_axis: float = _wgs84.a_axis,
    flat: float = _wgs84.flat,
    lln: str = "han-wahr",
    reference: str = "CE",
    **kwargs,
):
    r"""
    Calculates the crustal deformation induced by ocean tide loading
    via spherical harmonic summation :cite:p:`Desai:2014ee,Ray:2025uo`

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset with spatial coordinates
    Ylms: xarray.Dataset
        Dataset with spherical harmonic coefficients

            - ``alm``: tidal constituent in-phase components
            - ``blm``: tidal constituent out-of-phase components
    lmax: int or None, default None
        Upper bound of spherical harmonic degrees
    a_axis: float, default 6378136.3
        Semi-major axis of the Earth (meters)
    flat: float, default 1.0/298.257223563
        Ellipsoidal flattening
    lln: str, default 'han-wahr'
        name of the Load Love number dataset to use

            - ``'han-wahr'``: :cite:t:`Han:1995go`
            - ``'gegout'``: :cite:t:`Gegout:2010gc`
            - ``'wang-prem'``: :cite:t:`Wang:2012gc`
    reference: str, default 'CE'
        Reference frame of degree 1 load Love numbers

            - ``'CF'``: Center of Surface Figure
            - ``'CL'``: Center of Surface Lateral Figure
            - ``'CH'``: Center of Surface Height Figure
            - ``'CM'``: Center of Mass of Earth System
            - ``'CE'``: Center of Mass of Solid Earth

    Returns
    -------
    ds: xr.Dataset
        Dataset containing tidal harmonic constants
    """
    # verify load Love number table name
    if lln not in _lln_table.keys():
        raise ValueError(f"Unknown load Love number dataset {lln}")
    # verify maximum degree and order
    if lmax is None:
        lmax = int(Ylms.l.max().values)
    if lmax < 2:
        raise ValueError("Degree of truncation should be at least 2")
    # truncate to degree and order lmax
    if lmax < Ylms.l.max():
        Ylms = Ylms.where((Ylms.l <= lmax) & (Ylms.m <= lmax), drop=True)
    # extract harmonics and convert to datasets
    alm = Ylms.alm.conj().to_dataset(dim="constituent")
    blm = Ylms.blm.conj().to_dataset(dim="constituent")
    # list of tidal constituents
    constituents = alm.tmd.constituents
    # angular frequencies for constituents
    omega = pyTMD.constituents.frequency(constituents, **kwargs)

    # Earth parameters and physical constants
    # average radius of the Earth with same volume as ellipsoid [m]
    rad_e = a_axis * np.power(1.0 - flat, 1.0 / 3.0)
    # convert from geodetic latitude to geocentric latitude
    geolat = pyTMD.spatial.geocentric_latitude(ds.y, flat=flat)
    # calculate colatitude and longitude (radians)
    theta = np.radians(90.0 - geolat)
    lmda = np.radians(ds.x)
    # convert longitudes to range 0:360 (if previously -180:180)
    lmda = lmda.where(lmda >= 0, lmda + 2.0 * np.pi, drop=False)

    # spherical harmonic degree and order
    l = np.arange(lmax + 1)
    m = np.arange(lmax + 1)
    # calculate polynomials using Martin Mohlenkamp's relation
    Plm, dPlm = pyTMD.math.legendreP(lmax, np.cos(theta))
    # read load Love numbers from table
    hl, kl, ll = pyTMD.earth.load_love_numbers(
        _lln_table[lln], reference=reference, lmax=lmax
    )
    # convert to xarray data arrays
    Plm = xr.DataArray(
        Plm,
        dims=("l", "m", "y"),
        coords={"l": l, "m": m, "y": ds.y},
    )
    hl = xr.DataArray(
        hl,
        dims=("l",),
        coords={"l": l},
    )
    # allocate for frequency-dependent load Love numbers adjustments
    dh = xr.DataArray(
        np.zeros((lmax + 1, lmax + 1)),
        dims=("l", "m"),
        coords={"l": l, "m": m},
    )

    # calculate cos/sin of lambda arrays using Euler's formula
    m_lmda = np.exp(1j * Plm.m.dot(lmda))

    # create output dataset
    tmp = xr.Dataset(coords=ds.coords)
    # for each constituent
    for i, c in enumerate(constituents):
        # adjust load Love numbers for frequency dependence
        dh[2, 1], dk, dl = pyTMD.earth.adjust_load_love_numbers(omega[i])
        # degree dependent factors for converting to crustal deformation
        # taking into account frequency dependence of load Love numbers
        dfactor = rad_e * (hl + dh)
        # summation over all spherical harmonic degrees
        p_real = Plm.dot(dfactor * alm[c], dim="l")
        p_imag = Plm.dot(dfactor * blm[c], dim="l")
        # summation of cosine and sine harmonics
        # (dropping the imaginary component)
        d_real = p_real.dot(m_lmda, dim="m").real
        d_imag = p_imag.dot(m_lmda, dim="m").real
        # summation of in-phase and out-of-phase components
        tmp[c] = d_real + 1j * d_imag

    # copy attributes from spherical harmonic dataset
    tmp.attrs.update(Ylms.attrs)
    # check if chunks were present in spherical harmonic dataset
    if hasattr(Ylms, "chunks") and Ylms.chunks is not None:
        tmp = tmp.chunk("auto")
    # return the spatial dataset of tidal constituents
    return tmp


def _seawater_density(
    temperature: np.ndarray,
    salinity: np.ndarray,
):
    r"""
    Calculates the density of sea water using the EOS-80 model
    :cite:p:`UNESCO:1981um,Fofonoff:1983wi`

    Parameters
    ----------
    temperature: np.ndarray
        Sea water temperature (\ |degree| C)
    salinity: np.ndarray
        Sea water salinity (unitness)

    Returns
    -------
    rho_w: np.ndarray
        Sea water density (kg/m\ :sup:`3`)

    .. |degree|    unicode:: U+00B0 .. DEGREE SIGN
    """
    # fresh water density at atmospheric pressure as function of temperature
    # coefficients from equation 14 of Fofonoff (1983) derived from Bigg (1967)
    a = np.array(
        [999.842594, 6.793952e-2, -9.095290e-3, -1.120083e-6, 6.536332e-9]
    )
    # polynomial coefficients involving salinity
    b = np.array([0.824493, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9])
    c = np.array([-5.72466e-3, 1.0227e-4, -1.6546e-6])
    d = np.array([4.8314e-4])
    # seawater density at atmospheric pressure
    # equation 13 of Fofonoff (1983)
    rho_w = (
        pyTMD.math.polynomial_sum(a, temperature)
        + np.power(salinity, 1.0) * pyTMD.math.polynomial_sum(b, temperature)
        + np.power(salinity, 1.5) * pyTMD.math.polynomial_sum(c, temperature)
        + np.power(salinity, 2.0) * pyTMD.math.polynomial_sum(d, temperature)
    )
    # return the sea water density
    return rho_w
