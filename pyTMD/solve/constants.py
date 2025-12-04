#!/usr/bin/env python
u"""
constants.py
Written by Tyler Sutterley (12/2025)
Routines for estimating the harmonic constants for ocean tides

REFERENCES:
    G. D. Egbert and S. Erofeeva, "Efficient Inverse Modeling of Barotropic
        Ocean Tides", Journal of Atmospheric and Oceanic Technology, (2002).
    M. G. G. Foreman, J. Y. Cherniawsky, and V. A. Ballantyne,
        "Versatile Harmonic Tidal Analysis: Improvements and Applications",
        Journal of Atmospheric and Oceanic Technology, (2009).
    R. D. Ray, "A global ocean tide model from TOPEX/POSEIDON altimetry:
        GOT99.2", NASA Technical Memorandum 209478, (1999).   

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://scipy.org

PROGRAM DEPENDENCIES:
    astro.py: computes the basic astronomical mean longitudes
    constituents.py: calculates constituent parameters and nodal arguments  
    predict.py: predict tide values using harmonic constants

UPDATE HISTORY:
    Updated 12/2025: merge minor constituent inference into main iteration loop
    Updated 11/2025: output as xarray Dataset of complex harmonic constants
    Updated 10/2025: added option to infer minor constituents (post-fit)
    Updated 08/2025: use numpy degree to radian conversions
    Updated 06/2025: verify that height values are all finite
    Updated 05/2025: added option to include higher order polynomials
    Updated 09/2024: added bounded options for least squares solvers
    Updated 08/2024: use nodal arguments for all non-OTIS model type cases
    Updated 01/2024: moved to solve subdirectory
    Written 12/2023
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import xarray as xr
import scipy.optimize
import pyTMD.constituents
import pyTMD.predict

__all__ = [
    'constants'
]

# number of days between MJD and the tide epoch (1992-01-01T00:00:00)
_mjd_tide = 48622.0

def constants(t: float | np.ndarray,
        ht: np.ndarray,
        constituents: str | list | np.ndarray,
        deltat: float | np.ndarray = 0.0,
        corrections: str = 'OTIS',
        solver: str = 'lstsq',
        order: int = 0,
        infer_minor: bool = False,
        minor_constituents: list = [],
        bounds: tuple = (-np.inf, np.inf),
        max_iter: int | None = None,
        infer_iter: int = 1
    ):
    """
    Estimate the harmonic constants for a time series
    :cite:p:`Egbert:2002ge,Foreman:2009bg,Ray:1999vm`

    Parameters
    ----------
    t: float or np.ndarray
        days relative to 1992-01-01T00:00:00
    ht: np.ndarray
        input time series (elevation or currents)
    constituents: str, list or np.ndarray
        tidal constituent ID(s)
    deltat: float or np.ndarray, default 0.0
        time correction for converting to Ephemeris Time (days)
    corrections: str, default 'OTIS'
        use nodal corrections from OTIS/ATLAS or GOT/FES models
    solver: str, default 'lstsq'
        least squares solver to use

        - ``'lstsq'``: least squares solution
        - ``'gelsy'``: complete orthogonal factorization
        - ``'gelss'``: singular value decomposition (SVD)
        - ``'gelsd'``: SVD with divide and conquer method
        - ``'bvls'``: bounded-variable least-squares
    order: int, default 0
        degree of the polynomial to add to fit
    infer_minor: bool, default False
        infer minor tidal constituents
    minor_constituents: list or None, default None
        Specify constituents to infer
    bounds: tuple, default (None, None)
        Lower and upper bounds on parameters for ``'bvls'``
    max_iter: int or None, default None
        Maximum number of iterations for ``'bvls'``
    infer_iter: int, default 1
        Maximum number of iterations for inferring minor constituents

    Returns
    -------
    ds: xr.Dataset
        xarray dataset of complex harmonic constants
    """
    # check if input constituents is a string
    if isinstance(constituents, str):
        constituents = [constituents]
    # verify height and time variables
    t = np.ravel(t)
    ht = np.ravel(ht)
    # reduce height and time variables to finite values
    if not np.isfinite(ht).all():
        valid, = np.nonzero(np.isfinite(t) & np.isfinite(ht))
        t = t[valid]
        ht = ht[valid]
    # check that there are enough values for a time series fit
    nt = len(t)
    nc = len(constituents)
    if (nt <= 2*nc):
        raise ValueError('Not enough values for fit')
    # check that the number of time values matches the number of height values
    if (nt != len(ht)):
        raise ValueError('Dimension mismatch between input variables')

    # load the nodal corrections
    # convert time to Modified Julian Days (MJD)
    pu, pf, G = pyTMD.constituents.arguments(t + _mjd_tide, constituents,
        deltat=deltat, corrections=corrections)

    # create design matrix
    M = []
    # build polynomial functions for design matrix
    for o in range(order+1):
        # add polynomial term
        M.append(np.power(t, o))
    # add constituent terms
    for k,c in enumerate(constituents):
        if corrections in ('OTIS', 'ATLAS', 'TMD3', 'netcdf'):
            amp, ph, omega, alpha, species = \
                pyTMD.constituents._constituent_parameters(c)
            th = omega*t*86400.0 + ph + pu[:,k]
        else:
            th = np.radians(G[:,k]) + pu[:,k]
        # add constituent to design matrix
        M.append(pf[:,k]*np.cos(th))
        M.append(-pf[:,k]*np.sin(th))
    # take the transpose of the design matrix
    M = np.transpose(M)

    # initial heights for fits
    hcorr = np.copy(ht)
    # parameter array
    p = np.zeros(M.shape[1])
    # total number of fit iterations
    # first iteration: solve for major constituents
    # all others: remove contribution of minor constituents
    iterations = np.maximum(2, 1 + infer_iter) if infer_minor else 1
    # for each solution iteration
    for i in range(iterations):
        # remove inferred minor constituents from height time series
        if (i > 0) and infer_minor:
            # indices for the sine and cosine terms
            # skip over the polynomial terms
            isin = 2*np.arange(nc) + order + 2
            icos = 2*np.arange(nc) + order + 1
            # complex model amplitudes for major constituents
            darr = xr.DataArray(p[icos] + 1j*p[isin],
                coords=dict(constituent=constituents),
                dims=['constituent'])
            ds = darr.to_dataset(dim='constituent')
            # inferred minor constituent time series
            hminor = pyTMD.predict.infer_minor(t, ds,
                deltat=deltat, corrections=corrections,
                minor=minor_constituents)
            # corrected height (without minor constituents)
            hcorr = ht - hminor.values
        # use a least-squares fit to solve for parameters
        # can optionally use a bounded-variable least-squares fit
        if (solver == 'lstsq'):
            p[:], res, rnk, s = np.linalg.lstsq(M, hcorr, rcond=-1)
        elif solver in ('gelsd', 'gelsy', 'gelss'):
            p[:], res, rnk, s = scipy.linalg.lstsq(M, hcorr,
                lapack_driver=solver)
        elif (solver == 'bvls'):
            p[:] = scipy.optimize.lsq_linear(M, hcorr,
                method=solver, bounds=bounds,
                max_iter=max_iter).x

    # calculate amplitude and phase for each constituent
    ds = xr.Dataset(coords=dict(constituent=constituents))
    # for each constituent
    for k,c in enumerate(constituents):
        # indices for the sine and cosine terms
        # skip over the polynomial terms
        isin = 2*k + order + 2
        icos = 2*k + order + 1
        ds[c] = xr.DataArray(p[icos] + 1j*p[isin])
    # return the xarray dataset
    return ds
