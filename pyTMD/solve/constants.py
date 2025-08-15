#!/usr/bin/env python
u"""
constants.py
Written by Tyler Sutterley (08/2025)
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
    arguments.py: loads nodal corrections for tidal constituents
    astro.py: computes the basic astronomical mean longitudes

UPDATE HISTORY:
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
import scipy.optimize
import pyTMD.arguments

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
        bounds: tuple = (-np.inf, np.inf),
        max_iter: int | None = None
    ):
    """
    Estimate the harmonic constants for an elevation time series
    :cite:p:`Egbert:2002ge,Foreman:2009bg,Ray:1999vm`

    Parameters
    ----------
    t: float or np.ndarray
        days relative to 1992-01-01T00:00:00
    ht: np.ndarray
        elevation time series (meters)
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
    bounds: tuple, default (None, None)
        Lower and upper bounds on parameters for ``'bvls'``
    max_iter: int or None, default None
        Maximum number of iterations for ``'bvls'``

    Returns
    -------
    amp: np.ndarray
        amplitude of each harmonic constant (meters)
    phase: np.ndarray
        phase of each harmonic constant (degrees)
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
    pu, pf, G = pyTMD.arguments.arguments(t + _mjd_tide, constituents,
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
                pyTMD.arguments._constituent_parameters(c)
            th = omega*t*86400.0 + ph + pu[:,k]
        else:
            th = np.radians(G[:,k]) + pu[:,k]
        # add constituent to design matrix
        M.append(pf[:,k]*np.cos(th))
        M.append(-pf[:,k]*np.sin(th))
    # take the transpose of the design matrix
    M = np.transpose(M)

    # use a least-squares fit to solve for parameters
    # can optionally use a bounded-variable least-squares fit
    if (solver == 'lstsq'):
        p, res, rnk, s = np.linalg.lstsq(M, ht, rcond=-1)
    elif solver in ('gelsd', 'gelsy', 'gelss'):
        p, res, rnk, s = scipy.linalg.lstsq(M, ht,
            lapack_driver=solver)
    elif (solver == 'bvls'):
        p = scipy.optimize.lsq_linear(M, ht,
            method=solver, bounds=bounds,
            max_iter=max_iter).x

    # calculate amplitude and phase for each constituent
    amp = np.zeros((nc))
    ph = np.zeros((nc))
    # for each constituent
    for k,c in enumerate(constituents):
        # indices for the sine and cosine terms
        # skip over the polynomial terms
        isin = 2*k + order + 2
        icos = 2*k + order + 1
        # calculate amplitude and phase
        amp[k] = np.abs(1j*p[isin] + p[icos])
        ph[k] = np.arctan2(-p[isin], p[icos])
    # convert phase to degrees
    phase = np.degrees(ph)
    phase[phase < 0] += 360.0

    # return the amplitude and phase
    return (amp, phase)
