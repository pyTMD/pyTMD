#!/usr/bin/env python
u"""
ellipse.py
Written by Tyler Sutterley (05/2025)
Expresses the amplitudes and phases for the u and v components in terms of
    four ellipse parameters using Foreman's formula

CALLING SEQUENCE:
    umajor,uminor,uincl,uphase = pyTMD.ellipse.ellipse(u,v)

INPUTS:
    u: zonal current (EW)
    v: meridional current (NS)

OUTPUTS:
    umajor: amplitude of the semimajor semi-axis
    uminor: amplitude of the semiminor semi-axis
    uincl: angle of inclination of the northern semimajor semi-axis
    uphase: phase lag of the maximum current behind the maximum tidal potential
        of the individual constituent

REFERENCE:
    M. G. G. Foreman and R. F. Henry, "The harmonic analysis of tidal model time
        series", Advances in Water Resources, 12(3), 109-120, (1989).
        https://doi.org/10.1016/0309-1708(89)90017-1

UPDATE HISTORY:
    Updated 06/2025: added function to calculate x and y coordinates of ellipse
    Updated 01/2024: added inverse function to get currents from parameters
        use complex algebra to calculate tidal ellipse parameters
    Updated 09/2023: renamed to ellipse.py (from tidal_ellipse.py)
    Updated 03/2023: add basic variable typing to function inputs
    Updated 04/2022: updated docstrings to numpy documentation format
    Written 07/2020
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ellipse",
    "inverse",
    "_xy"
]

def ellipse(u: np.ndarray, v: np.ndarray):
    """
    Expresses the amplitudes and phases for the u and v components in terms of
    four ellipse parameters using Foreman's formula :cite:p:`Foreman:1989dt`

    Parameters
    ----------
    u: np.ndarray
        zonal current (EW)
    v: np.ndarray
        meridional current (NS)

    Returns
    -------
    umajor: np.ndarray
        amplitude of the semi-major axis
    uminor: np.ndarray
        amplitude of the semi-minor axis
    uincl: np.ndarray
        angle of inclination of the northern semi-major axis
    uphase: np.ndarray
        phase lag of the maximum current behind the maximum tidal potential
    """
    # validate inputs
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    # wp, wm: complex radius of positively and negatively rotating vectors
    wp = (u + 1j*v)/2.0
    wm = np.conj(u - 1j*v)/2.0
    # ap, am: amplitudes of positively and negatively rotating vectors
    ap = np.abs(wp)
    am = np.abs(wm)
    # ep, em: phases of positively and negatively rotating vectors
    ep = np.angle(wp, deg=True)
    em = np.angle(wm, deg=True)
    # determine the amplitudes of the semimajor and semiminor axes
    # using Foreman's formula
    umajor = (ap + am)
    uminor = (ap - am)
    # determine the inclination and phase using Foreman's formula
    uincl = (em + ep)/2.0
    uphase = (em - ep)/2.0
    # adjust orientation of ellipse
    k = (uincl//180.0)
    uincl -= 180.0*k
    uphase += 180.0*k
    uphase = np.mod(uphase, 360.0)
    # return values
    return (umajor, uminor, uincl, uphase)

def inverse(
        umajor: np.ndarray,
        uminor: np.ndarray,
        uincl: np.ndarray,
        uphase: np.ndarray
    ):
    """
    Calculates currents u, v using the four tidal ellipse
    parameters from Foreman's formula :cite:p:`Foreman:1989dt`

    Parameters
    ----------
    umajor: np.ndarray
        amplitude of the semi-major axis
    uminor: np.ndarray
        amplitude of the semi-minor axis
    uincl: np.ndarray
        angle of inclination of the northern semi-major axis
    uphase: np.ndarray
        phase lag of the maximum current behind the maximum tidal potential

    Returns
    -------
    u: np.ndarray
        zonal current (EW)
    v: np.ndarray
        meridional current (NS)
    """
    # validate inputs
    umajor = np.atleast_1d(umajor)
    uminor = np.atleast_1d(uminor)
    # convert inclination and phase to radians
    uincl = np.atleast_1d(uincl)*np.pi/180.0
    uphase = np.atleast_1d(uphase)*np.pi/180.0
    # ep, em: phases of positively and negatively rotating vectors
    ep = (uincl - uphase)
    em = (uincl + uphase)
    # ap, am: amplitudes of positively and negatively rotating vectors
    ap = (umajor + uminor)/2.0
    am = (umajor - uminor)/2.0
    # wp, wm: complex radius of positively and negatively rotating vectors
    wp = ap * np.exp(1j*ep)
    wm = am * np.exp(1j*em)
    # calculate complex currents
    u = wp + np.conj(wm)
    v = -1j*(wp - np.conj(wm))
    # return values
    return (u, v)

def _xy(
        umajor: float | np.ndarray,
        uminor: float | np.ndarray,
        uincl: float | np.ndarray,
        uphase: float | np.ndarray | None = None,
        xy: tuple = (0.0, 0.0),
        n: int | None = 1000,
    ):
    """
    Calculates the x and y coordinates of the tidal ellipse

    Parameters
    ----------
    umajor: np.ndarray
        amplitude of the semi-major axis
    uminor: np.ndarray
        amplitude of the semi-minor axis
    uincl: np.ndarray
        angle of inclination of the northern semi-major axis
    uphase: np.ndarray
        phase lag of the maximum current behind the maximum tidal potential
    xy: tuple, default (0.0, 0.0)
        center of the ellipse (x, y)
    n: int, default 1000
        number of points to calculate along the ellipse

    Returns
    -------
    x: np.ndarray
        x coordinates of the tidal ellipse
    y: np.ndarray
        y coordinates of the tidal ellipse
    """
    # validate inputs
    phi = uincl*np.pi/180.0
    # calculate the angle of the ellipse
    if uphase is not None:
        # use the phase lag and inclination
        th  = (uphase + uincl)*np.pi/180.0 
    else:
        # use a full rotation
        th = np.linspace(0, 2*np.pi, n)
    # calculate x and y coordinates
    x = xy[0] + umajor*np.cos(th)*np.cos(phi) - uminor*np.sin(th)*np.sin(phi)
    y = xy[1] + umajor*np.cos(th)*np.sin(phi) + uminor*np.sin(th)*np.cos(phi)
    return (x, y)
