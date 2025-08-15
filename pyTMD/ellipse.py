#!/usr/bin/env python
u"""
ellipse.py
Written by Tyler Sutterley (08/2025)
Expresses the amplitudes and phases for the u and v components in terms of
    four ellipse parameters using Foreman's formula

CALLING SEQUENCE:
    umajor,uminor,uincl,uphase = pyTMD.ellipse.ellipse(u,v)

INPUTS:
    u: zonal current (EW)
    v: meridional current (NS)

OUTPUTS:
    major: amplitude of the semimajor semi-axis
    minor: amplitude of the semiminor semi-axis
    incl: angle of inclination of the northern semimajor semi-axis
    phase: phase lag of the maximum current behind the maximum tidal potential
        of the individual constituent

REFERENCE:
    M. G. G. Foreman and R. F. Henry, "The harmonic analysis of tidal model time
        series", Advances in Water Resources, 12(3), 109-120, (1989).
        https://doi.org/10.1016/0309-1708(89)90017-1

UPDATE HISTORY:
    Updated 08/2025: use divmod to adjust orientation of ellipse
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
    major: np.ndarray
        amplitude of the semi-major axis
    minor: np.ndarray
        amplitude of the semi-minor axis
    incl: np.ndarray
        angle of inclination of the northern semi-major axis
    phase: np.ndarray
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
    major = (ap + am)
    minor = (ap - am)
    # determine the inclination and phase using Foreman's formula
    incl = (em + ep)/2.0
    phase = (em - ep)/2.0
    # adjust orientation of ellipse
    k, incl = np.divmod(incl, 180.0)
    phase = np.mod(phase + 180.0*k, 360.0)
    # return values
    return (major, minor, incl, phase)

def inverse(
        major: np.ndarray,
        minor: np.ndarray,
        incl: np.ndarray,
        phase: np.ndarray
    ):
    """
    Calculates currents u, v using the four tidal ellipse
    parameters from Foreman's formula :cite:p:`Foreman:1989dt`

    Parameters
    ----------
    major: np.ndarray
        amplitude of the semi-major axis
    minor: np.ndarray
        amplitude of the semi-minor axis
    incl: np.ndarray
        angle of inclination of the northern semi-major axis
    phase: np.ndarray
        phase lag of the maximum current behind the maximum tidal potential

    Returns
    -------
    u: np.ndarray
        zonal current (EW)
    v: np.ndarray
        meridional current (NS)
    """
    # validate inputs
    major = np.atleast_1d(major)
    minor = np.atleast_1d(minor)
    # convert inclination and phase to radians
    incl = np.radians(np.atleast_1d(incl))
    phase = np.radians(np.atleast_1d(phase))
    # ep, em: phases of positively and negatively rotating vectors
    ep = (incl - phase)
    em = (incl + phase)
    # ap, am: amplitudes of positively and negatively rotating vectors
    ap = (major + minor)/2.0
    am = (major - minor)/2.0
    # wp, wm: complex radius of positively and negatively rotating vectors
    wp = ap * np.exp(1j*ep)
    wm = am * np.exp(1j*em)
    # calculate complex currents
    u = wp + np.conj(wm)
    v = -1j*(wp - np.conj(wm))
    # return values
    return (u, v)

def _xy(
        major: float | np.ndarray,
        minor: float | np.ndarray,
        incl: float | np.ndarray,
        **kwargs
    ):
    """
    Calculates the x and y coordinates of the tidal ellipse

    Parameters
    ----------
    major: np.ndarray
        amplitude of the semi-major axis
    minor: np.ndarray
        amplitude of the semi-minor axis
    incl: np.ndarray
        angle of inclination of the northern semi-major axis
    phase: np.ndarray or None, default None
        phase lag of the maximum current behind the maximum tidal potential
    xy: tuple, default (0.0, 0.0)
        center of the ellipse (x, y)
    N: int or None, default None
        number of points to calculate along the ellipse

    Returns
    -------
    x: np.ndarray
        x coordinates of the tidal ellipse
    y: np.ndarray
        y coordinates of the tidal ellipse
    """
    # set default number of points
    kwargs.setdefault('phase', None)
    kwargs.setdefault('xy', (0.0, 0.0))
    kwargs.setdefault('N', 1000)
    # validate inputs
    phi = np.radians(np.atleast_1d(incl))
    # calculate the angle of the ellipse
    if kwargs['phase'] is not None:
        # use the phase lag and inclination
        th = np.radians(kwargs['phase'] + incl)
    else:
        # use a full rotation
        th = np.linspace(0, 2*np.pi, kwargs['N'])
    # calculate x and y coordinates
    x = kwargs['xy'][0] + \
        major*np.cos(th)*np.cos(phi) - \
        minor*np.sin(th)*np.sin(phi)
    y = kwargs['xy'][1] + \
        major*np.cos(th)*np.sin(phi) + \
        minor*np.sin(th)*np.cos(phi)
    return (x, y)
