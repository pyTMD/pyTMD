#!/usr/bin/env python
u"""
math.py
Written by Tyler Sutterley (11/2025)
Special functions of mathematical physics

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/

UPDATE HISTORY:
    Updated 11/2025: legendre now returns both polynomials and derivatives
    Updated 09/2025: added degree 4 to legendre polynomials function
    Updated 08/2025: add asec2rad and masec2rad functions for arcseconds
    Updated 07/2025: add deriv and phase arguments to sph_harm function
        add Legendre polynomial derivatives with respect to theta
    Updated 04/2025: use numpy power function over using pow for consistency
    Updated 01/2025: added function for fully-normalized Legendre polynomials
    Updated 12/2024: added function to calculate an aliasing frequency
    Written 11/2024
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.special import factorial

__all__ = [
    "asec2rad",
    "masec2rad",
    "polynomial_sum",
    "normalize_angle",
    "rotate",
    "aliasing",
    "legendre",
    "sph_harm"
]

def asec2rad(
        x: float | np.ndarray,
    ):
    """
    Convert angles from arcseconds to radians

    Parameters
    ----------
    x: float or np.ndarray
        Input angle in arcseconds
    """
    return np.radians(x / 3600.0)

def masec2rad(
        x: float | np.ndarray,
    ):
    """
    Convert angles from microarcseconds to radians

    Parameters
    ----------
    x: float or np.ndarray
        Input angle in microarcseconds
    """
    return np.radians(x / 3.6e9)

# PURPOSE: calculate the sum of a polynomial function of time
def polynomial_sum(
        coefficients: list | np.ndarray,
        t: np.ndarray
    ):
    """
    Calculates the sum of a polynomial function using Horner's method
    :cite:p:`Horner:1819br`

    Parameters
    ----------
    coefficients: list or np.ndarray
        leading coefficient of polynomials of increasing order
    t: np.ndarray
        delta time in units for a given astronomical longitudes calculation
    """
    # convert time to array if importing a single value
    t = np.atleast_1d(t)
    return np.sum([c * (t ** i) for i, c in enumerate(coefficients)], axis=0)

def normalize_angle(
        theta: float | np.ndarray,
        circle: float = 360.0
    ):
    """
    Normalize an angle to a single rotation

    Parameters
    ----------
    theta: float or np.ndarray
        Angle to normalize
    circle: float, default 360.0
        Circle of the angle
    """
    return np.mod(theta, circle)

def rotate(
        theta: float | np.ndarray,
        axis: str = 'x'
    ):
    """
    Rotate a 3-dimensional matrix about a given axis

    Parameters
    ----------
    theta: float or np.ndarray
        Angle of rotation in radians
    axis: str, default 'x'
        Axis of rotation (``'x'``, ``'y'``, or ``'z'``)
    """
    # allocate for output rotation matrix
    R = np.zeros((3, 3, len(np.atleast_1d(theta))))
    if (axis.lower() == 'x'):
        # rotate about x-axis
        R[0,0,:] = 1.0
        R[1,1,:] = np.cos(theta)
        R[1,2,:] = np.sin(theta)
        R[2,1,:] = -np.sin(theta)
        R[2,2,:] = np.cos(theta)
    elif (axis.lower() == 'y'):
        # rotate about y-axis
        R[0,0,:] = np.cos(theta)
        R[0,2,:] = -np.sin(theta)
        R[1,1,:] = 1.0
        R[2,0,:] = np.sin(theta)
        R[2,2,:] = np.cos(theta)
    elif (axis.lower() == 'z'):
        # rotate about z-axis
        R[0,0,:] = np.cos(theta)
        R[0,1,:] = np.sin(theta)
        R[1,0,:] = -np.sin(theta)
        R[1,1,:] = np.cos(theta)
        R[2,2,:] = 1.0
    else:
        raise ValueError(f'Invalid axis {axis}')
    # return the rotation matrix
    return R

def aliasing(
        f: float,
        fs: float
    ) -> float:
    """
    Calculate the aliasing frequency of a signal

    Parameters
    ----------
    f: float
        Frequency of the signal
    fs: float
        Sampling frequency of the signal

    Returns
    -------
    fa: float
        Aliasing frequency of the signal
    """
    fa = np.abs(f - fs*np.round(f/fs))
    return fa

def legendre(
        l: int,
        x: np.ndarray,
        m: int = 0,
    ):
    """
    Computes associated Legendre functions and their first-derivatives
    for a particular degree and order 
    :cite:p:`Munk:1966go,HofmannWellenhof:2006hy`

    Parameters
    ----------
    l: int
        degree of the Legendre polynomials (0 to 4)
    x: np.ndarray
        elements ranging from -1 to 1

        Typically ``cos(theta)``, where ``theta`` is the colatitude in radians
    m: int, default 0
        order of the Legendre polynomials (0 to ``l``)

    Returns
    -------
    Plm: np.ndarray
        Legendre polynomials of degree ``l`` and order ``m``
    dPlm: np.ndarray
        first derivative of Legendre polynomials with respect to ``theta``
    """
    # maximum degree and order
    lmax = 4
    # verify values are integers
    l = np.int64(l)
    m = np.int64(m)
    # assert values
    assert (l >= 0) and (l <= lmax), f'Degree must be between 0 and {lmax}'
    assert (m >= 0) and (m <= l), 'Order must be between 0 and l'
    # verify x is array
    if isinstance(x, list):
        x = np.atleast_1d(x)
    # if x is the cos of colatitude, u is the sine
    u = np.sqrt(1.0 - x**2)
    # since tides only use low-degree harmonics:
    # functions are hard coded rather than using a recursion relation
    if (l == 0) and (m == 0):
        Plm = 1.0
        dPlm = 0.0
    elif (l == 1) and (m == 0):
        Plm = x
        dPlm = -u
    elif (l == 1) and (m == 1):
        Plm = u
        dPlm = x
    elif (l == 2) and (m == 0):
        Plm = 0.5*(3.0*x**2 - 1.0)
        dPlm = -3.0*u*x
    elif (l == 2) and (m == 1):
        Plm = 3.0*x*u
        dPlm = 3.0*(1.0 - 2.0*u**2)
    elif (l == 2) and (m == 2):
        Plm = 3.0*u**2
        dPlm = 6.0*u*x
    elif (l == 3) and (m == 0):
        Plm = 0.5*(5.0*x**2 - 3.0)*x
        dPlm = u*(1.5 - 7.5*x**2)
    elif (l == 3) and (m == 1):
        Plm = 1.5*(5.0*x**2 - 1.0)*u
        dPlm = -1.5*x*(10.0*u**2 - 5.0*x**2 + 1.0)
    elif (l == 3) and (m == 2):
        Plm = 15.0*x*u**2
        dPlm = 15.0*u*(3.0*x**2 - 1.0)
    elif (l == 3) and (m == 3):
        Plm = 15.0*u**3
        dPlm = 45.0*x*u**2
    elif (l == 4) and (m == 0):
        Plm = 0.125*(35.0*x**4 - 30.0*x**2 + 3.0)
        dPlm = -2.5*(7.0*x**2 - 3.0)*u*x
    elif (l == 4) and (m == 1):
        Plm = 2.5*(7.0*x**2 - 3.0)*u*x
        dPlm = 2.5*(28.0*x**4 - 27.0*x**2 + 3.0)
    elif (l == 4) and (m == 2):
        Plm = 7.5*(7.0*x**2 - 1.0)*u**2
        dPlm = (105*x**2 - 105*u**2 - 15.0)*u*x
    elif (l == 4) and (m == 3):
        Plm = 105.0*x*u**3
        dPlm = (420.0*x**3 - 105.0)*u**2
    elif (l == 4) and (m == 4):
        Plm = 105.0*u**4
        dPlm = 420.0*x*u**3
    # apply Condon-Shortley phase
    Plm *= np.power(-1.0, m)
    dPlm *= np.power(-1.0, m)
    # return the associated legendre functions
    return Plm, dPlm

def sph_harm(
        l: int,
        theta: np.ndarray,
        phi: np.ndarray,
        m: int = 0,
        phase: float = 0.0,
    ):
    """
    Computes the spherical harmonics for a particular degree
    and order :cite:p:`Munk:1966go,HofmannWellenhof:2006hy`

    Parameters
    ----------
    l: int
        degree of the spherical harmonics (0 to 4)
    theta: np.ndarray
        colatitude in radians
    phi: np.ndarray
        longitude in radians
    m: int, default 0
        order of the spherical harmonics (0 to ``l``)
    phase: float, default 0.0
        phase shift in radians

    Returns
    -------
    Ylm: np.ndarray
        complex spherical harmonics of degree ``l`` and order ``m``
    dYlm: np.ndarray
        first derivative of spherical harmonics with respect to ``theta``
    """
    # calculate associated Legendre functions and derivatives
    Plm, dPlm = legendre(l, np.cos(theta), m=m)
    # normalization from Munk and Cartwright (1966) equation A5
    norm = np.sqrt(factorial(l - m)/factorial(l + m))
    # normalized spherical harmonics of degree l and order m
    dfactor = np.sqrt((2.0*l + 1.0)/(4.0*np.pi))
    Ylm = dfactor*norm*Plm*np.exp(1j*m*phi + 1j*phase)
    dYlm = dfactor*norm*dPlm*np.exp(1j*m*phi + 1j*phase)
    # return values
    return Ylm, dYlm
