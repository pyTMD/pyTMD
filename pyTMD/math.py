#!/usr/bin/env python
"""
math.py
Written by Tyler Sutterley (03/2026)
Special functions of mathematical physics

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/

UPDATE HISTORY:
    Updated 03/2026: add radius and scalar product functions
        calculate Legendre polynomials using Hofmann-Wellenhof (2006) eq. 1.67
    Updated 02/2026: add inverse functions for converting radians to arcseconds
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
from scipy.special import factorial

__all__ = [
    "asec2rad",
    "masec2rad",
    "rad2asec",
    "rad2masec",
    "polynomial_sum",
    "normalize_angle",
    "radius",
    "rotate",
    "scalar_product",
    "aliasing",
    "legendre",
    "_assoc_legendre",
    "_condon_shortley",
    "_legendre_norm",
    "sph_harm",
]


def asec2rad(x: float | np.ndarray):
    """
    Convert angles from arcseconds to radians

    Parameters
    ----------
    x: float or np.ndarray
        Input angle (arcseconds)
    """
    return np.radians(x / 3600.0)


def masec2rad(x: float | np.ndarray):
    """
    Convert angles from microarcseconds to radians

    Parameters
    ----------
    x: float or np.ndarray
        Input angle (microarcseconds)
    """
    return np.radians(x / 3.6e9)


def rad2asec(x: float | np.ndarray):
    """
    Convert angles from radians to arcseconds

    Parameters
    ----------
    x: float or np.ndarray
        Input angle (radians)
    """
    return 3600.0 * np.degrees(x)


def rad2masec(x: float | np.ndarray):
    """
    Convert angles from radians to microarcseconds

    Parameters
    ----------
    x: float or np.ndarray
        Input angle (radians)
    """
    return 3.6e9 * np.degrees(x)


# PURPOSE: calculate the sum of a polynomial function of time
def polynomial_sum(
    coefficients: list | np.ndarray,
    t: np.ndarray,
):
    """
    Calculates the sum of a polynomial function using Horner's method
    :cite:p:`Horner:1819br`

    Parameters
    ----------
    coefficients: list or np.ndarray
        Leading coefficient of polynomials of increasing order
    t: np.ndarray
        Time for a given astronomical longitudes calculation
    """
    # convert time to array if importing a single value
    t = np.atleast_1d(t)
    return np.sum([c * (t**i) for i, c in enumerate(coefficients)], axis=0)


def normalize_angle(
    theta: float | np.ndarray,
    circle: float = 360.0,
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


def radius(
    x: float | np.ndarray,
    y: float | np.ndarray,
    z: float | np.ndarray,
):
    """
    Calculate the radius from the origin to a point in 3-dimensional space

    Parameters
    ----------
    x: float or np.ndarray
        x-coordinate of the point
    y: float or np.ndarray
        y-coordinate of the point
    z: float or np.ndarray
        z-coordinate of the point
    """
    return np.sqrt(x**2 + y**2 + z**2)


def rotate(
    theta: float | np.ndarray,
    axis: str = "x",
):
    """
    Rotate a 3-dimensional matrix about a given axis

    Parameters
    ----------
    theta: float or np.ndarray
        Angle of rotation (radians)
    axis: str, default 'x'
        Axis of rotation (``'x'``, ``'y'``, or ``'z'``)
    """
    # allocate for output rotation matrix
    R = np.zeros((3, 3, len(np.atleast_1d(theta))))
    if axis.lower() == "x":
        # rotate about x-axis
        R[0, 0, :] = 1.0
        R[1, 1, :] = np.cos(theta)
        R[1, 2, :] = np.sin(theta)
        R[2, 1, :] = -np.sin(theta)
        R[2, 2, :] = np.cos(theta)
    elif axis.lower() == "y":
        # rotate about y-axis
        R[0, 0, :] = np.cos(theta)
        R[0, 2, :] = -np.sin(theta)
        R[1, 1, :] = 1.0
        R[2, 0, :] = np.sin(theta)
        R[2, 2, :] = np.cos(theta)
    elif axis.lower() == "z":
        # rotate about z-axis
        R[0, 0, :] = np.cos(theta)
        R[0, 1, :] = np.sin(theta)
        R[1, 0, :] = -np.sin(theta)
        R[1, 1, :] = np.cos(theta)
        R[2, 2, :] = 1.0
    else:
        raise ValueError(f"Invalid axis {axis}")
    # return the rotation matrix
    return R


def scalar_product(
    x: float | np.ndarray,
    y: float | np.ndarray,
    z: float | np.ndarray,
    u: float | np.ndarray,
    v: float | np.ndarray,
    w: float | np.ndarray,
):
    """
    Calculate the scalar product of two vectors in 3-dimensional space

    Parameters
    ----------
    x: float or np.ndarray
        x-coordinate of the first vector
    y: float or np.ndarray
        y-coordinate of the first vector
    z: float or np.ndarray
        z-coordinate of the first vector
    u: float or np.ndarray
        x-coordinate of the second vector
    v: float or np.ndarray
        y-coordinate of the second vector
    w: float or np.ndarray
        z-coordinate of the second vector
    """
    return x * u + y * v + z * w


def aliasing(
    f: float,
    fs: float,
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
    fa = np.abs(f - fs * np.round(f / fs))
    return fa


def legendre(
    l: int,
    x: np.ndarray,
    m: int = 0,
    norm: float = 1.0,
):
    r"""
    Computes associated Legendre functions and their first-derivatives
    for a particular degree and order
    :cite:p:`Munk:1966go,HofmannWellenhof:2006hy`

    Parameters
    ----------
    l: int
        Degree of the Legendre polynomials
    x: np.ndarray
        Elements ranging from -1 to 1

        Typically :math:`\cos(\theta)`, where :math:`\theta`
        is the colatitude
    m: int, default 0
        Order of the Legendre polynomials (:math:`0` to :math:`l`)
    norm: float, default 1.0
        Normalization to apply to outputs

    Returns
    -------
    Plm: np.ndarray
        Legendre polynomials of degree :math:`l` and order :math:`m`
    dPlm: np.ndarray
        First derivative of spherical harmonics with respect to
        :math:`\theta`
    """
    # verify values are integers
    l = np.int64(l)
    m = np.int64(m)
    # assert values
    assert (m >= 0) and (m <= l), "Order must be between 0 and l"
    # verify x is array
    if isinstance(x, list):
        x = np.atleast_1d(x)
    # function 1.67 from Hofmann-Wellenhof (2006)
    Plm = _assoc_legendre(l, m, x)
    # if x is the cos of colatitude, u is the sine
    u = np.sqrt(1.0 - x**2)
    # calculate first derivative
    # this will initially have a singularity at the poles
    Pm1 = _assoc_legendre(l - 1, m, x)
    # ignore divide by zero and invalid value warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        dPlm = (l * x * Plm - (l + m) * Pm1) / u
    # handle singularity at the poles (x = +/-1) for order 1 terms
    pole = -0.5 * np.power(x, l) * l * (l + 1) if m == 1 else 0.0
    dPlm = np.where(np.isclose(u, 0.0), pole, dPlm)
    # return the associated legendre functions
    # and their first derivatives with respect to theta
    return norm * Plm, norm * dPlm


def _assoc_legendre(
    l: int,
    m: int,
    x: np.ndarray,
):
    r"""
    Computes associated Legendre polynomials using equation 1.67
    from :cite:t:`HofmannWellenhof:2006hy`

    Parameters
    ----------
    l: int
        Degree of the Legendre polynomials
    m: int
        Order of the Legendre polynomials (:math:`0` to :math:`l`)
    x: np.ndarray
        Elements ranging from -1 to 1

        Typically :math:`\cos(\theta)`, where :math:`\theta`
        is the colatitude

    Returns
    -------
    Plm: np.ndarray
        Legendre polynomials of degree :math:`l` and order :math:`m`
    """
    # verify values are integers
    l = np.int64(l)
    m = np.int64(m)
    # return 0 for invalid values
    if m > l:
        return 0.0
    # verify x is array
    if isinstance(x, list):
        x = np.atleast_1d(x)
    # if x is the cos of colatitude, u is the sine
    u = np.sqrt(1.0 - x**2)
    # calculate un-normalized polynomials
    # function 1.67 from Hofmann-Wellenhof (2006)
    P = 0.0
    r = int((l - m) // 2)
    for k in range(r + 1):
        P += (
            np.power(-1.0, k)
            * factorial(2.0 * l - 2.0 * k)
            / factorial(k)
            / factorial(l - k)
            / factorial(l - m - 2.0 * k)
            * np.power(x, l - m - 2.0 * k)
        )
    # calculate for degree l and order m
    Plm = P * np.power(2.0, -l) * np.power(u, m)
    # apply Condon-Shortley phase
    Plm *= _condon_shortley(m)
    # return the associated legendre polynomials
    return Plm


def _condon_shortley(m: int):
    """
    Computes the Condon-Shortley phase

    Parameters
    ----------
    m: int
        Order of the Legendre polynomials
    """
    return np.power(-1.0, m)


def _legendre_norm(l: int, m: int):
    """
    Calculates the Legendre Polynomial normalization from
    :cite:t:`Munk:1966go`

    Parameters
    ----------
    l: int
        Degree of the Legendre polynomials
    m: int
        Order of the Legendre polynomials (:math:`0` to :math:`l`)
    """
    # normalization from Munk and Cartwright (1966) equation A5
    return np.sqrt(factorial(l - m) / factorial(l + m))


def sph_harm(
    l: int,
    theta: np.ndarray,
    phi: np.ndarray,
    m: int = 0,
    phase: float = 0.0,
):
    r"""
    Computes the spherical harmonics for a particular degree
    and order :cite:p:`Munk:1966go,HofmannWellenhof:2006hy`

    Parameters
    ----------
    l: int
        Degree of the spherical harmonics
    theta: np.ndarray
        Colatitude (radians)
    phi: np.ndarray
        Longitude (radians)
    m: int, default 0
        Order of the spherical harmonics (:math:`0` to :math:`l`)
    phase: float, default 0.0
        Phase shift (radians)

    Returns
    -------
    Ylm: np.ndarray
        Complex spherical harmonics of degree :math:`l` and order :math:`m`
    dYlm: np.ndarray
        First derivative of spherical harmonics with respect to
        :math:`\theta`
    """
    # normalization from Munk and Cartwright (1966) equation A5
    norm = _legendre_norm(l, m)
    # calculate associated Legendre functions and derivatives
    Plm, dPlm = legendre(l, np.cos(theta), m=m, norm=norm)
    # normalized spherical harmonics of degree l and order m
    dfactor = np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi))
    Ylm = dfactor * Plm * np.exp(1j * m * phi + 1j * phase)
    dYlm = dfactor * dPlm * np.exp(1j * m * phi + 1j * phase)
    # return values
    return Ylm, dYlm
