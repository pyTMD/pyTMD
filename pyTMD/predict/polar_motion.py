#!/usr/bin/env python
"""
polar_motion.py
Written by Tyler Sutterley (04/2026)
Prediction routines for pole tides and Earth Orientation Parameters (EOPs)

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
    math.py: Special functions of mathematical physics

UPDATE HISTORY:
    Updated 04/2026: parallel outputs from earth_orientation and length_of_day
    Written 03/2026: split up prediction functions into separate files
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import pyTMD.astro
import pyTMD.constituents
import pyTMD.math
import timescale.eop

__all__ = [
    "load_pole_tide",
    "ocean_pole_tide",
    "earth_orientation",
    "length_of_day",
]

# number of days between MJD and the tide epoch (1992-01-01T00:00:00)
_mjd_tide = 48622.0
# number of days between MJD and the J2000 epoch
_mjd_j2000 = 51544.5
# Julian century
_century = 36525.0


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
    UXYZ: xr.Dataset,
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
    UXYZ: xarray.Dataset
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


def earth_orientation(
    t: np.ndarray,
    deltat: float | np.ndarray = 0.0,
):
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
    # delaunay arguments
    args = ["l", "lp", "F", "D", "omega", "gamma", "k"]
    arguments = xr.DataArray(
        np.c_[l, lp, F, D, omega, gamma, K],
        dims=["time", "argument"],
        coords=dict(
            time=np.atleast_1d(MJD),
            argument=args,
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
    # convert delaunay coefficients to DataArray
    delaunay_table = xr.DataArray(
        delaunay_table,
        dims=["argument", "constituent"],
        coords=dict(
            argument=args,
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
    # convert EOP corrections to DataArray
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
    corrections = dEOP.real * phase.real + dEOP.imag * phase.imag
    # calculate angular frequency of constituents
    omegas = pyTMD.constituents.frequency(constituents)
    # create output Dataset from DataArray objects
    ds = xr.Dataset()
    # polar motion corrections in X and Y (arcseconds)
    ds["dX"] = 1e-3 * corrections.sel(EOP="dX")
    ds["dX"].attrs["units"] = "arcseconds"
    ds["dX"].attrs["long_name"] = "anomaly in polar motion in X"
    ds["dY"] = 1e-3 * corrections.sel(EOP="dY")
    ds["dY"].attrs["units"] = "arcseconds"
    ds["dY"].attrs["long_name"] = "anomaly in polar motion in Y"
    # delta UT1-TAI (seconds)
    ds["dUT"] = 1e-4 * corrections.sel(EOP="dUT")
    ds["dUT"].attrs["units"] = "seconds"
    ds["dUT"].attrs["long_name"] = "anomaly in UT1-TAI"
    # period of constituent (days)
    periods = 2.0 * np.pi / (86400.0 * omegas)
    ds["period"] = ("constituent", periods)
    ds["period"].attrs["units"] = "days"
    # return the variations in earth rotation
    return ds


# PURPOSE: estimate variations in length of day
def length_of_day(
    t: np.ndarray,
    deltat: float | np.ndarray = 0.0,
):
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
    # astronomical arguments
    # note the sign change to go from N to N'
    args = ["tau", "s", "h", "p", "n", "pp", "k"]
    arguments = np.c_[tau, s, h, p, -n, pp, k]
    # convert arguments to DataArray
    arguments = xr.DataArray(
        arguments,
        dims=["time", "argument"],
        coords=dict(
            time=np.atleast_1d(MJD),
            argument=args,
        ),
    )
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
        coords=dict(argument=args),
    )
    # equilibrium phase converted to radians
    G = np.radians(arguments.dot(coefficients))
    # calculate length of day corrections
    dUT = ZROT["UTc"] * np.cos(G) + ZROT["UTs"] * np.sin(G)
    dLOD = ZROT["dLODc"] * np.cos(G) + ZROT["dLODs"] * np.sin(G)
    # create output Dataset from DataArray objects
    ds = xr.Dataset(coords=dict(time=np.atleast_1d(MJD)))
    # delta UT1-TAI (seconds)
    ds["dUT"] = 1e-6 * dUT
    ds["dUT"].attrs["units"] = "seconds"
    ds["dUT"].attrs["long_name"] = "anomaly in UT1-TAI"
    # delta LOD (seconds per day)
    ds["dLOD"] = 1e-6 * dLOD
    ds["dLOD"].attrs["units"] = "seconds per day"
    ds["dLOD"].attrs["long_name"] = "excess length of day"
    # period of constituent (days)
    ds["period"] = ("constituent", ZROT["period"])
    ds["period"].attrs["units"] = "days"
    # return the variations in earth rotation
    return ds
