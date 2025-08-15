#!/usr/bin/env python
u"""
spatial.py
Written by Tyler Sutterley (08/2025)

Spatial transformation routines

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

UPDATE HISTORY:
    Updated 08/2025: convert angles with numpy radians and degrees functions
    Updated 03/2025: add more ellipsoidal parameters to datum class
    Updated 02/2025: major refactor to move io routines out of this module
    Updated 12/2024: add latitude and longitude as potential dimension names
    Updated 11/2024: added function to calculate the altitude and azimuth
    Updated 09/2024: deprecation fix case where an array is output to scalars
    Updated 08/2024: changed from 'geotiff' to 'GTiff' and 'cog' formats
        added functions to convert to and from East-North-Up coordinates
    Updated 07/2024: added functions to convert to and from DMS
    Updated 06/2024: added function to write parquet files with metadata
    Updated 05/2024: added function to read from parquet files
        allowing for decoding of the geometry column from WKB
        deprecation update to use exceptions with osgeo osr
    Updated 04/2024: use timescale for temporal operations
        use wrapper to importlib for optional dependencies
    Updated 03/2024: can calculate polar stereographic distortion for distances
    Updated 02/2024: changed class name for ellipsoid parameters to datum
    Updated 10/2023: can read from netCDF4 or HDF5 variable groups
        apply no formatting to columns in ascii file output
    Updated 09/2023: add function to invert field mapping keys and values
        use datetime64[ns] for parsing dates from ascii files
    Updated 08/2023: remove possible crs variables from output fields list
        place PyYAML behind try/except statement to reduce build size
    Updated 05/2023: use datetime parser within pyTMD.time module
    Updated 04/2023: copy inputs in cartesian to not modify original arrays
        added iterative methods for converting from cartesian to geodetic
        allow netCDF4 and HDF5 outputs to be appended to existing files
        using pathlib to define and expand paths
    Updated 03/2023: add basic variable typing to function inputs
    Updated 02/2023: use outputs from constants class for WGS84 parameters
        include more possible dimension names for gridded and drift outputs
    Updated 01/2023: added default field mapping for reading from netCDF4/HDF5
        split netCDF4 output into separate functions for grid and drift types
    Updated 12/2022: add software information to output HDF5 and netCDF4
    Updated 11/2022: place some imports within try/except statements
        added encoding for writing ascii files
        use f-strings for formatting verbose or ascii output
    Updated 10/2022: added datetime parser for ascii time columns
    Updated 06/2022: added field_mapping options to netCDF4 and HDF5 reads
        added from_file wrapper function to read from particular formats
    Updated 04/2022: add option to reduce input GDAL raster datasets
        updated docstrings to numpy documentation format
        use gzip virtual filesystem for reading compressed geotiffs
        include utf-8 encoding in reads to be windows compliant
    Updated 03/2022: add option to specify output GDAL driver
    Updated 01/2022: use iteration breaks in convert ellipsoid function
        remove fill_value attribute after creating netCDF4 and HDF5 variables
    Updated 11/2021: added empty cases to netCDF4 and HDF5 output for crs
        try to get grid mapping attributes from netCDF4 and HDF5
    Updated 10/2021: add pole case in stereographic area scale calculation
        using python logging for handling verbose output
    Updated 09/2021: can calculate height differences between ellipsoids
    Updated 07/2021: added function for determining input variable type
    Updated 03/2021: added polar stereographic area scale calculation
        add routines for converting to and from cartesian coordinates
        replaced numpy bool/int to prevent deprecation warnings
    Updated 01/2021: add streaming from bytes for ascii, netCDF4, HDF5, geotiff
        set default time for geotiff files to 0
    Updated 12/2020: added module for converting ellipsoids
    Updated 11/2020: output data as masked arrays if containing fill values
        add functions to read from and write to geotiff image formats
    Written 09/2020
"""
from __future__ import annotations

import warnings
import numpy as np

__all__ = [
    "data_type",
    "datum",
    "convert_ellipsoid",
    "compute_delta_h",
    "wrap_longitudes",
    "to_dms",
    "from_dms",
    "to_cartesian",
    "to_sphere",
    "to_geodetic",
    "_moritz_iterative",
    "_bowring_iterative",
    "_zhu_closed_form",
    "to_ENU",
    "from_ENU",
    "to_horizontal",
    "to_zenith",
    "scale_factors",
]

def data_type(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> str:
    """
    Determines input data type based on variable dimensions

    Parameters
    ----------
    x: np.ndarray
        x-dimension coordinates
    y: np.ndarray
        y-dimension coordinates
    t: np.ndarray
        time-dimension coordinates

    Returns
    -------
    string denoting input data type

        - ``'time series'``
        - ``'drift'``
        - ``'grid'``
    """
    xsize = np.size(x)
    ysize = np.size(y)
    tsize = np.size(t)
    if (xsize == 1) and (ysize == 1) and (tsize >= 1):
        return 'time series'
    elif (xsize == ysize) & (xsize == tsize):
        return 'drift'
    elif (np.ndim(x) > 1) & (xsize == ysize):
        return 'grid'
    elif (xsize != ysize):
        return 'grid'
    else:
        raise ValueError('Unknown data type')

_ellipsoids = ['CLK66', 'CLK80', 'GRS67', 'GRS80', 'WGS60', 'WGS66', 'WGS72',
    'WGS84', 'ATS77', 'NAD27', 'NAD83', 'INTER', 'KRASS', 'SGS90',
    'AIRY', 'MAIRY', 'HLMRT', 'HOUGH', 'HGH80', 'MERIT', 'TOPEX',
    'EGM96', 'IAG75', 'IAU64', 'IAU76', 'IERS89', 'IERS']
_units = ['MKS', 'CGS']

class datum:
    """
    Class for gravitational and ellipsoidal parameters
    :cite:p:`HofmannWellenhof:2006hy,Urban:2013vl`

    Parameters
    ----------
    ellipsoid: str, default 'WGS84'
        Reference ellipsoid name

            - ``'CLK66'``: Clarke 1866
            - ``'CLK80'``: Clarke 1880
            - ``'GRS67'``: Geodetic Reference System 1967
            - ``'GRS80'``: Geodetic Reference System 1980
            - ``'HGH80'``: Hughes 1980 Ellipsoid
            - ``'WGS60'``: World Geodetic System 1960
            - ``'WGS66'``: World Geodetic System 1966
            - ``'WGS72'``: World Geodetic System 1972
            - ``'WGS84'``: World Geodetic System 1984
            - ``'ATS77'``: Quasi-earth centred ellipsoid for ATS77
            - ``'NAD27'``: North American Datum 1927
            - ``'NAD83'``: North American Datum 1983
            - ``'INTER'``: International
            - ``'KRASS'``: Krassovsky (USSR)
            - ``'SGS90'``: Soviet Geodetic System 1990
            - ``'HLMRT'``: Helmert 1906 Ellipsoid
            - ``'HOUGH'``: Hough 1960 Ellipsoid
            - ``'AIRY'``: Airy (1830)
            - ``'MAIRY'``: Modified Airy (1849)
            - ``'MERIT'``: MERIT 1983 ellipsoid
            - ``'TOPEX'``: TOPEX/POSEIDON ellipsoid
            - ``'EGM96'``: EGM 1996 gravity model
            - ``'IAG75'``: International Association of Geodesy (1975)
            - ``'IAU64'``: International Astronomical Union (1964)
            - ``'IAU76'``: International Astronomical Union (1976)
            - ``'IERS89'``: IERS Numerical Standards (1989)
            - ``'IERS'``: IERS Numerical Standards (2010)
    units: str, default `MKS`
        Output units

            - ``'MKS'``: meters, kilograms, seconds
            - ``'CGS'``: centimeters, grams, seconds

    Attributes
    ----------
    a_axis: float
        Semi-major axis of the ellipsoid
    flat: float
        Flattening of the ellipsoid
    omega: float
        Angular velocity of the Earth
    GM: float
        Geocentric gravitational constant
    """
    np.seterr(invalid='ignore')
    def __init__(self, **kwargs):
        # set default keyword arguments
        kwargs.setdefault('ellipsoid', 'WGS84')
        kwargs.setdefault('units', 'MKS')
        kwargs.setdefault('a_axis', None)
        kwargs.setdefault('flat', None)
        kwargs.setdefault('GM', None)
        kwargs.setdefault('omega', None)
        # set ellipsoid name and units
        self.units = kwargs['units'].upper()
        if ((kwargs['a_axis'] is not None) and (kwargs['flat'] is not None)):
            self.name = 'user_defined'
        else:
            self.name = kwargs['ellipsoid'].upper()
        # validate ellipsoid and units
        assert self.name in _ellipsoids + ['user_defined']
        assert self.units in _units

        # set parameters for ellipsoid
        if self.name in ('CLK66', 'NAD27'):
            # Clarke 1866
            self.a_axis = 6378206.4# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/294.9786982# flattening of the ellipsoid

        elif (self.name == 'CLK80'):
            # Clarke 1880
            self.a_axis = 6378249.145# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/293.465# flattening of the ellipsoid

        elif self.name in ('GRS80', 'NAD83'):
            # Geodetic Reference System 1980
            # North American Datum 1983
            self.a_axis = 6378135.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.26# flattening of the ellipsoid
            self.GM = 3.986005e14# [m^3/s^2] Geocentric Gravitational Constant

        elif (self.name == 'GRS67'):
            # Geodetic Reference System 1967
            self.a_axis = 6378160.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.247167427# flattening of the ellipsoid
            self.GM = 3.98603e14# [m^3/s^2] Geocentric Gravitational Constant
            self.omega = 7292115.1467e-11# angular velocity of the Earth [rad/s]

        elif (self.name == 'WGS60'):
            # World Geodetic System 1960
            self.a_axis = 6378165.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.30# flattening of the ellipsoid

        elif (self.name == 'WGS66'):
            # World Geodetic System 1966
            self.a_axis = 6378145.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.25# flattening of the ellipsoid

        elif (self.name == 'WGS72'):
            # World Geodetic System 1972
            self.a_axis = 6378135.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.26# flattening of the ellipsoid

        elif (self.name == 'WGS84'):
            # World Geodetic System 1984
            self.a_axis = 6378137.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.257223563# flattening of the ellipsoid

        elif (self.name == 'ATS77'):
            # Quasi-earth centred ellipsoid for ATS77
            self.a_axis = 6378135.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.257# flattening of the ellipsoid

        elif (self.name == 'KRASS'):
            # Krassovsky (USSR)
            self.a_axis = 6378245.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.3# flattening of the ellipsoid

        elif (self.name == 'SGS90'):
            # Soviet Geodetic System 1990
            self.a_axis = 6378136.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.2578393# flattening of the ellipsoid

        elif (self.name == 'INTER'):
            # International 1924
            self.a_axis = 6378388.0# [m] semimajor axis of the ellipsoid
            self.flat = 1/297.0# flattening of the ellipsoid

        elif (self.name == 'AIRY'):
            # Airy 1830 Ellipsoid
            self.a_axis = 6377563.396# [m] semimajor axis of the ellipsoid
            self.flat = 1/299.3249646# flattening of the ellipsoid

        elif (self.name == 'MAIRY'):
            # Modified Airy 1849 Ellipsoid
            self.a_axis = 6377340.189# [m] semimajor axis of the ellipsoid
            self.flat = 1/299.3249646# flattening of the ellipsoid

        elif (self.name == 'HLMRT'):
            # Helmert 1906 Ellipsoid
            self.a_axis = 6378200.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.3# flattening of the ellipsoid

        elif (self.name == 'HOUGH'):
            # Hough 1960 Ellipsoid
            self.a_axis = 6378270.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/297.0# flattening of the ellipsoid

        elif (self.name == 'HGH80'):
            # Hughes 1980 Ellipsoid
            self.a_axis = 6378273.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.279411123064# flattening of the ellipsoid

        elif (self.name == 'MERIT'):
            # MERIT 1983 ellipsoid
            self.a_axis = 6378137.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.257# flattening of the ellipsoid

        elif (self.name == 'TOPEX'):
            # TOPEX/POSEIDON ellipsoid
            self.a_axis = 6378136.3# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.257# flattening of the ellipsoid
            self.GM = 3.986004415e14# [m^3/s^2]

        elif (self.name == 'EGM96'):
            # EGM 1996 gravity model
            self.a_axis = 6378136.3# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.256415099# flattening of the ellipsoid
            self.GM = 3.986004415e14# [m^3/s^2]

        elif (self.name == 'IAG75'):
            # International Association of Geodesy (IAG 1975)
            self.a_axis = 6378140.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.257# flattening of the ellipsoid

        elif (self.name == 'IAU64'):
            # International Astronomical Union (IAU 1964)
            self.a_axis = 6378160.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.25# flattening of the ellipsoid

        elif (self.name == 'IAU76'):
            # International Astronomical Union (IAU 1964)
            self.a_axis = 6378140.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.257# flattening of the ellipsoid

        elif (self.name == 'IERS89'):
            # IERS Numerical Standards (1989)
            self.a_axis = 6378136.0# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.257# flattening of the ellipsoid

        elif (self.name == 'IERS'):
            # IERS Numerical Standards (2010)
            self.a_axis = 6378136.6# [m] semimajor axis of the ellipsoid
            self.flat = 1.0/298.25642# flattening of the ellipsoid

        elif (self.name == 'user_defined'):
            # custom datum
            self.a_axis = np.float64(kwargs['a_axis'])
            self.flat = np.float64(kwargs['flat'])

        # set default parameters if not listed as part of ellipsoid
        # Geocentric Gravitational Constant
        if kwargs['GM'] is not None:
            # user defined Geocentric Gravitational Constant
            self.GM = np.float64(kwargs['GM'])
        elif self.name not in ('GRS80', 'GRS67', 'NAD83', 'TOPEX', 'EGM96'):
            # for ellipsoids not listing the Geocentric Gravitational Constant
            self.GM = 3.986004418e14# [m^3/s^2]

        # angular velocity of the Earth
        if kwargs['omega'] is not None:
            # user defined angular velocity of the Earth
            self.omega = np.float64(kwargs['omega'])
        elif self.name not in ('GRS67'):
            # for ellipsoids not listing the angular velocity of the Earth
            self.omega = 7292115e-11# [rad/s]

        # universal gravitational constant [N*m^2/kg^2]
        self.G = 6.67430e-11

        # standard gravitational acceleration [m/s^2]
        # (World Meteorological Organization)
        self.gamma = 9.80665

        # convert units to CGS
        if (self.units == 'CGS'):
            self.a_axis *= 100.0
            self.GM *= 1e6
            self.G *= 1000.0 # [dyn*cm^2/g^2]
            self.gamma *= 100.0

    # mean radius of the Earth having the same volume
    # (4pi/3)R^3 = (4pi/3)(a^2)b = (4pi/3)(a^3)(1 - f)
    @property
    def rad_e(self) -> float:
        """Average radius of the Earth with same volume as ellipsoid
        """
        return self.a_axis*(1.0 - self.flat)**(1.0/3.0)

    # semiminor axis of the ellipsoid
    @property
    def b_axis(self) -> float:
        """Semi-minor axis of the ellipsoid
        """
        return (1.0 - self.flat)*self.a_axis

    # Ratio between ellipsoidal axes
    @property
    def ratio(self) -> float:
        """Ratio between ellipsoidal axes
        """
        return (1.0 - self.flat)

    # Polar radius of curvature
    @property
    def rad_p(self) -> float:
        """Polar radius of curvature
        """
        return self.a_axis/(1.0 - self.flat)

    # Linear eccentricity
    @property
    def ecc(self) -> float:
        """Linear eccentricity
        """
        return np.sqrt((2.0*self.flat - self.flat**2)*self.a_axis**2)

    # first numerical eccentricity
    @property
    def ecc1(self) -> float:
        """First numerical eccentricity
        """
        return self.ecc/self.a_axis

    # second numerical eccentricity
    @property
    def ecc2(self) -> float:
        """Second numerical eccentricity
        """
        return self.ecc/self.b_axis

    # m parameter [omega^2*a^2*b/(GM)]
    # p. 70, Eqn.(2-137)
    @property
    def m(self) -> float:
        """m Parameter
        """
        return self.omega**2*((1 - self.flat)*self.a_axis**3)/self.GM

    # flattening f2 component
    # p. 80, Eqn.(2-200)
    @property
    def f2(self) -> float:
        """f2 component
        """
        return -self.flat + (5.0/2.0)*self.m + (1.0/2.0)*self.flat**2.0 - \
            (26.0/7.0)*self.flat*self.m + (15.0/4.0)*self.m**2.0

    # flattening f4 component
    # p. 80, Eqn.(2-200)
    @property
    def f4(self) -> float:
        """f4 component
        """
        return -(1.0/2.0)*self.flat**2.0 + (5.0/2.0)*self.flat*self.m

    # q
    # p. 67, Eqn.(2-113)
    @property
    def q(self) -> float:
        """q Parameter
        """
        return 0.5*((1.0 + 3.0/(self.ecc2**2))*np.arctan(self.ecc2)-3.0/self.ecc2)

    # q_0
    # p. 67, Eqn.(2-113)
    @property
    def q0(self) -> float:
        r"""q\ :sub:`0` Parameter
        """
        return 3*(1.0 + 1.0/(self.ecc2**2)) * \
            (1.0 -1.0/self.ecc2*np.arctan(self.ecc2)) - 1.0

    # J_2 p. 75 Eqn.(2-167), p. 76 Eqn.(2-172)
    @property
    def J2(self) -> float:
        """Oblateness coefficient
        """
        return (self.ecc1**2)*(1.0 - 2.0*self.m*self.ecc2/(15.0*self.q))/3.0

    # Normalized C20 harmonic
    # p. 60, Eqn.(2-80)
    @property
    def C20(self) -> float:
        r"""Normalized C\ :sub:`20` harmonic
        """
        return -self.J2/np.sqrt(5.0)

    # Normal gravity at the equator
    # p. 79, Eqn.(2-286)
    @property
    def gamma_a(self) -> float:
        """Normal gravity at the equator
        """
        return (self.GM/(self.a_axis*self.b_axis)) * \
            (1.0 - (3.0/2.0)*self.m - (3.0/14.0)*self.ecc2**2.0*self.m)

    # Normal gravity at the pole
    # p. 79, Eqn.(2-286)
    @property
    def gamma_b(self) -> float:
        """Normal gravity at the pole
        """
        return (self.GM/(self.a_axis**2)) * \
            (1.0 + self.m + (3.0/7.0)*self.ecc2**2.0*self.m)

    # Normal gravity at location
    # p. 80, Eqn.(2-199)
    def gamma_0(self, theta) -> float:
        """Normal gravity at colatitudes

        Parameters
        ----------
        theta: float
            Colatitudes in radians
        """
        return self.gamma_a*(1.0 + self.f2*np.cos(theta)**2.0 + self.f4*np.cos(theta)**4.0)

    # Normal gravity at location
    # p. 82, Eqn.(2-215)
    def gamma_h(self, theta, height) -> float:
        """Normal gravity at colatitudes and heights

        Parameters
        ----------
        theta: float
            Colatitudes in radians
        height: float
            Height above ellipsoid
        """
        return self.gamma_0(theta) * \
            (1.0 - (2.0/self.a_axis) * \
            (1.0 + self.flat + self.m - 2.0*self.flat*np.cos(theta)**2.0)*height +
            (3.0/self.a_axis**2.0)*height**2.0)

    # ratio between gravity at pole versus gravity at equator
    @property
    def dk(self) -> float:
        """Ratio between gravity at pole versus gravity at equator
        """
        return self.b_axis*self.gamma_b/(self.a_axis*self.gamma_b) - 1.0

    # Normal potential at the ellipsoid
    # p. 68, Eqn.(2-123)
    @property
    def U0(self) -> float:
        """Normal potential at the ellipsoid
        """
        return self.GM/self.ecc*np.arctan(self.ecc2) + \
            (1.0/3.0)*self.omega**2*self.a_axis**2

    # Surface area of the reference ellipsoid
    @property
    def area(self) -> float:
        """Surface area of the ellipsoid
        """
        return np.pi*self.a_axis**2.0 * \
            (2.0 + ((1.0 - self.ecc1**2)/self.ecc1) *
                np.log((1.0 + self.ecc1)/(1.0 - self.ecc1)))

    # Volume of the reference ellipsoid
    @property
    def volume(self) -> float:
        """Volume of the ellipsoid
        """
        return (4.0*np.pi/3.0)*(self.a_axis**3.0)*(1.0 - self.ecc1**2.0)**0.5

    # Average density
    @property
    def rho_e(self) -> float:
        """Average density
        """
        return self.GM/(self.G*self.volume)

    def __str__(self):
        """String representation of the ``datum`` object
        """
        properties = ['pyTMD.datum']
        properties.append(f"    name: {self.name}")
        properties.append(f"    units: {self.units}")
        return '\n'.join(properties)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

def convert_ellipsoid(
        lat1: np.ndarray,
        h1: np.ndarray,
        a1: float,
        f1: float,
        a2: float,
        f2: float,
        eps: float = 1e-12,
        itmax: int = 10
    ):
    """
    Convert latitudes and heights to a different ellipsoid using
    Newton-Raphson :cite:p:`Meeus:1991vh`

    Parameters
    ----------
    lat1: np.ndarray
        latitude of input ellipsoid in degrees
    h1: np.ndarray
        height above input ellipsoid in meters
    a1: float
        semi-major axis of input ellipsoid
    f1: float
        flattening of input ellipsoid
    a2: float
        semi-major axis of output ellipsoid
    f2: float
        flattening of output ellipsoid
    eps: float, default 1e-12
        tolerance to prevent division by small numbers and
        to determine convergence
    itmax: int, default 10
        maximum number of iterations to use in Newton-Raphson

    Returns
    -------
    lat2: np.ndarray
        latitude of output ellipsoid in degrees
    h2: np.ndarray
        height above output ellipsoid in meters
    """
    if (len(lat1) != len(h1)):
        raise ValueError('lat and h have incompatible dimensions')
    # semiminor axis of input and output ellipsoid
    b1 = (1.0 - f1)*a1
    b2 = (1.0 - f2)*a2
    # initialize output arrays
    npts = len(lat1)
    lat2 = np.zeros((npts))
    h2 = np.zeros((npts))
    # for each point
    for N in range(npts):
        # force lat1 into range -90 <= lat1 <= 90
        if (np.abs(lat1[N]) > 90.0):
            lat1[N] = np.sign(lat1[N])*90.0
        # handle special case near the equator
        # lat2 = lat1 (latitudes congruent)
        # h2 = h1 + a1 - a2
        if (np.abs(lat1[N]) < eps):
            lat2[N] = np.copy(lat1[N])
            h2[N] = h1[N] + a1 - a2
        # handle special case near the poles
        # lat2 = lat1 (latitudes congruent)
        # h2 = h1 + b1 - b2
        elif ((90.0 - np.abs(lat1[N])) < eps):
            lat2[N] = np.copy(lat1[N])
            h2[N] = h1[N] + b1 - b2
        # handle case if latitude is within 45 degrees of equator
        elif (np.abs(lat1[N]) <= 45):
            # convert lat1 to radians
            lat1r = np.radians(lat1[N])
            sinlat1 = np.sin(lat1r)
            coslat1 = np.cos(lat1r)
            # prevent division by very small numbers
            coslat1 = np.copy(eps) if (coslat1 < eps) else coslat1
            # calculate tangent
            tanlat1 = sinlat1 / coslat1
            u1 = np.arctan(b1 / a1 * tanlat1)
            hpr1sin = b1 * np.sin(u1) + h1[N] * sinlat1
            hpr1cos = a1 * np.cos(u1) + h1[N] * coslat1
            # set initial value for u2
            u2 = np.copy(u1)
            # setup constants
            k0 = b2 * b2 - a2 * a2
            k1 = a2 * hpr1cos
            k2 = b2 * hpr1sin
            # perform newton-raphson iteration to solve for u2
            # cos(u2) will not be close to zero since abs(lat1) <= 45
            for i in range(0, itmax+1):
                cosu2 = np.cos(u2)
                fu2 = k0 * np.sin(u2) + k1 * np.tan(u2) - k2
                fu2p = k0 * cosu2 + k1 / (cosu2 * cosu2)
                if (np.abs(fu2p) < eps):
                    break
                else:
                    delta = fu2 / fu2p
                    u2 -= delta
                    if (np.abs(delta) < eps):
                        break
            # convert latitude to degrees and verify values between +/- 90
            lat2r = np.arctan(a2 / b2 * np.tan(u2))
            lat2[N] = np.clip(np.degrees(lat2r), -90.0, 90.0)
            # calculate height
            h2[N] = (hpr1cos - a2 * np.cos(u2)) / np.cos(lat2r)
        # handle final case where latitudes are between 45 degrees and pole
        else:
            # convert lat1 to radians
            lat1r = np.radians(lat1[N])
            sinlat1 = np.sin(lat1r)
            coslat1 = np.cos(lat1r)
            # prevent division by very small numbers
            coslat1 = np.copy(eps) if (coslat1 < eps) else coslat1
            # calculate tangent
            tanlat1 = sinlat1 / coslat1
            u1 = np.arctan(b1 / a1 * tanlat1)
            hpr1sin = b1 * np.sin(u1) + h1[N] * sinlat1
            hpr1cos = a1 * np.cos(u1) + h1[N] * coslat1
            # set initial value for u2
            u2 = np.copy(u1)
            # setup constants
            k0 = a2 * a2 - b2 * b2
            k1 = b2 * hpr1sin
            k2 = a2 * hpr1cos
            # perform newton-raphson iteration to solve for u2
            # sin(u2) will not be close to zero since abs(lat1) > 45
            for i in range(0, itmax+1):
                sinu2 = np.sin(u2)
                fu2 = k0 * np.cos(u2) + k1 / np.tan(u2) - k2
                fu2p = -1 * (k0 * sinu2 + k1 / (sinu2 * sinu2))
                if (np.abs(fu2p) < eps):
                    break
                else:
                    delta = fu2 / fu2p
                    u2 -= delta
                    if (np.abs(delta) < eps):
                        break
            # convert latitude to degrees and verify values between +/- 90
            lat2r = np.arctan(a2 / b2 * np.tan(u2))
            lat2[N] = np.clip(np.degrees(lat2r), -90.0, 90.0)
            # calculate height
            h2[N] = (hpr1sin - b2 * np.sin(u2)) / np.sin(lat2r)

    # return the latitude and height
    return (lat2, h2)

def compute_delta_h(
        lat: np.ndarray,
        a1: float,
        f1: float,
        a2: float,
        f2: float
    ):
    """
    Compute difference in elevation for two ellipsoids at a given
    latitude using a simplified empirical relation :cite:p:`Meeus:1991vh`

    Parameters
    ----------
    lat: np.ndarray
        latitudes (degrees north)
    a1: float
        semi-major axis of input ellipsoid
    f1: float
        flattening of input ellipsoid
    a2: float
        semi-major axis of output ellipsoid
    f2: float
        flattening of output ellipsoid

    Returns
    -------
    delta_h: np.ndarray
        difference in elevation for two ellipsoids
    """
    # force latitudes to be within -90 to 90 and convert to radians
    phi = np.radians(np.clip(lat, -90.0, 90.0))
    # semi-minor axis of input and output ellipsoid
    b1 = (1.0 - f1)*a1
    b2 = (1.0 - f2)*a2
    # compute differences in semi-major and semi-minor axes
    delta_a = a2 - a1
    delta_b = b2 - b1
    # compute differences between ellipsoids
    # delta_h = -(delta_a * cos(phi)^2 + delta_b * sin(phi)^2)
    delta_h = -(delta_a*np.cos(phi)**2 + delta_b*np.sin(phi)**2)
    return delta_h

def wrap_longitudes(lon: float | np.ndarray):
    """
    Wraps longitudes to range from -180 to +180

    Parameters
    ----------
    lon: float or np.ndarray
        longitude (degrees east)
    """
    phi = np.arctan2(np.sin(np.radians(lon)), np.cos(np.radians(lon)))
    # convert phi from radians to degrees
    return np.degrees(phi)

def to_dms(d: np.ndarray):
    """
    Convert decimal degrees to degrees, minutes and seconds

    Parameters
    ----------
    d: np.ndarray
        decimal degrees

    Returns
    -------
    degree: np.ndarray
        degrees
    minute: np.ndarray
        minutes (arcminutes)
    second: np.ndarray
        seconds (arcseconds)
    """
    sign = np.sign(d)
    minute, second = np.divmod(np.abs(d)*3600.0, 60.0)
    degree, minute = np.divmod(minute, 60.0)
    return (sign*degree, minute, second)

def from_dms(
        degree: np.ndarray,
        minute: np.ndarray,
        second: np.ndarray
    ):
    """
    Convert degrees, minutes and seconds to decimal degrees

    Parameters
    ----------
    degree: np.ndarray
        degrees
    minute: np.ndarray
        minutes (arcminutes)
    second: np.ndarray
        seconds (arcseconds)

    Returns
    -------
    d: np.ndarray
        decimal degrees
    """
    sign = np.sign(degree)
    d = np.abs(degree) + minute/60.0 + second/3600.0
    return sign*d

# get WGS84 parameters
_wgs84 = datum(ellipsoid='WGS84', units='MKS')

def to_cartesian(
        lon: np.ndarray,
        lat: np.ndarray,
        h: float | np.ndarray = 0.0,
        a_axis: float = _wgs84.a_axis,
        flat: float = _wgs84.flat
    ):
    """
    Converts geodetic coordinates to Cartesian coordinates

    Parameters
    ----------
    lon: np.ndarray
        longitude (degrees east)
    lat: np.ndarray
        latitude (degrees north)
    h: float or np.ndarray, default 0.0
        height above ellipsoid (or sphere)
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid

        for spherical coordinates set to radius of the Earth
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening

        for spherical coordinates set to 0
    """
    # verify axes and copy to not modify inputs
    singular_values = (np.ndim(lon) == 0)
    lon = np.atleast_1d(np.copy(lon)).astype(np.float64)
    lat = np.atleast_1d(np.copy(lat)).astype(np.float64)
    # fix coordinates to be 0:360
    lon[lon < 0] += 360.0
    # Linear eccentricity and first numerical eccentricity
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    ecc1 = lin_ecc/a_axis
    # convert from geodetic latitude to geocentric latitude
    # geodetic latitude in radians
    latitude_geodetic_rad = np.radians(lat)
    # prime vertical radius of curvature
    N = a_axis/np.sqrt(1.0 - ecc1**2.0*np.sin(latitude_geodetic_rad)**2.0)
    # calculate X, Y and Z from geodetic latitude and longitude
    X = (N + h) * np.cos(latitude_geodetic_rad) * np.cos(np.radians(lon))
    Y = (N + h) * np.cos(latitude_geodetic_rad) * np.sin(np.radians(lon))
    Z = (N * (1.0 - ecc1**2.0) + h) * np.sin(latitude_geodetic_rad)
    # return the cartesian coordinates
    # flattened to singular values if necessary
    if singular_values:
        return (X[0], Y[0], Z[0])
    else:
        return (X, Y, Z)

def to_sphere(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Convert from cartesian coordinates to spherical coordinates

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    """
    # verify axes and copy to not modify inputs
    singular_values = (np.ndim(x) == 0)
    x = np.atleast_1d(np.copy(x)).astype(np.float64)
    y = np.atleast_1d(np.copy(y)).astype(np.float64)
    z = np.atleast_1d(np.copy(z)).astype(np.float64)
    # calculate radius
    rad = np.sqrt(x**2.0 + y**2.0 + z**2.0)
    # calculate angular coordinates
    # phi: azimuthal angle
    phi = np.arctan2(y, x)
    # th: polar angle
    th = np.arccos(z/rad)
    # convert to degrees and fix to 0:360
    lon = np.degrees(phi)
    if np.any(lon < 0):
        lt0 = np.nonzero(lon < 0)
        lon[lt0] += 360.0
    # convert to degrees and fix to -90:90
    lat = 90.0 - np.degrees(th)
    np.clip(lat, -90, 90, out=lat)
    # return longitude, latitude and radius
    # flattened to singular values if necessary
    if singular_values:
        return (lon[0], lat[0], rad[0])
    else:
        return (lon, lat, rad)

def to_geodetic(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        a_axis: float = _wgs84.a_axis,
        flat: float = _wgs84.flat,
        method: str = 'bowring',
        eps: float = np.finfo(np.float64).eps,
        iterations: int = 10
    ):
    """
    Convert from cartesian coordinates to geodetic coordinates
    using either iterative or closed-form methods

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    method: str, default 'bowring'
        method to use for conversion

            - ``'moritz'``: iterative solution
            - ``'bowring'``: iterative solution
            - ``'zhu'``: closed-form solution
    eps: float, default np.finfo(np.float64).eps
        tolerance for iterative methods
    iterations: int, default 10
        maximum number of iterations
    """
    # verify axes and copy to not modify inputs
    singular_values = (np.ndim(x) == 0)
    x = np.atleast_1d(np.copy(x)).astype(np.float64)
    y = np.atleast_1d(np.copy(y)).astype(np.float64)
    z = np.atleast_1d(np.copy(z)).astype(np.float64)
    # calculate the geodetic coordinates using the specified method
    if (method.lower() == 'moritz'):
        lon, lat, h = _moritz_iterative(x, y, z,
            a_axis=a_axis,
            flat=flat,
            eps=eps,
            iterations=iterations)
    elif (method.lower() == 'bowring'):
        lon, lat, h = _bowring_iterative(x, y, z,
            a_axis=a_axis,
            flat=flat,
            eps=eps,
            iterations=iterations)
    elif (method.lower() == 'zhu'):
        lon, lat, h = _zhu_closed_form(x, y, z,
            a_axis=a_axis,
            flat=flat)
    else:
        raise ValueError(f'Unknown conversion method: {method}')
    # return longitude, latitude and height
    # flattened to singular values if necessary
    if singular_values:
        return (lon[0], lat[0], h[0])
    else:
        return (lon, lat, h)

def _moritz_iterative(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        a_axis: float = _wgs84.a_axis,
        flat: float = _wgs84.flat,
        eps: float = np.finfo(np.float64).eps,
        iterations: int = 10
    ):
    """
    Convert from cartesian coordinates to geodetic coordinates
    using the iterative solution of :cite:t:`HofmannWellenhof:2006hy`

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    eps: float, default np.finfo(np.float64).eps
        tolerance for iterative method
    iterations: int, default 10
        maximum number of iterations
    """
    # Linear eccentricity and first numerical eccentricity
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    ecc1 = lin_ecc/a_axis
    # calculate longitude
    lon = np.degrees(np.arctan2(y, x))
    # set initial estimate of height to 0
    h = np.zeros_like(lon)
    h0 = np.inf*np.ones_like(lon)
    # calculate radius of parallel
    p = np.sqrt(x**2 + y**2)
    # initial estimated value for phi using h=0
    phi = np.arctan(z/(p*(1.0 - ecc1**2)))
    # iterate to tolerance or to maximum number of iterations
    i = 0
    while np.any(np.abs(h - h0) > eps) and (i <= iterations):
        # copy previous iteration of height
        h0 = np.copy(h)
        # calculate radius of curvature
        N = a_axis/np.sqrt(1.0 - ecc1**2 * np.sin(phi)**2)
        # estimate new value of height
        h = p/np.cos(phi) - N
        # estimate new value for latitude using heights
        phi = np.arctan(z/(p*(1.0 - ecc1**2*N/(N + h))))
        # add to iterator
        i += 1
    # convert latitudes and fix values
    lat = np.clip(np.degrees(phi), -90.0, 90.0)
    # return longitude, latitude and height
    return (lon, lat, h)

def _bowring_iterative(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        a_axis: float = _wgs84.a_axis,
        flat: float = _wgs84.flat,
        eps: float = np.finfo(np.float64).eps,
        iterations: int = 10
    ):
    """
    Convert from cartesian coordinates to geodetic coordinates using
    the iterative solution of :cite:t:`Bowring:1976jh,Bowring:1985du`

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    eps: float, default np.finfo(np.float64).eps
        tolerance for iterative method
    iterations: int, default 10
        maximum number of iterations
    """
    # semiminor axis of the WGS84 ellipsoid [m]
    b_axis = (1.0 - flat)*a_axis
    # Linear eccentricity
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    # square of first and second numerical eccentricity
    e12 = lin_ecc**2/a_axis**2
    e22 = lin_ecc**2/b_axis**2
    # calculate longitude
    lon = np.degrees(np.arctan2(y, x))
    # calculate radius of parallel
    p = np.sqrt(x**2 + y**2)
    # initial estimated value for reduced parametric latitude
    u = np.arctan(a_axis*z/(b_axis*p))
    # initial estimated value for latitude
    phi = np.arctan((z + e22*b_axis*np.sin(u)**3) /
        (p - e12*a_axis*np.cos(u)**3))
    phi0 = np.inf*np.ones_like(lon)
    # iterate to tolerance or to maximum number of iterations
    i = 0
    while np.any(np.abs(phi - phi0) > eps) and (i <= iterations):
        # copy previous iteration of phi
        phi0 = np.copy(phi)
        # calculate reduced parametric latitude
        u = np.arctan(b_axis*np.tan(phi)/a_axis)
        # estimate new value of latitude
        phi = np.arctan((z + e22*b_axis*np.sin(u)**3) /
            (p - e12*a_axis*np.cos(u)**3))
        # add to iterator
        i += 1
    # calculate final radius of curvature
    N = a_axis/np.sqrt(1.0 - e12 * np.sin(phi)**2)
    # estimate final height (Bowring, 1985)
    h = p*np.cos(phi) + z*np.sin(phi) - a_axis**2/N
    # convert latitudes and fix values
    lat = np.clip(np.degrees(phi), -90.0, 90.0)
    # return longitude, latitude and height
    return (lon, lat, h)

def _zhu_closed_form(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        a_axis: float = _wgs84.a_axis,
        flat: float = _wgs84.flat,
    ):
    """
    Convert from cartesian coordinates to geodetic coordinates
    using the closed-form solution of :cite:t:`Zhu:1993ja`

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    """
    # semiminor axis of the WGS84 ellipsoid [m]
    b_axis = (1.0 - flat)*a_axis
    # Linear eccentricity
    lin_ecc = np.sqrt((2.0*flat - flat**2)*a_axis**2)
    # square of first numerical eccentricity
    e12 = lin_ecc**2/a_axis**2
    # calculate longitude
    lon = np.degrees(np.arctan2(y, x))
    # calculate radius of parallel
    w = np.sqrt(x**2 + y**2)
    # allocate for output latitude and height
    lat = np.zeros_like(lon)
    h = np.zeros_like(lon)
    if np.any(w == 0):
        # special case where w == 0 (exact polar solution)
        ind, = np.nonzero(w == 0)
        h[ind] = np.sign(z[ind])*z[ind] - b_axis
        lat[ind] = 90.0*np.sign(z[ind])
    else:
        # all other cases
        ind, = np.nonzero(w != 0)
        l = e12/2.0
        m = (w[ind]/a_axis)**2.0
        n = ((1.0 - e12)*z[ind]/b_axis)**2.0
        i = -(2.0*l**2 + m + n)/2.0
        k = (l**2.0 - m - n)*l**2.0
        q = (1.0/216.0)*(m + n - 4.0*l**2)**3.0 + m*n*l**2.0
        D = np.sqrt((2.0*q - m*n*l**2)*m*n*l**2)
        B = i/3.0 - (q + D)**(1.0/3.0) - (q - D)**(1.0/3.0)
        t = np.sqrt(np.sqrt(B**2-k) - (B + i)/2.0) - \
            np.sign(m - n)*np.sqrt((B - i)/2.0)
        wi = w/(t + l)
        zi = (1.0 - e12)*z[ind]/(t - l)
        # calculate latitude and height
        lat[ind] = np.degrees(np.arctan2(zi, ((1.0 - e12)*wi)))
        h[ind] = np.sign(t-1.0+l)*np.sqrt((w-wi)**2.0 + (z[ind]-zi)**2.0)
    # return longitude, latitude and height
    return (lon, lat, h)

def to_ENU(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        lon0: float | np.ndarray = 0.0,
        lat0: float | np.ndarray = 0.0,
        h0: float | np.ndarray = 0.0,
        a_axis: float = _wgs84.a_axis,
        flat: float = _wgs84.flat,
    ):
    """
    Convert from Earth-Centered Earth-Fixed (ECEF) cartesian coordinates
    to East-North-Up coordinates (ENU)

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    lon0: float or np.ndarray, default 0.0
        reference longitude (degrees east)
    lat0: float or np.ndarray, default 0.0
        reference latitude (degrees north)
    h0: float or np.ndarray, default 0.0
        reference height (meters)
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening

    Returns
    -------
    E: np.ndarray
        east coordinates
    N: np.ndarray
        north coordinates
    U: np.ndarray
        up coordinates
    """
    # verify axes and copy to not modify inputs
    singular_values = (np.ndim(x) == 0)
    x = np.atleast_1d(np.copy(x)).astype(np.float64)
    y = np.atleast_1d(np.copy(y)).astype(np.float64)
    z = np.atleast_1d(np.copy(z)).astype(np.float64)
    # convert latitude and longitude to ECEF
    X0, Y0, Z0 = to_cartesian(lon0, lat0, h=h0, a_axis=a_axis, flat=flat)
    # latitude and longitude in radians
    theta0 = np.radians(lat0)
    lambda0 = np.radians(lon0)
    # calculate the rotation matrix
    R = np.zeros((3, 3))
    R[0,0] = -np.sin(lambda0)
    R[0,1] = np.cos(lambda0)
    R[0,2] = 0.0
    R[1,0] = -np.sin(theta0)*np.cos(lambda0)
    R[1,1] = -np.sin(theta0)*np.sin(lambda0)
    R[1,2] = np.cos(theta0)
    R[2,0] = np.cos(theta0)*np.cos(lambda0)
    R[2,1] = np.cos(theta0)*np.sin(lambda0)
    R[2,2] = np.sin(theta0)
    # calculate the ENU coordinates
    E, N, U = np.dot(R, np.vstack((x - X0, y - Y0, z - Z0)))
    # return the ENU coordinates
    # flattened to singular values if necessary
    if singular_values:
        return (E[0], N[0], U[0])
    else:
        return (E, N, U)

def from_ENU(
        E: np.ndarray,
        N: np.ndarray,
        U: np.ndarray,
        lon0: float | np.ndarray = 0.0,
        lat0: float | np.ndarray = 0.0,
        h0: float | np.ndarray = 0.0,
        a_axis: float = _wgs84.a_axis,
        flat: float = _wgs84.flat,
    ):
    """
    Convert from East-North-Up coordinates (ENU) to
    Earth-Centered Earth-Fixed (ECEF) cartesian coordinates

    Parameters
    ----------
    E, np.ndarray
        east coordinates
    N, np.ndarray
        north coordinates
    U, np.ndarray
        up coordinates
    lon0: float or np.ndarray, default 0.0
        reference longitude (degrees east)
    lat0: float or np.ndarray, default 0.0
        reference latitude (degrees north)
    h0: float or np.ndarray, default 0.0
        reference height (meters)
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening

    Returns
    -------
    x, float
        cartesian x-coordinates
    y, float
        cartesian y-coordinates
    z, float
        cartesian z-coordinates
    """
    # verify axes and copy to not modify inputs
    singular_values = (np.ndim(E) == 0)
    E = np.atleast_1d(np.copy(E)).astype(np.float64)
    N = np.atleast_1d(np.copy(N)).astype(np.float64)
    U = np.atleast_1d(np.copy(U)).astype(np.float64)
    # convert latitude and longitude to ECEF
    X0, Y0, Z0 = to_cartesian(lon0, lat0, h=h0, a_axis=a_axis, flat=flat)
    # latitude and longitude in radians
    theta0 = np.radians(lat0)
    lambda0 = np.radians(lon0)
    # calculate the rotation matrix
    R = np.zeros((3, 3))
    R[0,0] = -np.sin(lambda0)
    R[1,0] = np.cos(lambda0)
    R[2,0] = 0.0
    R[0,1] = -np.sin(theta0)*np.cos(lambda0)
    R[1,1] = -np.sin(theta0)*np.sin(lambda0)
    R[2,1] = np.cos(theta0)
    R[0,2] = np.cos(theta0)*np.cos(lambda0)
    R[1,2] = np.cos(theta0)*np.sin(lambda0)
    R[2,2] = np.sin(theta0)
    # calculate the ECEF coordinates
    x, y, z = np.dot(R, np.vstack((E, N, U)))
    # add reference coordinates
    x += X0
    y += Y0
    z += Z0
    # return the ECEF coordinates
    # flattened to singular values if necessary
    if singular_values:
        return (x[0], y[0], z[0])
    else:
        return (x, y, z)

def to_horizontal(
        E: np.ndarray,
        N: np.ndarray,
        U: np.ndarray,
    ):
    """
    Convert from East-North-Up coordinates (ENU) to a
    celestial horizontal coordinate system (alt-az)

    Parameters
    ----------
    E: np.ndarray
        east coordinates
    N: np.ndarray
        north coordinates
    U: np.ndarray
        up coordinates

    Returns
    -------
    alpha: np.ndarray
        altitude (elevation) angle in degrees
    phi: np.ndarray
        azimuth angle in degrees
    D: np.ndarray
        distance from observer to object in meters
    """
    # calculate distance to object
    # convert coordinates to unit vectors
    D = np.sqrt(E**2 + N**2 + U**2)
    # altitude (elevation) angle in degrees
    alpha = np.degrees(np.arcsin(U/D))
    # azimuth angle in degrees (fixed to 0 to 360)
    phi = np.mod(np.degrees(np.arctan2(E/D, N/D)), 360.0)
    return (alpha, phi, D)

def to_zenith(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        lon0: float | np.ndarray = 0.0,
        lat0: float | np.ndarray = 0.0,
        h0: float | np.ndarray = 0.0,
        a_axis: float = _wgs84.a_axis,
        flat: float = _wgs84.flat,
    ):
    """
    Calculate zenith angle of an object from Earth-Centered
    Earth-Fixed (ECEF) cartesian coordinates

    Parameters
    ----------
    x, np.ndarray
        cartesian x-coordinates
    y, np.ndarray
        cartesian y-coordinates
    z, np.ndarray
        cartesian z-coordinates
    lon0: float or np.ndarray, default 0.0
        reference longitude (degrees east)
    lat0: float or np.ndarray, default 0.0
        reference latitude (degrees north)
    h0: float or np.ndarray, default 0.0
        reference height (meters)
    a_axis: float, default 6378137.0
        semimajor axis of the ellipsoid
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening

    Returns
    -------
    zenith: np.ndarray
        zenith angle of object in degrees
    """
    # convert from ECEF to ENU
    E, N, U = to_ENU(x, y, z, lon0=lon0, lat0=lat0, h0=h0,
        a_axis=a_axis, flat=flat)
    # convert from ENU to horizontal coordinates
    alpha, phi, D = to_horizontal(E, N, U)
    # calculate zenith angle in degrees
    zenith = 90.0 - alpha
    # return zenith angle
    return zenith

def scale_areas(*args, **kwargs):
    warnings.warn("Deprecated. Please use pyTMD.spatial.scale_factors instead",
        DeprecationWarning)
    return scale_factors(*args, **kwargs)

def scale_factors(
        lat: np.ndarray,
        flat: float = _wgs84.flat,
        reference_latitude: float = 70.0,
        metric: str = 'area'
    ):
    """
    Calculates scaling factors to account for polar stereographic
    distortion including special case of at the exact pole
    :cite:p:`Snyder:1982gf`

    Parameters
    ----------
    lat: np.ndarray
        latitude (degrees north)
    flat: float, default 1.0/298.257223563
        ellipsoidal flattening
    reference_latitude: float, default 70.0
        reference latitude (true scale latitude)
    metric: str, default 'area'
        metric to calculate scaling factors

            - ``'distance'``: scale factors for distance
            - ``'area'``: scale factors for area

    Returns
    -------
    scale: np.ndarray
        scaling factors at input latitudes
    """
    assert metric.lower() in ['distance', 'area'], 'Unknown metric'
    # convert latitude from degrees to positive radians
    theta = np.radians(np.abs(lat))
    # convert reference latitude from degrees to positive radians
    theta_ref = np.radians(np.abs(reference_latitude))
    # square of the eccentricity of the ellipsoid
    # ecc2 = (1-b**2/a**2) = 2.0*flat - flat^2
    ecc2 = 2.0*flat - flat**2
    # eccentricity of the ellipsoid
    ecc = np.sqrt(ecc2)
    # calculate ratio at input latitudes
    m = np.cos(theta)/np.sqrt(1.0 - ecc2*np.sin(theta)**2)
    t = np.tan(np.pi/4.0 - theta/2.0)/((1.0 - ecc*np.sin(theta)) / \
        (1.0 + ecc*np.sin(theta)))**(ecc/2.0)
    # calculate ratio at reference latitude
    mref = np.cos(theta_ref)/np.sqrt(1.0 - ecc2*np.sin(theta_ref)**2)
    tref = np.tan(np.pi/4.0 - theta_ref/2.0)/((1.0 - ecc*np.sin(theta_ref)) / \
        (1.0 + ecc*np.sin(theta_ref)))**(ecc/2.0)
    # distance scaling
    k = (mref/m)*(t/tref)
    kp = 0.5*mref*np.sqrt(((1.0+ecc)**(1.0+ecc))*((1.0-ecc)**(1.0-ecc)))/tref
    if (metric.lower() == 'distance'):
        # distance scaling
        scale = np.where(np.isclose(theta, np.pi/2.0), 1.0/kp, 1.0/k)
    elif (metric.lower() == 'area'):
        # area scaling
        scale = np.where(np.isclose(theta, np.pi/2.0), 1.0/(kp**2), 1.0/(k**2))
    return scale
