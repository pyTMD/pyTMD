#!/usr/bin/env python
u"""
compute.py
Written by Tyler Sutterley (10/2025)
Calculates tidal elevations for correcting elevation or imagery data
Calculates tidal currents at locations and times

Ocean and Load Tides
Uses OTIS format tidal solutions provided by Oregon State University and ESR
    http://volkov.oce.orst.edu/tides/region.html
    https://www.esr.org/research/polar-tide-models/list-of-polar-tide-models/
    ftp://ftp.esr.org/pub/datasets/tmd/
Global Tide Model (GOT) solutions provided by Richard Ray at GSFC
or Finite Element Solution (FES) models provided by AVISO

Long-Period Equilibrium Tides (LPET)
Calculates long-period equilibrium tidal elevations for correcting
elevation or imagery data from the summation of fifteen spectral lines
    https://doi.org/10.1111/j.1365-246X.1973.tb03420.x

Load Pole Tides (LPT)
Calculates radial load pole tide displacements following IERS Convention
(2010) guidelines for correcting elevation or imagery data
    https://iers-conventions.obspm.fr/chapter7.php

Ocean Pole Tides (OPT)
Calculates radial ocean pole load tide displacements following IERS Convention
(2010) guidelines for correcting elevation or imagery data
    https://iers-conventions.obspm.fr/chapter7.php

Solid Earth Tides (SET)
Calculates radial Solid Earth tide displacements following IERS Convention
(2010) guidelines for correcting elevation or imagery data
    https://iers-conventions.obspm.fr/chapter7.php
Or by using a tide potential catalog following Cartwright and Tayler (1971)

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    time.py: utilities for calculating time operations
    spatial: utilities for reading, writing and operating on spatial data
    utilities.py: download and management utilities for syncing files
    astro.py: computes the basic astronomical mean longitudes
    constituents.py: calculates constituent parameters and nodal arguments  
    crs.py: Coordinate Reference System (CRS) routines
    predict.py: predict tide values using harmonic constants
    io/model.py: retrieves tide model parameters for named tide models
    io/OTIS.py: extract tidal harmonic constants from OTIS tide models
    io/ATLAS.py: extract tidal harmonic constants from netcdf models
    io/GOT.py: extract tidal harmonic constants from GSFC GOT models
    io/FES.py: extract tidal harmonic constants from FES tide models
    interpolate.py: interpolation routines for spatial data

UPDATE HISTORY:
    Updated 10/2025: change default directory for tide models to cache
    Updated 09/2025: added wrapper for calculating solid earth tides
        using a tide potential catalog following Cartwright and Tayler (1971)
    Updated 08/2025: convert angles with numpy radians and degrees functions
        pass kwargs to computation of long-period equilibrium tides
        use timescale shortcut wrapper functions to create Timescale objects
    Updated 07/2025: mask mean pole values prior to valid epoch of convention
        add a default directory for tide models
    Updated 05/2025: added option to select constituents to read from model
    Updated 12/2024: moved check points function as compute.tide_masks
    Updated 11/2024: expose buffer distance for cropping tide model data
    Updated 10/2024: compute delta times based on corrections type
        simplify by using wrapper functions to read and interpolate constants
        added option to append equilibrium amplitudes for node tides
    Updated 09/2024: use JSON database for known model parameters
        drop support for the ascii definition file format
        use model class attributes for file format and corrections
        add keyword argument to select nodal corrections type
        fix to use case insensitive assertions of string argument values
        add model attribute for tide model bulk frequencies
    Updated 08/2024: allow inferring only specific minor constituents
        use prediction functions for pole tides in cartesian coordinates
        use rotation matrix to convert from cartesian to spherical
    Updated 07/2024: assert that data type is a known value
        make number of days to convert JD to MJD a variable
        added option to crop tide models to the domain of the input data
        added option to use JSON format definition files
        renamed format for ATLAS to ATLAS-compact
        renamed format for netcdf to ATLAS-netcdf
        renamed format for FES to FES-netcdf and added FES-ascii
        renamed format for GOT to GOT-ascii and added GOT-netcdf
        drop use of heights when converting to cartesian coordinates
        use prediction function to calculate cartesian tide displacements
    Updated 06/2024: use np.clongdouble instead of np.longcomplex
    Updated 04/2024: use wrapper to importlib for optional dependencies
    Updated 02/2024: changed class name for ellipsoid parameters to datum
    Updated 01/2024: made the inference of minor constituents an option
        refactored lunisolar ephemerides functions
        renamed module to compute and added tidal currents function
    Updated 12/2023: use new crs class for coordinate reprojection
    Updated 08/2023: changed ESR netCDF4 format to TMD3 format
    Updated 05/2023: use timescale class for time conversion operations
        use defaults from eop module for pole tide and EOP files
        add option for using higher resolution ephemerides from JPL
    Updated 04/2023: added function for radial solid earth tides
        using pathlib to define and expand paths
    Updated 03/2023: add basic variable typing to function inputs
        added function for long-period equilibrium tides
        added function for radial load pole tides
        added function for radial ocean pole tides
    Updated 12/2022: refactored tide read and prediction programs
    Updated 11/2022: place some imports within try/except statements
        use f-strings for formatting verbose or ascii output
    Updated 05/2022: added ESR netCDF4 formats to list of model types
        updated keyword arguments to read tide model programs
        added option to apply flexure to heights for applicable models
    Updated 04/2022: updated docstrings to numpy documentation format
    Updated 12/2021: added function to calculate a tidal time series
        verify coordinate dimensions for each input data type
        added option for converting from LORAN times to UTC
    Updated 09/2021: refactor to use model class for files and attributes
    Updated 07/2021: can use numpy datetime arrays as input time variable
        added function for determining the input spatial variable type
        added check that tide model directory is accessible
    Updated 06/2021: added new Gr1km-v2 1km Greenland model from ESR
        add try/except for input projection strings
    Updated 05/2021: added option for extrapolation cutoff in kilometers
    Updated 03/2021: added TPXO9-atlas-v4 in binary OTIS format
        simplified netcdf inputs to be similar to binary OTIS read program
    Updated 02/2021: replaced numpy bool to prevent deprecation warning
    Updated 12/2020: added valid data extrapolation with nearest_extrap
    Updated 11/2020: added model constituents from TPXO9-atlas-v3
    Updated 08/2020: using builtin time operations.
        calculate difference in leap seconds from start of epoch
        using conversion protocols following pyproj-2 updates
    Updated 07/2020: added function docstrings, FES2014 and TPXO9-atlas-v2
        use merged delta time files combining biannual, monthly and daily files
    Updated 03/2020: added TYPE, TIME, FILL_VALUE and METHOD options
    Written 03/2020
"""
from __future__ import print_function, annotations

import logging
import pathlib
import numpy as np
from io import IOBase
import scipy.interpolate
import pyTMD.crs
import pyTMD.io
import pyTMD.io.model
import pyTMD.predict
import pyTMD.spatial
import pyTMD.utilities
import timescale.eop
import timescale.time
# attempt imports
pyproj = pyTMD.utilities.import_dependency('pyproj')
xr = pyTMD.utilities.import_dependency('xarray')

__all__ = [
    "corrections",
    "tide_elevations",
    "tide_currents",
    "tide_masks",
    "LPET_elevations",
    "LPT_displacements",
    "OPT_displacements",
    "SET_displacements"
]

# number of days between the Julian day epoch and MJD
_jd_mjd = 2400000.5

# default working data directory for tide models
_default_directory = pyTMD.utilities.get_cache_path()

# PURPOSE: wrapper function for computing values
def corrections(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        CORRECTION: str = 'ocean',
        **kwargs
    ):
    """
    Wrapper function to compute tide corrections at points and times

    Parameters
    ----------
    x: np.ndarray
        x-coordinates in projection EPSG
    y: np.ndarray
        y-coordinates in projection EPSG
    delta_time: np.ndarray
        seconds since EPOCH or datetime array
    CORRECTION: str, default 'ocean'
        Correction type to compute

            - ``'ocean'``: ocean tide from model constituents
            - ``'load'``: load tide from model constituents
            - ``'LPET'``: long-period equilibrium tide
            - ``'LPT'``: solid earth load pole tide
            - ``'OPT'``: ocean pole tide
            - ``'SET'``: solid earth tide
    **kwargs: dict
        keyword arguments for correction functions

    Returns
    -------
    values: np.ndarray
        tidal correction at coordinates and time in meters
    """
    if CORRECTION.lower() in ('ocean', 'load'):
        return tide_elevations(x, y, delta_time, **kwargs)
    elif (CORRECTION.upper() == 'LPET'):
        return LPET_elevations(x, y, delta_time, **kwargs)
    elif (CORRECTION.upper() == 'LPT'):
        return LPT_displacements(x, y, delta_time, **kwargs)
    elif (CORRECTION.upper() == 'OPT'):
        return OPT_displacements(x, y, delta_time, **kwargs)
    elif (CORRECTION.upper() == 'SET'):
        return SET_displacements(x, y, delta_time, **kwargs)
    else:
        raise ValueError(f'Unrecognized correction type: {CORRECTION}')

# PURPOSE: compute tides at points and times using tide model algorithms
def tide_elevations(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        DIRECTORY: str | pathlib.Path | None = _default_directory,
        MODEL: str | None = None,
        DEFINITION_FILE: str | pathlib.Path | IOBase | None = None,
        CROP: bool = False,
        BOUNDS: list | np.ndarray | None = None,
        BUFFER: int | float = 0,
        EPSG: str | int = 4326,
        EPOCH: list | tuple = (2000, 1, 1, 0, 0, 0),
        TYPE: str | None = 'drift',
        TIME: str = 'UTC',
        METHOD: str = 'linear',
        EXTRAPOLATE: bool = False,
        CUTOFF: int | float = 10.0,
        CORRECTIONS: str | None = None,
        CONSTITUENTS: list | None = None,
        INFER_MINOR: bool = True,
        MINOR_CONSTITUENTS: list | None = None,
        APPEND_NODE: bool = False,
        APPLY_FLEXURE: bool = False,
        **kwargs
    ):
    """
    Compute ocean or load tides at points and times from
    model constituents

    Parameters
    ----------
    x: np.ndarray
        x-coordinates in projection EPSG
    y: np.ndarray
        y-coordinates in projection EPSG
    delta_time: np.ndarray
        seconds since EPOCH or datetime array
    DIRECTORY: str or NoneType, default None
        working data directory for tide models
    MODEL: str or NoneType, default None
        Tide model to use in correction
    DEFINITION_FILE: str, pathlib.Path, io.IOBase or NoneType, default None
        Tide model definition file for use
    CROP: bool, default False
        Crop tide model data to (buffered) bounds
    BOUNDS: list, np.ndarray or NoneType, default None
        Boundaries for cropping tide model data
    BUFFER: int or float, default 0
        Buffer distance for cropping tide model data
    EPSG: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    EPOCH: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    TYPE: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    TIME: str, default 'UTC'
        Time type if need to compute leap seconds to convert to UTC

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    METHOD: str
        Interpolation method

            - ```bilinear```: quick bilinear interpolation
            - ```spline```: scipy bivariate spline interpolation
            - ```linear```, ```nearest```: scipy regular grid interpolations

    EXTRAPOLATE: bool, default False
        Extrapolate with nearest-neighbors
    CUTOFF: int or float, default 10.0
        Extrapolation cutoff in kilometers

        Set to ``np.inf`` to extrapolate for all points
    CORRECTIONS: str or None, default None
        Nodal correction type, default based on model
    CONSTITUENTS: list or None, default None
        Specify constituents to read from model
    INFER_MINOR: bool, default True
        Infer the height values for minor tidal constituents
    MINOR_CONSTITUENTS: list or None, default None
        Specify constituents to infer
    APPEND_NODE: bool, default False
        Append equilibrium amplitudes for node tides
    APPLY_FLEXURE: bool, default False
        Apply ice flexure scaling factor to height values

        Only valid for models containing flexure fields

    Returns
    -------
    tide: xarray.DataArray
        tidal elevation in meters
    """

    # check that tide directory is accessible
    if DIRECTORY is not None:
        DIRECTORY = pathlib.Path(DIRECTORY).expanduser()
        if not DIRECTORY.exists():
            raise FileNotFoundError("Invalid tide directory")

    # validate input arguments
    assert TIME.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert METHOD.lower() in ('bilinear', 'spline', 'linear', 'nearest')

    # get parameters for tide model
    if DEFINITION_FILE is not None:
        model = pyTMD.io.model(DIRECTORY).from_file(DEFINITION_FILE)
    else:
        model = pyTMD.io.model(DIRECTORY).from_database(MODEL)
    # open dataset
    ds = model.open_dataset(type='z')
    # append_node=APPEND_NODE, apply_flexure=APPLY_FLEXURE
    # subset to constituents
    if CONSTITUENTS:
        ds = ds.subset(CONSTITUENTS)

    # determine input data type based on variable dimensions
    if not TYPE:
        TYPE = pyTMD.spatial.data_type(x, y, delta_time)
    assert TYPE.lower() in ('grid', 'drift', 'time series')
    # convert coordinates to xarray DataArrays
    # in coordinate reference system of model
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        X, Y = ds.tmd.transform(x, y, crs=EPSG)
    elif (TYPE.lower() == 'grid') and (np.size(x) != np.size(y)):
        gridx, gridy = np.meshgrid(x, y)
        mx, my = ds.tmd.transform(gridx, gridy, crs=EPSG)
        X = xr.DataArray(mx, dims=('y','x'))
        Y = xr.DataArray(my, dims=('y','x'))
    elif (TYPE.lower() == 'grid'):
        mx, my = ds.tmd.transform(x, y, crs=EPSG)
        X = xr.DataArray(mx, dims=('y','x'))
        Y = xr.DataArray(my, dims=('y','x'))
    elif (TYPE.lower() == 'drift'):
        mx, my = ds.tmd.transform(x, y, crs=EPSG)
        X = xr.DataArray(mx, dims=('time'))
        Y = xr.DataArray(my, dims=('time'))
    elif (TYPE.lower() == 'time series'):
        mx, my = ds.tmd.transform(x, y, crs=EPSG)
        X = xr.DataArray(mx, dims=('station'))
        Y = xr.DataArray(my, dims=('station'))

    # crop tide model dataset to bounds
    if CROP:
        # default bounds if cropping data
        xmin, xmax = np.min(X), np.max(X)
        ymin, ymax = np.min(Y), np.max(Y)
        bounds = BOUNDS or [xmin, xmax, ymin, ymax]
        # crop dataset to buffered bounds
        ds = ds.tmd.crop(bounds, buffer=BUFFER)

    # convert delta times or datetimes objects to timescale
    if (TIME.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=EPOCH, standard=TIME)

    # nodal corrections to apply
    nodal_corrections = CORRECTIONS or model.corrections
    # minor constituents to infer
    minor_constituents = MINOR_CONSTITUENTS or model.minor
    # delta time (TT - UT1) for tide model
    if nodal_corrections in ('OTIS','ATLAS','TMD3','netcdf'):
        # use delta time at 2000.0 to match TMD outputs
        deltat = np.zeros_like(ts.tt_ut1)
    else:
        # use interpolated delta times
        deltat = ts.tt_ut1

    # interpolate model to grid points
    local = ds.tmd.interp(X, Y, method=METHOD,
        extrapolate=EXTRAPOLATE, cutoff=CUTOFF)
    # calculate tide values for input data type
    tide = local.tmd.predict(ts.tide, deltat=deltat,
        corrections=nodal_corrections)
    # calculate values for minor constituents by inference
    if INFER_MINOR:
        # add major and minor components
        tide += local.tmd.infer(ts.tide, deltat=deltat,
            corrections=nodal_corrections,
            minor=minor_constituents)
    # return the ocean or load tide correction
    return tide

# PURPOSE: compute tides at points and times using tide model algorithms
def tide_currents(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        DIRECTORY: str | pathlib.Path | None = _default_directory,
        MODEL: str | None = None,
        DEFINITION_FILE: str | pathlib.Path | IOBase | None = None,
        CROP: bool = False,
        BOUNDS: list | np.ndarray | None = None,
        BUFFER: int | float = 0,
        EPSG: str | int = 4326,
        EPOCH: list | tuple = (2000, 1, 1, 0, 0, 0),
        TYPE: str | None = 'drift',
        TIME: str = 'UTC',
        METHOD: str = 'linear',
        EXTRAPOLATE: bool = False,
        CUTOFF: int | float = 10.0,
        CORRECTIONS: str | None = None,
        CONSTITUENTS: list | None = None,
        INFER_MINOR: bool = True,
        MINOR_CONSTITUENTS: list | None = None,
        **kwargs
    ):
    """
    Compute ocean tide currents at points and times from
    model constituents

    Parameters
    ----------
    x: np.ndarray
        x-coordinates in projection EPSG
    y: np.ndarray
        y-coordinates in projection EPSG
    delta_time: np.ndarray
        seconds since EPOCH or datetime array
    DIRECTORY: str or NoneType, default None
        working data directory for tide models
    MODEL: str or NoneType, default None
        Tide model to use in correction
    DEFINITION_FILE: str, pathlib.Path, io.IOBase or NoneType, default None
        Tide model definition file for use
    CROP: bool, default False
        Crop tide model data to (buffered) bounds
    BOUNDS: list, np.ndarray or NoneType, default None
        Boundaries for cropping tide model data
    BUFFER: int or float, default 0
        Buffer distance for cropping tide model data
    EPSG: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    EPOCH: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    TYPE: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    TIME: str, default 'UTC'
        Time type if need to compute leap seconds to convert to UTC

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    METHOD: str
        Interpolation method

            - ```bilinear```: quick bilinear interpolation
            - ```spline```: scipy bivariate spline interpolation
            - ```linear```, ```nearest```: scipy regular grid interpolations

    EXTRAPOLATE: bool, default False
        Extrapolate with nearest-neighbors
    CUTOFF: int or float, default 10.0
        Extrapolation cutoff in kilometers

        Set to ``np.inf`` to extrapolate for all points
    CORRECTIONS: str or None, default None
        Nodal correction type, default based on model
    CONSTITUENTS: list or None, default None
        Specify constituents to read from model
    INFER_MINOR: bool, default True
        Infer the height values for minor tidal constituents
    MINOR_CONSTITUENTS: list or None, default None
        Specify constituents to infer

    Returns
    -------
    tide: xr.DataTree
        tidal currents in cm/s

        u: xr.Dataset
            zonal velocities
        v: xr.Dataset
            meridional velocities
    """

    # check that tide directory is accessible
    if DIRECTORY is not None:
        DIRECTORY = pathlib.Path(DIRECTORY).expanduser()
        if not DIRECTORY.exists():
            raise FileNotFoundError("Invalid tide directory")

    # validate input arguments
    assert TIME.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert METHOD.lower() in ('bilinear', 'spline', 'linear', 'nearest')

    # get parameters for tide model
    if DEFINITION_FILE is not None:
        model = pyTMD.io.model(DIRECTORY).from_file(DEFINITION_FILE)
    else:
        model = pyTMD.io.model(DIRECTORY).from_database(MODEL)
    # open datatree with model currents
    dtree = model.open_datatree(type=['u', 'v'])
    # subset to constituents
    if CONSTITUENTS:
        dtree = dtree.subset(CONSTITUENTS)

    # determine input data type based on variable dimensions
    if not TYPE:
        TYPE = pyTMD.spatial.data_type(x, y, delta_time)
    assert TYPE.lower() in ('grid', 'drift', 'time series')
    # convert coordinates to xarray DataArrays
    # in coordinate reference system of model
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        X, Y = dtree.tmd.transform(x, y, crs=EPSG)
    elif (TYPE.lower() == 'grid') and (np.size(x) != np.size(y)):
        gridx, gridy = np.meshgrid(x, y)
        mx, my = dtree.tmd.transform(gridx, gridy, crs=EPSG)
        X = xr.DataArray(mx, dims=('y','x'))
        Y = xr.DataArray(my, dims=('y','x'))
    elif (TYPE.lower() == 'grid'):
        mx, my = dtree.tmd.transform(x, y, crs=EPSG)
        X = xr.DataArray(mx, dims=('y','x'))
        Y = xr.DataArray(my, dims=('y','x'))
    elif (TYPE.lower() == 'drift'):
        mx, my = dtree.tmd.transform(x, y, crs=EPSG)
        X = xr.DataArray(mx, dims=('time'))
        Y = xr.DataArray(my, dims=('time'))
    elif (TYPE.lower() == 'time series'):
        mx, my = dtree.tmd.transform(x, y, crs=EPSG)
        X = xr.DataArray(mx, dims=('station'))
        Y = xr.DataArray(my, dims=('station'))

    # crop tide model datasets to bounds
    if CROP:
        # default bounds if cropping data
        xmin, xmax = np.min(X), np.max(X)
        ymin, ymax = np.min(Y), np.max(Y)
        bounds = BOUNDS or [xmin, xmax, ymin, ymax]
        # crop datatree to buffered bounds
        dtree = dtree.tmd.crop(bounds, buffer=BUFFER)

    # convert delta times or datetimes objects to timescale
    if (TIME.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=EPOCH, standard=TIME)

    # nodal corrections to apply
    nodal_corrections = CORRECTIONS or model.corrections
    # minor constituents to infer
    minor_constituents = MINOR_CONSTITUENTS or model.minor
    # delta time (TT - UT1) for tide model
    if nodal_corrections in ('OTIS','ATLAS','TMD3','netcdf'):
        # use delta time at 2000.0 to match TMD outputs
        deltat = np.zeros_like(ts.tt_ut1)
    else:
        # use interpolated delta times
        deltat = ts.tt_ut1

    # python dictionary with tide model data
    tide = xr.DataTree()
    # iterate over u and v currents
    for key, ds in dtree.items():
        # convert component to dataset
        ds = ds.to_dataset()
        # interpolate model to grid points
        local = ds.tmd.interp(X, Y, method=METHOD,
            extrapolate=EXTRAPOLATE, cutoff=CUTOFF)
        # calculate tide values for input data type
        tide[key] = local.tmd.predict(ts.tide, deltat=deltat,
            corrections=nodal_corrections)
        # calculate values for minor constituents by inference
        if INFER_MINOR:
            # add major and minor components
            tide[key] += local.tmd.infer(ts.tide, deltat=deltat,
                corrections=nodal_corrections,
                minor=minor_constituents)
    # return the ocean tide currents
    return tide

# PURPOSE: check if points are within a tide model domain
def tide_masks(x: np.ndarray, y: np.ndarray,
        DIRECTORY: str | pathlib.Path | None = _default_directory,
        MODEL: str | None = None,
        DEFINITION_FILE: str | pathlib.Path | IOBase | None = None,
        EPSG: str | int = 4326,
        TYPE: str | None = 'drift',
        METHOD: str = 'linear',
        **kwargs
    ):
    """
    Check if points are within a tide model domain

    Parameters
    ----------
    x: np.ndarray
        x-coordinates in projection EPSG
    y: np.ndarray
        y-coordinates in projection EPSG
    DIRECTORY: str or NoneType, default None
        working data directory for tide models
    MODEL: str or NoneType, default None
        Tide model to use
    DEFINITION_FILE: str or NoneType, default None
        Tide model definition file for use
    EPSG: str or int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    METHOD: str, default 'spline'
        interpolation method

            - ```bilinear```: quick bilinear interpolation
            - ```spline```: scipy bivariate spline interpolation
            - ```linear```, ```nearest```: scipy regular grid interpolations

    Returns
    -------
    mask: xr.DataArray
        ocean tide mask
    """

    # check that tide directory is accessible
    if DIRECTORY is not None:
        DIRECTORY = pathlib.Path(DIRECTORY).expanduser()
        if not DIRECTORY.exists():
            raise FileNotFoundError("Invalid tide directory")

    # get parameters for tide model
    if DEFINITION_FILE is not None:
        model = pyTMD.io.model(DIRECTORY).from_file(DEFINITION_FILE)
    else:
        model = pyTMD.io.model(DIRECTORY).from_database(MODEL)
    # reduce list of constituents to only those required for mask
    if model.multifile:
        model.parse_constituents()
        model.reduce_constituents(model.constituents[0])
    # open model as dataset
    ds = model.open_dataset(type='z')
    
    # reform coordinate dimensions for input grids
    # or verify coordinate dimension shapes
    assert TYPE.lower() in ('grid', 'drift', 'time series')
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        # converting x,y to model coordinates
        X, Y = ds.tmd.transform(x, y, crs=EPSG)
    if (TYPE.lower() == 'grid') and (np.size(x) != np.size(y)):
        # convert to meshgrid
        gridx, gridy = np.meshgrid(x, y)
        # converting x,y to model coordinates
        mx, my = ds.tmd.transform(gridx, gridy, crs=EPSG)
        # convert to xarray DataArrays
        X = xr.DataArray(mx, dims=('y','x'))
        Y = xr.DataArray(my, dims=('y','x'))
    elif (TYPE.lower() == 'grid'):
        # converting x,y to model coordinates
        mx, my = ds.tmd.transform(x, y, crs=EPSG)
        # convert to xarray DataArrays
        X = xr.DataArray(mx, dims=('y','x'))
        Y = xr.DataArray(my, dims=('y','x'))
    elif (TYPE.lower() == 'drift'):
        mx, my = ds.tmd.transform(x, y, crs=EPSG)
        X = xr.DataArray(mx, dims=('time'))
        Y = xr.DataArray(my, dims=('time'))
    elif (TYPE.lower() == 'time series'):
        mx, my = ds.tmd.transform(x, y, crs=EPSG)
        X = xr.DataArray(mx, dims=('station'))
        Y = xr.DataArray(my, dims=('station'))
        
    # interpolate model mask to grid points
    local = ds.tmd.interp(X, Y, method=METHOD)
    # get name of first listed constituent
    c = local.tmd.constituents[0]
    mz = np.logical_not(local[c].real.isnull()).astype(bool)
    # return mask
    return mz

# PURPOSE: compute long-period equilibrium tidal elevations
def LPET_elevations(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        EPSG: str | int = 4326,
        EPOCH: list | tuple = (2000, 1, 1, 0, 0, 0),
        TYPE: str | None = 'drift',
        TIME: str = 'UTC',
        **kwargs
    ):
    """
    Compute long-period equilibrium tidal elevations at points and times

    Parameters
    ----------
    x: np.ndarray
        x-coordinates in projection EPSG
    y: np.ndarray
        y-coordinates in projection EPSG
    delta_time: np.ndarray
        seconds since EPOCH or datetime array
    EPSG: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    EPOCH: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    TYPE: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    TIME: str, default 'UTC'
        Time type if need to compute leap seconds to convert to UTC

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    FILL_VALUE: float, default np.nan
        Output invalid value

    Returns
    -------
    tide_lpe: np.ndarray
        long-period equilibrium tide at coordinates and time in meters
    """

    # validate input arguments
    assert TIME.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    # determine input data type based on variable dimensions
    if not TYPE:
        TYPE = pyTMD.spatial.data_type(x, y, delta_time)
    assert TYPE.lower() in ('grid', 'drift', 'time series')
    # transformer for converting to WGS84 Latitude and Longitude
    crs1 = pyproj.CRS.from_user_input(EPSG)
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # convert coordinates to xarray DataArrays
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        longitude, latitude = transformer.transform(x, y)
    elif (TYPE.lower() == 'grid') and (np.size(x) != np.size(y)):
        gridx, gridy = np.meshgrid(x, y)
        lon, lat = transformer.transform(gridx, gridy)
        longitude = xr.DataArray(lon, dims=('y','x'))
        latitude = xr.DataArray(lat, dims=('y','x'))
    elif (TYPE.lower() == 'grid'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('y','x'))
        latitude = xr.DataArray(lat, dims=('y','x'))
    elif (TYPE.lower() == 'drift'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('time'))
        latitude = xr.DataArray(lat, dims=('time'))
    elif (TYPE.lower() == 'time series'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('station'))
        latitude = xr.DataArray(lat, dims=('station'))
    # create dataset
    ds = xr.Dataset(coords={'x': longitude, 'y': latitude})

    # verify that delta time is an array
    delta_time = np.atleast_1d(delta_time)
    # convert delta times or datetimes objects to timescale
    if (TIME.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=EPOCH, standard=TIME)

    # predict long-period equilibrium tides at time
    LPET = pyTMD.predict.equilibrium_tide(ts.tide, ds,
        deltat=ts.tt_ut1,
        **kwargs
    )
    # return the long-period equilibrium tide elevations
    return LPET

# PURPOSE: compute radial load pole tide displacements
# following IERS Convention (2010) guidelines
def LPT_displacements(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        EPSG: str | int = 4326,
        EPOCH: list | tuple = (2000, 1, 1, 0, 0, 0),
        TYPE: str | None = 'drift',
        TIME: str = 'UTC',
        ELLIPSOID: str = 'WGS84',
        CONVENTION: str = '2018',
        VARIABLE: str = 'R',
        FILL_VALUE: float = np.nan,
        **kwargs
    ):
    """
    Compute radial load pole tide displacements at points and times
    following IERS Convention (2010) guidelines

    Parameters
    ----------
    x: np.ndarray
        x-coordinates in projection EPSG
    y: np.ndarray
        y-coordinates in projection EPSG
    delta_time: np.ndarray
        seconds since EPOCH or datetime array
    EPSG: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    EPOCH: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    TYPE: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    TIME: str, default 'UTC'
        Time type if need to compute leap seconds to convert to UTC

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    ELLIPSOID: str, default 'WGS84'
        Ellipsoid for calculating Earth parameters
    CONVENTION: str, default '2018'
        IERS Mean or Secular Pole Convention

            - ``'2003'``
            - ``'2010'``
            - ``'2015'``
            - ``'2018'``
    VARIABLE: str: default 'R'
        Output variable to extract from dataset

            - ``'N'``: north displacement
            - ``'E'``: east displacement
            - ``'R'``: radial displacement
    FILL_VALUE: float, default np.nan
        Output invalid value

    Returns
    -------
    Srad: np.ndarray
        solid earth pole tide at coordinates and time in meters
    """

    # validate input arguments
    assert TIME.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert ELLIPSOID.upper() in pyTMD.spatial._ellipsoids
    assert CONVENTION.isdigit() and CONVENTION in timescale.eop._conventions
    # determine input data type based on variable dimensions
    if not TYPE:
        TYPE = pyTMD.spatial.data_type(x, y, delta_time)
    assert TYPE.lower() in ('grid', 'drift', 'time series')
    # transformer for converting to WGS84 Latitude and Longitude
    crs1 = pyproj.CRS.from_user_input(EPSG)
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # convert coordinates to xarray DataArrays
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        longitude, latitude = transformer.transform(x, y)
    elif (TYPE.lower() == 'grid') and (np.size(x) != np.size(y)):
        gridx, gridy = np.meshgrid(x, y)
        lon, lat = transformer.transform(gridx, gridy)
        longitude = xr.DataArray(lon, dims=('y','x'))
        latitude = xr.DataArray(lat, dims=('y','x'))
    elif (TYPE.lower() == 'grid'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('y','x'))
        latitude = xr.DataArray(lat, dims=('y','x'))
    elif (TYPE.lower() == 'drift'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('time'))
        latitude = xr.DataArray(lat, dims=('time'))
    elif (TYPE.lower() == 'time series'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('station'))
        latitude = xr.DataArray(lat, dims=('station'))
    # create dataset
    ds = xr.Dataset(coords={'x': longitude, 'y': latitude})

    # verify that delta time is an array
    delta_time = np.atleast_1d(delta_time)
    # convert delta times or datetimes objects to timescale
    if (TIME.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=EPOCH, standard=TIME)

    # number of time points
    nt = len(ts)

    # earth and physical parameters for ellipsoid
    units = pyTMD.spatial.datum(ellipsoid=ELLIPSOID, units='MKS')
    # tidal love/shida numbers appropriate for the load tide
    hb2 = 0.6207
    lb2 = 0.0836

    # convert from geodetic latitude to geocentric latitude
    # calculate X, Y and Z from geodetic latitude and longitude
    X,Y,Z = pyTMD.spatial.to_cartesian(ds.x, ds.y,
        a_axis=units.a_axis, flat=units.flat)
    XYZ = xr.Dataset(
        data_vars={
            'X': (ds.dims, X),
            'Y': (ds.dims, Y),
            'Z': (ds.dims, Z)
        },
        coords=ds.coords
    ) 
    # geocentric colatitude (radians)
    theta = np.pi/2.0 - np.arctan(XYZ.Z / np.sqrt(XYZ.X**2.0 + XYZ.Y**2.0))
    # calculate longitude (radians)
    phi = np.arctan2(XYZ.Y, XYZ.X)

    # compute normal gravity at spatial location
    # p. 80, Eqn.(2-199)
    gamma_0 = units.gamma_0(theta)

    # rotation matrix for converting to/from cartesian coordinates
    R = xr.Dataset()
    R[0,0] = np.cos(phi)*np.cos(theta)
    R[0,1] = -np.sin(phi)
    R[0,2] = np.cos(phi)*np.sin(theta)
    R[1,0] = np.sin(phi)*np.cos(theta)
    R[1,1] = np.cos(phi)
    R[1,2] = np.sin(phi)*np.sin(theta)
    R[2,0] = -np.sin(theta)
    R[2,1] = xr.zeros_like(theta)
    R[2,2] = np.cos(theta)

    # calculate load pole tides in cartesian coordinates
    dxi = pyTMD.predict.load_pole_tide(ts.tide, XYZ,
        deltat=ts.tt_ut1,
        gamma_0=gamma_0,
        omega=units.omega,
        h2=hb2,
        l2=lb2,
        convention=CONVENTION
    )

    # rotate displacements from cartesian coordinates
    S = xr.Dataset()
    S['N'] = R[0,0]*dxi['X'] + R[1,0]*dxi['Y'] + R[2,0]*dxi['Z']
    S['E'] = R[0,1]*dxi['X'] + R[1,1]*dxi['Y'] + R[2,1]*dxi['Z']
    S['R'] = R[0,2]*dxi['X'] + R[1,2]*dxi['Y'] + R[2,2]*dxi['Z']
    # return the load pole tide displacements for variable
    return S[VARIABLE]

# PURPOSE: compute radial load pole tide displacements
# following IERS Convention (2010) guidelines
def OPT_displacements(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        EPSG: str | int = 4326,
        EPOCH: list | tuple = (2000, 1, 1, 0, 0, 0),
        TYPE: str | None = 'drift',
        TIME: str = 'UTC',
        ELLIPSOID: str = 'WGS84',
        CONVENTION: str = '2018',
        METHOD: str = 'linear',
        VARIABLE: str = 'R',
        FILL_VALUE: float = np.nan,
        **kwargs
    ):
    """
    Compute radial ocean pole tide displacements at points and times
    following IERS Convention (2010) guidelines

    Parameters
    ----------
    x: np.ndarray
        x-coordinates in projection EPSG
    y: np.ndarray
        y-coordinates in projection EPSG
    delta_time: np.ndarray
        seconds since EPOCH or datetime array
    EPSG: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    EPOCH: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    TYPE: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    TIME: str, default 'UTC'
        Time type if need to compute leap seconds to convert to UTC

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    ELLIPSOID: str, default 'WGS84'
        Ellipsoid for calculating Earth parameters
    CONVENTION: str, default '2018'
        IERS Mean or Secular Pole Convention

            - ``'2003'``
            - ``'2010'``
            - ``'2015'``
            - ``'2018'``
    METHOD: str
        Interpolation method

            - ```bilinear```: quick bilinear interpolation
            - ```spline```: scipy bivariate spline interpolation
            - ```linear```, ```nearest```: scipy regular grid interpolations
    VARIABLE: str: default 'R'
        Output variable to extract from dataset

            - ``'N'``: north displacement
            - ``'E'``: east displacement
            - ``'R'``: radial displacement
    FILL_VALUE: float, default np.nan
        Output invalid value

    Returns
    -------
    Urad: np.ndarray
        ocean pole tide at coordinates and time in meters
    """

    # validate input arguments
    assert TIME.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert ELLIPSOID.upper() in pyTMD.spatial._ellipsoids
    assert CONVENTION.isdigit() and CONVENTION in timescale.eop._conventions
    assert METHOD.lower() in ('bilinear', 'spline', 'linear', 'nearest')
    # determine input data type based on variable dimensions
    if not TYPE:
        TYPE = pyTMD.spatial.data_type(x, y, delta_time)
    assert TYPE.lower() in ('grid', 'drift', 'time series')
    # transformer for converting to WGS84 Latitude and Longitude
    crs1 = pyproj.CRS.from_user_input(EPSG)
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # convert coordinates to xarray DataArrays
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        longitude, latitude = transformer.transform(x, y)
    elif (TYPE.lower() == 'grid') and (np.size(x) != np.size(y)):
        gridx, gridy = np.meshgrid(x, y)
        lon, lat = transformer.transform(gridx, gridy)
        longitude = xr.DataArray(lon, dims=('y','x'))
        latitude = xr.DataArray(lat, dims=('y','x'))
    elif (TYPE.lower() == 'grid'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('y','x'))
        latitude = xr.DataArray(lat, dims=('y','x'))
    elif (TYPE.lower() == 'drift'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('time'))
        latitude = xr.DataArray(lat, dims=('time'))
    elif (TYPE.lower() == 'time series'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('station'))
        latitude = xr.DataArray(lat, dims=('station'))
    # create dataset
    ds = xr.Dataset(coords={'x': longitude, 'y': latitude})

    # verify that delta time is an array
    delta_time = np.atleast_1d(delta_time)
    # convert delta times or datetimes objects to timescale
    if (TIME.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time.flatten())
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=EPOCH, standard=TIME)
    # number of time points
    nt = len(ts)

    # earth and physical parameters for ellipsoid
    units = pyTMD.spatial.datum(ellipsoid=ELLIPSOID, units='MKS')
    # mean equatorial gravitational acceleration [m/s^2]
    ge = 9.7803278
    # density of sea water [kg/m^3]
    rho_w = 1025.0
    # tidal love number differential (1 + kl - hl) for pole tide frequencies
    gamma = 0.6870 + 0.0036j

    # convert from geodetic latitude to geocentric latitude
    # calculate X, Y and Z from geodetic latitude and longitude
    X,Y,Z = pyTMD.spatial.to_cartesian(ds.x, ds.y,
        a_axis=units.a_axis, flat=units.flat)
    XYZ = xr.Dataset(
        data_vars={
            'X': (ds.dims, X),
            'Y': (ds.dims, Y),
            'Z': (ds.dims, Z)
        },
        coords=ds.coords
    )
    # geocentric colatitude (radians)
    theta = np.pi/2.0 - np.arctan(XYZ.Z / np.sqrt(XYZ.X**2.0 + XYZ.Y**2.0))
    # calculate longitude (radians)
    phi = np.arctan2(XYZ.Y, XYZ.X)
    # geocentric latitude (degrees)
    latitude_geocentric = 90.0 - np.degrees(theta)

    # read and interpolate ocean pole tide map from Desai (2002)
    IERS = pyTMD.io.IERS.open_dataset()
    Umap = IERS.interp(x=ds.x, y=latitude_geocentric, method=METHOD)

    # rotation matrix for converting to/from cartesian coordinates
    R = xr.Dataset()
    R[0,0] = np.cos(phi)*np.cos(theta)
    R[0,1] = -np.sin(phi)
    R[0,2] = np.cos(phi)*np.sin(theta)
    R[1,0] = np.sin(phi)*np.cos(theta)
    R[1,1] = np.cos(phi)
    R[1,2] = np.sin(phi)*np.sin(theta)
    R[2,0] = -np.sin(theta)
    R[2,1] = xr.zeros_like(theta)
    R[2,2] = np.cos(theta)

    # calculate pole tide displacements in Cartesian coordinates
    UXYZ = xr.Dataset()
    UXYZ['X'] = R[0,0]*Umap['N'] + R[0,1]*Umap['E'] + R[0,2]*Umap['R']
    UXYZ['Y'] = R[1,0]*Umap['N'] + R[1,1]*Umap['E'] + R[1,2]*Umap['R']
    UXYZ['Z'] = R[2,0]*Umap['N'] + R[2,1]*Umap['E'] + R[2,2]*Umap['R']

    # calculate ocean pole tides in cartesian coordinates
    dxi = pyTMD.predict.ocean_pole_tide(ts.tide, UXYZ,
        deltat=ts.tt_ut1,
        a_axis=units.a_axis,
        gamma_0=ge,
        GM=units.GM,
        omega=units.omega,
        rho_w=rho_w,
        g2=gamma,
        convention=CONVENTION
    )

    # rotate displacements from cartesian coordinates
    U = xr.Dataset()
    U['N'] = R[0,0]*dxi['X'] + R[1,0]*dxi['Y'] + R[2,0]*dxi['Z']
    U['E'] = R[0,1]*dxi['X'] + R[1,1]*dxi['Y'] + R[2,1]*dxi['Z']
    U['R'] = R[0,2]*dxi['X'] + R[1,2]*dxi['Y'] + R[2,2]*dxi['Z']
    # return the ocean pole tide displacements for variable
    return U[VARIABLE]

# PURPOSE: compute solid earth tidal elevations
def SET_displacements(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        METHOD: str = 'ephemerides',
        **kwargs
    ):
    """
    Compute solid earth tidal elevations (body tides) at points and times

    Parameters
    ----------
    x: np.ndarray
        x-coordinates in projection EPSG
    y: np.ndarray
        y-coordinates in projection EPSG
    delta_time: np.ndarray
        seconds since EPOCH or datetime array
    METHOD: str, default 'IERS'
        method for calculating solid earth tidal elevations

            - ``'ephemerides'``: following :cite:t:`Petit:2010tp` guidelines
            - ``'catalog'``: using tide potential catalogs
    """
    if (METHOD.lower() == 'ephemerides'):
        return _ephemeride_SET(
            x, y, delta_time,
            **kwargs
        )
    elif (METHOD.lower() == 'catalog'):
        return _catalog_SET(
            x, y, delta_time,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid METHOD: {METHOD}")

# PURPOSE: compute solid earth tides following IERS conventions
def _ephemeride_SET(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        EPSG: str | int = 4326,
        EPOCH: list | tuple = (2000, 1, 1, 0, 0, 0),
        TYPE: str | None = 'drift',
        TIME: str = 'UTC',
        ELLIPSOID: str = 'WGS84',
        TIDE_SYSTEM: str = 'tide_free',
        EPHEMERIDES: str = 'approximate',
        VARIABLE: str = 'R',
        **kwargs
    ):
    """
    Compute solid earth tidal elevations at points and times
    following IERS Convention (2010) guidelines

    Parameters
    ----------
    x: np.ndarray
        x-coordinates in projection EPSG
    y: np.ndarray
        y-coordinates in projection EPSG
    delta_time: np.ndarray
        seconds since EPOCH or datetime array
    EPSG: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    EPOCH: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    TYPE: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    TIME: str, default 'UTC'
        Time type if need to compute leap seconds to convert to UTC

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    ELLIPSOID: str, default 'WGS84'
        Ellipsoid for calculating Earth parameters
    TIDE_SYSTEM: str, default 'tide_free'
        Permanent tide system for the output solid Earth tide

            - ``'tide_free'``: no permanent direct and indirect tidal potentials
            - ``'mean_tide'``: permanent tidal potentials (direct and indirect)
    EPHEMERIDES: str, default 'approximate'
        Ephemerides for calculating Earth parameters

            - ``'approximate'``: approximate lunisolar parameters
            - ``'JPL'``: computed from JPL ephmerides kernel
    VARIABLE: str: default 'R'
        Output variable to extract from dataset

            - ``'N'``: north displacement
            - ``'E'``: east displacement
            - ``'R'``: radial displacement

    Returns
    -------
    tide_se: np.ndarray
        solid earth tide at coordinates and time in meters
    """

    # validate input arguments
    assert TIME.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert TIDE_SYSTEM.lower() in ('mean_tide', 'tide_free')
    assert EPHEMERIDES.lower() in ('approximate', 'jpl')
    # determine input data type based on variable dimensions
    if not TYPE:
        TYPE = pyTMD.spatial.data_type(x, y, delta_time)
    assert TYPE.lower() in ('grid', 'drift', 'time series')
    # transformer for converting to WGS84 Latitude and Longitude
    crs1 = pyproj.CRS.from_user_input(EPSG)
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # convert coordinates to xarray DataArrays
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        longitude, latitude = transformer.transform(x, y)
    elif (TYPE.lower() == 'grid') and (np.size(x) != np.size(y)):
        gridx, gridy = np.meshgrid(x, y)
        lon, lat = transformer.transform(gridx, gridy)
        longitude = xr.DataArray(lon, dims=('y','x'))
        latitude = xr.DataArray(lat, dims=('y','x'))
    elif (TYPE.lower() == 'grid'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('y','x'))
        latitude = xr.DataArray(lat, dims=('y','x'))
    elif (TYPE.lower() == 'drift'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('time'))
        latitude = xr.DataArray(lat, dims=('time'))
    elif (TYPE.lower() == 'time series'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('station'))
        latitude = xr.DataArray(lat, dims=('station'))
    # create dataset
    ds = xr.Dataset(coords={'x': longitude, 'y': latitude})

    # verify that delta time is an array
    delta_time = np.atleast_1d(delta_time)
    # convert delta times or datetimes objects to timescale
    if (TIME.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=EPOCH, standard=TIME)

    # earth and physical parameters for ellipsoid
    units = pyTMD.spatial.datum(ellipsoid=ELLIPSOID, units='MKS')

    # convert input coordinates to cartesian
    X,Y,Z = pyTMD.spatial.to_cartesian(ds.x, ds.y,
        a_axis=units.a_axis, flat=units.flat)
    XYZ = xr.Dataset(
        data_vars={
            'X': (ds.dims, X),
            'Y': (ds.dims, Y),
            'Z': (ds.dims, Z)
        },
        coords=ds.coords
    )
    # geocentric colatitude (radians)
    theta = np.pi/2.0 - np.arctan(XYZ.Z / np.sqrt(XYZ.X**2.0 + XYZ.Y**2.0))
    # calculate longitude (radians)
    phi = np.arctan2(XYZ.Y, XYZ.X)

    # compute ephemerides for lunisolar coordinates
    SX, SY, SZ = pyTMD.astro.solar_ecef(ts.MJD, ephemerides=EPHEMERIDES)
    LX, LY, LZ = pyTMD.astro.lunar_ecef(ts.MJD, ephemerides=EPHEMERIDES)
    # create datasets for lunisolar coordinates
    SXYZ = xr.Dataset(
        data_vars={
            'X': (['time'], SX),
            'Y': (['time'], SY),
            'Z': (['time'], SZ)
        },
        coords=dict(time=np.atleast_1d(ts.MJD))
    )
    LXYZ = xr.Dataset(
        data_vars={
            'X': (['time'], LX),
            'Y': (['time'], LY),
            'Z': (['time'], LZ)
        },
        coords=dict(time=np.atleast_1d(ts.MJD))
    )

    # rotation matrix for converting to/from cartesian coordinates
    R = xr.Dataset()
    R[0,0] = np.cos(phi)*np.cos(theta)
    R[0,1] = -np.sin(phi)
    R[0,2] = np.cos(phi)*np.sin(theta)
    R[1,0] = np.sin(phi)*np.cos(theta)
    R[1,1] = np.cos(phi)
    R[1,2] = np.sin(phi)*np.sin(theta)
    R[2,0] = -np.sin(theta)
    R[2,1] = xr.zeros_like(theta)
    R[2,2] = np.cos(theta)

    # calculate radial displacement at time
    # predict solid earth tides (cartesian)
    dxi = pyTMD.predict.solid_earth_tide(ts.tide, XYZ, SXYZ, LXYZ,
        deltat=ts.tt_ut1,
        a_axis=units.a_axis,
        tide_system=TIDE_SYSTEM
    )
    # rotate displacements from cartesian coordinates
    SE = xr.Dataset()
    SE['N'] = R[0,0]*dxi['X'] + R[1,0]*dxi['Y'] + R[2,0]*dxi['Z']
    SE['E'] = R[0,1]*dxi['X'] + R[1,1]*dxi['Y'] + R[2,1]*dxi['Z']
    SE['R'] = R[0,2]*dxi['X'] + R[1,2]*dxi['Y'] + R[2,2]*dxi['Z']
    # return the solid earth tide displacements for variable
    return SE[VARIABLE]

# PURPOSE: compute body tides following Cartwright and Tayler (1971)
def _catalog_SET(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        EPSG: str | int = 4326,
        EPOCH: list | tuple = (2000, 1, 1, 0, 0, 0),
        TYPE: str | None = 'drift',
        TIME: str = 'UTC',
        CATALOG: str = 'CTE1973',
        TIDE_SYSTEM: str = 'tide_free',
        EPHEMERIDES: str = 'IERS',
        INCLUDE_PLANETS: bool = False,
        VARIABLE: str = 'R',
        **kwargs
    ):
    """
    Compute solid earth tidal elevations at points and times
    using a tide-potential catalog following :cite:t:`Cartwright:1971iz`

    Parameters
    ----------
    x: np.ndarray
        x-coordinates in projection EPSG
    y: np.ndarray
        y-coordinates in projection EPSG
    delta_time: np.ndarray
        seconds since EPOCH or datetime array
    EPSG: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    EPOCH: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    TYPE: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    TIME: str, default 'UTC'
        Time type if need to compute leap seconds to convert to UTC

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    CATALOG: str, default 'CTE1973'
        Name of the tide potential catalog

            - ``'CTE1973'``: :cite:t:`Cartwright:1973em`
            - ``'HW1995'``: :cite:t:`Hartmann:1995jp`
            - ``'T1987'``: :cite:t:`Tamura:1987tp`
            - ``'W1990'``: Woodworth updates to ``'CTE1973'``
    TIDE_SYSTEM: str, default 'tide_free'
        Permanent tide system for the output solid Earth tide

            - ``'tide_free'``: no permanent direct and indirect tidal potentials
            - ``'mean_tide'``: permanent tidal potentials (direct and indirect)
    EPHEMERIDES: str, default 'IERS'
        Method for calculating astronomical mean longitudes

            - ``'Cartwright'``: use coefficients from David Cartwright
            - ``'Meeus'``: use coefficients from Meeus Astronomical Algorithms
            - ``'ASTRO5'``: use Meeus Astronomical coefficients from ``ASTRO5``
            - ``'IERS'``: convert from IERS Delaunay arguments
    INCLUDE_PLANETS: bool, default False
        Include tide potentials from planetary bodies
    VARIABLE: str: default 'R'
        Output variable to extract from dataset

            - ``'N'``: north displacement
            - ``'E'``: east displacement
            - ``'R'``: radial displacement

    Returns
    -------
    tide_se: np.ndarray
        solid earth tide at coordinates and time in meters
    """

    # validate input arguments
    assert TIME.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert TIDE_SYSTEM.lower() in ('mean_tide', 'tide_free')
    assert CATALOG in pyTMD.predict._tide_potential_table.keys()
    assert EPHEMERIDES.lower() in ('cartwright', 'meeus', 'astro5', 'iers')
    # determine input data type based on variable dimensions
    if not TYPE:
        TYPE = pyTMD.spatial.data_type(x, y, delta_time)
    assert TYPE.lower() in ('grid', 'drift', 'time series')
    # transformer for converting to WGS84 Latitude and Longitude
    crs1 = pyproj.CRS.from_user_input(EPSG)
    crs2 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
    # convert coordinates to xarray DataArrays
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        longitude, latitude = transformer.transform(x, y)
    elif (TYPE.lower() == 'grid') and (np.size(x) != np.size(y)):
        gridx, gridy = np.meshgrid(x, y)
        lon, lat = transformer.transform(gridx, gridy)
        longitude = xr.DataArray(lon, dims=('y','x'))
        latitude = xr.DataArray(lat, dims=('y','x'))
    elif (TYPE.lower() == 'grid'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('y','x'))
        latitude = xr.DataArray(lat, dims=('y','x'))
    elif (TYPE.lower() == 'drift'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('time'))
        latitude = xr.DataArray(lat, dims=('time'))
    elif (TYPE.lower() == 'time series'):
        lon, lat = transformer.transform(x, y)
        longitude = xr.DataArray(lon, dims=('station'))
        latitude = xr.DataArray(lat, dims=('station'))
    # create dataset
    ds = xr.Dataset(coords={'x': longitude, 'y': latitude})

    # verify that delta time is an array
    delta_time = np.atleast_1d(delta_time)
    # convert delta times or datetimes objects to timescale
    if (TIME.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=EPOCH, standard=TIME)

    # calculate body tides
    SE = pyTMD.predict.body_tide(ts.tide, ds, 
        deltat=ts.tt_ut1,
        method=EPHEMERIDES,
        tide_system=TIDE_SYSTEM,
        catalog=CATALOG,
        include_planets=INCLUDE_PLANETS,
        **kwargs
    )

    # return the solid earth tide displacements for variable
    return SE[VARIABLE]
