#!/usr/bin/env python
u"""
compute.py
Written by Tyler Sutterley (12/2025)
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
    h5netcdf: Pythonic interface to netCDF4 via h5py
        https://h5netcdf.org/
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
    predict.py: predict tide values using harmonic constants
    io/model.py: retrieves tide model parameters for named tide models
    io/OTIS.py: extract tidal harmonic constants from OTIS tide models
    io/ATLAS.py: extract tidal harmonic constants from netcdf models
    io/GOT.py: extract tidal harmonic constants from GSFC GOT models
    io/FES.py: extract tidal harmonic constants from FES tide models
    interpolate.py: interpolation routines for spatial data

UPDATE HISTORY:
    Updated 12/2025: use coords functions to convert x and y to DataArrays
    Updated 11/2025: use xarray DataArrays for input coordinates
        outputs from prediction functions will be also be DataArrays
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

import pathlib
import numpy as np
import xarray as xr
from io import IOBase
import pyTMD.io
import pyTMD.predict
import pyTMD.spatial
import pyTMD.utilities
import timescale.eop
import timescale.time
# attempt imports
pyproj = pyTMD.utilities.import_dependency('pyproj')

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

# default working data directory for tide models
_default_directory = pyTMD.utilities.get_cache_path()

# PURPOSE: wrapper function for computing values
def corrections(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        correction: str = 'ocean',
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
    correction: str, default 'ocean'
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
    if correction.lower() in ('ocean', 'load'):
        return tide_elevations(x, y, delta_time, **kwargs)
    elif (correction.upper() == 'LPET'):
        return LPET_elevations(x, y, delta_time, **kwargs)
    elif (correction.upper() == 'LPT'):
        return LPT_displacements(x, y, delta_time, **kwargs)
    elif (correction.upper() == 'OPT'):
        return OPT_displacements(x, y, delta_time, **kwargs)
    elif (correction.upper() == 'SET'):
        return SET_displacements(x, y, delta_time, **kwargs)
    else:
        raise ValueError(f'Unrecognized correction type: {correction}')

# PURPOSE: compute tides at points and times using tide model algorithms
def tide_elevations(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        directory: str | pathlib.Path | None = _default_directory,
        model: str | None = None,
        definition_file: str | pathlib.Path | IOBase | None = None,
        crop: bool = False,
        bounds: list | np.ndarray | None = None,
        buffer: int | float = 0,
        crs: str | int = 4326,
        epoch: list | tuple = (2000, 1, 1, 0, 0, 0),
        type: str | None = 'drift',
        standard: str = 'UTC',
        method: str = 'linear',
        extrapolate: bool = False,
        cutoff: int | float = 10.0,
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
    directory: str or NoneType, default None
        working data directory for tide models
    model: str or NoneType, default None
        Tide model to use in correction
    definition_file: str, pathlib.Path, io.IOBase or NoneType, default None
        Tide model definition file for use
    chunks: int, dict, str, or None, default None
        variable chunk sizes for dask (see ``xarray.open_dataset``)
    crop: bool, default False
        Crop tide model data to (buffered) bounds
    bounds: list, np.ndarray or NoneType, default None
        Boundaries for cropping tide model data
    buffer: int or float, default 0
        Buffer distance for cropping tide model data
    crs: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate reference system
    epoch: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    type: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    standard: str, default 'UTC'
        Time standard of input temporal data

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    method: str = 'linear'
        Interpolation method from xarray

            - ``'linear'``: linear interpolation for regular grids
            - ``'nearest'``: nearest-neighbor interpolation

    extrapolate: bool, default False
        Extrapolate with nearest-neighbors
    cutoff: int or float, default 10.0
        Extrapolation cutoff in kilometers

        Set to ``np.inf`` to extrapolate for all points
    corrections: str or None, default None
        Nodal correction type, default based on model
    constituents: list or None, default None
        Specify constituents to read from model
    infer_minor: bool, default True
        Infer the height values for minor tidal constituents
    minor_constituents: list or None, default None
        Specify constituents to infer
    append_node: bool, default False
        Append equilibrium amplitudes for node tides
    apply_flexure: bool, default False
        Apply ice flexure scaling factor to height values

        Only valid for models containing flexure fields

    Returns
    -------
    tide: xarray.DataArray
        tidal elevation in meters
    """
    # default keyword arguments
    kwargs.setdefault('chunks', None)
    kwargs.setdefault('corrections', None)
    kwargs.setdefault('constituents', None)
    kwargs.setdefault('infer_minor', True)
    kwargs.setdefault('minor_constituents', None)
    kwargs.setdefault('append_node', False)
    kwargs.setdefault('apply_flexure', False)

    # check that tide directory is accessible
    if (directory is not None):
        directory = pyTMD.utilities.Path(directory).resolve()
        if directory.is_local() and not directory.exists():
            raise FileNotFoundError("Directory not found")

    # validate input arguments
    assert standard.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert method.lower() in ('linear', 'nearest')

    # get parameters for tide model
    if definition_file is not None:
        m = pyTMD.io.model(directory).from_file(definition_file)
    else:
        m = pyTMD.io.model(directory).from_database(model)
    # open dataset
    ds = m.open_dataset(group='z', chunks=kwargs['chunks'],
        append_node=kwargs['append_node'])
    # apply flexure field to each constituent
    if kwargs['apply_flexure']:
        for c in ds.tmd.constituents:
            ds[c] *= ds['flexure']
    # subset to constituents
    if kwargs['constituents']:
        ds = ds.tmd.subset(kwargs['constituents'])

    # determine input data type based on variable dimensions
    if not type:
        type = pyTMD.spatial.data_type(x, y, delta_time)
    assert type.lower() in ('grid', 'drift', 'time series')
    # convert coordinates to xarray DataArrays
    # in coordinate reference system of model
    X, Y = ds.tmd.coords_as(x, y, type=type, crs=crs)

    # crop tide model dataset to bounds
    if crop and bounds is None:
        # default bounds if cropping data
        xmin, xmax = np.min(X), np.max(X)
        ymin, ymax = np.min(Y), np.max(Y)
        # crop dataset to buffered default bounds
        ds = ds.tmd.crop([xmin, xmax, ymin, ymax], buffer=buffer)
    elif crop:
        # crop dataset to buffered bounds
        ds = ds.tmd.crop(bounds, buffer=buffer) 

    # convert delta times or datetimes objects to timescale
    if (standard.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=epoch, standard=standard)

    # nodal corrections to apply
    nodal_corrections = kwargs['corrections'] or m.corrections
    # minor constituents to infer
    minor_constituents = kwargs['minor_constituents'] or m.minor
    # delta time (TT - UT1) for tide model
    if nodal_corrections in ('OTIS','ATLAS','TMD3','netcdf'):
        # use delta time at 2000.0 to match TMDv2.5 outputs
        deltat = np.zeros_like(ts.tt_ut1)
    else:
        # use interpolated delta times
        deltat = ts.tt_ut1

    # interpolate model to grid points
    local = ds.tmd.interp(X, Y, method=method,
        extrapolate=extrapolate, cutoff=cutoff)
    # calculate tide values for input data type
    tide = local.tmd.predict(ts.tide, deltat=deltat,
        corrections=nodal_corrections)
    # calculate values for minor constituents by inference
    if kwargs['infer_minor']:
        # add major and minor components
        tide += local.tmd.infer(ts.tide, deltat=deltat,
            corrections=nodal_corrections,
            minor=minor_constituents)
    # return the ocean or load tide correction
    return tide

# PURPOSE: compute tides at points and times using tide model algorithms
def tide_currents(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        directory: str | pathlib.Path | None = _default_directory,
        model: str | None = None,
        definition_file: str | pathlib.Path | IOBase | None = None,
        crop: bool = False,
        bounds: list | np.ndarray | None = None,
        buffer: int | float = 0,
        crs: str | int = 4326,
        epoch: list | tuple = (2000, 1, 1, 0, 0, 0),
        type: str | None = 'drift',
        standard: str = 'UTC',
        method: str = 'linear',
        extrapolate: bool = False,
        cutoff: int | float = 10.0,
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
    directory: str or NoneType, default None
        working data directory for tide models
    model: str or NoneType, default None
        Tide model to use in correction
    definition_file: str, pathlib.Path, io.IOBase or NoneType, default None
        Tide model definition file for use
    chunks: int, dict, str, or None, default None
        variable chunk sizes for dask (see ``xarray.open_dataset``)
    crop: bool, default False
        Crop tide model data to (buffered) bounds
    bounds: list, np.ndarray or NoneType, default None
        Boundaries for cropping tide model data
    buffer: int or float, default 0
        Buffer distance for cropping tide model data
    crs: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    epoch: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    type: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    standard: str, default 'UTC'
        Time standard of input temporal data

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    method: str
        Interpolation method from xarray

            - ``'linear'``: linear interpolation for regular grids
            - ``'nearest'``: nearest-neighbor interpolation

    extrapolate: bool, default False
        Extrapolate with nearest-neighbors
    cutoff: int or float, default 10.0
        Extrapolation cutoff in kilometers

        Set to ``np.inf`` to extrapolate for all points
    corrections: str or None, default None
        Nodal correction type, default based on model
    constituents: list or None, default None
        Specify constituents to read from model
    infer_minor: bool, default True
        Infer the height values for minor tidal constituents
    minor_constituents: list or None, default None
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
    # default keyword arguments
    kwargs.setdefault('chunks', None)
    kwargs.setdefault('corrections', None)
    kwargs.setdefault('constituents', None)
    kwargs.setdefault('infer_minor', True)
    kwargs.setdefault('minor_constituents', None)

    # check that tide directory is accessible
    if (directory is not None):
        directory = pyTMD.utilities.Path(directory).resolve()
        if directory.is_local() and not directory.exists():
            raise FileNotFoundError("Directory not found")

    # validate input arguments
    assert standard.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert method.lower() in ('linear', 'nearest')

    # get parameters for tide model
    if definition_file is not None:
        m = pyTMD.io.model(directory).from_file(definition_file)
    else:
        m = pyTMD.io.model(directory).from_database(model)
    # open datatree with model currents
    dtree = m.open_datatree(group=['u', 'v'], chunks=kwargs['chunks'])
    # subset to constituents
    if kwargs['constituents']:
        dtree = dtree.tmd.subset(kwargs['constituents'])

    # determine input data type based on variable dimensions
    if not type:
        type = pyTMD.spatial.data_type(x, y, delta_time)
    assert type.lower() in ('grid', 'drift', 'time series')
    # convert coordinates to xarray DataArrays
    # in coordinate reference system of model
    X, Y = dtree.tmd.coords_as(x, y, type=type, crs=crs)

    # crop tide model datatree to bounds
    if crop and bounds is None:
        # default bounds if cropping data
        xmin, xmax = np.min(X), np.max(X)
        ymin, ymax = np.min(Y), np.max(Y)
        # crop datatree to buffered default bounds
        dtree = dtree.tmd.crop([xmin, xmax, ymin, ymax], buffer=buffer)
    elif crop:
        # crop datatree to buffered bounds
        dtree = dtree.tmd.crop(bounds, buffer=buffer)

    # convert delta times or datetimes objects to timescale
    if (standard.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=epoch, standard=standard)

    # nodal corrections to apply
    nodal_corrections = kwargs['corrections'] or m.corrections
    # minor constituents to infer
    minor_constituents = kwargs['minor_constituents'] or m.minor
    # delta time (TT - UT1) for tide model
    if nodal_corrections in ('OTIS','ATLAS','TMD3','netcdf'):
        # use delta time at 2000.0 to match TMDv2.5 outputs
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
        local = ds.tmd.interp(X, Y, method=method,
            extrapolate=extrapolate, cutoff=cutoff)
        # calculate tide values for input data type
        tide[key] = local.tmd.predict(ts.tide, deltat=deltat,
            corrections=nodal_corrections)
        # calculate values for minor constituents by inference
        if kwargs['infer_minor']:
            # add major and minor components
            tide[key] += local.tmd.infer(ts.tide, deltat=deltat,
                corrections=nodal_corrections,
                minor=minor_constituents)
    # return the tidal currents
    return tide

# PURPOSE: check if points are within a tide model domain
def tide_masks(x: np.ndarray, y: np.ndarray,
        directory: str | pathlib.Path | None = _default_directory,
        model: str | None = None,
        definition_file: str | pathlib.Path | IOBase | None = None,
        crs: str | int = 4326,
        type: str | None = 'drift',
        method: str = 'linear',
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
    directory: str or NoneType, default None
        working data directory for tide models
    model: str or NoneType, default None
        Tide model to use
    definition_file: str or NoneType, default None
        Tide model definition file for use
    crs: str or int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    method: str, default 'linear'
        interpolation method from xarray

            - ``'linear'``: linear interpolation for regular grids
            - ``'nearest'``: nearest-neighbor interpolation

    Returns
    -------
    mask: xr.DataArray
        ocean tide mask
    """

    # check that tide directory is accessible
    if (directory is not None):
        directory = pyTMD.utilities.Path(directory).resolve()
        if directory.is_local() and not directory.exists():
            raise FileNotFoundError("Directory not found")

    # get parameters for tide model
    if definition_file is not None:
        m = pyTMD.io.model(directory).from_file(definition_file)
    else:
        m = pyTMD.io.model(directory).from_database(model, group='z')
    # reduce list of constituents to only those required for mask
    if m.multifile:
        m.parse_constituents()
        m.reduce_constituents(m.constituents[0])
    # open model as dataset
    ds = m.open_dataset(group='z')
    
    # determine input data type based on variable dimensions
    assert type.lower() in ('grid', 'drift', 'time series')
    # convert coordinates to xarray DataArrays
    # in coordinate reference system of model
    X, Y = ds.tmd.coords_as(x, y, type=type, crs=crs)
        
    # interpolate model mask to grid points
    local = ds.tmd.interp(X, Y, method=method)
    # get name of first listed constituent
    c = local.tmd.constituents[0]
    mask = np.logical_not(local[c].real.isnull()).astype(bool)
    # return mask
    return mask

# PURPOSE: compute long-period equilibrium tidal elevations
def LPET_elevations(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        crs: str | int = 4326,
        epoch: list | tuple = (2000, 1, 1, 0, 0, 0),
        type: str | None = 'drift',
        standard: str = 'UTC',
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
    crs: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    epoch: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    type: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    standard: str, default 'UTC'
        Time standard of input temporal data

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC

    Returns
    -------
    LPET: xr.DataArray
        long-period equilibrium tide at coordinates and time in meters
    """

    # validate input arguments
    assert standard.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    # determine input data type based on variable dimensions
    if not type:
        type = pyTMD.spatial.data_type(x, y, delta_time)
    assert type.lower() in ('grid', 'drift', 'time series')
    # convert coordinates to xarray DataArrays
    # in WGS84 Latitude and Longitude
    longitude, latitude = pyTMD.io.dataset._coords(x, y, type=type,
        source_crs=crs, target_crs=4326)
    # create dataset
    ds = xr.Dataset(coords={'x': longitude, 'y': latitude})

    # verify that delta time is an array
    delta_time = np.atleast_1d(delta_time)
    # convert delta times or datetimes objects to timescale
    if (standard.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=epoch, standard=standard)

    # predict long-period equilibrium tides at time
    LPET = pyTMD.predict.equilibrium_tide(ts.tide, ds,
        deltat=ts.tt_ut1, **kwargs)
    # return the long-period equilibrium tide elevations
    return LPET

# PURPOSE: compute radial load pole tide displacements
# following IERS Convention (2010) guidelines
def LPT_displacements(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        crs: str | int = 4326,
        epoch: list | tuple = (2000, 1, 1, 0, 0, 0),
        type: str | None = 'drift',
        standard: str = 'UTC',
        ellipsoid: str = 'WGS84',
        convention: str = '2018',
        variable: str | list = 'R',
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
    crs: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    epoch: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    type: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    standard: str, default 'UTC'
        Time standard of input temporal data

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    ellipsoid: str, default 'WGS84'
        Ellipsoid name for calculating Earth parameters
    convention: str, default '2018'
        IERS Mean or Secular Pole Convention

            - ``'2003'``
            - ``'2010'``
            - ``'2015'``
            - ``'2018'``
    variable: str or list, default 'R'
        Output variable(s) to extract from dataset

            - ``'N'``: north displacement
            - ``'E'``: east displacement
            - ``'R'``: radial displacement

    Returns
    -------
    S: xr.DataArray
        solid earth pole tide at coordinates and time in meters
    """

    # validate input arguments
    assert standard.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert ellipsoid.upper() in pyTMD.spatial._ellipsoids
    assert convention.isdigit() and convention in timescale.eop._conventions
    # determine input data type based on variable dimensions
    if not type:
        type = pyTMD.spatial.data_type(x, y, delta_time)
    assert type.lower() in ('grid', 'drift', 'time series')
    # convert coordinates to xarray DataArrays
    # in WGS84 Latitude and Longitude
    longitude, latitude = pyTMD.io.dataset._coords(x, y, type=type,
        source_crs=crs, target_crs=4326)
    # create dataset
    ds = xr.Dataset(coords={'x': longitude, 'y': latitude})

    # verify that delta time is an array
    delta_time = np.atleast_1d(delta_time)
    # convert delta times or datetimes objects to timescale
    if (standard.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=epoch, standard=standard)

    # earth and physical parameters for ellipsoid
    units = pyTMD.spatial.datum(ellipsoid=ellipsoid, units='MKS')
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
        convention=convention
    )

    # rotate displacements from cartesian coordinates
    S = xr.Dataset()
    S['N'] = R[0,0]*dxi['X'] + R[1,0]*dxi['Y'] + R[2,0]*dxi['Z']
    S['E'] = R[0,1]*dxi['X'] + R[1,1]*dxi['Y'] + R[2,1]*dxi['Z']
    S['R'] = R[0,2]*dxi['X'] + R[1,2]*dxi['Y'] + R[2,2]*dxi['Z']
    # set attributes for output variables
    for var in S.data_vars:
        S[var].attrs['units'] = dxi['X'].attrs.get('units', 'meters')
    # return the load pole tide displacements for variable(s)
    return S[variable]

# PURPOSE: compute radial load pole tide displacements
# following IERS Convention (2010) guidelines
def OPT_displacements(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        crs: str | int = 4326,
        epoch: list | tuple = (2000, 1, 1, 0, 0, 0),
        type: str | None = 'drift',
        standard: str = 'UTC',
        ellipsoid: str = 'WGS84',
        convention: str = '2018',
        method: str = 'linear',
        variable: str | list = 'R',
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
    crs: str | int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    epoch: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    type: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    standard: str, default 'UTC'
        Time standard of input temporal data

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    ellipsoid: str, default 'WGS84'
        Ellipsoid name for calculating Earth parameters
    convention: str, default '2018'
        IERS Mean or Secular Pole Convention

            - ``'2003'``
            - ``'2010'``
            - ``'2015'``
            - ``'2018'``
    method: str
        Interpolation method from xarray

            - ``'linear'``: linear interpolation for regular grids
            - ``'nearest'``: nearest-neighbor interpolation
    variable: str or list, default 'R'
        Output variable(s) to extract from dataset

            - ``'N'``: north displacement
            - ``'E'``: east displacement
            - ``'R'``: radial displacement

    Returns
    -------
    U: xr.DataArray
        ocean pole tide at coordinates and time in meters
    """

    # validate input arguments
    assert standard.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert ellipsoid.upper() in pyTMD.spatial._ellipsoids
    assert convention.isdigit() and convention in timescale.eop._conventions
    assert method.lower() in ('linear', 'nearest')
    # determine input data type based on variable dimensions
    if not type:
        type = pyTMD.spatial.data_type(x, y, delta_time)
    assert type.lower() in ('grid', 'drift', 'time series')
    # convert coordinates to xarray DataArrays
    # in WGS84 Latitude and Longitude
    longitude, latitude = pyTMD.io.dataset._coords(x, y, type=type,
        source_crs=crs, target_crs=4326)
    # create dataset
    ds = xr.Dataset(coords={'x': longitude, 'y': latitude})

    # verify that delta time is an array
    delta_time = np.atleast_1d(delta_time)
    # convert delta times or datetimes objects to timescale
    if (standard.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time.flatten())
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=epoch, standard=standard)

    # earth and physical parameters for ellipsoid
    units = pyTMD.spatial.datum(ellipsoid=ellipsoid, units='MKS')
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
    Umap = IERS.interp(x=ds.x, y=latitude_geocentric, method=method)

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
        convention=convention
    )

    # rotate displacements from cartesian coordinates
    U = xr.Dataset()
    U['N'] = R[0,0]*dxi['X'] + R[1,0]*dxi['Y'] + R[2,0]*dxi['Z']
    U['E'] = R[0,1]*dxi['X'] + R[1,1]*dxi['Y'] + R[2,1]*dxi['Z']
    U['R'] = R[0,2]*dxi['X'] + R[1,2]*dxi['Y'] + R[2,2]*dxi['Z']
    # set attributes for output variables
    for var in U.data_vars:
        U[var].attrs['units'] = dxi['X'].attrs.get('units', 'meters')
    # return the ocean pole tide displacements for variable(s)
    return U[variable]

# PURPOSE: compute solid earth tidal elevations
def SET_displacements(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        method: str = 'ephemerides',
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
    method: str, default 'IERS'
        method for calculating solid earth tidal elevations

            - ``'ephemerides'``: following :cite:t:`Petit:2010tp` guidelines
            - ``'catalog'``: using tide potential catalogs
    """
    if (method.lower() == 'ephemerides'):
        return _ephemeride_SET(
            x, y, delta_time,
            **kwargs
        )
    elif (method.lower() == 'catalog'):
        return _catalog_SET(
            x, y, delta_time,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid calculation method: {method}")

# PURPOSE: compute solid earth tides following IERS conventions
def _ephemeride_SET(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        crs: str | int = 4326,
        epoch: list | tuple = (2000, 1, 1, 0, 0, 0),
        type: str | None = 'drift',
        standard: str = 'UTC',
        ellipsoid: str = 'WGS84',
        tide_system: str = 'tide_free',
        ephemerides: str = 'approximate',
        variable: str | list = 'R',
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
    crs: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    epoch: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    type: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    standard: str, default 'UTC'
        Time standard of input temporal data

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    ellipsoid: str, default 'WGS84'
        Ellipsoid name for calculating Earth parameters
    tide_system: str, default 'tide_free'
        Permanent tide system for the output solid Earth tide

            - ``'tide_free'``: no permanent direct and indirect tidal potentials
            - ``'mean_tide'``: permanent tidal potentials (direct and indirect)
    ephemerides: str, default 'approximate'
        Ephemerides for calculating Earth parameters

            - ``'approximate'``: approximate lunisolar parameters
            - ``'JPL'``: computed from JPL ephmerides kernel
    variable: str or list, default 'R'
        Output variable(s) to extract from dataset

            - ``'N'``: north displacement
            - ``'E'``: east displacement
            - ``'R'``: radial displacement

    Returns
    -------
    SE: xr.DataArray
        solid earth tide at coordinates and time in meters
    """

    # validate input arguments
    assert standard.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert tide_system.lower() in ('mean_tide', 'tide_free')
    assert ephemerides.lower() in ('approximate', 'jpl')
    # determine input data type based on variable dimensions
    if not type:
        type = pyTMD.spatial.data_type(x, y, delta_time)
    assert type.lower() in ('grid', 'drift', 'time series')
    # convert coordinates to xarray DataArrays
    # in WGS84 Latitude and Longitude
    longitude, latitude = pyTMD.io.dataset._coords(x, y, type=type,
        source_crs=crs, target_crs=4326)
    # create dataset
    ds = xr.Dataset(coords={'x': longitude, 'y': latitude})

    # verify that delta time is an array
    delta_time = np.atleast_1d(delta_time)
    # convert delta times or datetimes objects to timescale
    if (standard.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=epoch, standard=standard)

    # earth and physical parameters for ellipsoid
    units = pyTMD.spatial.datum(ellipsoid=ellipsoid, units='MKS')

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
    SX, SY, SZ = pyTMD.astro.solar_ecef(ts.MJD, ephemerides=ephemerides)
    LX, LY, LZ = pyTMD.astro.lunar_ecef(ts.MJD, ephemerides=ephemerides)
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
        tide_system=tide_system
    )
    # rotate displacements from cartesian coordinates
    SE = xr.Dataset()
    SE['N'] = R[0,0]*dxi['X'] + R[1,0]*dxi['Y'] + R[2,0]*dxi['Z']
    SE['E'] = R[0,1]*dxi['X'] + R[1,1]*dxi['Y'] + R[2,1]*dxi['Z']
    SE['R'] = R[0,2]*dxi['X'] + R[1,2]*dxi['Y'] + R[2,2]*dxi['Z']
    # set attributes for output variables
    for var in SE.data_vars:
        SE[var].attrs['units'] = dxi['X'].attrs.get('units', 'meters')
    # return the solid earth tide displacements for variable(s)
    return SE[variable]

# PURPOSE: compute body tides following Cartwright and Tayler (1971)
def _catalog_SET(
        x: np.ndarray, y: np.ndarray, delta_time: np.ndarray,
        crs: str | int = 4326,
        epoch: list | tuple = (2000, 1, 1, 0, 0, 0),
        type: str | None = 'drift',
        standard: str = 'UTC',
        catalog: str = 'CTE1973',
        tide_system: str = 'tide_free',
        ephemerides: str = 'IERS',
        include_planets: bool = False,
        variable: str | list = 'R',
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
    crs: int, default: 4326 (WGS84 Latitude and Longitude)
        Input coordinate system
    epoch: tuple, default (2000,1,1,0,0,0)
        Time period for calculating delta times
    type: str or NoneType, default 'drift'
        Input data type

            - ``None``: determined from input variable dimensions
            - ``'drift'``: drift buoys or satellite/airborne altimetry
            - ``'grid'``: spatial grids or images
            - ``'time series'``: time series at a single point
    standard: str, default 'UTC'
        Time standard of input temporal data

            - ``'GPS'``: leap seconds needed
            - ``'LORAN'``: leap seconds needed (LORAN = GPS + 9 seconds)
            - ``'TAI'``: leap seconds needed (TAI = GPS + 19 seconds)
            - ``'UTC'``: no leap seconds needed
            - ``'datetime'``: numpy datatime array in UTC
    catalog: str, default 'CTE1973'
        Name of the tide potential catalog

            - ``'CTE1973'``: :cite:t:`Cartwright:1973em`
            - ``'HW1995'``: :cite:t:`Hartmann:1995jp`
            - ``'T1987'``: :cite:t:`Tamura:1987tp`
            - ``'W1990'``: Woodworth updates to ``'CTE1973'``
    tide_system: str, default 'tide_free'
        Permanent tide system for the output solid Earth tide

            - ``'tide_free'``: no permanent direct and indirect tidal potentials
            - ``'mean_tide'``: permanent tidal potentials (direct and indirect)
    ephemerides: str, default 'IERS'
        Method for calculating astronomical mean longitudes

            - ``'Cartwright'``: use coefficients from David Cartwright
            - ``'Meeus'``: use coefficients from Meeus Astronomical Algorithms
            - ``'ASTRO5'``: use Meeus Astronomical coefficients from ``ASTRO5``
            - ``'IERS'``: convert from IERS Delaunay arguments
    include_planets: bool, default False
        Include tide potentials from planetary bodies
    variable: str | list, default 'R'
        Output variable(s) to extract from dataset

            - ``'N'``: north displacement
            - ``'E'``: east displacement
            - ``'R'``: radial displacement

    Returns
    -------
    SE: xr.DataArray
        solid earth tide at coordinates and time in meters
    """

    # validate input arguments
    assert standard.lower() in ('gps', 'loran', 'tai', 'utc', 'datetime')
    assert tide_system.lower() in ('mean_tide', 'tide_free')
    assert catalog in pyTMD.predict._tide_potential_table.keys()
    assert ephemerides.lower() in ('cartwright', 'meeus', 'astro5', 'iers')
    # determine input data type based on variable dimensions
    if not type:
        type = pyTMD.spatial.data_type(x, y, delta_time)
    assert type.lower() in ('grid', 'drift', 'time series')
    # convert coordinates to xarray DataArrays
    # in WGS84 Latitude and Longitude
    longitude, latitude = pyTMD.io.dataset._coords(x, y, type=type,
        source_crs=crs, target_crs=4326)
    # create dataset
    ds = xr.Dataset(coords={'x': longitude, 'y': latitude})

    # verify that delta time is an array
    delta_time = np.atleast_1d(delta_time)
    # convert delta times or datetimes objects to timescale
    if (standard.lower() == 'datetime'):
        ts = timescale.from_datetime(delta_time)
    else:
        ts = timescale.from_deltatime(delta_time,
            epoch=epoch, standard=standard)

    # calculate body tides
    SE = pyTMD.predict.body_tide(ts.tide, ds, 
        deltat=ts.tt_ut1,
        method=ephemerides,
        tide_system=tide_system,
        catalog=catalog,
        include_planets=include_planets,
        **kwargs
    )

    # return the solid earth tide displacements for variable(s)
    return SE[variable]
