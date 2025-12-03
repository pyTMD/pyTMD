#!/usr/bin/env python
u"""
OTIS.py
Written by Tyler Sutterley (11/2025)

Reads OTIS format tidal solutions provided by Oregon State University and ESR
    http://volkov.oce.orst.edu/tides/region.html
    https://www.esr.org/research/polar-tide-models/list-of-polar-tide-models/
    ftp://ftp.esr.org/pub/datasets/tmd/

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 11/2025: near-complete rewrite of program to use xarray
    Updated 10/2025: refactored binary read programs
        added option to use memory mapping for reading large files
    Updated 08/2025: use numpy degree to radian conversions
        added option to gap fill when reading constituent grids
    Updated 05/2025: added option to select constituents to read from model
    Updated 12/2024: released version of TMD3 has different variable names
    Updated 11/2024: expose buffer distance for cropping tide model data
    Updated 10/2024: save latitude and longitude to output constituent object
        fix error when using default bounds in extract_constants
    Updated 09/2024: using new JSON dictionary format for model projections
    Updated 08/2024: revert change and assume crop bounds are projected
    Updated 07/2024: added crop and bounds keywords for trimming model data
        convert the crs of bounds when cropping model data
    Updated 06/2024: change int32 to int to prevent overflows with numpy 2.0
    Updated 02/2024: don't overwrite hu and hv in _interpolate_zeta
        changed variable for setting global grid flag to is_global
    Updated 01/2024: construct currents masks differently if not global
        renamed currents masks and bathymetry interpolation functions
    Updated 12/2023: use new crs class for coordinate reprojection
    Updated 10/2023: fix transport variable entry for TMD3 models
    Updated 09/2023: prevent overwriting ATLAS compact x and y coordinates
    Updated 08/2023: changed ESR netCDF4 format to TMD3 format
    Updated 04/2023: using pathlib to define and expand tide model paths
    Updated 03/2023: add basic variable typing to function inputs
        new function name for converting coordinate reference systems
    Updated 12/2022: refactor tide read programs under io
        new functions to read and interpolate from constituents class
        refactored interpolation routines into new module
    Updated 11/2022: place some imports within try/except statements
        fix variable reads for ATLAS compact data formats
        use f-strings for formatting verbose or ascii output
    Updated 10/2022: invert current tide masks to be True for invalid points
    Updated 06/2022: unit updates in the ESR netCDF4 format
    Updated 05/2022: add functions for using ESR netCDF4 format models
        changed keyword arguments to camel case
    Updated 04/2022: updated docstrings to numpy documentation format
        use longcomplex data format to be windows compliant
    Updated 03/2022: invert tide mask to be True for invalid points
        add separate function for resampling ATLAS compact global model
        decode ATLAS compact constituents for Python3 compatibility
        reduce iterative steps when combining ATLAS local models
    Updated 02/2022: use ceiling of masks for interpolation
    Updated 07/2021: added checks that tide model files are accessible
    Updated 06/2021: fix tidal currents for bilinear interpolation
        check for nan points when reading elevation and transport files
    Updated 05/2021: added option for extrapolation cutoff in kilometers
    Updated 03/2021: add extrapolation check where there are no invalid points
        prevent ComplexWarning for fill values when calculating amplitudes
        can read from single constituent TPXO9 ATLAS binary files
        replaced numpy bool/int to prevent deprecation warnings
    Updated 02/2021: set invalid values to nan in extrapolation
        replaced numpy bool to prevent deprecation warning
    Updated 12/2020: added valid data extrapolation with nearest_extrap
    Updated 09/2020: set bounds error to false for regular grid interpolations
        adjust dimensions of input coordinates to be iterable
        use masked arrays with atlas models and grids. make 2' grid with nearest
    Updated 08/2020: check that interpolated points are within range of model
        replaced griddata interpolation with scipy regular grid interpolators
    Updated 07/2020: added function docstrings. separate bilinear interpolation
        update griddata interpolation. changed type variable to keyword argument
    Updated 06/2020: output currents as numpy masked arrays
        use argmin and argmax in bilinear interpolation
    Updated 11/2019: interpolate heights and fluxes to numpy masked arrays
    Updated 09/2019: output as numpy masked arrays instead of nan-filled arrays
    Updated 01/2019: decode constituents for Python3 compatibility
    Updated 08/2018: added option grid for using ATLAS outputs that
        combine both global and localized tidal solutions
        added multivariate spline interpolation option
    Updated 07/2018: added different interpolation methods
    Updated 09/2017: Adapted for Python
"""
from __future__ import division, annotations

import pyproj
import pathlib
import warnings
import numpy as np
import xarray as xr
import pyTMD.utilities
# attempt imports
dask = pyTMD.utilities.import_dependency('dask')
dask_available = pyTMD.utilities.dependency_available('dask')

# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = [
    'open_dataset',
    'open_mfdataset',
    'open_otis_dataset',
    'open_atlas_dataset',
    'open_tmd3_dataset',
    'open_otis_grid',
    'open_otis_elevation',
    'open_otis_transport',
    'open_atlas_grid',
    'open_atlas_elevation',
    'open_atlas_transport',
    'read_raw_binary',
    'write_raw_binary',
    'OTISDataset',
    'OTISDataTree',
    'ATLASDataset',
]

# variable attributes
_attributes = dict(
    z = dict(
        bathymetry = dict(
            standard_name="ocean_depth",
            long_name="ocean depth",
            units="m",
        ),
        mask = dict(
            standard_name="sea_binary_mask",
            long_name="land-sea mask",
            units="1",
        ),
        elevation = dict(
            standard_name="tide_height",
            long_name="tidal_elevation",
            units="m",
        )
    ),
    u = dict(
        bathymetry=dict(
            standard_name="ocean_depth",
            long_name="ocean depth at u-locations",
            units="m",
        ),
        mask = dict(
            standard_name="sea_binary_mask",
            long_name="land-sea mask at u-locations",
            units="1",
        ),
        current = dict(
            standard_name="eastward_tidal_current",
            long_name="zonal tidal currents on u-grid",
            units="m/s",
        ),
        transport = dict(
            standard_name="eastward_tidal_transport",
            long_name="zonal tidal transports on u-grid",
            units="m**2/s",
        )
    ),
    v = dict(
        bathymetry = dict(
            standard_name="ocean_depth",
            long_name="ocean depth at v-locations",
            units="m",
        ),
        mask = dict(
            standard_name="sea_binary_mask",
            long_name="land-sea mask at v-locations",
            units="1",
        ),
        current = dict(
            standard_name="northward_tidal_current",
            long_name="meridional tidal currents on v-grid",
            units="m/s",
        ),
        transport = dict(
            standard_name="northward_tidal_transport",
            long_name="meridional tidal transports on v-grid",
            units="m**2/s",
        )
    )
)

# PURPOSE: read tide model files
def open_dataset(
        model_file: str | list | pathlib.Path,
        grid_file: str | pathlib.Path | None = None,
        format: str = 'OTIS',
        **kwargs,
    ):
    """
    Open OTIS/ATLAS/TMD3 model files

    Parameters
    ----------
    model_file: str or pathlib.Path
        input model file
    grid_file: str, pathlib.Path or None, default None
        input model grid file
    format: str, default 'OTIS'
        Model format

            - ``'ATLAS'``
            - ``'OTIS'``
            - ``'TMD3'``
    group: str, default 'z'
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: zonal currents
            - ``'U'``: zonal depth-averaged transport
            - ``'v'``: meridional currents
            - ``'V'``: meridional depth-averaged transport
    **kwargs: dict
        additional keyword arguments for opening files

    Returns
    -------
    ds: xarray.Dataset
        tide model data
    """
    # set default keyword arguments
    kwargs.setdefault('group', 'z')
    group = kwargs.get('group').lower()
    # open file(s) as xarray dataset
    if (format == 'OTIS'):
        # OTIS (single or multi-file)
        ds = open_otis_dataset(model_file,
            grid_file=grid_file,
            **kwargs
        )
    elif (format == 'ATLAS'):
        # ATLAS-compact
        ds = open_atlas_dataset(model_file,
            grid_file=grid_file,
            **kwargs
        )
    elif (format == 'TMD3'):
        # TMD3 netCDF4
        ds = open_tmd3_dataset(model_file, **kwargs)
    # add attributes
    ds.attrs['format'] = format
    # convert transports to currents if necessary
    if kwargs['group'] in ('u','v'):
        # convert transports to currents and update attributes
        for c in ds.tmd.constituents:
            ds[c] /= ds['bathymetry']
            ds[c].attrs.update(_attributes[group]['current'])
    # return xarray dataset
    return ds

# PURPOSE: read a list of model files
def open_mfdataset(
        model_files: list[str] | list[pathlib.Path],
        group: str = 'z',
        **kwargs
    ):
    """
    Open multiple OTIS model files

    Parameters
    ----------
    model_files: list of str or pathlib.Path
        list of OTIS model files
    group: str, default 'z'
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: zonal currents
            - ``'U'``: zonal depth-averaged transport
            - ``'v'``: meridional currents
            - ``'V'``: meridional depth-averaged transport
    parallel: bool, default False
        Open files in parallel using ``dask.delayed``
    **kwargs: dict
        additional keyword arguments for opening OTIS files

    Returns
    -------
    ds: xarray.Dataset
        OTIS tide model data
    """
    # set default keyword arguments
    kwargs.setdefault('parallel', False)
    parallel = kwargs.get('parallel') and dask_available
    # read each file as xarray dataset and append to list
    if (group == 'z') and parallel:
        # elevations
        opener = dask.delayed(open_otis_elevation)
        d, = dask.compute([opener(f, **kwargs) for f in model_files])
    elif group == 'z':
        # elevations
        d = [open_otis_elevation(f, **kwargs) for f in model_files]
    elif group in ('u','U') and parallel:
        # transports are returned as (u,v)
        opener = dask.delayed(open_otis_transport)
        d, = dask.compute([opener(f, **kwargs)[0] for f in model_files])
    elif group in ('u','U'):
        # transports are returned as (u,v)
        d = [open_otis_transport(f, **kwargs)[0] for f in model_files]
    elif group in ('v','V') and parallel:
        # transports are returned as (u,v)
        opener = dask.delayed(open_otis_transport)
        d, = dask.compute([opener(f, **kwargs)[1] for f in model_files])
    elif group in ('v','V'):
        # transports are returned as (u,v)
        d = [open_otis_transport(f, **kwargs)[1] for f in model_files]
    # merge datasets
    ds = xr.merge(d, compat='override')
    # add attributes
    ds.attrs['group'] = group.upper() if group in ('u','v') else group
    # return xarray dataset
    return ds

def open_otis_dataset(
        model_file: str | list | pathlib.Path,
        grid_file: str | pathlib.Path,
        group: str | None = 'z',
        **kwargs
    ):
    """
    Open OTIS model files

    Parameters
    ----------
    model_file: str, list or pathlib.Path
        input model file(s)
    grid_file: str, pathlib.Path
        input model grid file
    group: str, default 'z'
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: zonal currents
            - ``'U'``: zonal depth-averaged transport
            - ``'v'``: meridional currents
            - ``'V'``: meridional depth-averaged transport
    crs: int, str or dict, default 4326
        coordinate reference system for the model data
    use_mmap: bool, default False
        use memory mapping to read data

    Returns
    -------
    ds: xarray.Dataset
        combined OTIS model data
    """
    # default coordinate reference system
    crs = kwargs.get('crs', 4326)
    # open grid file
    ds1 = open_otis_grid(grid_file, **kwargs)
    # add attributes
    ds1.attrs['crs'] = pyproj.CRS.from_user_input(crs).to_dict()
    # open model file(s)
    if isinstance(model_file, list):
        # multi-file datasets
        ds2 = open_mfdataset(model_file, group=group, **kwargs)
    elif (group == 'z'):
        # elevations
        ds2 = open_otis_elevation(model_file, **kwargs)
    elif group in ('u', 'U'):
        # transports are returned as (u,v)
        ds2 = open_otis_transport(model_file, **kwargs)[0]
    elif group in ('v', 'V'):
        # transports are returned as (u,v)
        ds2 = open_otis_transport(model_file, **kwargs)[1]
    # merge datasets
    ds = ds1.otis.merge(ds2, group=group)
    # add attributes
    ds.attrs['group'] = group.upper() if group in ('u','v') else group
    # return xarray dataset
    return ds

def open_atlas_dataset(
        model_file: str | pathlib.Path,
        grid_file: str | pathlib.Path,
        group: str | None = 'z',
        chunks: int | dict | str | None = None,
        use_mmap : bool = False,
        **kwargs
    ):
    """
    Open ATLAS model files

    Parameters
    ----------
    model_file: str or pathlib.Path
        input model file
    grid_file: str, pathlib.Path
        input model grid file
    group: str, default 'z'
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: zonal currents
            - ``'U'``: zonal depth-averaged transport
            - ``'v'``: meridional currents
            - ``'V'``: meridional depth-averaged transport
    crs: int, str or dict, default 4326
        coordinate reference system for the model data
    chunks: int, dict, str, or None, default None
        coerce output to specified chunks
    use_mmap: bool, default False
        use memory mapping to read data

    Returns
    -------
    ds: xarray.Dataset
        combined ATLAS model data
    """
    # default coordinate reference system
    crs = kwargs.get('crs', 4326)
    # open grid file
    dsg, dtg = open_atlas_grid(grid_file, use_mmap=use_mmap)
    ds1 = dsg.compact.combine_local(dtg, chunks=chunks)
    # add attributes
    ds1.attrs['crs'] = pyproj.CRS.from_user_input(crs).to_dict()
    # open model file(s)
    if (group == 'z'):
        # elevations are returned as (z, localz)
        dsh, dth = open_atlas_elevation(model_file, use_mmap=use_mmap)
        ds2 = dsh.compact.combine_local(dth, chunks=chunks)
    elif group in ('u', 'U'):
        # transports are returned as (u, v, localu, localv)
        dsu, dtu, dsv, dtv = open_atlas_transport(model_file, use_mmap=use_mmap)
        ds2 = dsu.compact.combine_local(dtu, chunks=chunks)
    elif group in ('v', 'V'):
        # transports are returned as (u, v, localu, localv)
        dsu, dtu, dsv, dtv = open_atlas_transport(model_file, use_mmap=use_mmap)
        ds2 = dsv.compact.combine_local(dtv, chunks=chunks)
    # merge datasets
    ds = xr.merge([ds1, ds2], compat='override')
    # add attributes
    ds.attrs['group'] = group.upper() if group in ('u','v') else group
    # return xarray dataset
    return ds

# PURPOSE: read TMD3 netCDF4 files
def open_tmd3_dataset(
        input_file: str | pathlib.Path,
        group: str | None = 'z',
        chunks: int | dict | str | None = None,
        **kwargs
    ):
    """
    Open TMD3-formatted netCDF4 files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input TMD3 netCDF4 file
    group: str, default 'z'
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: zonal currents
            - ``'U'``: zonal depth-averaged transport
            - ``'v'``: meridional currents
            - ``'V'``: meridional depth-averaged transport
    chunks: int, dict, str, or None, default None
        variable chunk sizes for dask (see ``xarray.open_dataset``)

    Returns
    -------
    ds: xarray.Dataset
        TMD3 tide model data
    """
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if input_file.is_local() and not input_file.exists():
        raise FileNotFoundError(f'File not found: {input_file}')
    # read the netCDF4-format tide grid file
    tmp = xr.open_dataset(input_file, mask_and_scale=True, chunks=chunks)
    # replace constituents array with names
    constituents = tmp.constituents.attrs['constituent_order'].split()
    tmp['constituents'] = constituents
    # coordinate reference system
    spatial_proj4 = tmp.mapping.attrs.get('spatial_proj4', 4326)
    # flip y orientation to be monotonically increasing
    tmp = tmp.reindex(y=tmp.y[::-1])
    # convert imaginary component to negative to match convention
    # get units attributes for model group
    if (group == 'z'):
        ds = (tmp['hRe'] + -1j*tmp['hIm']).to_dataset(dim='constituents')
        units = tmp['hRe'].attrs.get('units')
    elif group in ('U','u'):
        ds = (tmp['URe'] + -1j*tmp['UIm']).to_dataset(dim='constituents')
        units = tmp['URe'].attrs.get('units')
    elif group in ('V','v'):
        ds = (tmp['VRe'] + -1j*tmp['VIm']).to_dataset(dim='constituents')
        units = tmp['VRe'].attrs.get('units')
    # read water column thickness, mask and flexure
    ds['mask'] = tmp['mask']
    # convert bathymetry to float and rename to match
    ds['bathymetry'] = tmp.wct.astype('f')
    # convert flexure from percent to scale factor
    ds['flexure'] = tmp.flexure.astype('f') / 100.0
    # add attributes
    for con in ds.data_vars:
        ds[con].attrs['units'] = units
    ds.attrs['crs'] = pyproj.CRS.from_user_input(spatial_proj4).to_dict()
    ds.attrs['group'] = group.upper() if group in ('u','v') else group
    # return xarray dataset
    return ds

# PURPOSE: read OTIS grid files
def open_otis_grid(
        input_file: str | pathlib.Path,
        chunks: int | dict | str | None = None,
        use_mmap: bool = False,
        **kwargs
    ):
    """
    Open OTIS model grid files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input OTIS grid file
    chunks: int, dict, str, or None, default None
        coerce output to specified chunks
    use_mmap: bool, default False
        use memory mapping to read data

    Returns
    -------
    ds: xarray.Dataset
        OTIS grid data
    """
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if input_file.is_local() and not input_file.exists():
        raise FileNotFoundError(f'File not found: {input_file}')
    # set initial offset (skip 4 bytes)
    offset = 4
    # read data as big endian
    # get model dimensions   
    nx, ny = read_raw_binary(input_file,
        dtype=np.dtype('>i4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    # extract x and y limits (could be latitude and longitude)
    ylim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    xlim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    # read dt from file
    dt, = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(1,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 4
    # convert longitudinal limits (if x == longitude)
    if (xlim[0] < 0) & (xlim[1] < 0) & (dt > 0):
        xlim += 360.0
    # grid bounding box (xmin, xmax, ymin, ymax)
    bounds = np.array([*xlim, *ylim], dtype='>f4')
    # x and y coordinate spacing
    dx = (xlim[1] - xlim[0])/nx
    dy = (ylim[1] - ylim[0])/ny
    # create x and y arrays arrays
    x = np.linspace(xlim[0] + dx/2.0, xlim[1] - dx/2.0, nx)
    y = np.linspace(ylim[0] + dy/2.0, ylim[1] - dy/2.0, ny)
    # read nob from file
    nob, = read_raw_binary(input_file,
        dtype=np.dtype('>i4'),
        shape=(1,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 4
    # skip 8 bytes
    offset += 8
    # read iob from file
    if (nob == 0):
        iob = []
        offset += 4
    else:
        iob = read_raw_binary(input_file,
            dtype='>i4',
            shape=(nob, 2),
            use_mmap=use_mmap,
            offset=offset,
            order='C'
        )
        offset += 2*4*nob
    # skip 8 bytes
    offset += 8
    # read hz matrix
    hz = read_raw_binary(input_file,
        dtype='>f4',
        shape=(ny, nx),
        use_mmap=use_mmap,
        offset=offset,
        order='C'
    )
    offset += 4*nx*ny
    # skip 8 bytes
    offset += 8
    # read mz matrix (1: wet point, 0: dry point)
    mz = read_raw_binary(input_file,
        dtype='>i4',
        shape=(ny, nx),
        use_mmap=use_mmap,
        offset=offset,
        order='C'
    )
    offset += 4*nx*ny
    # update mask for cases where bathymetry is zero or negative
    mz = np.minimum(mz, (hz > 0))
    # data dictionary
    grid = dict(dims=('y', 'x'), coords={}, data_vars={})
    grid["coords"]["y"] = dict(data=y.copy(), dims='y')
    grid["coords"]["x"] = dict(data=x.copy(), dims='x')
    for field in ["bathymetry", "mask"]:
        grid["data_vars"][field] = dict(dims=('y', 'x'))
    # store the data
    grid["data_vars"]["bathymetry"]["data"] = hz
    grid["data_vars"]["mask"]["data"] = mz
    # convert to xarray Dataset from the data dictionary
    ds = xr.Dataset.from_dict(grid)
    # coerce to specified chunks
    if chunks is not None:
        ds = ds.chunk(chunks)
    # add attributes
    ds.attrs['dt'] = dt.copy()
    ds.attrs['iob'] = iob.copy()
    ds.attrs['bounds'] = bounds.copy()
    for field in ['mask', 'bathymetry']:
        ds[field].attrs.update(_attributes['z'][field])
    # return xarray dataset
    return ds

# PURPOSE: read OTIS elevation files
def open_otis_elevation(
        input_file: str | pathlib.Path,
        chunks: int | dict | str | None = None,
        use_mmap: bool = False,
        **kwargs
    ):
    """
    Read OTIS tidal elevation files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input OTIS elevation file
    chunks: int, dict, str, or None, default None
        coerce output to specified chunks
    use_mmap: bool, default False
        use memory mapping to read data

    Returns
    -------
    ds: xarray.Dataset
        OTIS tidal elevation data
    """
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if input_file.is_local() and not input_file.exists():
        raise FileNotFoundError(f'File not found: {input_file}')
    # set initial offset
    offset = 0
    # read data as big endian
    ll,nx,ny,nc = read_raw_binary(input_file,
        dtype=np.dtype('>i4'),
        shape=(4,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 4*4
    # offset for x and y limits
    ylim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    xlim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    # grid bounding box (xmin, xmax, ymin, ymax)
    bounds = np.array([*xlim, *ylim], dtype='>f4')
    # x and y coordinate spacing
    dx = (xlim[1] - xlim[0])/nx
    dy = (ylim[1] - ylim[0])/ny
    # create x and y arrays arrays
    x = np.linspace(xlim[0] + dx/2.0, xlim[1] - dx/2.0, nx)
    y = np.linspace(ylim[0] + dy/2.0, ylim[1] - dy/2.0, ny)
    # read constituent name
    constituents = read_raw_binary(input_file,
        dtype='|S4',
        shape=(nc, ),
        use_mmap=use_mmap,
        offset=offset,
        order='C'
    )
    # add to offset
    offset += int(nc)*4
    # data dictionary
    h = dict(dims=('y', 'x'), coords={}, data_vars={})
    h["coords"]["y"] = dict(data=y.copy(), dims='y')
    h["coords"]["x"] = dict(data=x.copy(), dims='x')
    # read constituents from file
    for ic in range(nc):
        # get constituent name
        field = constituents[ic].decode("utf8").rstrip()
        h["data_vars"][field] = dict(dims=('y', 'x'))
        # skip records to constituent
        offset += 8 
        # read elevations for constituent
        temp = read_raw_binary(input_file,
                dtype='>f4',
                shape=(ny, nx, 2),
                use_mmap=use_mmap,
                offset=offset,
                order='C'
        )
        # real and imaginary components of elevation
        Z = np.ma.array(temp[:,:,0] + 1j*temp[:,:,1])
        # update mask for nan values
        Z.mask = np.isnan(Z.data) | (np.abs(Z.data) == 0)
        # replace masked values with fill value
        Z.data[Z.mask] = Z.fill_value
        # store the data
        h["data_vars"][field]["data"] = Z
        # skip to next constituent
        offset += 4*2*nx*ny
    # convert to xarray Dataset from the data dictionary
    ds = xr.Dataset.from_dict(h)
    # coerce to specified chunks
    if chunks is not None:
        ds = ds.chunk(chunks)
    # add attributes
    ds.attrs['bounds'] = bounds.copy()
    for field in ds.data_vars:
        ds[field].attrs.update(_attributes['z']['elevation'])
    # return xarray dataset
    return ds

# PURPOSE: read OTIS transport files
def open_otis_transport(
        input_file: str | pathlib.Path,
        chunks: int | dict | str | None = None,
        use_mmap: bool = False,
        **kwargs
    ):
    """
    Read OTIS tidal transport files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input OTIS transport file
    chunks: int, dict, str, or None, default None
        coerce output to specified chunks
    use_mmap: bool, default False
        use memory mapping to read data

    Returns
    -------
    dsu: xarray.Dataset
        OTIS zonal tidal transport data
    dsv: xarray.Dataset
        OTIS meridional tidal transport data
    """
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if input_file.is_local() and not input_file.exists():
        raise FileNotFoundError(f'File not found: {input_file}')
    # set initial offset
    offset = 0
    # read data as big endian
    ll,nx,ny,nc = read_raw_binary(input_file,
        dtype=np.dtype('>i4'),
        shape=(4,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 4*4
    # offset for x and y limits
    ylim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    xlim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    # grid bounding box (xmin, xmax, ymin, ymax)
    bounds = np.array([*xlim, *ylim], dtype='>f4')
    # x and y coordinate spacing
    dx = (xlim[1] - xlim[0])/nx
    dy = (ylim[1] - ylim[0])/ny
    # create x and y arrays arrays
    x = np.linspace(xlim[0] + dx/2.0, xlim[1] - dx/2.0, nx)
    y = np.linspace(ylim[0] + dy/2.0, ylim[1] - dy/2.0, ny)
    # read constituents from file
    constituents = read_raw_binary(input_file,
        dtype='|S4',
        shape=(nc, ),
        use_mmap=use_mmap,
        offset=offset,
        order='C'
    )
    # add to offset
    offset += int(nc)*4
    # u and v dictionaries
    u = dict(dims=('y', 'x'), coords={}, data_vars={})
    v = dict(dims=('y', 'x'), coords={}, data_vars={})
    u["coords"]["y"] = dict(data=y.copy(), dims='y')
    u["coords"]["x"] = dict(data=x - dx/2.0, dims='x')
    v["coords"]["y"] = dict(data=y - dy/2.0, dims='y')
    v["coords"]["x"] = dict(data=x.copy(), dims='x')
    # read constituents from file
    for ic in range(nc):
        # get constituent name
        field = constituents[ic].decode("utf8").rstrip()
        u["data_vars"][field] = dict(dims=('y', 'x'))
        v["data_vars"][field] = dict(dims=('y', 'x'))
        # skip records to constituent
        offset += 8 
        # read elevations for constituent
        temp = read_raw_binary(input_file,
                dtype='>f4',
                shape=(ny, nx, 4),
                use_mmap=use_mmap,
                offset=offset,
                order='C'
        )
        # real and imaginary components of transport
        U = np.ma.array(temp[:,:,0] + 1j*temp[:,:,1])
        V = np.ma.array(temp[:,:,2] + 1j*temp[:,:,3])
        # update mask for nan values
        U.mask = np.isnan(U.data) | (np.abs(U.data) == 0)
        V.mask = np.isnan(V.data) | (np.abs(V.data) == 0)
        # replace masked values with fill value
        U.data[U.mask] = U.fill_value
        V.data[V.mask] = V.fill_value
        # store the data
        u["data_vars"][field]["data"] = U
        v["data_vars"][field]["data"] = V
        # skip to next constituent
        offset += 4*4*nx*ny
    # convert to xarray Datasets from the data dictionaries
    dsu = xr.Dataset.from_dict(u)
    dsv = xr.Dataset.from_dict(v)
    # coerce to specified chunks
    if chunks is not None:
        dsu = dsu.chunk(chunks)
        dsv = dsv.chunk(chunks)
    # add attributes
    dsu.attrs['bounds'] = bounds.copy()
    dsv.attrs['bounds'] = bounds.copy()
    for field in dsu.data_vars:
        dsu[field].attrs.update(_attributes['u']['transport'])
    for field in dsv.data_vars:
        dsv[field].attrs.update(_attributes['v']['transport'])
    # return xarray datasets
    return dsu, dsv

# PURPOSE: read ATLAS-compressed grid files
def open_atlas_grid(
        input_file: str | pathlib.Path,
        use_mmap: bool = False,
        **kwargs
    ):
    """
    Open ATLAS-compressed model grid files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input ATLAS grid file
    use_mmap: bool, default False
        use memory mapping to read data

    Returns
    -------
    ds: xarray.Dataset
        ATLAS global grid data
    dtree: xarray.DataTree
        local ATLAS grid solutions
    """
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if input_file.is_local() and not input_file.exists():
        raise FileNotFoundError(f'File not found: {input_file}')
    # get file information
    file_info = input_file.stat()
    # set initial offset (skip 4 bytes)
    offset = 4
    # read data as big endian
    # get model dimensions   
    nx, ny = read_raw_binary(input_file,
        dtype=np.dtype('>i4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    # extract x and y limits (could be latitude and longitude)
    ylim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    xlim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    # read dt from file
    dt, = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(1,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 4
    # grid bounding box (xmin, xmax, ymin, ymax)
    bounds = np.array([*xlim, *ylim], dtype='>f4')
    # x and y coordinate spacing
    dx = (xlim[1] - xlim[0])/nx
    dy = (ylim[1] - ylim[0])/ny
    # create x and y arrays
    x = np.linspace(xlim[0] + dx/2.0, xlim[1] - dx/2.0, nx)
    y = np.linspace(ylim[0] + dy/2.0, ylim[1] - dy/2.0, ny)
    # read nob from file
    nob, = read_raw_binary(input_file,
        dtype=np.dtype('>i4'),
        shape=(1,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 4
    # skip 8 bytes
    offset += 8
    # read iob from file
    if (nob == 0):
        iob = []
        offset += 4
    else:
        iob = read_raw_binary(input_file,
            dtype='>i4',
            shape=(nob, 2),
            use_mmap=use_mmap,
            offset=offset,
            order='C'
        )
        offset += 2*4*nob
    # skip 8 bytes
    offset += 8
    # read hz matrix
    hz = read_raw_binary(input_file,
        dtype='>f4',
        shape=(ny, nx),
        use_mmap=use_mmap,
        offset=offset,
        order='C'
    )
    offset += 4*nx*ny
    # skip 8 bytes
    offset += 8
    # read mz matrix (1: wet point, 0: dry point)
    mz = read_raw_binary(input_file,
        dtype='>i4',
        shape=(ny, nx),
        use_mmap=use_mmap,
        offset=offset,
        order='C'
    )
    offset += 4*nx*ny
    # update mask for cases where bathymetry is zero or negative
    mz = np.minimum(mz, (hz > 0))
    # skip 8 bytes
    offset += 8
    # read pmask matrix
    pmask = read_raw_binary(input_file,
        dtype='>i4',
        shape=(ny, nx),
        use_mmap=use_mmap,
        offset=offset,
        order='C'
    )
    offset += 4*nx*ny
    # skip 8 bytes
    offset += 8

    # data dictionary
    grid = dict(dims=('y', 'x'), coords={}, data_vars={})
    grid["coords"]["y"] = dict(data=y.copy(), dims='y')
    grid["coords"]["x"] = dict(data=x.copy(), dims='x')
    for field in ["bathymetry", "mask", "local"]:
        grid["data_vars"][field] = dict(dims=('y', 'x'))
    # store the data
    grid["data_vars"]["bathymetry"]["data"] = hz
    grid["data_vars"]["mask"]["data"] = mz
    grid["data_vars"]["local"]["data"] = pmask
    # convert to xarray Dataset from the data dictionary
    ds = xr.Dataset.from_dict(grid)
    # add attributes
    ds.attrs['dt'] = dt.copy()
    ds.attrs['iob'] = iob.copy()
    ds.attrs['bounds'] = bounds.copy()
    for field in ['mask', 'bathymetry']:
        ds[field].attrs.update(_attributes['z'][field])

    # read local models
    nmod = 0
    dtree = xr.DataTree()
    # while the file position is not at the end of file
    while (offset < file_info.st_size):
        # add 1 to number of models
        nmod += 1
        # get local model dimensions
        # and number of valid depth indices
        NX, NY, ND = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(3,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 3*4
        # extract local x and y limits
        YLIM = read_raw_binary(input_file,
            dtype=np.dtype('>f4'),
            shape=(2,),
            use_mmap=use_mmap,
            offset=offset
        ).astype('>f8')
        offset += 2*4
        XLIM = read_raw_binary(input_file,
            dtype=np.dtype('>f4'),
            shape=(2,),
            use_mmap=use_mmap,
            offset=offset
        ).astype('>f8')
        offset += 2*4
        # local bounding box (xmin, xmax, ymin, ymax)
        BOUNDS = np.array([*XLIM, *YLIM], dtype='>f4')
        # x and y coordinate spacing
        DX = (XLIM[1] - XLIM[0])/NX
        DY = (YLIM[1] - YLIM[0])/NY
        # create local x and y arrays
        X = np.linspace(XLIM[0] + DX/2.0, XLIM[1] - DX/2.0, NX)
        Y = np.linspace(YLIM[0] + DY/2.0, YLIM[1] - DY/2.0, NY)
        # extract region name
        temp = read_raw_binary(input_file,
            dtype='|S4',
            shape=(5, ),
            use_mmap=use_mmap,
            offset=offset,
            order='C'
        )
        name = b''.join(temp).decode("utf8").strip()
        offset += 5*4
        # skip 8 bytes
        offset += 8
        # extract local valid indices
        indx = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(ND,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 4*ND
        indy = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(ND,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 4*ND
        # reduce coordinates to valid points
        gridx, gridy = np.meshgrid(X, Y)
        XD = gridx[indy-1, indx-1]
        YD = gridy[indy-1, indx-1]
        # skip 8 bytes
        offset += 8
        # extract depth for valid points
        depth = read_raw_binary(input_file,
            dtype=np.dtype('>f4'),
            shape=(ND, ),
            use_mmap=use_mmap,
            offset=offset,
            order='C'
        )
        offset += 4*ND
        # skip 8 bytes
        offset += 8
        # save to dictionary
        local = dict(dims=('i', ), coords={}, data_vars={})
        local["data_vars"]["y"] = dict(data=YD.copy(), dims='i')
        local["data_vars"]["x"] = dict(data=XD.copy(), dims='i')
        for field in ["bathymetry",]:
            local["data_vars"][field] = dict(dims=('i', ))
        # store the data
        local["data_vars"]["bathymetry"]["data"] = depth
        # convert to xarray Dataset from the data dictionary
        dtree[name] = xr.Dataset.from_dict(local)
        dtree[name].attrs['bounds'] = BOUNDS.copy()
    # return xarray dataset (global) and datatree (local)
    return (ds, dtree)

# PURPOSE: read ATLAS-compressed elevation files
def open_atlas_elevation(
        input_file: str | pathlib.Path,
        use_mmap: bool = False,
        **kwargs
    ):
    """
    Open ATLAS-compressed tidal elevation files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input ATLAS elevation file
    use_mmap: bool, default False
        use memory mapping to read data

    Returns
    -------
    ds: xarray.Dataset
        ATLAS global tidal elevation data
    dtree: xarray.DataTree
        local ATLAS tidal elevation solutions
    """
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if input_file.is_local() and not input_file.exists():
        raise FileNotFoundError(f'File not found: {input_file}')
    # get file information
    file_info = input_file.stat()
    # set initial offset
    offset = 0
    # read data as big endian
    ll,nx,ny,nc = read_raw_binary(input_file,
        dtype=np.dtype('>i4'),
        shape=(4,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 4*4
    # offset for x and y limits
    ylim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    xlim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    # grid bounding box (xmin, xmax, ymin, ymax)
    bounds = np.array([*xlim, *ylim], dtype='>f4')
    # x and y coordinate spacing
    dx = (xlim[1] - xlim[0])/nx
    dy = (ylim[1] - ylim[0])/ny
    # create x and y arrays
    x = np.linspace(xlim[0] + dx/2.0, xlim[1] - dx/2.0, nx)
    y = np.linspace(ylim[0] + dy/2.0, ylim[1] - dy/2.0, ny)
    # read constituents from file
    constituents = read_raw_binary(input_file,
        dtype='|S4',
        shape=(nc, ),
        use_mmap=use_mmap,
        offset=offset,
        order='C'
    )
    # add to offset
    offset += int(nc)*4
    # data dictionary
    h = dict(dims=('y', 'x'), coords={}, data_vars={})
    h["coords"]["y"] = dict(data=y.copy(), dims='y')
    h["coords"]["x"] = dict(data=x.copy(), dims='x')
    # read constituents from file
    for ic in range(nc):
        # get constituent name
        field = constituents[ic].decode("utf8").rstrip()
        h["data_vars"][field] = dict(dims=('y', 'x'))
        # skip records to constituent
        offset += 8 
        # read elevations for constituent
        temp = read_raw_binary(input_file,
                dtype='>f4',
                shape=(ny, nx, 2),
                use_mmap=use_mmap,
                offset=offset,
                order='C'
        )
        # real and imaginary components of elevation
        Z = np.ma.array(temp[:,:,0] + 1j*temp[:,:,1])
        # update mask for nan values
        Z.mask = np.isnan(Z.data) | (np.abs(Z.data) == 0)
        # replace masked values with fill value
        Z.data[Z.mask] = Z.fill_value
        # store the data
        h["data_vars"][field]["data"] = Z
        # skip to next constituent
        offset += 4*2*nx*ny
    # convert to xarray Dataset from the data dictionary
    ds = xr.Dataset.from_dict(h)
    # add attributes
    ds.attrs['bounds'] = bounds.copy()
    for field in ds.data_vars:
        ds[field].attrs.update(_attributes['z']['elevation'])
    offset += 4

    # read local models
    nmod = 0
    dtree = xr.DataTree()
    # while the file position is not at the end of file
    while (offset < file_info.st_size):
        # add 1 to number of models
        offset += 4
        nmod += 1
        # get local model dimensions, number of constituents
        # and number of valid elevation indices
        NX, NY, NC, NZ = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(4,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 4*4
        # extract local x and y limits
        YLIM = read_raw_binary(input_file,
            dtype=np.dtype('>f4'),
            shape=(2,),
            use_mmap=use_mmap,
            offset=offset
        ).astype('>f8')
        offset += 2*4
        XLIM = read_raw_binary(input_file,
            dtype=np.dtype('>f4'),
            shape=(2,),
            use_mmap=use_mmap,
            offset=offset
        ).astype('>f8')
        offset += 2*4
        # local bounding box (xmin, xmax, ymin, ymax)
        BOUNDS = np.array([*XLIM, *YLIM], dtype='>f4')
        # x and y coordinate spacing
        DX = (XLIM[1] - XLIM[0])/NX
        DY = (YLIM[1] - YLIM[0])/NY
        # create local x and y arrays
        X = np.linspace(XLIM[0] + DX/2.0, XLIM[1] - DX/2.0, NX)
        Y = np.linspace(YLIM[0] + DY/2.0, YLIM[1] - DY/2.0, NY)
        # extract constituents for localized solution
        CONSTITUENTS = read_raw_binary(input_file,
            dtype='|S4',
            shape=(NC, ),
            use_mmap=use_mmap,
            offset=offset,
            order='C'
        )
        offset += NC*4
        # extract region name
        temp = read_raw_binary(input_file,
            dtype='|S4',
            shape=(5, ),
            use_mmap=use_mmap,
            offset=offset,
            order='C'
        )
        name = b''.join(temp).decode("utf8").strip()
        offset += 5*4
        # skip 8 bytes
        offset += 8
        # extract local valid indices
        indx = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(NZ,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 4*NZ
        indy = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(NZ,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 4*NZ
        # reduce coordinates to valid points
        gridx, gridy = np.meshgrid(X, Y)
        XZ = gridx[indy-1, indx-1]
        YZ = gridy[indy-1, indx-1]
        # skip 4 bytes
        offset += 4
        # save to dictionary
        local = dict(dims=('i', ), coords={}, data_vars={})
        local["data_vars"]["y"] = dict(data=YZ.copy(), dims='i')
        local["data_vars"]["x"] = dict(data=XZ.copy(), dims='i')
        # read constituents from file
        for IC in range(NC):
            # get constituent name
            field = CONSTITUENTS[IC].decode("utf8").rstrip()
            local["data_vars"][field] = dict(dims=('i', ))
            # skip 4 bytes
            offset += 4
            # read elevation for constituent
            temp = read_raw_binary(input_file,
                dtype='>f4',
                shape=(NZ, 2),
                use_mmap=use_mmap,
                offset=offset,
                order='C'
            )
            offset += 4*2*NZ
            # skip 4 bytes
            offset += 4
            # create local elevation
            Z = np.zeros((NZ), dtype=np.complex64)
            # real and imaginary components of elevation
            Z.real[:] = temp[:,0]
            Z.imag[:] = temp[:,1]
            # store the data variable
            local["data_vars"][field]["data"] = Z
        # convert to xarray Dataset from the data dictionary
        dtree[name] = xr.Dataset.from_dict(local)
        dtree[name].attrs['bounds'] = BOUNDS.copy()
    # return xarray dataset (global) and datatree (local)
    return (ds, dtree)

# PURPOSE: read ATLAS-compressed transport files
def open_atlas_transport(
        input_file: str | pathlib.Path,
        use_mmap: bool = False,
        **kwargs
    ):
    """
    Open ATLAS-compressed tidal transport files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input ATLAS transport file
    use_mmap: bool, default False
        use memory mapping to read data

    Returns
    -------
    dsu: xarray.Dataset
        ATLAS global zonal tidal transport data
    dsv: xarray.Dataset
        ATLAS global meridional tidal transport data
    dtu: xarray.DataTree
        local ATLAS zonal tidal transport solutions
    dtv: xarray.DataTree
        local ATLAS meridional tidal transport solutions
    """
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if input_file.is_local() and not input_file.exists():
        raise FileNotFoundError(f'File not found: {input_file}')
    # get file information
    file_info = input_file.stat()
    # set initial offset
    offset = 0
    # read data as big endian
    ll,nx,ny,nc = read_raw_binary(input_file,
        dtype=np.dtype('>i4'),
        shape=(4,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 4*4
    # offset for x and y limits
    ylim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    xlim = read_raw_binary(input_file,
        dtype=np.dtype('>f4'),
        shape=(2,),
        use_mmap=use_mmap,
        offset=offset
    )
    offset += 2*4
    # grid bounding box (xmin, xmax, ymin, ymax)
    bounds = np.array([*xlim, *ylim], dtype='>f4')
    # x and y coordinate spacing
    dx = (xlim[1] - xlim[0])/nx
    dy = (ylim[1] - ylim[0])/ny
    # create x and y arrays
    x = np.linspace(xlim[0] + dx/2.0, xlim[1] - dx/2.0, nx)
    y = np.linspace(ylim[0] + dy/2.0, ylim[1] - dy/2.0, ny)
    # read constituents from file
    constituents = read_raw_binary(input_file,
        dtype='|S4',
        shape=(nc, ),
        use_mmap=use_mmap,
        offset=offset,
        order='C'
    )
    # add to offset
    offset += int(nc)*4
    # u and v dictionaries
    u = dict(dims=('y', 'x'), coords={}, data_vars={})
    v = dict(dims=('y', 'x'), coords={}, data_vars={})
    u["coords"]["y"] = dict(data=y.copy(), dims='y')
    u["coords"]["x"] = dict(data=x - dx/2.0, dims='x')
    v["coords"]["y"] = dict(data=y - dy/2.0, dims='y')
    v["coords"]["x"] = dict(data=x.copy(), dims='x')
    # read constituents from file
    for ic in range(nc):
        # get constituent name
        field = constituents[ic].decode("utf8").rstrip()
        u["data_vars"][field] = dict(dims=('y', 'x'))
        v["data_vars"][field] = dict(dims=('y', 'x'))
        # skip records to constituent
        offset += 8 
        # read elevations for constituent
        temp = read_raw_binary(input_file,
                dtype='>f4',
                shape=(ny, nx, 4),
                use_mmap=use_mmap,
                offset=offset,
                order='C'
        )
        # real and imaginary components of transport
        U = np.ma.array(temp[:,:,0] + 1j*temp[:,:,1])
        V = np.ma.array(temp[:,:,2] + 1j*temp[:,:,3])
        # update mask for nan values
        U.mask = np.isnan(U.data) | (np.abs(U.data) == 0)
        V.mask = np.isnan(V.data) | (np.abs(V.data) == 0)
        # replace masked values with fill value
        U.data[U.mask] = U.fill_value
        V.data[V.mask] = V.fill_value
        # store the data
        u["data_vars"][field]["data"] = U
        v["data_vars"][field]["data"] = V
        # skip to next constituent
        offset += 4*4*nx*ny
    # convert to xarray Datasets from the data dictionaries
    dsu = xr.Dataset.from_dict(u)
    dsv = xr.Dataset.from_dict(v)
    # add attributes
    dsu.attrs['bounds'] = bounds.copy()
    dsv.attrs['bounds'] = bounds.copy()
    for field in dsu.data_vars:
        dsu[field].attrs.update(_attributes['u']['transport'])
    for field in dsv.data_vars:
        dsv[field].attrs.update(_attributes['v']['transport'])
    offset += 4

    # read local models
    nmod = 0
    dtu = xr.DataTree()
    dtv = xr.DataTree()
    # while the file position is not at the end of file
    while (offset < file_info.st_size):
        # add 1 to number of models
        offset += 4
        nmod += 1
        # get local model dimensions, number of constituents
        # and number of valid elevation indices
        NX, NY, NC, NU, NV = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(5,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 5*4
        # extract local x and y limits
        YLIM = read_raw_binary(input_file,
            dtype=np.dtype('>f4'),
            shape=(2,),
            use_mmap=use_mmap,
            offset=offset
        ).astype('>f8')
        offset += 2*4
        XLIM = read_raw_binary(input_file,
            dtype=np.dtype('>f4'),
            shape=(2,),
            use_mmap=use_mmap,
            offset=offset
        ).astype('>f8')
        offset += 2*4
        # local bounding box (xmin, xmax, ymin, ymax)
        BOUNDS = np.array([*XLIM, *YLIM], dtype='>f4')
        # x and y coordinate spacing
        DX = (XLIM[1] - XLIM[0])/NX
        DY = (YLIM[1] - YLIM[0])/NY
        # x and y coordinate spacing
        DX = (XLIM[1] - XLIM[0])/NX
        DY = (YLIM[1] - YLIM[0])/NY
        # create local x and y arrays
        X = np.linspace(XLIM[0] + DX/2.0, XLIM[1] - DX/2.0, NX)
        Y = np.linspace(YLIM[0] + DY/2.0, YLIM[1] - DY/2.0, NY)
        # extract constituents for localized solution
        CONSTITUENTS = read_raw_binary(input_file,
            dtype='|S4',
            shape=(NC, ),
            use_mmap=use_mmap,
            offset=offset,
            order='C'
        )
        offset += NC*4
        # extract region name
        temp = read_raw_binary(input_file,
            dtype='|S4',
            shape=(5, ),
            use_mmap=use_mmap,
            offset=offset,
            order='C'
        )
        name = b''.join(temp).decode("utf8").strip()
        offset += 5*4
        # skip 8 bytes
        offset += 8
        # extract local valid indices for zonal transports
        iux = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(NU,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 4*NU
        iuy = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(NU,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 4*NU
        # skip 8 bytes
        offset += 8
        # extract local valid indices for meridional transports
        ivx = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(NV,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 4*NV
        ivy = read_raw_binary(input_file,
            dtype=np.dtype('>i4'),
            shape=(NV,),
            use_mmap=use_mmap,
            offset=offset
        )
        offset += 4*NV
        # skip 4 bytes
        offset += 4
        # reduce coordinates to valid points
        gridx, gridy = np.meshgrid(X, Y)
        XU = gridx[iuy-1, iux-1] - DX/2.0
        YU = gridy[iuy-1, iux-1]
        XV = gridx[ivy-1, ivx-1]
        YV = gridy[ivy-1, ivx-1] - DY/2.0
        # u and v dictionaries
        lclu = dict(dims=('i', ), coords={}, data_vars={})
        lclv = dict(dims=('i', ), coords={}, data_vars={})
        lclu["data_vars"]["y"] = dict(data=YU.copy(), dims='i')
        lclu["data_vars"]["x"] = dict(data=XU.copy(), dims='i')
        lclv["data_vars"]["y"] = dict(data=YV.copy(), dims='i')
        lclv["data_vars"]["x"] = dict(data=XV.copy(), dims='i')
        # read constituents from file
        for IC in range(NC):
            # get constituent name
            field = CONSTITUENTS[IC].decode("utf8").rstrip()
            # skip 4 bytes
            offset += 4
            # read zonal transport for constituent
            temp = read_raw_binary(input_file,
                dtype='>f4',
                shape=(NU, 2),
                use_mmap=use_mmap,
                offset=offset,
                order='C'
            )
            offset += 4*2*NU
            # create local zonal transport
            U = np.ma.zeros((NY, NX), dtype=np.complex64)
            U.mask = np.ones((NY, NX), dtype=bool)
            # real and imaginary components of u transport
            U.data.real[iuy-1,iux-1] = temp[:,0]
            U.data.imag[iuy-1,iux-1] = temp[:,1]
            U.mask[iuy-1,iux-1] = False
            # skip 8 bytes
            offset += 8
            # read meridional transport for constituent
            temp = read_raw_binary(input_file,
                dtype='>f4',
                shape=(NV, 2),
                use_mmap=use_mmap,
                offset=offset,
                order='C'
            )
            offset += 4*2*NV
            # skip 4 bytes
            offset += 4
            # create local meridional transport
            V = np.ma.zeros((NY, NX), dtype=np.complex64)
            V.mask = np.ones((NY, NX), dtype=bool)
            # real and imaginary components of v transport
            V.data.real[ivy-1,ivx-1] = temp[:,0]
            V.data.imag[ivy-1,ivx-1] = temp[:,1]
            V.mask[ivy-1,ivx-1] = False
            # store the data variables
            lclu["data_vars"][field] = {}
            lclu["data_vars"][field]["dims"] = ('y', 'x')
            lclu["data_vars"][field]["data"] = U
            lclv["data_vars"][field] = {}
            lclv["data_vars"][field]["dims"] = ('y', 'x')
            lclv["data_vars"][field]["data"] = V
        # convert to xarray Datasets from the data dictionaries
        dtu[name] = xr.Dataset.from_dict(lclu)
        dtv[name] = xr.Dataset.from_dict(lclv)
        dtu[name].attrs['bounds'] = BOUNDS.copy()
        dtv[name].attrs['bounds'] = BOUNDS.copy()
    # return xarray datasets (global) and datatrees (local)
    return (dsu, dsv, dtu, dtv)

# PURPOSE: read a variable from a raw binary file
# with the option to use memory-mapping
def read_raw_binary(
        path: str | pathlib.Path,
        dtype: np.dtype | str,
        shape: tuple,
        use_mmap: bool = False,
        offset: int = 0,
        order: str = 'C',
        **kwargs
    ):
    """
    Read a variable from a raw binary file

    Parameters
    ----------
    path: str or pathlib.Path
        path to input file
    dtype: numpy.dtype or str
        variable data type
    shape: tuple
        shape of the data
    use_mmap: bool, default False
        create a memory-map of the variable 
    offset: int, default 0
        offset to apply on read
    order: str, default 'C'
        memory layout of array

    Returns
    -------
    var: numpy.ndarray
        data variable
    """
    # open the file and read the variable
    with open(path, mode='rb') as fid:
        if use_mmap:
            # use memory-mapping
            var = np.memmap(fid,
                dtype=np.dtype(dtype),
                mode='r',
                offset=offset,
                shape=shape,
                order=order
            )
        else:
            # read variable directly
            count = np.prod(shape)
            var = np.fromfile(fid,
                dtype=np.dtype(dtype),
                offset=offset,
                count=count
            )
            var = var.reshape(shape, order=order)
    # verify data shape
    var.shape = shape
    return var

# PURPOSE: write a variable to a raw binary file with memory-mapping
def write_raw_binary(
        path: str | pathlib.Path,
        variable: np.ndarray,
        offset: int = 0,
        order: str = 'C',
        **kwargs
    ):
    """
    Write a variable to a raw binary file with memory-mapping

    Parameters
    ----------
    path: str or pathlib.Path
        path to input file
    variable: numpy.ndarray
        data variable to write
    offset: int, default 0
        offset to apply on read
    order: str, default 'C'
        memory layout of array

    Returns
    -------
    var: numpy.ndarray
        data variable
    """
    # convert variable to array
    variable = np.array(variable)
    # set default keyword arguments
    kwargs.setdefault('shape', variable.shape)
    kwargs.setdefault('dtype', variable.dtype)
    # reshape variable
    variable = variable.reshape(kwargs['shape'], order=order)
    # use memory-mapping to write variable
    var = np.memmap(path,
        dtype=np.dtype(kwargs['dtype']),
        mode='r+',
        offset=offset,
        shape=kwargs['shape'],
        order=order
    )
    var[:] = variable.astype(kwargs['dtype'])
    var.flush()

# PURPOSE: OTIS utilities for xarray Datasets
@xr.register_dataset_accessor('otis')
class OTISDataset:
    """Accessor for extending an ``xarray.Dataset`` for OTIS tidal models
    """
    def __init__(self, ds):
        # initialize dataset
        self._ds = ds

    # PURPOSE: interpolate grid variables to u and v nodes
    def merge(self, ds, group: str = 'z'):
        """
        Interpolate grid variables from zeta nodes to u and v nodes

        Parameters
        ----------
        ds: xarray.Dataset
            OTIS tide model data
        group: str, default 'z'
            Tidal variable of input dataset

                - ``'z'``: heights
                - ``'u'``: zonal currents
                - ``'U'``: zonal depth-averaged transport
                - ``'v'``: meridional currents
                - ``'V'``: meridional depth-averaged transport
        """
        # wrap mask if global
        mode = 'wrap' if self.is_global else 'edge'
        if group in ('u', 'U'):
            # calculate Dataset on u grids
            # pad and roll the mask and bathymetry
            tmp = self._ds.pad(x=(1, 0), mode=mode).rolling(x=2)
            mask = tmp.min()['mask'].isel(x=slice(1, None))
            bathymetry = tmp.mean()['bathymetry'].isel(x=slice(1, None))
            # assign to dataset
            ds['mask'] = (ds.dims, mask.values)
            ds['bathymetry'] = ds['mask']*bathymetry.values
            for field in ['mask', 'bathymetry']:
                ds[field].attrs.update(_attributes['u'][field])
        elif group in ('v', 'V'):
            # calculate Dataset on v grids
            # pad and roll the mask and bathymetry
            tmp = self._ds.pad(y=(1, 0), mode='edge').rolling(y=2)
            mask = tmp.min()['mask'].isel(y=slice(1, None))
            bathymetry = tmp.mean()['bathymetry'].isel(y=slice(1, None))
            # assign to dataset
            ds['mask'] = (ds.dims, mask.values)
            ds['bathymetry'] = ds['mask']*bathymetry.values
            for field in ['mask', 'bathymetry']:
                ds[field].attrs.update(_attributes['v'][field])
        else:
            # merge without interpolation
            ds = xr.merge([self._ds, ds], compat='override')
        # set coordinate reference system
        ds.attrs['crs'] = self.crs.to_dict()
        # return the updated datasets
        return ds

    @property
    def crs(self):
        """Coordinate reference system of the ``Dataset``
        """
        # return the CRS of the dataset
        # default is EPSG:4326 (WGS84)
        CRS = self._ds.attrs.get('crs', 4326)
        return pyproj.CRS.from_user_input(CRS)

    @property
    def is_global(self) -> bool:
        """Determine if the dataset covers a global domain
        """
        # grid spacing in x-direction
        self._dx = self._ds.x[1] - self._ds.x[0]
        # check if global grid
        cyclic = np.isclose(self._ds.x[-1] - self._ds.x[0], 360.0 - self._dx)
        return self.crs.is_geographic and cyclic

# PURPOSE: OTIS utilities for xarray datatrees
@xr.register_datatree_accessor('otis')
class OTISDataTree:
    """Accessor for extending an ``xarray.DataTree`` for OTIS tidal models
    """
    def __init__(self, dtree):
        # initialize datatree
        self._dtree = dtree

    # PURPOSE: output grid file in OTIS format
    def to_grid(self, path: str | pathlib.Path):
        """
        Writes OTIS-format grid files

        Parameters
        ----------
        path: str or pathlib.Path
            output OTIS grid file name
        """
        # tilde-expand output file
        path = pyTMD.utilities.Path(path).resolve()
        path.touch()
        # offset in output file
        offset = 0
        # get c-grid data from elevation dataset
        ds = self._dtree['z'].to_dataset()
        # get dimensions
        nob = len(ds.attrs['iob'])
        ny = len(ds['y'])
        nx = len(ds['x'])
        # record length
        record_length = 32
        # write header to file
        header = np.array([record_length, nx, ny], dtype='>i4')
        write_raw_binary(path, header,
            shape=(3,),
            dtype='>i4',
            offset=offset
        )
        offset += 4*3
        # extract bounds and write to file
        write_raw_binary(path, ds.attrs['bounds'],
            shape=(4,),
            dtype='>f4',
            offset=offset
        )
        offset += 4*4
        # extract time step and write to file
        write_raw_binary(path, ds.attrs['dt'],
            shape=(1,),
            dtype='>f4',
            offset=offset
        )
        offset += 4
        # write number of open boundaries to file
        write_raw_binary(path, nob,
            shape=(1,),
            dtype='>i4',
            offset=offset
        )
        offset += 4
        offset += 8
        if (nob == 0):
            offset += 4
        else:
            write_raw_binary(path, ds.attrs['iob'],
                shape=(nob,2),
                dtype='>i4',
                offset=offset
            )
            offset += 4*2*nob
        # write depth to file
        offset += 8
        write_raw_binary(path, ds['bathymetry'].to_numpy(),
            dtype='>f4',
            offset=offset
        )
        offset += 4*nx*ny
        offset += 4
        # write mask to file
        offset += 4
        write_raw_binary(path, ds['mask'].to_numpy(),
            dtype='>i4',
            offset=offset
        )
        offset += 4*nx*ny
        # end variable
        write_raw_binary(path, record_length,
            shape=(1,),
            dtype='>i4',
            offset=offset
        )
        offset += 4

    # PURPOSE: output elevation file in OTIS format
    def to_elevation(self,
            path: str | pathlib.Path,
            **kwargs
        ):
        """
        Writes OTIS-format elevation files

        Parameters
        ----------
        path: str or pathlib.Path
            output OTIS elevation file name
        """
        # tilde-expand output file
        path = pyTMD.utilities.Path(path).resolve()
        path.touch()
        # offset in output file
        offset = 0
        # get z data
        ds = kwargs.get('z', self._dtree['z'].to_dataset())
        # get dimensions
        ny = len(ds['y'])
        nx = len(ds['x'])
        nc = len(ds.tmd.constituents)
        # length of header: allow for 4 character >i c_id strings
        header_length = 4*(7 + nc)
        # write header to file
        header = np.array([header_length, nx, ny, nc], dtype='>i4')
        write_raw_binary(path, header,
            shape=(4,),
            dtype='>i4',
            offset=offset
        )
        offset += 4*4
        # extract bounds and write to file
        write_raw_binary(path, ds.attrs['bounds'],
            shape=(4,),
            dtype='>f4',
            offset=offset
        )
        offset += 4*4
        # write constituent names to file
        write_raw_binary(path, ds.tmd.constituents,
            shape=(nc,), 
            dtype='|S4',
            offset=offset
        )
        offset += 4*nc
        offset += 4
        # write each constituent to file
        for c in ds.tmd.constituents:
            offset += 4
            # merge real and imaginary components of elevation
            temp = np.zeros((ny, nx, 2), dtype='>f4')
            temp[:,:,0] = ds[c].real.values
            temp[:,:,1] = ds[c].imag.values
            write_raw_binary(path, temp,
                dtype='>f4',
                offset=offset
            )
            offset += 4*2*nx*ny
            offset += 4
        # end variable
        write_raw_binary(path, header_length,
            shape=(1,),
            dtype='>i4',
            offset=offset
        )
        offset += 4

    # PURPOSE: output transport file in OTIS format
    def to_transport(self,
            path: str | pathlib.Path,
            **kwargs
        ):
        """
        Writes OTIS-format transport files

        Parameters
        ----------
        path: str or pathlib.Path
            output OTIS elevation file name
        """
        # tilde-expand output file
        path = pyTMD.utilities.Path(path).resolve()
        path.touch()
        # offset in output file
        offset = 0
        # get u and v data
        dsu = kwargs.get('u', self._dtree['u'].to_dataset())
        dsv = kwargs.get('v', self._dtree['v'].to_dataset())
        # get dimensions
        ny = len(dsu['y'])
        nx = len(dsu['x'])
        nc = len(dsu.tmd.constituents)
        # length of header: allow for 4 character >i c_id strings
        header_length = 4*(7 + nc)
        # write header to file
        header = np.array([header_length, nx, ny, nc], dtype='>i4')
        write_raw_binary(path, header,
            shape=(4,),
            dtype='>i4',
            offset=offset
        )
        offset += 4*4
        # extract bounds and write to file
        write_raw_binary(path, dsu.attrs['bounds'],
            shape=(4,),
            dtype='>f4',
            offset=offset
        )
        offset += 4*4
        # write constituent names to file
        write_raw_binary(path, dsu.tmd.constituents,
            shape=(nc,),
            dtype='|S4',
            offset=offset
        )
        offset += 4*nc
        offset += 4
        # write each constituent to file
        for c in dsu.tmd.constituents:
            offset += 4
            # merge real and imaginary components of u and v transports
            temp = np.zeros((ny, nx, 4), dtype='>f4')
            temp[:,:,0] = dsu[c].real.values
            temp[:,:,1] = dsu[c].imag.values
            temp[:,:,2] = dsv[c].real.values
            temp[:,:,3] = dsv[c].imag.values
            write_raw_binary(path, temp,
                dtype='>f4',
                offset=offset
            )
            offset += 4*4*nx*ny
            offset += 4
        # end variable
        write_raw_binary(path, header_length,
            shape=(1,),
            dtype='>i4',
            offset=offset
        )
        offset += 4

    # PURPOSE: output elevation file in OTIS format
    def to_mfelevation(self, directory: str | pathlib.Path):
        """
        Writes OTIS-format singular elevation files

        Parameters
        ----------
        directory: str or pathlib.Path
            output directory for OTIS elevation files
        """
        # tilde-expand output directory
        directory = pyTMD.utilities.Path(directory).resolve()
        # get z data
        ds = self._dtree['z'].to_dataset()
        # write each constituent to file
        for c in ds.tmd.constituents:
            path = directory.joinpath(c)
            self.to_elevation(path, z=ds[[c]])

    # PURPOSE: output elevation file in OTIS format
    def to_mftransport(self, directory: str | pathlib.Path):
        """
        Writes OTIS-format singular transport files

        Parameters
        ----------
        directory: str or pathlib.Path
            output directory for OTIS transport files
        """
        # tilde-expand output directory
        directory = pyTMD.utilities.Path(directory).resolve()
        # get u and v data
        dsu = self._dtree['u'].to_dataset()
        dsv = self._dtree['v'].to_dataset()
        # write each constituent to file
        for c in dsu.tmd.constituents:
            path = directory.joinpath(c)
            self.to_transport(path, u=dsu[[c]], v=dsv[[c]])

# PURPOSE: ATLAS-compact utilities for xarray Datasets
@xr.register_dataset_accessor('compact')
class ATLASDataset:
    """
    Accessor for extending an ``xarray.Dataset`` for ATLAS-compact
    tidal models
    """
    def __init__(self, ds, spacing: float | list[float] = 1.0/30.0):
        # initialize dataset
        self._ds = ds
        self._dx, self._dy = np.broadcast_to(np.atleast_1d(spacing), (2,))
        
    def refine(self):
        """Refine data resolution to a finer resolution
        """
        # create coordinate DataArrays
        x = xr.DataArray(self._x, dims='x')
        y = xr.DataArray(self._y, dims='y')
        # interpolate global model to refined grid
        ds = self._ds.interp(x=x, y=y)
        ds.attrs['bounds'] = np.array(self.__bounds__)
        return ds

    def combine_local(self,
            dtree: xr.DataTree,
            chunks: int | dict | str | None = None
        ):
        """Combine ATLAS model solutions into a single xarray Dataset

        Parameters
        ----------
        dtree: xarray.DataTree
            Local ATLAS model solutions
        chunks: int, dict, str, or None, default None
            coerce output to specified chunks

        Returns
        -------
        ds: xarray.Dataset
            combined ATLAS tidal data
        """
        # create refined dataset
        ds = self.refine()
        # for each local model
        for region, local in dtree.items():
            # check if any model longitudes are -180:180
            X = local.x.where(local.x >= 0, local.x + 360.0, drop=False)
            # local indices in global model
            indx = ((X - ds.x.min())//self._dx).astype('i')
            indy = ((local.y - ds.y.min())//self._dy).astype('i')
            # for each data variable in the global model
            for key in ds.data_vars.keys():
                # check if data variable is in the local model
                if key in local.data_vars:
                    # replace global data with local data at valid indices
                    ds[key][indy, indx] = local[key][:]
                elif (key == 'mask'):
                    # replace global mask with local mask at valid indices
                    ds[key][indy, indx] = True
        # coerce to specified chunks
        if chunks is not None:
            ds = ds.chunk(chunks)
        # return combined xarray Dataset
        return ds   

    @property
    def shape(self):
        """Grid dimensions"""
        nx = np.round(360.0/self._dx).astype(int)
        ny = np.round(180.0/self._dy).astype(int)
        return np.array([ny, nx])
    
    @property
    def size(self):
        """Grid size"""
        return np.prod(self.shape)

    @property
    def _x(self):
        """Refined x-coordinates"""
        return np.linspace(self.__xlim__[0], self.__xlim__[1], self.shape[1])

    @property
    def _y(self):
        """Refined y-coordinates"""
        return np.linspace(self.__ylim__[0], self.__ylim__[1], self.shape[0])

    @property
    def __xlim__(self):
        """Limits for x-coordinates"""
        return [self._dx/2.0, 360.0 - self._dx/2.0]    
    
    @property
    def __ylim__(self):
        """Limits for y-coordinates"""
        return [-90.0 + self._dy/2.0, 90.0 - self._dy/2.0]

    @property
    def __bounds__(self):
        """Bounding box for refined grid"""
        return np.array([*self.__xlim__, *self.__ylim__])
