#!/usr/bin/env python
u"""
ATLAS.py
Written by Tyler Sutterley (11/2025)

Reads netCDF4 ATLAS tidal solutions provided by Oregon State University

PYTHON DEPENDENCIES:
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 11/2025: near-complete rewrite of program to use xarray
    Updated 08/2025: use numpy degree to radian conversions
        added option to gap fill when reading constituent grids
    Updated 11/2024: expose buffer distance for cropping tide model data
    Updated 10/2024: fix error when using default bounds in extract_constants
    Updated 07/2024: added crop and bounds keywords for trimming model data
    Updated 02/2024: changed variable for setting global grid flag to is_global
    Updated 10/2023: add generic wrapper function for reading constituents
    Updated 04/2023: using pathlib to define and expand tide model paths
    Updated 03/2023: add basic variable typing to function inputs
    Updated 12/2022: refactor tide read programs under io
        new functions to read and interpolate from constituents class
        new functions to output ATLAS formatted netCDF4 files
        refactored interpolation routines into new module
    Updated 11/2022: place some imports within try/except statements
        use f-strings for formatting verbose or ascii output
    Updated 07/2022: fix setting of masked array data to NaN
    Updated 05/2022: reformat arguments to extract_netcdf_constants definition
        changed keyword arguments to camel case
    Updated 04/2022: updated docstrings to numpy documentation format
    Updated 12/2021: adjust longitude convention based on model longitude
    Updated 09/2021: fix cases where there is no mask on constituent files
    Updated 07/2021: added check that tide model files are accessible
    Updated 06/2021: add warning for tide models being entered as string
    Updated 05/2021: added option for extrapolation cutoff in kilometers
    Updated 03/2021: add extrapolation check where there are no invalid points
        prevent ComplexWarning for fill values when calculating amplitudes
        simplified inputs to be similar to binary OTIS read program
    Updated 02/2021: set invalid values to nan in extrapolation
        replaced numpy bool to prevent deprecation warning
    Updated 12/2020: added valid data extrapolation with nearest_extrap
        replace tostring with tobytes to fix DeprecationWarning
    Updated 11/2020: create function to read bathymetry and spatial coordinates
    Updated 09/2020: set bounds error to false for regular grid interpolations
        adjust dimensions of input coordinates to be iterable
        reduce number of interpolations by copying bathymetry mask to variables
    Updated 08/2020: replaced griddata with scipy regular grid interpolators
    Updated 07/2020: added function docstrings. separate bilinear interpolation
        changed TYPE variable to keyword argument. update griddata interpolation
    Updated 06/2020: use argmin and argmax in bilinear interpolation
    Written 09/2019
"""
from __future__ import annotations

import gzip
import pathlib
import xarray as xr

__all__ = [
    'open_dataset',
    'open_mfdataset',
    'open_atlas_grid',
    'open_atlas_dataset',
    'ATLASDataTree',
]

def open_dataset(model_files: list[str] | list[pathlib.Path],
        grid_file: str | pathlib.Path,
        **kwargs
    ):
    """
    Open ATLAS tide model file

    Parameters
    ----------
    model_files: list of str or pathlib.Path
        list of ATLAS model files
    grid_file: str or pathlib.path
        ATLAS model grid file
    **kwargs: dict
        additional keyword arguments for opening ATLAS files

    Returns
    -------
    ds: xarray.Dataset
        ATLAS tide model data
    """
    # set default keyword arguments
    kwargs.setdefault('scale', 1.0)
    # read ATLAS grid and model files
    ds1 = open_atlas_grid(grid_file, **kwargs)
    # scale constituent data to output units
    ds2 = open_mfdataset(model_files, **kwargs)
    # merge datasets
    ds = xr.merge([ds1, ds2])
    # return xarray dataset
    return ds

# PURPOSE: read a list of ATLAS netCDF4 files
def open_mfdataset(
        model_files: list[str] | list[pathlib.Path],
        **kwargs
    ):
    """
    Open multiple ATLAS model files

    Parameters
    ----------
    model_files: list of str or pathlib.Path
        list of ATLAS model files
    **kwargs: dict
        additional keyword arguments for opening ATLAS files

    Returns
    -------
    ds: xarray.Dataset
        ATLAS tide model data
    """
    # read each file and store constituents in list
    d = [open_atlas_dataset(f, **kwargs) for f in model_files]
    # merge datasets
    ds = xr.merge(d)
    # return xarray dataset
    return ds

def open_atlas_grid(
        grid_file: str | pathlib.Path,
        type: str = 'z',
        compressed: bool = False,
        **kwargs
    ):
    """
    Open ATLAS model grid file

    Parameters
    ----------
    grid_file: str or pathlib.Path
        ATLAS model grid file
    type: str, default 'z'
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: horizontal transport velocities
            - ``'U'``: horizontal depth-averaged transport
            - ``'v'``: vertical transport velocities
            - ``'V'``: vertical depth-averaged transport
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    ds: xarray.Dataset
        ATLAS tide model data
    """
    # tilde-expand input file
    grid_file = pathlib.Path(grid_file).expanduser()
    # read the netCDF4-format tide elevation file
    if compressed:
        # read gzipped netCDF4 file
        f = gzip.open(grid_file, 'rb')
        tmp = xr.open_dataset(f, mask_and_scale=True)
    else:
        tmp = xr.open_dataset(grid_file, mask_and_scale=True)
    # read bathymetry and coordinates for variable type
    if (type == 'z'):
        # get bathymetry at nodes
        ds = tmp['hz'].T.to_dataset(name='bathymetry')
        ds.coords['x'] = tmp['lon_z']
        ds.coords['y'] = tmp['lat_z']
    elif type in ('U','u'):
        # get bathymetry at nodes
        ds = tmp['hu'].T.to_dataset(name='bathymetry')
        ds.coords['x'] = tmp['lon_u']
        ds.coords['y'] = tmp['lat_u']
    elif type in ('V','v'):
        # get bathymetry at nodes
        ds = tmp['hv'].T.to_dataset(name='bathymetry')
        ds.coords['x'] = tmp['lon_v']
        ds.coords['y'] = tmp['lat_v']
    # mask invalid bathymetries
    ds = ds.where(ds.bathymetry != 0, None, drop=False)
    # close open gzip file if compressed
    f.close() if kwargs['compressed'] else None
    # return xarray dataset
    return ds
    
# PURPOSE: reads ATLAS netCDF4 files
def open_atlas_dataset(
        input_file: str | pathlib.Path,
        type: str = 'z',
        scale: float = 1.0,
        compressed: bool = False,
        **kwargs
    ):
    """
    Open ATLAS-formatted netCDF4 files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input ATLAS file
    type: str, default 'z'
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: horizontal transport velocities
            - ``'U'``: horizontal depth-averaged transport
            - ``'v'``: vertical transport velocities
            - ``'V'``: vertical depth-averaged transport
    scale: float, default 1.0
        Scaling factor for converting to output units
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    ds: xarray.Dataset
        ATLAS tide model data
    """
    # tilde-expand input file
    input_file = pathlib.Path(input_file).expanduser()
    # read the netCDF4-format tide elevation file
    if compressed:
        # read gzipped netCDF4 file
        f = gzip.open(input_file, 'rb')
        tmp = xr.open_dataset(f, mask_and_scale=True)
    else:
        tmp = xr.open_dataset(input_file, mask_and_scale=True)
    # constituent name
    constituent = tmp['con'].values.tobytes().decode('utf-8').strip()
    if (type == 'z'):
        ds = (tmp['hRe'].T + -1j*tmp['hIm'].T).to_dataset(name=constituent)
        ds.coords['x'] = tmp['lon_z']
        ds.coords['y'] = tmp['lat_z']
    elif type in ('U','u'):
        ds = (tmp['URe'].T + -1j*tmp['UIm'].T).to_dataset(name=constituent)
        ds.coords['x'] = tmp['lon_u']
        ds.coords['y'] = tmp['lat_u']
    elif type in ('V','v'):
        ds = (tmp['VRe'].T + -1j*tmp['VIm'].T).to_dataset(name=constituent)
        ds.coords['x'] = tmp['lon_v']
        ds.coords['y'] = tmp['lat_v']
    # close open gzip file if compressed
    f.close() if kwargs['compressed'] else None
    # return xarray dataset scaled to output units
    return ds*scale

@xr.register_datatree_accessor('atlas')
class ATLASDataTree:
    """
    Accessor for extending an ``xarray.DataTree`` for ATLAS-netcdf
    tidal models
    """
    def __init__(self, dtree):
        self._dtree = dtree
