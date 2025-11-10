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
import pint
import pathlib
import datetime
import xarray as xr
import pyTMD.version

__all__ = [
    'open_dataset',
    'open_mfdataset',
    'open_atlas_grid',
    'open_atlas_dataset',
    'ATLASDataset',
    'ATLASDataTree',
]

# pint unit registry
__ureg__ = pint.UnitRegistry()
# default units for currents
_default_units = {
    'u': 'cm/s',
    'v': 'cm/s',
}

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
    kwargs.setdefault('type', 'z')
    # read ATLAS grid and model files
    ds1 = open_atlas_grid(grid_file, **kwargs)
    ds2 = open_mfdataset(model_files, **kwargs)
    # merge datasets
    ds = xr.merge([ds1, ds2], compat='override')
    ds2.attrs['units'] = ds2.attrs['units']
    # convert transports to currents if necessary
    if kwargs['type'] in ('u','v'):
        # get units for constituents and bathymetry
        quantity = 1.0*(__ureg__.parse_units(ds2.attrs['units']) /
            __ureg__.parse_units(ds1.bathymetry.attrs['units']))
        # conversion factor for outputs units
        base_units = quantity.to(_default_units[kwargs['type']])
        scale_factor = base_units.magnitude
        # convert transports to currents in output units
        ds[ds.tmd.constituents] *= scale_factor/ds['bathymetry']
        # update units attributes
        ds.attrs['units'] = base_units.units
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
    ds = xr.merge(d, compat='override')
    # return xarray dataset
    return ds

def open_atlas_grid(
        grid_file: str | pathlib.Path,
        type: str = 'z',
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
            - ``'u'``: zonal currents
            - ``'U'``: zonal depth-averaged transport
            - ``'v'``: meridional currents
            - ``'V'``: meridional depth-averaged transport
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    ds: xarray.Dataset
        ATLAS tide model data
    """
    # set default keyword arguments
    kwargs.setdefault('compressed', False)
    # tilde-expand input file
    grid_file = pathlib.Path(grid_file).expanduser()
    # read the netCDF4-format tide elevation file
    if kwargs['compressed']:
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
    # add attributes
    ds.attrs['type'] = type
    ds.attrs['format'] = 'ATLAS'
    # close open gzip file if compressed
    f.close() if kwargs['compressed'] else None
    # return xarray dataset
    return ds
    
# PURPOSE: reads ATLAS netCDF4 files
def open_atlas_dataset(
        input_file: str | pathlib.Path,
        type: str = 'z',
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
            - ``'u'``: zonal currents
            - ``'U'``: zonal depth-averaged transport
            - ``'v'``: meridional currents
            - ``'V'``: meridional depth-averaged transport
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    ds: xarray.Dataset
        ATLAS tide model data
    """
    # set default keyword arguments
    kwargs.setdefault('compressed', False)
    # tilde-expand input file
    input_file = pathlib.Path(input_file).expanduser()
    # read the netCDF4-format tide elevation file
    if kwargs['compressed']:
        # read gzipped netCDF4 file
        f = gzip.open(input_file, 'rb')
        tmp = xr.open_dataset(f, mask_and_scale=True)
    else:
        tmp = xr.open_dataset(input_file, mask_and_scale=True)
    # constituent name
    con = tmp['con'].values.astype('|S').tobytes().decode('utf-8').strip()
    if (type == 'z'):
        ds = (tmp['hRe'].T + -1j*tmp['hIm'].T).to_dataset(name=con)
        ds.coords['x'] = tmp['lon_z']
        ds.coords['y'] = tmp['lat_z']
        ds.attrs['units'] = tmp['hRe'].attrs.get('units')
    elif type in ('U','u'):
        ds = (tmp['uRe'].T + -1j*tmp['uIm'].T).to_dataset(name=con)
        ds.coords['x'] = tmp['lon_u']
        ds.coords['y'] = tmp['lat_u']
        ds.attrs['units'] = tmp['uRe'].attrs.get('units')
    elif type in ('V','v'):
        ds = (tmp['vRe'].T + -1j*tmp['vIm'].T).to_dataset(name=con)
        ds.coords['x'] = tmp['lon_v']
        ds.coords['y'] = tmp['lat_v']
        ds.attrs['units'] = tmp['vRe'].attrs.get('units')
    # add attributes
    ds.attrs['type'] = type
    ds.attrs['format'] = 'ATLAS'
    # close open gzip file if compressed
    f.close() if kwargs['compressed'] else None
    # return xarray dataset
    return ds

# PURPOSE: ATLAS-netcdf utilities for xarray Datasets
@xr.register_dataset_accessor('atlas')
class ATLASDataset:
    """
    Accessor for extending an ``xarray.Dataset`` for ATLAS-netcdf
    tidal models
    """
    def __init__(self, ds):
        self._ds = ds

    # PURPOSE: output grid file in ATLAS netCDF format
    def to_grid(self,
            path: str | pathlib.Path,
            mode: str = 'w',
            encoding: dict = {"zlib": True, "complevel": 9},
            **kwargs
        ):
        """
        Writes grid data to netCDF4 files in ATLAS format

        Parameters
        ----------
        path: str or pathlib.Path
            output ATLAS-netcdf grid file name
        mode: str, default 'w'
            netCDF4 file mode
        encoding: dict, default {"zlib": True, "complevel": 9}
            netCDF4 variable compression settings
        **kwargs: dict
            additional keyword arguments for xarray netCDF4 writer
        """
        # tilde-expand output file
        path = pathlib.Path(path).expanduser()
        # set variable names for type
        type = self._ds.attrs['type'].lower()
        depth_key = f'h{type}'
        lon_key = f'lon_{type}'
        lat_key = f'lat_{type}'
        # set default encoding
        kwargs.setdefault('encoding', {depth_key: encoding})
        # coordinate remapping
        mapping_coords = dict(x=lon_key, y=lat_key)
        attrs = {lon_key: {}, lat_key: {}, depth_key: {}}
        # set variable attributes
        attrs[lon_key]['units'] = 'degrees_east'
        attrs[lon_key]['long_name'] = f'longitude of {type.upper()} nodes'
        attrs[lat_key]['units'] = 'degrees_north'
        attrs[lat_key]['long_name'] = f'latitude of {type.upper()} nodes'
        attrs[depth_key]['units'] = 'meters'
        attrs[depth_key]['long_name'] = f'Bathymetry at {type.upper()} nodes'
        attrs[depth_key]['field'] = 'bath, scalar'
        # create output xarray dataset
        ds = xr.Dataset()
        ds[depth_key] = self._ds['bathymetry']
        # remap coordinates to ATLAS convention
        ds = ds.rename(mapping_coords)
        # add global attributes
        ds.attrs['title'] = 'ATLAS bathymetry data'
        ds.attrs['type'] = 'OTIS grid file'
        ds.attrs['date_created'] = datetime.datetime.now().isoformat()
        ds.attrs['software_reference'] = pyTMD.version.project_name
        ds.attrs['software_version'] = pyTMD.version.full_version
        # set variable attributes
        for key, value in attrs.items():
            ds[key].attrs.update(value)
        # output to netCDF4 file
        ds.to_netcdf(path, mode=mode, **kwargs)

    # PURPOSE: output tidal constituent data in ATLAS netCDF format
    def to_netcdf(self,
            path: str | pathlib.Path,
            mode: str = 'w',
            encoding: dict = {"zlib": True, "complevel": 9},
            **kwargs
        ):
        """
        Writes tidal constituents to netCDF4 files in ATLAS format

        Parameters
        ----------
        path: str or pathlib.Path
            output directory for ATLAS-netcdf files
        mode: str, default 'w'
            netCDF4 file mode
        encoding: dict, default {"zlib": True, "complevel": 9}
            netCDF4 variable compression settings
        **kwargs: dict
            additional keyword arguments for xarray netCDF4 writer
        """
        # tilde-expand output directory
        path = pathlib.Path(path).expanduser()
        # set variable names 
        type = self._ds.attrs['type'].lower()
        type_key = dict(z='h', u='U', v='V')[type]
        lon_key = f'lon_{type}'
        lat_key = f'lat_{type}'
        # set default encoding
        default_encoding = {f'{type_key}{c}': encoding for c in ('Re','Im')}
        kwargs.setdefault('encoding', default_encoding)
        # coordinate remapping
        mapping_coords = dict(x=lon_key, y=lat_key)
        attrs = {lon_key: {}, lat_key: {}}
        # set variable attributes
        attrs[lon_key]['units'] = 'degrees_east'
        attrs[lon_key]['long_name'] = f'longitude of {type.upper()} nodes'
        attrs[lat_key]['units'] = 'degrees_north'
        attrs[lat_key]['long_name'] = f'latitude of {type.upper()} nodes'
        # build variable attributes for real and imaginary components
        for key, val in dict(Re='Real part', Im='Imag part').items():
            # variable units and long_name attributes
            if (type == 'z'):
                long_name = f'Tidal elevation complex amplitude, {val}'
            elif (type == 'u'):
                long_name = f'Tidal WE transport complex amplitude, {val}'
            elif (type == 'v'):
                long_name = f'Tidal SN transport complex amplitude, {val}'
            # variable field description
            fields = []
            fields.append(f'{key}({type_key}), scalar')
            fields.append(f'amp=abs({type_key}Re+i*{type_key}Im)')
            fields.append(f'GMT phase=atan2(-{type_key}Im,{type_key}Re)/pi*180')
            # set variable attributes
            attrs[f'{type_key}{key}'] = {}
            attrs[f'{type_key}{key}']['units'] = self._ds.attrs['units']
            attrs[f'{type_key}{key}']['long_name'] = long_name
            attrs[f'{type_key}{key}']['field'] = '; '.join(fields)
        # create output xarray dataset for each constituent
        for v in self._ds.tmd.constituents:
            # create xarray dataset
            ds = xr.Dataset()
            # extract real and imaginary components
            ds[f'{type_key}Re'] = self._ds[v].real.values()
            ds[f'{type_key}Im'] = self._ds[v].imag.values()
            # remap coordinates to ATLAS convention
            ds = ds.rename(mapping_coords)
            # update variable attributes
            for att_name, att_val in attrs.items():
                ds[att_name].attrs.update(att_val)
            # add global attributes
            if type == 'z':
                ds.attrs['title'] = 'ATLAS tidal elevation file'
                ds.attrs['type'] = 'OTIS elevation file'
            elif type in ('u', 'v'):
                ds.attrs['title'] = 'ATLAS tidal SN and WE transports file'
                ds.attrs['type'] = 'OTIS transport file'
            ds.attrs['date_created'] = datetime.datetime.now().isoformat()
            ds.attrs['software_reference'] = pyTMD.version.project_name
            ds.attrs['software_version'] = pyTMD.version.full_version
            # write ATLAS netCDF4 file
            FILE = path.joinpath(f"{v}.nc")
            ds.to_netcdf(FILE, mode=mode, **kwargs)

# PURPOSE: ATLAS-netcdf utilities for xarray DataTrees
@xr.register_datatree_accessor('atlas')
class ATLASDataTree:
    """
    Accessor for extending an ``xarray.DataTree`` for ATLAS-netcdf
    tidal models
    """
    def __init__(self, dtree):
        self._dtree = dtree

    def to_netcdf(self,
            grid_file: str | pathlib.Path,
            directory: str | pathlib.Path | None = None,
            **kwargs
        ):
        """
        Writes netCDF4 files in ATLAS format

        Parameters
        ----------
        grid_file: str or pathlib.Path
            output ATLAS-netcdf grid file
        directory: str or pathlib.Path
            output directory for ATLAS-netcdf files
        **kwargs: dict
            additional keyword arguments for netCDF4 writer
        """
        # tilde-expand grid file
        grid_file = pathlib.Path(grid_file).expanduser()
        # set default output directory
        directory = grid_file.parent if directory is None else directory
        # for each model type
        for type in ('z', 'u', 'v'):
            # get xarray dataset for type
            ds = self._dtree[type].to_dataset()
            # write in append mode to add type to same grid and directory
            # output grid file
            ds.atlasnc.to_grid(grid_file,
                type=type,
                mode='a',
                **kwargs)
            # output constituent files
            ds.atlasnc.to_netcdf(directory,
                type=type,
                mode='a',
                **kwargs
            )
