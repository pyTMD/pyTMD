#!/usr/bin/env python
u"""
FES.py
Written by Tyler Sutterley (11/2025)

Reads ascii and netCDF4 files for FES tidal solutions provided by AVISO
    https://www.aviso.altimetry.fr/data/products/auxiliary-products/
        global-tide-fes.html

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 11/2025: near-complete rewrite of program to use xarray
    Updated 10/2025: simplify ascii read function to use masked_equal
    Updated 08/2025: use numpy degree to radian conversions
        added option to gap fill when reading constituent grids
    Updated 11/2024: expose buffer distance for cropping tide model data
    Updated 10/2024: fix error when using default bounds in extract_constants
    Updated 07/2024: added new FES2022 to available known model versions
        FES2022 have masked longitudes, only extract longitude data
        FES2022 extrapolated data have zeroed out inland water bodies
        added crop and bounds keywords for trimming model data
    Updated 01/2024: attempt to extract constituent IDs from filenames
    Updated 06/2023: extract ocean tide model variables for FES2012
    Updated 04/2023: added global HAMTIDE11 model
        using pathlib to define and expand tide model paths
    Updated 03/2023: add basic variable typing to function inputs
    Updated 12/2022: refactor tide read programs under io
        new functions to read and interpolate from constituents class
        new functions to output FES formatted netCDF4 files
        refactored interpolation routines into new module
    Updated 11/2022: place some imports within try/except statements
        use f-strings for formatting verbose or ascii output
    Updated 05/2022: reformat arguments to extract_FES_constants definition
        changed keyword arguments to camel case
    Updated 04/2022: updated docstrings to numpy documentation format
        include utf-8 encoding in reads to be windows compliant
        fix netCDF4 masks for nan values
    Updated 01/2022: added global Empirical Ocean Tide model (EOT20)
    Updated 12/2021: adjust longitude convention based on model longitude
    Updated 07/2021: added check that tide model files are accessible
    Updated 06/2021: add warning for tide models being entered as string
    Updated 05/2021: added option for extrapolation cutoff in kilometers
    Updated 03/2021: add extrapolation check where there are no invalid points
        prevent ComplexWarning for fill values when calculating amplitudes
        simplified inputs to be similar to binary OTIS read program
        replaced numpy bool/int to prevent deprecation warnings
        use uuid for reading from gzipped netCDF4 files
    Updated 02/2021: set invalid values to nan in extrapolation
        replaced numpy bool to prevent deprecation warning
    Updated 12/2020: added nearest-neighbor data extrapolation
    Updated 09/2020: set bounds error to false for regular grid interpolations
        adjust dimensions of input coordinates to be iterable
    Updated 08/2020: replaced griddata with scipy regular grid interpolators
    Written 07/2020
"""
from __future__ import division, annotations

import re
import gzip
import pathlib
import datetime
import numpy as np
import xarray as xr
import pyTMD.constituents
import pyTMD.utilities
# attempt imports
dask = pyTMD.utilities.import_dependency('dask')
dask_available = pyTMD.utilities.dependency_available('dask')

__all__ = [
    'open_mfdataset',
    'open_fes_dataset',
    'open_fes_ascii',
    'open_fes_netcdf',
    'FESDataset',
]

# PURPOSE: read a list of FES ASCII or netCDF4 files
def open_mfdataset(
        model_files: list[str] | list[pathlib.Path],
        parallel: bool = False,
        **kwargs,
    ):
    """
    Open multiple FES model files

    Parameters
    ----------
    model_files: list of str or pathlib.Path
        list of FES model files
    parallel: bool, default False
        Open files in parallel using ``dask.delayed``
    kwargs: dict
        additional keyword arguments for opening FES files

    Returns
    -------
    ds: xarray.Dataset
        FES tide model data
    """
    # merge multiple granules
    if parallel and dask_available:
        opener = dask.delayed(open_fes_dataset)
    else:
        opener = open_fes_dataset
    # read each file as xarray dataset and append to list
    d = [opener(f, **kwargs) for f in model_files]
    # read datasets as dask arrays
    if parallel and dask_available:
        d, = dask.compute(d)
    # merge datasets
    ds = xr.merge(d, compat='override')
    # return xarray dataset
    return ds

# PURPOSE: reads a FES ASCII or netCDF4 file
def open_fes_dataset(
        input_file: str | pathlib.Path,
        **kwargs
    ):
    """
    Open FES-formatted model files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input transport file
    format: str, default 'netcdf'
        Model format

            - ``'ascii'``: FES ascii format
            - ``'netcdf'``: FES netCDF4 format
    **kwargs: dict
        additional keyword arguments for opening FES files
    
    Returns
    -------
    ds: xarray.Dataset
        FES tide model data
    """
    # detect file format if not provided
    if kwargs.get('format', None) is None:
        kwargs['format'] = pyTMD.utilities.detect_format(input_file)
    # detect if file is compressed if not provided
    if kwargs.get('compressed', None) is None:
        kwargs['compressed'] = pyTMD.utilities.detect_compression(input_file)
    # open FES files based on format
    if kwargs['format'] == 'ascii':
        # FES ascii constituent files
        ds = open_fes_ascii(input_file, **kwargs)
    elif kwargs['format'] == 'netcdf':
        # FES netCDF4 constituent files
        ds = open_fes_netcdf(input_file, **kwargs)
    else:
        raise ValueError(f"Unrecognized file format: {kwargs['format']}") 
    # return xarray dataset
    return ds

# PURPOSE: read FES ASCII files
def open_fes_ascii(
        input_file: str | pathlib.Path,
        chunks: int | dict | str | None = None,
        **kwargs
    ):
    """
    Open FES-formatted ASCII files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input ascii file
    chunks: int, dict, str, or None, default None
        coerce output to specified chunks
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    ds: xarray.Dataset
        FES tide model data
    """
    # set default keyword arguments
    kwargs.setdefault('compressed', False)
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if input_file.is_local() and not input_file.exists():
        raise FileNotFoundError(f'File not found: {input_file}')
    # read the ASCII-format tide elevation file
    if kwargs['compressed']:
        # read gzipped ascii file
        with gzip.open(input_file, 'rb') as f:
            file_contents = f.read().splitlines()
    else:
        with open(input_file, mode="r", encoding='utf8') as f:
            file_contents = f.read().splitlines()
    # parse model file for constituent identifier
    cons = pyTMD.constituents._parse_name(input_file.stem)
    # longitude range (lonmin, lonmax)
    lonmin, lonmax = np.array(file_contents[0].split(), dtype=np.float64)
    # latitude range (latmin, latmax)
    latmin, latmax = np.array(file_contents[1].split(), dtype=np.float64)
    # grid step size (dlon, dlat)
    dlon, dlat = np.array(file_contents[2].split(), dtype=np.float64)
    # grid dimensions (nlon, nlat)
    nlon, nlat = np.array(file_contents[3].split(), dtype=int)
    # mask fill value
    masked_values = file_contents[4].split()
    fill_value = np.float32(masked_values[0])
    # number of columns in ascii file
    ncol = 30
    # create latitude and longitude arrays
    lat = np.linspace(latmin, latmax, nlat)
    lon = np.linspace(lonmin, lonmax, nlon)
    # data dictionary
    var = dict(dims=('y', 'x'), coords={}, data_vars={})
    var["coords"]["y"] = dict(data=lat.copy(), dims='y')
    var["coords"]["x"] = dict(data=lon.copy(), dims='x')
    # input amplitude and phase
    amp = np.zeros((nlat,nlon), dtype=np.float32)
    ph = np.zeros((nlat,nlon), dtype=np.float32)
    # starting line to fill amplitude and phase variables
    i1 = 5
    # for each latitude
    for i in range(nlat):
        for j in range(nlon//ncol):
            j1 = j*ncol
            # amplitude and phase are on two separate rows
            amp[i,j1:j1+ncol] = np.array(file_contents[i1].split())
            ph[i,j1:j1+ncol] = np.array(file_contents[i1+1].split())
            i1 += 2
        # add last rows of tidal variables
        j1 = (j+1)*ncol
        j2 = nlon % ncol
        # amplitude and phase are on two separate rows
        amp[i,j1:j1+j2] = np.array(file_contents[i1].split())
        ph[i,j1:j1+j2] = np.array(file_contents[i1+1].split())
        i1 += 2
    # convert to masked arrays
    amp = np.ma.masked_equal(amp, fill_value)
    ph = np.ma.masked_equal(ph, fill_value)
    # store the data variables
    var["data_vars"][cons] = {}
    var["data_vars"][cons]["dims"] = ('y', 'x')
    var["data_vars"][cons]["data"] = amp*np.exp(-1j*ph*np.pi/180.0)
    # convert to xarray Dataset from the data dictionary
    ds = xr.Dataset.from_dict(var)
    # coerce to specified chunks
    if chunks is not None:
        ds = ds.chunk(chunks)
    # add attributes
    ds.attrs['group'] = kwargs['group']
    # return xarray dataset
    return ds

# PURPOSE: read FES netCDF4 files
def open_fes_netcdf(
        input_file: str | pathlib.Path,
        chunks: int | dict | str | None = None,
        **kwargs
    ):
    """
    Open FES-formatted netCDF4 files

    Parameters
    ----------
    input_file: str or pathlib.Path
        model file
    group: str or NoneType, default None
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: zonal currents
            - ``'v'``: meridional currents
    chunks: int, dict, str, or None, default None
        variable chunk sizes for dask (see ``xarray.open_dataset``)
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    ds: xarray.Dataset
        FES tide model data
    """
    # set default keyword arguments
    kwargs.setdefault('group', 'z')
    kwargs.setdefault('compressed', False)
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if input_file.is_local() and not input_file.exists():
        raise FileNotFoundError(f'File not found: {input_file}')
    # read the netCDF4-format tide elevation file
    if kwargs['compressed']:
        # read gzipped netCDF4 file
        f = gzip.open(input_file, 'rb')
        tmp = xr.open_dataset(f, mask_and_scale=True, chunks=chunks)
    else:
        tmp = xr.open_dataset(input_file, mask_and_scale=True, chunks=chunks)
    # parse model file for constituent identifier
    cons = pyTMD.constituents._parse_name(input_file.stem)
    # amplitude and phase components for different versions
    if 'Ha' in tmp.variables:
        # FES2012 variable names
        mapping_coords = dict(lon='x', lat='y')
        mapping_amp = dict(z='Ha', u='Ua', v='Va')
        mapping_ph = dict(z='Hg', u='Ug', v='Vg')
    elif 'amplitude' in tmp.variables:
        # FES2014/2022 variable names
        mapping_coords = dict(lon='x', lat='y')
        mapping_amp = dict(z='amplitude', u='Ua', v='Va')
        mapping_ph = dict(z='phase', u='Ug', v='Vg')
    elif 'AMPL' in tmp.variables:
        # HAMTIDE11 variable names
        mapping_coords = dict(LON='x', LAT='y')
        mapping_amp = dict(z='AMPL', u='UAMP', v='VAMP')
        mapping_ph = dict(z='PHAS', u='UPHA', v='VPHA')
    # amplitude and phase variable names
    amp_key = mapping_amp[kwargs['group']]
    phase_key = mapping_ph[kwargs['group']]
    # mask where amplitude or phase are zero
    valid_values = (tmp[amp_key] != 0) & (tmp[phase_key] != 0)
    amplitude = tmp[amp_key].where(valid_values, drop=False)
    phase = tmp[phase_key].where(valid_values, drop=False)
    # create output xarray dataset for file
    ds = xr.Dataset()
    # calculate complex form of constituent oscillation
    ds[cons] = amplitude*np.exp(-1j*phase*np.pi/180.0)
    # rename coordinates
    ds = ds.rename(mapping_coords)
    # add attributes
    ds.attrs['group'] = kwargs['group']
    ds[cons].attrs['units'] = tmp[amp_key].attrs.get('units', '')
    # return xarray dataset
    return ds

# PURPOSE: FES utilities for xarray Datasets
@xr.register_dataset_accessor('fes')
class FESDataset:
    """Accessor for extending an ``xarray.Dataset`` for FES tidal models
    """
    def __init__(self, ds):
        self._ds = ds

    # PURPOSE: output tidal constituent file in FES2014/2022 format
    def to_netcdf(self,
            path: str | pathlib.Path,
            mode: str = 'w',
            encoding: dict = {"zlib": True, "complevel": 9},
            **kwargs
        ):
        """
        Writes tidal constituents to netCDF4 files in FES2014/2022 format

        Parameters
        ----------
        path: str | pathlib.Path
            output directory for netCDF4 files
        mode: str, default 'w'
            netCDF4 file mode
        encoding: dict, default {"zlib": True, "complevel": 9}
            netCDF4 variable compression settings
        **kwargs: dict
            additional keyword arguments for xarray netCDF4 writer
        """
        # tilde-expand output path
        path = pyTMD.utilities.Path(path).resolve()
        # set variable names and units for group
        if (self._ds.attrs['group'] == 'z'):
            amp_key = 'amplitude'
            phase_key = 'phase'
        elif (self._ds.attrs['group'] == 'u'):
            amp_key = 'Ua'
            phase_key = 'Ug'
        elif (self._ds.attrs['group'] == 'v'):
            amp_key = 'Va'
            phase_key = 'Vg'
        # set default encoding
        kwargs.setdefault('encoding', {amp_key: encoding, phase_key: encoding})
        # coordinate remapping
        mapping_coords = dict(x='lon', y='lat')
        attrs = dict(lon={}, lat={})
        attrs['lon']['axis'] = 'X'
        attrs['lon']['units'] = 'degrees_east'
        attrs['lon']['long_name'] = 'longitude'
        attrs['lat']['axis'] = 'Y'
        attrs['lat']['units'] = 'degrees_north'
        attrs['lat']['long_name'] = 'latitude'
        # for each variable
        for v in self._ds.data_vars.keys():
            # create xarray dataset
            ds = xr.Dataset()
            # calculate amplitude and phase
            ds[amp_key] = self._ds[v].tmd.amplitude
            ds[phase_key] = self._ds[v].tmd.phase
            ds[amp_key].attrs['units'] = self._ds[v].attrs.get('units', '')
            ds[phase_key].attrs['units'] = 'degrees'
            ds[amp_key].attrs['long_name'] = f'Tide amplitude at {v} frequency'
            ds[phase_key].attrs['long_name'] = f'Tide phase at {v} frequency'
            # define and fill constituent ID
            ds['con'] = v.ljust(4).encode('utf8')
            ds['con'].attrs['_Encoding'] = 'utf8'
            ds['con'].attrs['long_name'] = 'tidal constituent'
            # remap coordinates to FES convention
            ds = ds.rename(mapping_coords)
            # update variable attributes
            for att_name, att_val in attrs.items():
                ds[att_name].attrs.update(att_val)
            # add global attributes
            ds.attrs['title'] = 'FES tidal constituent data'
            ds.attrs['date_created'] = datetime.datetime.now().isoformat()
            ds.attrs['software_reference'] = pyTMD.version.project_name
            ds.attrs['software_version'] = pyTMD.version.full_version
            # write FES netCDF4 file
            FILE = path.joinpath(f"{v}.nc")
            ds.to_netcdf(FILE, mode=mode, **kwargs)
