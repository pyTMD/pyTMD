#!/usr/bin/env python
u"""
IERS.py
Written by Tyler Sutterley (11/2025)

Reads ocean pole load tide coefficients provided by IERS
http://maia.usno.navy.mil/conventions/2010/2010_official/chapter7/tn36_c7.pdf
http://maia.usno.navy.mil/conventions/2010/2010_update/chapter7/icc7.pdf

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

REFERENCES:
    S. Desai, "Observing the pole tide with satellite altimetry", Journal of
        Geophysical Research: Oceans, 107(C11), 2002. doi: 10.1029/2001JC001224
    S. Desai, J. Wahr and B. Beckley "Revisiting the pole tide for and from
        satellite altimetry", Journal of Geodesy, 89(12), p1233-1243, 2015.
        doi: 10.1007/s00190-015-0848-7

UPDATE HISTORY:
    Updated 11/2025: near-complete rewrite of program to use xarray
    Updated 08/2024: convert outputs to be in -180:180 longitude convention
        added function to interpolate ocean pole tide values to coordinates
        renamed from ocean_pole_tide to IERS
    Updated 06/2024: use np.clongdouble instead of np.longcomplex
    Updated 05/2023: add default for ocean pole tide file
    Updated 04/2023: using pathlib to define and expand paths
    Updated 03/2023: add basic variable typing to function inputs
    Updated 12/2022: refactor ocean pole tide read programs under io
    Updated 04/2022: updated docstrings to numpy documentation format
        use longcomplex data format to be windows compliant
    Updated 07/2021: added check that ocean pole tide file is accessible
    Updated 03/2021: replaced numpy bool/int to prevent deprecation warnings
    Updated 08/2020: output north load and east load deformation components
    Updated 07/2020: added function docstrings
    Updated 12/2018: Compatibility updates for Python3
    Written 09/2017
"""
from __future__ import annotations

import re
import gzip
import pathlib
import warnings
import numpy as np
import xarray as xr
import pyTMD.utilities
# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
# attempt imports
pyproj = pyTMD.utilities.import_dependency('pyproj')

__all__ = [
    "open_dataset",
]

# ocean pole tide file from Desai (2002) and IERS conventions
_ocean_pole_tide_file = pyTMD.utilities.get_data_path(
    ['data','opoleloadcoefcmcor.txt.gz']
)

# PURPOSE: read real and imaginary ocean pole tide coefficients
def open_dataset(
        input_file: str | pathlib.Path = _ocean_pole_tide_file,
        chunks: int | dict | str | None = None,
        **kwargs
    ):
    """
    Open Ocean Pole Tide ASCII files from
    :cite:p:`Desai:2002ev,Desai:2015jr`

    Parameters
    ----------
    input_file: str or pathlib.Path
        Ocean pole tide file
    chunks: int | dict | str | None, default None
        coerce output to specified chunks
    crs: str | int | dict, default 4326
        Coordinate reference system
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    ds: xarray.Dataset
        Ocean pole tide data
    """
    # set default keyword arguments
    kwargs.setdefault('compressed', True)
    # default coordinate reference system is EPSG:4326 (WGS84)
    crs = kwargs.get('crs', 4326)
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if input_file.is_local() and not input_file.exists():
        raise FileNotFoundError(f'File not found: {input_file}')
    # read compressed ocean pole tide file
    if kwargs['compressed']:
        # read gzipped ascii file
        with gzip.open(input_file, 'rb') as f:
            file_contents = f.read().decode('utf8').splitlines()
    else:
        with open(input_file, mode='r', encoding='utf8') as f:
            file_contents = f.read().splitlines()

    # counts the number of lines in the header
    count = 0
    # parse over header text
    parameters = {}
    HEADER = True
    while HEADER:
        # file line at count
        line = file_contents[count]
        # detect the end of the header text
        HEADER = not bool(re.match(r'---------', line))
        # parse key-value pairs from header
        key, _, val = line.partition('=')
        parameters[key.strip().lower()] = val.strip()
        # add 1 to counter
        count += 1

    # grid parameters and dimensions
    dlon = float(parameters['longitude_step_degrees'])
    dlat = float(parameters['latitude_step_degrees'])
    nlon = int(parameters['number_longitude_grid_points'])
    nlat = int(parameters['number_latitude_grid_points'])
    # create grid vectors (coerce to -180:180 longitude convention)
    lon_start = -180.0 + dlon/2.0
    lat_start = float(parameters['first_latitude_degrees'])
    lon = lon_start + np.arange(nlon)*dlon
    lat = lat_start + np.arange(nlat)*dlat
    # data dictionary
    var = dict(dims=('y', 'x'), coords={}, data_vars={})
    var["coords"]["y"] = dict(data=lat.copy(), dims='y')
    var["coords"]["x"] = dict(data=lon.copy(), dims='x')
    # allocate for output grid maps
    for key in ['R','N','E']:
        var["data_vars"][key] = {}
        var["data_vars"][key]["dims"] = ('y', 'x')
        data = np.zeros((nlat, nlon), dtype=np.clongdouble)
        var["data_vars"][key]["data"] = data

    # read lines of file and add to output variables
    for i,line in enumerate(file_contents[count:]):
        # read line of ocean pole tide file
        ln,lt,urr,uri,unr,uni,uer,uei = np.array(line.split(), dtype='f8')
        # calculate indices of output grid
        # coerce to -180:180 longitude convention
        ilon = int(np.mod(ln - lon_start, 360.0)//dlon) 
        ilat = int((lt - lat_start)//dlat)
        # assign ocean pole tide coefficients to output variables
        var["data_vars"]["R"]["data"][ilat,ilon] = urr + 1j*uri
        var["data_vars"]["N"]["data"][ilat,ilon] = unr + 1j*uni
        var["data_vars"]["E"]["data"][ilat,ilon] = uer + 1j*uei
    # convert to xarray Dataset from the data dictionary
    ds = xr.Dataset.from_dict(var)
    # coerce to specified chunks
    if chunks is not None:
        ds = ds.chunk(chunks)
    # add attributes
    ds.attrs['crs'] = pyproj.CRS.from_user_input(crs).to_dict()
    # return xarray dataset
    return ds
