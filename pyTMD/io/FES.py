#!/usr/bin/env python
u"""
FES.py
Written by Tyler Sutterley (12/2022)

Reads files for a tidal model and makes initial calculations to run tide program
Includes functions to extract tidal harmonic constants from the
    FES (Finite Element Solution) tide models for given locations
ascii and netCDF4 files can be been compressed using gzip

Reads ascii and netCDF4 FES tidal solutions provided by AVISO
    https://www.aviso.altimetry.fr/data/products/auxiliary-products/
        global-tide-fes.html

INPUTS:
    ilon: longitude to interpolate
    ilat: latitude to interpolate
    model_files: list of model files for each constituent

OPTIONS:
    type: tidal variable to run
        z: heights
        u: horizontal transport velocities
        v: vertical transport velocities
    version: model version to run
        FES1999
        FES2004
        FES2012
        FES2014
        EOT20
    method: interpolation method
        bilinear: quick bilinear interpolation
        spline: scipy bivariate spline interpolation
        linear, nearest: scipy regular grid interpolations
    extrapolate: extrapolate model using nearest-neighbors
    cutoff: extrapolation cutoff in kilometers
        set to np.inf to extrapolate for all points
    compressed: input files are gzip compressed
    scale: scaling factor for converting to output units

OUTPUTS:
    amplitude: amplitudes of tidal constituents
    phase: phases of tidal constituents

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    netCDF4: Python interface to the netCDF C library
         https://unidata.github.io/netcdf4-python/netCDF4/index.html

PROGRAM DEPENDENCIES:
    bilinear_interp.py: bilinear interpolation of data to coordinates
    nearest_extrap.py: nearest-neighbor extrapolation of data to coordinates

UPDATE HISTORY:
    Updated 12/2022: refactor tide read programs under io
        new functions to read and interpolate from constituents class
        new functions to output FES formatted netCDF4 files
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
import os
import copy
import gzip
import uuid
import logging
import datetime
import warnings
import numpy as np
import scipy.interpolate
import pyTMD.version
import pyTMD.io.constituents
from pyTMD.bilinear_interp import bilinear_interp
from pyTMD.nearest_extrap import nearest_extrap
from pyTMD.utilities import get_git_revision_hash

# attempt imports
try:
    import netCDF4
except (ImportError, ModuleNotFoundError) as e:
    warnings.filterwarnings("always")
    warnings.warn("netCDF4 not available")
    warnings.warn("Some functions will throw an exception if called")
# ignore warnings
warnings.filterwarnings("ignore")

# PURPOSE: extract harmonic constants from tide models at coordinates
def extract_constants(ilon, ilat, model_files=None, **kwargs):
    """
    Reads files for a FES ascii or netCDF4 tidal model

    Makes initial calculations to run the tide program

    Spatially interpolates tidal constituents to input coordinates

    Parameters
    ----------
    ilon: float
        longitude to interpolate
    ilat: float
        latitude to interpolate
    model_files: list or NoneType, default None
        list of model files for each constituent
    type: str, default 'z'
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: horizontal transport velocities
            - ``'v'``: vertical transport velocities
    version: str or NoneType, default None
        Model version to read

            - ``'FES1999'``
            - ``'FES2004'``
            - ``'FES2012'``
            - ``'FES2014'``
            - ``'EOT20'``
    method: str, default 'spline'
        Interpolation method

            - ``'bilinear'``: quick bilinear interpolation
            - ``'spline'``: scipy bivariate spline interpolation
            - ``'linear'``, ``'nearest'``: scipy regular grid interpolations
    extrapolate: bool, default False
        Extrapolate model using nearest-neighbors
    cutoff: float, default 10.0
        Extrapolation cutoff in kilometers

        Set to ``np.inf`` to extrapolate for all points
    compressed: bool, default False
        Input files are gzip compressed
    scale: float, default 1.0
        Scaling factor for converting to output units

    Returns
    -------
    amplitude: float
        amplitudes of tidal constituents
    phase: float
        phases of tidal constituents
    """
    # set default keyword arguments
    kwargs.setdefault('type', 'z')
    kwargs.setdefault('version', None)
    kwargs.setdefault('method', 'spline')
    kwargs.setdefault('extrapolate', False)
    kwargs.setdefault('cutoff', 10.0)
    kwargs.setdefault('compressed', False)
    kwargs.setdefault('scale', 1.0)
    # raise warnings for deprecated keyword arguments
    deprecated_keywords = dict(TYPE='type',VERSION='version',
        METHOD='method',EXTRAPOLATE='extrapolate',CUTOFF='cutoff',
        GZIP='compressed',SCALE='scale')
    for old,new in deprecated_keywords.items():
        if old in kwargs.keys():
            warnings.warn(f"""Deprecated keyword argument {old}.
                Changed to '{new}'""", DeprecationWarning)
            # set renamed argument to not break workflows
            kwargs[new] = copy.copy(kwargs[old])

    # raise warning if model files are entered as a string
    if isinstance(model_files,str):
        warnings.warn("Tide model is entered as a string")
        model_files = [model_files]

    # adjust dimensions of input coordinates to be iterable
    ilon = np.atleast_1d(np.copy(ilon))
    ilat = np.atleast_1d(np.copy(ilat))
    # number of points
    npts = len(ilon)
    # number of constituents
    nc = len(model_files)

    # amplitude and phase
    amplitude = np.ma.zeros((npts,nc))
    amplitude.mask = np.zeros((npts,nc),dtype=bool)
    ph = np.ma.zeros((npts,nc))
    ph.mask = np.zeros((npts,nc),dtype=bool)
    # read and interpolate each constituent
    for i, fi in enumerate(model_files):
        # check that model file is accessible
        if not os.access(os.path.expanduser(fi), os.F_OK):
            raise FileNotFoundError(os.path.expanduser(fi))
        # read constituent from elevation file
        if kwargs['version'] in ('FES1999','FES2004'):
            # FES ascii constituent files
            hc,lon,lat = read_ascii_file(os.path.expanduser(fi), **kwargs)
        elif kwargs['version'] in ('FES2012','FES2014','EOT20'):
            # FES netCDF4 constituent files
            hc,lon,lat = read_netcdf_file(os.path.expanduser(fi), **kwargs)
        # adjust longitudinal convention of input latitude and longitude
        # to fit tide model convention
        if (np.min(ilon) < 0.0) & (np.max(lon) > 180.0):
            # input points convention (-180:180)
            # tide model convention (0:360)
            ilon[ilon<0.0] += 360.0
        elif (np.max(ilon) > 180.0) & (np.min(lon) < 0.0):
            # input points convention (0:360)
            # tide model convention (-180:180)
            ilon[ilon>180.0] -= 360.0

        # grid step size of tide model
        dlon = lon[1] - lon[0]
        # replace original values with extend arrays/matrices
        if np.isclose(lon[-1] - lon[0], 360.0 - dlon):
            lon = extend_array(lon, dlon)
            hc = extend_matrix(hc)
        # determine if any input points are outside of the model bounds
        invalid = (ilon < lon.min()) | (ilon > lon.max()) | \
                  (ilat < lat.min()) | (ilat > lat.max())

        # interpolated complex form of constituent oscillation
        hci = np.ma.zeros((npts), dtype=hc.dtype, fill_value=hc.fill_value)
        hci.mask = np.zeros((npts),dtype=bool)
        # interpolate amplitude and phase of the constituent
        if (kwargs['method'] == 'bilinear'):
            # replace invalid values with nan
            hc.data[hc.mask] = np.nan
            # use quick bilinear to interpolate values
            hci.data[:] = bilinear_interp(lon, lat, hc, ilon, ilat,
                dtype=hc.dtype)
            # replace nan values with fill_value
            hci.mask[:] |= np.isnan(hci.data)
            hci.data[hci.mask] = hci.fill_value
        elif (kwargs['method'] == 'spline'):
            # interpolate complex form of the constituent
            # use scipy splines to interpolate values
            f1 = scipy.interpolate.RectBivariateSpline(lon, lat,
                hc.data.real.T, kx=1, ky=1)
            f2 = scipy.interpolate.RectBivariateSpline(lon, lat,
                hc.data.imag.T, kx=1, ky=1)
            f3 = scipy.interpolate.RectBivariateSpline(lon, lat,
                hc.mask.T, kx=1, ky=1)
            hci.data.real[:] = f1.ev(ilon,ilat)
            hci.data.imag[:] = f2.ev(ilon,ilat)
            hci.mask[:] = f3.ev(ilon,ilat).astype(bool)
            # replace invalid values with fill_value
            hci.data[hci.mask] = hci.fill_value
        else:
            # interpolate complex form of the constituent
            # use scipy regular grid to interpolate values
            r1 = scipy.interpolate.RegularGridInterpolator((lat,lon),
                hc.data, method=kwargs['method'], bounds_error=False,
                fill_value=hci.fill_value)
            r2 = scipy.interpolate.RegularGridInterpolator((lat,lon),
                hc.mask, method=kwargs['method'], bounds_error=False,
                fill_value=1)
            hci.data[:] = r1.__call__(np.c_[ilat,ilon])
            hci.mask[:] = np.ceil(r2.__call__(np.c_[ilat,ilon])).astype(bool)
            # replace invalid values with fill_value
            hci.mask[:] |= (hci.data == hci.fill_value)
            hci.data[hci.mask] = hci.fill_value
        # extrapolate data using nearest-neighbors
        if kwargs['extrapolate'] and np.any(hci.mask):
            # find invalid data points
            inv, = np.nonzero(hci.mask)
            # replace invalid values with nan
            hc.data[hc.mask] = np.nan
            # extrapolate points within cutoff of valid model points
            hci[inv] = nearest_extrap(lon, lat, hc, ilon[inv], ilat[inv],
                dtype=hc.dtype, cutoff=kwargs['cutoff'])
        # convert amplitude from input units to meters
        amplitude.data[:,i] = np.abs(hci.data)*kwargs['scale']
        amplitude.mask[:,i] = np.copy(hci.mask)
        # phase of the constituent in radians
        ph.data[:,i] = np.arctan2(-np.imag(hci.data),np.real(hci.data))
        ph.mask[:,i] = np.copy(hci.mask)
        # update mask to invalidate points outside model domain
        amplitude.mask[:,i] |= invalid
        ph.mask[:,i] |= invalid

    # convert phase to degrees
    phase = ph*180.0/np.pi
    phase.data[phase.data < 0] += 360.0
    # replace data for invalid mask values
    amplitude.data[amplitude.mask] = amplitude.fill_value
    phase.data[phase.mask] = phase.fill_value
    # return the interpolated values
    return (amplitude, phase)

# PURPOSE: read harmonic constants from tide models
def read_constants(model_files=None, **kwargs):
    """
    Reads files for a FES ascii or netCDF4 tidal model

    Parameters
    ----------
    model_files: list or NoneType, default None
        list of model files for each constituent
    type: str, default 'z'
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: horizontal transport velocities
            - ``'v'``: vertical transport velocities
    version: str or NoneType, default None
        Model version to read

            - ``'FES1999'``
            - ``'FES2004'``
            - ``'FES2012'``
            - ``'FES2014'``
            - ``'EOT20'``
    compressed: bool, default False
        Input files are gzip compressed

    Returns
    -------
    constituents: obj
        complex form of tide model constituents
    """
    # set default keyword arguments
    kwargs.setdefault('type', 'z')
    kwargs.setdefault('version', None)
    kwargs.setdefault('compressed', False)

    # raise warning if model files are entered as a string
    if isinstance(model_files,str):
        warnings.warn("Tide model is entered as a string")
        model_files = [model_files]

    # save output constituents
    constituents = pyTMD.io.constituents()
    # read each model constituent
    for i, fi in enumerate(model_files):
        # check that model file is accessible
        if not os.access(os.path.expanduser(fi), os.F_OK):
            raise FileNotFoundError(os.path.expanduser(fi))
        # read constituent from elevation file
        if kwargs['version'] in ('FES1999','FES2004'):
            # FES ascii constituent files
            hc,lon,lat = read_ascii_file(os.path.expanduser(fi), **kwargs)
        elif kwargs['version'] in ('FES2012','FES2014','EOT20'):
            # FES netCDF4 constituent files
            hc,lon,lat = read_netcdf_file(os.path.expanduser(fi), **kwargs)
        # grid step size of tide model
        dlon = lon[1] - lon[0]
        # replace original values with extend arrays/matrices
        if np.isclose(lon[-1] - lon[0], 360.0 - dlon):
            lon = extend_array(lon, dlon)
            hc = extend_matrix(hc)
        # append extended constituent
        constituents.append(str(i), hc)
        # set model coordinates
        setattr(constituents, 'longitude', lon)
        setattr(constituents, 'latitude', lat)

    # return the complex form of the model constituents
    return constituents

# PURPOSE: interpolate constants from tide models to input coordinates
def interpolate_constants(ilon, ilat, constituents, **kwargs):
    """
    Interpolate constants from FES tidal models to input coordinates

    Makes initial calculations to run the tide program

    Parameters
    ----------
    ilon: float
        longitude to interpolate
    ilat: float
        latitude to interpolate
    constituents: obj
        Tide model constituents (complex form)
    method: str, default 'spline'
        Interpolation method

            - ``'bilinear'``: quick bilinear interpolation
            - ``'spline'``: scipy bivariate spline interpolation
            - ``'linear'``, ``'nearest'``: scipy regular grid interpolations
    extrapolate: bool, default False
        Extrapolate model using nearest-neighbors
    cutoff: float, default 10.0
        Extrapolation cutoff in kilometers

        Set to ``np.inf`` to extrapolate for all points
    scale: float, default 1.0
        Scaling factor for converting to output units

    Returns
    -------
    amplitude: float
        amplitudes of tidal constituents
    phase: float
        phases of tidal constituents
    """
    # set default keyword arguments
    kwargs.setdefault('method', 'spline')
    kwargs.setdefault('extrapolate', False)
    kwargs.setdefault('cutoff', 10.0)
    kwargs.setdefault('scale', 1.0)
    # verify that constituents are valid class instance
    assert isinstance(constituents, pyTMD.io.constituents)
    # extract model coordinates
    lon = np.copy(constituents.longitude)
    lat = np.copy(constituents.latitude)

    # adjust dimensions of input coordinates to be iterable
    ilon = np.atleast_1d(np.copy(ilon))
    ilat = np.atleast_1d(np.copy(ilat))
    # adjust longitudinal convention of input latitude and longitude
    # to fit tide model convention
    if (np.min(ilon) < 0.0) & (np.max(lon) > 180.0):
        # input points convention (-180:180)
        # tide model convention (0:360)
        ilon[ilon<0.0] += 360.0
    elif (np.max(ilon) > 180.0) & (np.min(lon) < 0.0):
        # input points convention (0:360)
        # tide model convention (-180:180)
        ilon[ilon>180.0] -= 360.0
    # determine if any input points are outside of the model bounds
    invalid = (ilon < lon.min()) | (ilon > lon.max()) | \
              (ilat < lat.min()) | (ilat > lat.max())

    # number of points
    npts = len(ilon)
    # number of constituents
    nc = len(constituents)
    # amplitude and phase
    amplitude = np.ma.zeros((npts,nc))
    amplitude.mask = np.zeros((npts,nc), dtype=bool)
    ph = np.ma.zeros((npts,nc))
    ph.mask = np.zeros((npts,nc), dtype=bool)
    # default complex fill value
    fill_value = np.ma.default_fill_value(np.dtype(complex))
    # interpolate each constituent
    for i, c in enumerate(constituents.fields):
        # get model constituent
        hc = constituents.get(c)
        # interpolated complex form of constituent oscillation
        hci = np.ma.zeros((npts), dtype=hc.dtype, fill_value=hc.fill_value)
        hci.mask = np.zeros((npts),dtype=bool)
        # interpolate amplitude and phase of the constituent
        if (kwargs['method'] == 'bilinear'):
            # replace invalid values with nan
            hc.data[hc.mask] = np.nan
            # use quick bilinear to interpolate values
            hci.data[:] = bilinear_interp(lon, lat, hc, ilon, ilat,
                dtype=hc.dtype)
            # replace nan values with fill_value
            hci.mask[:] |= np.isnan(hci.data)
            hci.data[hci.mask] = hci.fill_value
        elif (kwargs['method'] == 'spline'):
            # replace invalid values with fill value
            hc.data[hc.mask] = fill_value
            # interpolate complex form of the constituent
            # use scipy splines to interpolate values
            f1=scipy.interpolate.RectBivariateSpline(lon, lat,
                hc.data.real.T, kx=1, ky=1)
            f2=scipy.interpolate.RectBivariateSpline(lon, lat,
                hc.data.imag.T, kx=1, ky=1)
            f3=scipy.interpolate.RectBivariateSpline(lon, lat,
                hc.mask.T, kx=1, ky=1)
            hci.data.real[:] = f1.ev(ilon,ilat)
            hci.data.imag[:] = f2.ev(ilon,ilat)
            hci.mask[:] = f3.ev(ilon,ilat).astype(bool)
            # replace invalid values with fill_value
            hci.data[hci.mask] = hci.fill_value
        else:
            # replace invalid values with fill value
            hc.data[hc.mask] = fill_value
            # interpolate complex form of the constituent
            # use scipy regular grid to interpolate values
            r1 = scipy.interpolate.RegularGridInterpolator((lon, lat),
                hc.data, method=kwargs['method'], bounds_error=False,
                fill_value=hci.fill_value)
            r2 = scipy.interpolate.RegularGridInterpolator((lon, lat),
                hc.mask, method=kwargs['method'], bounds_error=False,
                fill_value=1)
            hci.data[:] = r1.__call__(np.c_[ilat,ilon])
            hci.mask[:] = np.ceil(r2.__call__(np.c_[ilat,ilon])).astype(bool)
            # replace invalid values with fill_value
            hci.mask[:] |= (hci.data == hci.fill_value)
            hci.data[hci.mask] = hci.fill_value
        # extrapolate data using nearest-neighbors
        if kwargs['extrapolate'] and np.any(hci.mask):
            # find invalid data points
            inv, = np.nonzero(hci.mask)
            # replace invalid values with nan
            hc.data[hc.mask] = np.nan
            # extrapolate points within cutoff of valid model points
            hci[inv] = nearest_extrap(lon, lat, hc,
                ilon[inv], ilat[inv], dtype=hc.dtype,
                cutoff=kwargs['cutoff'])
        # convert amplitude from input units to meters
        amplitude.data[:,i] = np.abs(hci.data)*kwargs['scale']
        amplitude.mask[:,i] = np.copy(hci.mask)
        # phase of the constituent in radians
        ph.data[:,i] = np.arctan2(-np.imag(hci.data),np.real(hci.data))
        ph.mask[:,i] = np.copy(hci.mask)
        # update mask to invalidate points outside model domain
        amplitude.mask[:,i] |= invalid
        ph.mask[:,i] |= invalid

    # convert phase to degrees
    phase = ph*180.0/np.pi
    phase.data[phase.data < 0] += 360.0
    # replace data for invalid mask values
    amplitude.data[amplitude.mask] = amplitude.fill_value
    phase.data[phase.mask] = phase.fill_value
    # return the interpolated values
    return (amplitude, phase)

# PURPOSE: Extend a longitude array
def extend_array(input_array, step_size):
    """
    Extends a longitude array

    Parameters
    ----------
    input_array: float
        array to extend
    step_size: float
        step size between elements of array

    Returns
    -------
    temp: float
        extended array
    """
    n = len(input_array)
    temp = np.zeros((n+2), dtype=input_array.dtype)
    # extended array [x-1,x0,...,xN,xN+1]
    temp[0] = input_array[0] - step_size
    temp[1:-1] = input_array[:]
    temp[-1] = input_array[-1] + step_size
    return temp

# PURPOSE: Extend a global matrix
def extend_matrix(input_matrix):
    """
    Extends a global matrix

    Parameters
    ----------
    input_matrix: float
        matrix to extend

    Returns
    -------
    temp: float
        extended matrix
    """
    ny, nx = np.shape(input_matrix)
    temp = np.ma.zeros((ny,nx+2), dtype=input_matrix.dtype)
    temp[:,0] = input_matrix[:,-1]
    temp[:,1:-1] = input_matrix[:,:]
    temp[:,-1] = input_matrix[:,0]
    return temp

# PURPOSE: read FES ascii tide model grid files
def read_ascii_file(input_file, **kwargs):
    """
    Read FES (Finite Element Solution) tide model file

    Parameters
    ----------
    input_file: str
        model file
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    hc: complex form of tidal constituent oscillation
    lon: longitude of tidal model
    lat: latitude of tidal model
    """
    # set default keyword arguments
    kwargs.setdefault('compressed', False)
    # tilde-expand input file
    input_file = os.path.expanduser(input_file)
    # read input tide model file
    if kwargs['compressed']:
        # read gzipped ascii file
        with gzip.open(input_file, 'rb') as f:
            file_contents = f.read(input_file).splitlines()
    else:
        with open(input_file, mode="r", encoding='utf8') as f:
            file_contents = f.read().splitlines()
    # parse header text
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
    fill_value = np.float64(masked_values[0])
    # create output variables
    lat = np.linspace(latmin, latmax, nlat)
    lon = np.linspace(lonmin, lonmax, nlon)
    amp = np.ma.zeros((nlat,nlon), fill_value=fill_value, dtype=np.float32)
    ph = np.ma.zeros((nlat,nlon), fill_value=fill_value, dtype=np.float32)
    # create masks for output variables (0=valid)
    amp.mask = np.zeros((nlat,nlon),dtype=bool)
    ph.mask = np.zeros((nlat,nlon),dtype=bool)
    # starting line to fill amplitude and phase variables
    i1 = 5
    # for each latitude
    for i in range(nlat):
        for j in range(nlon//30):
            j1 = j*30
            amplitude_data = file_contents[i1].split()
            amp.data[i,j1:j1+30] = np.array(amplitude_data, dtype=np.float32)
            phase_data = file_contents[i1+1].split()
            ph.data[i,j1:j1+30] = np.array(phase_data, dtype=np.float32)
            i1 += 2
        # add last tidal variables
        j1 = (j+1)*30
        j2 = nlon % 30
        amplitude_data = file_contents[i1].split()
        amp.data[i,j1:j1+j2] = np.array(amplitude_data, dtype=np.float32)
        phase_data = file_contents[i1+1].split()
        ph.data[i,j1:j1+j2] = np.array(phase_data, dtype=np.float32)
        i1 += 2
    # calculate complex form of constituent oscillation
    mask = (amp.data == amp.fill_value) | (ph.data == ph.fill_value)
    hc = np.ma.array(amp*np.exp(-1j*ph*np.pi/180.0), mask=mask,
        fill_value=np.ma.default_fill_value(np.dtype(complex)))
    # return output variables
    return (hc, lon, lat)

# PURPOSE: read FES netCDF4 tide model files
def read_netcdf_file(input_file, **kwargs):
    """
    Read FES (Finite Element Solution) tide model netCDF4 file

    Parameters
    ----------
    input_file: str
        model file
    type: str or NoneType, default None
        Tidal variable to read

            - ``'z'``: heights
            - ``'u'``: horizontal transport velocities
            - ``'v'``: vertical transport velocities
    version: str or NoneType, default None
        FES model version
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    hc: complex
        complex form of tidal constituent oscillation
    lon: float
        longitude of tidal model
    lat: float
        latitude of tidal model
    """
    # set default keyword arguments
    kwargs.setdefault('type', None)
    kwargs.setdefault('version', None)
    kwargs.setdefault('compressed', False)
    # read the netcdf format tide elevation file
    if kwargs['compressed']:
        # read gzipped netCDF4 file
        f = gzip.open(os.path.expanduser(input_file),'rb')
        fileID = netCDF4.Dataset(uuid.uuid4().hex, 'r', memory=f.read())
    else:
        fileID = netCDF4.Dataset(os.path.expanduser(input_file), 'r')
    # variable dimensions for each model
    if kwargs['version'] in ('FES2012',):
        lon = fileID.variables['longitude'][:]
        lat = fileID.variables['latitude'][:]
    elif kwargs['version'] in ('FES2014','EOT20'):
        lon = fileID.variables['lon'][:]
        lat = fileID.variables['lat'][:]
    # amplitude and phase components for each type
    if (kwargs['type'] == 'z'):
        amp_key = 'amplitude'
        phase_key = 'phase'
    elif (kwargs['type'] == 'u'):
        amp_key = 'Ua'
        phase_key = 'Ug'
    elif (kwargs['type'] == 'v'):
        amp_key = 'Va'
        phase_key = 'Vg'
    # get amplitude and phase components
    amp = fileID.variables[amp_key][:]
    ph = fileID.variables[phase_key][:]
    # close the file
    fileID.close()
    f.close() if kwargs['compressed'] else None
    # calculate complex form of constituent oscillation
    mask = (amp.data == amp.fill_value) | \
        (ph.data == ph.fill_value) | \
        np.isnan(amp.data) | np.isnan(ph.data)
    hc = np.ma.array(amp*np.exp(-1j*ph*np.pi/180.0), mask=mask,
        fill_value=np.ma.default_fill_value(np.dtype(complex)))
    # return output variables
    return (hc, lon, lat)

# PURPOSE: output tidal constituent file in FES2014 format
def output_netcdf_file(FILE, hc, lon, lat, constituent, **kwargs):
    """
    Writes tidal constituent files in FES2014 netCDF format

    Parameters
    ----------
    FILE: str
        output FES model file name
    hc: complex
        Eulerian form of tidal constituent
    lon: float
        longitude coordinates
    lat: float
        latitude coordinates
    constituent: str
        tidal constituent ID
    type: str or NoneType, default None
        Tidal variable to output

            - ``'z'``: heights
            - ``'u'``: horizontal transport velocities
            - ``'v'``: vertical transport velocities
    """
    # set default keyword arguments
    kwargs.setdefault('type', None)
    # opening NetCDF file for writing
    fileID = netCDF4.Dataset(os.path.expanduser(FILE), 'w', format="NETCDF4")
    # define the NetCDF dimensions
    fileID.createDimension('lon', len(lon))
    fileID.createDimension('lat', len(lat))
    fileID.createDimension('nct', 4)
    # calculate amplitude and phase
    amp = np.abs(hc)
    ph = 180.0*np.arctan2(-np.imag(hc), np.real(hc))/np.pi
    # update masks and fill values
    amp.mask = np.copy(hc.mask)
    amp.data[amp.mask] = amp.fill_value
    ph.mask = np.copy(hc.mask)
    ph.data[ph.mask] = ph.fill_value
    # set variable names and units for type
    if (kwargs['type'] == 'z'):
        amp_key = 'amplitude'
        phase_key = 'phase'
        units = 'cm'
    elif (kwargs['type'] == 'u'):
        amp_key = 'Ua'
        phase_key = 'Ug'
        units = 'cm/s'
    elif (kwargs['type'] == 'v'):
        amp_key = 'Va'
        phase_key = 'Vg'
        units = 'cm/s'
    # defining the NetCDF variables
    nc = {}
    nc['lon'] = fileID.createVariable('lon', lon.dtype, ('lon',))
    nc['lat'] = fileID.createVariable('lat', lat.dtype, ('lat',))
    nc[amp_key] = fileID.createVariable(amp_key, amp.dtype,
        ('lat','lon',), fill_value=amp.fill_value, zlib=True)
    nc[phase_key] = fileID.createVariable(phase_key, ph.dtype,
        ('lat','lon',), fill_value=ph.fill_value, zlib=True)
    # filling the NetCDF variables
    nc['lon'][:] = lon[:]
    nc['lat'][:] = lat[:]
    nc['amplitude'][:] = amp[:]
    nc[phase_key][:] = ph[:]
    # set variable attributes for coordinates
    nc['lon'].setncattr('axis', 'X')
    nc['lon'].setncattr('units', 'degrees_east')
    nc['lon'].setncattr('long_name', 'longitude')
    nc['lat'].setncattr('axis', 'Y')
    nc['lat'].setncattr('units', 'degrees_north')
    nc['lat'].setncattr('long_name', 'latitude')
    # set variable attributes
    nc[amp_key].setncattr('units', units)
    long_name = f'Tide amplitude at {constituent} frequency'
    nc[amp_key].setncattr('long_name', long_name)
    nc[phase_key].setncattr('units', 'degrees')
    long_name = f'Tide phase at {constituent} frequency'
    nc[phase_key].setncattr('long_name', long_name)
    # define and fill constituent ID
    nc['con'] = fileID.createVariable('con', 'S1', ('nct',))
    con = [char.encode('utf8') for char in constituent.ljust(4)]
    nc['con'][:] = np.array(con, dtype='S1')
    nc['con'].setncattr('_Encoding', 'utf8')
    nc['con'].setncattr('long_name', "tidal constituent")
    # add global attributes
    fileID.title = "FES tide file"
    # add attribute for date created
    fileID.date_created = datetime.datetime.now().isoformat()
    # add attributes for software information
    fileID.software_reference = pyTMD.version.project_name
    fileID.software_version = pyTMD.version.full_version
    fileID.software_revision = get_git_revision_hash()
    # Output NetCDF structure information
    logging.info(FILE)
    logging.info(list(fileID.variables.keys()))
    # Closing the NetCDF file
    fileID.close()
