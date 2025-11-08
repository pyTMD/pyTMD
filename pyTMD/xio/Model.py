#!/usr/bin/env python
u"""
Model.py
Written by Tyler Sutterley (11/2025)
"""
import pint
import pyproj
import pyTMD.io
import numpy as np
import xarray as xr

# unit registry for converting to base units
__ureg__ = pint.UnitRegistry()

# PURPOSE: experimental extension of pyTMD.io.model for xarray I/O
class Model(pyTMD.io.model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def open_dataset(self, **kwargs):
        # import tide model functions
        # set default keyword arguments
        kwargs.setdefault('type', 'z')
        kwargs.setdefault('to_base_units', True)
        kwargs.setdefault('append_node', False)
        kwargs.setdefault('compressed', self.compressed)
        kwargs.setdefault('constituents', None)
        # model type
        mtype = kwargs['type'].lower()
        # extract model file
        model_file = self[mtype].get('model_file')
        # reduce constituents if specified
        self.reduce_constituents(kwargs['constituents'])
        if self.format in ('OTIS', 'ATLAS-compact', 'TMD3'):
            # open OTIS/TMD3/ATLAS-compact files as xarray Dataset
            ds = pyTMD.xio.OTIS.open_dataset(model_file, 
                grid_file=self[mtype].get('grid_file'),
                format=self.file_format,
                crs=self.crs, **kwargs)
        elif self.format in ('ATLAS-netcdf',):
            # open ATLAS netCDF4 files as xarray Dataset
            ds = pyTMD.xio.ATLAS.open_dataset(model_file, 
                grid_file=self[mtype].get('grid_file'),
                format=self.file_format, **kwargs)
        elif self.format in ('GOT-ascii', 'GOT-netcdf'):
            # open GOT ASCII/netCDF4 files as xarray Dataset
            ds = pyTMD.xio.GOT.open_mfdataset(model_file,
                format=self.file_format, **kwargs)
        elif self.format in ('FES-ascii', 'FES-netcdf'):
            # open FES ASCII/netCDF4 files as xarray Dataset
            ds = pyTMD.xio.FES.open_mfdataset(model_file,
                version=self.version, **kwargs)
        # add coordinate reference system to Dataset
        ds.attrs['crs'] = self.crs.to_dict()
        # set units attribute if not already set
        ds.attrs['units'] = ds.attrs.get('units', self[mtype].units)
        # get units from attributes
        quantity = 1.0*__ureg__.parse_units(ds.attrs['units'])
        # conversion for base units
        base_units = quantity.to_base_units()
        scale_factor = base_units.magnitude
        # convert to output units
        if kwargs['to_base_units'] and (scale_factor != 1.0):
            # scale constituents to base units
            ds[ds.tmd.constituents] = ds[ds.tmd.constituents]*scale_factor
            # update units attribute
            ds.attrs['units'] = str(base_units.units)
        # return xarray dataset
        return ds
    
    def interp(self, 
            x: np.ndarray,
            y: np.ndarray, 
            crs: str | int | dict = 4326, 
            method='linear',           
            extrapolate: bool = False,
            cutoff: int | float = np.inf,   
            **kwargs
        ):
        """
        Interpolate tidal constants to input coordinates
        
        Parameters
        ----------
        x: np.ndarray
            input x-coordinates
        y: np.ndarray
            input y-coordinates
        crs: str, int, or dict, default 4326
            Coordinate reference system of input coordinates
        method: str, default 'linear'
            Interpolation method
        extrapolate: bool, default False
            Flag to extrapolate values using nearest-neighbors
        cutoff: int or float, default np.inf
            Maximum distance for extrapolation
        **kwargs: dict
            Additional keyword arguments for reading the dataset
            
        Returns
        -------
        ds: xarray.Dataset
            interpolated tidal constants
        """
        # open dataset (if not already cached)
        if not hasattr(self, '_ds'):
            self._ds = self.open_dataset(**kwargs)
        # pad global grids along x-dimension (if necessary)
        if self._ds.tmd.is_global:
            self._ds = self._ds.tmd.pad(n=1)
        # transform input coordinates to model crs
        mx, my = self.transform(x, y, crs=crs)
        # convert to xarray DataArrays and ensure float64 precision
        if isinstance(x, xr.DataArray) and isinstance(y, xr.DataArray):
            # xarray DataArrays
            mx = xr.DataArray(mx.astype('f8'), dims=x.dims)
            my = xr.DataArray(my.astype('f8'), dims=y.dims)
        elif np.ndim(x) < 2:
            # 1D arrays or scalars
            mx = xr.DataArray(np.atleast_1d(mx).astype('f8'), dims='i')
            my = xr.DataArray(np.atleast_1d(my).astype('f8'), dims='i')
        elif np.ndim(x) == 2:
            # 2D arrays
            mx = xr.DataArray(mx.astype('f8'), dims=('y', 'x'))
            my = xr.DataArray(my.astype('f8'), dims=('y', 'x'))
        # interpolate tidal constants
        ds = self._ds.tmd.interp(x=mx, y=my, method=method,
            extrapolate=extrapolate, cutoff=cutoff)
        # return xarray dataset
        return ds

    def transform(self,
            i1: np.ndarray,
            i2: np.ndarray,
            crs: str | int | dict = 4326,
            **kwargs
        ):
        """
        Transform coordinates to/from the model coordinate reference system

        Parameters
        ----------
        i1: np.ndarray
            Input x-coordinates
        i2: np.ndarray
            Input y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of input coordinates
        direction: str, default 'FORWARD'
            Direction of transformation

            - ``'FORWARD'``: from input crs to model crs
            - ``'BACKWARD'``: from model crs to input crs

        Returns
        -------
        o1: np.ndarray
            Transformed x-coordinates
        o2: np.ndarray
            Transformed y-coordinates
        """
        # set the direction of the transformation
        kwargs.setdefault('direction', 'FORWARD')
        # get the coordinate reference system and transform
        source_crs = pyproj.CRS.from_user_input(crs)
        transformer = pyproj.Transformer.from_crs(
            source_crs, self.crs, always_xy=True)
        # convert coordinate reference system
        o1, o2 = transformer.transform(i1, i2, **kwargs)
        # return the transformed coordinates
        return (o1, o2)
