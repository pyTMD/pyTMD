#!/usr/bin/env python
u"""
dataset.py
Written by Tyler Sutterley (12/2025)
An xarray.Dataset extension for tidal model data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 12/2025: add coords functions to transform coordinates
    Updated 11/2025: get crs directly using pyproj.CRS.from_user_input
        set variable name to constituent for to_dataarray method
        added is_global property for models covering a global domain
        added pad function to pad global datasets along boundaries
        added inpaint function to fill missing data in datasets
    Updated 09/2025: added argument to limit the list of constituents
        when converting to an xarray DataArray
    Written 08/2025
"""
import warnings
import numpy as np
import xarray as xr
from pyTMD.utilities import import_dependency
# suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# attempt imports
pyproj = import_dependency('pyproj')
pint = import_dependency('pint')

__all__ = [
    'DataTree',
    'Dataset',
    'DataArray',
    '_transform',
    '_coords'
]

# pint unit registry
__ureg__ = pint.UnitRegistry()
# default units for pyTMD outputs
_default_units = {
    'elevation': 'm',
    'current': 'cm/s',
    'transport': 'm^2/s',
}

@xr.register_datatree_accessor('tmd')
class DataTree:
    """Accessor for extending an ``xarray.DataTree`` for tidal model data
    """
    def __init__(self, dtree):
        # initialize DataTree
        self._dtree = dtree

    def coords_as(self, 
            x: np.ndarray,
            y: np.ndarray,
            crs: str | int | dict = 4326,
            **kwargs
        ):
        """
        Transform coordinates into DataArrays in the DataTree
        coordinate reference system
        
        Parameters
        ----------
        x: np.ndarray
            Input x-coordinates
        y: np.ndarray
            Input y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of input coordinates

        Returns
        -------
        X: xarray.DataArray
            Transformed x-coordinates
        Y: xarray.DataArray
            Transformed y-coordinates
        """
        # convert coordinate reference system to that of the datatree
        # and format as xarray DataArray with appropriate dimensions
        X, Y = _coords(x, y,
            source_crs=crs,
            target_crs=self.crs,
            **kwargs
        )
        # return the transformed coordinates
        return X, Y

    def crop(self, *args, **kwargs):
        """
        Crop ``DataTree`` to input bounding box
        """
        # create copy of datatree
        dtree = self._dtree.copy()
        # crop each dataset in the datatree
        for key, ds in dtree.items():
            ds = ds.to_dataset()
            dtree[key] = ds.tmd.crop(*args, **kwargs)
        # return the datatree
        return dtree

    def inpaint(self, **kwargs):
        """
        Inpaint over missing data in ``DataTree``
        """
        # create copy of datatree
        dtree = self._dtree.copy()
        # inpaint each dataset in the datatree
        for key, ds in dtree.items():
            ds = ds.to_dataset()
            dtree[key] = ds.tmd.inpaint(**kwargs)
        # return the datatree
        return dtree

    def interp(self, x, y, **kwargs):
        """
        Interpolate ``DataTree`` to input coordinates
        
        Parameters
        ----------
        x: np.ndarray
            input x-coordinates
        y: np.ndarray
            input y-coordinates
        """
        # create copy of datatree
        dtree = self._dtree.copy()
        # interpolate each dataset in the datatree
        for key, ds in dtree.items():
            ds = ds.to_dataset()
            dtree[key] = ds.tmd.interp(x, y, **kwargs)
        # return the datatree
        return dtree

    def subset(self, c: str | list):
        """
        Reduce to a subset of constituents

        Parameters
        ----------
        c: str or list
            List of constituents names
        """
        # create copy of datatree
        dtree = self._dtree.copy()
        # subset each dataset in the datatree
        for key, ds in dtree.items():
            ds = ds.to_dataset()
            dtree[key] = ds.tmd.subset(c)
        # return the datatree
        return dtree

    def transform_as(self, x, y, crs=4326, **kwargs):
        """
        Transform coordinates to/from the datatree coordinate reference system

        Parameters
        ----------
        x: np.ndarray
            Input x-coordinates
        y: np.ndarray
            Input y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of input coordinates
        direction: str, default 'FORWARD'
            Direction of transformation

            - ``'FORWARD'``: from input crs to model crs
            - ``'INVERSE'``: from model crs to input crs
        
        Returns
        -------
        X: np.ndarray
            Transformed x-coordinates
        Y: np.ndarray
            Transformed y-coordinates
        """
        # convert coordinate reference system to that of the datatree
        X, Y = _transform(x, y,
            source_crs=crs,
            target_crs=self.crs,
            **kwargs
        )
        # return the transformed coordinates
        return (X, Y)
    
    def to_ellipse(self, **kwargs):
        """
        Expresses tidal currents in terms of four ellipse parameters

        - ``major``: amplitude of the semi-major axis
        - ``minor``: amplitude of the semi-minor axis
        - ``incl``: angle of inclination of the northern semi-major axis
        - ``phase``: phase lag of the current behind the tidal potential  
        """
        from pyTMD.ellipse import ellipse
        # get u and v components from datatree
        dsu = (self._dtree.get('u',None) or self._dtree.get('U',None)).to_dataset()
        dsv = (self._dtree.get('v',None) or self._dtree.get('V',None)).to_dataset()
        # calculate ellipse parameters for each constituent
        dmajor = xr.Dataset()
        dminor = xr.Dataset()
        dincl = xr.Dataset()
        dphase = xr.Dataset()
        # for each constituent in the u-component
        for c in dsu.tmd.constituents:
            # assert units between datasets are the same
            if (dsu[c].attrs.get('units','') != dsv[c].attrs.get('units','')):
                raise ValueError(f'Incompatible units for {c} in u and v datasets')
            # calculate ellipse parameters
            major, minor, incl, phase = ellipse(dsu[c].values, dsv[c].values)
            # create xarray DataArray for ellipse parameters
            dmajor[c] = xr.DataArray(major, dims=dsu[c].dims, coords=dsu.coords)
            dminor[c] = xr.DataArray(minor, dims=dsu[c].dims, coords=dsu.coords)
            dincl[c] = xr.DataArray(incl, dims=dsu[c].dims, coords=dsu.coords)
            dphase[c] = xr.DataArray(phase, dims=dsu[c].dims, coords=dsu.coords)
            # add attributes to each variable
            dmajor[c].attrs['units'] = dsu[c].attrs.get('units', '')
            dminor[c].attrs['units'] = dsu[c].attrs.get('units', '')
            dincl[c].attrs['units'] = 'degrees'
            dphase[c].attrs['units'] = 'degrees'
        # create output datatree
        dtree = xr.DataTree()
        # add datasets to output datatree
        dtree['major'] = dmajor
        dtree['minor'] = dminor
        dtree['incl'] = dincl
        dtree['phase'] = dphase
        # return datatree
        return dtree
    
    def from_ellipse(self, **kwargs):
        """
        Calculates tidal currents from the four ellipse parameters

        - ``major``: amplitude of the semi-major axis
        - ``minor``: amplitude of the semi-minor axis
        - ``incl``: angle of inclination of the northern semi-major axis
        - ``phase``: phase lag of the current behind the tidal potential  
        """
        from pyTMD.ellipse import inverse
        # get ellipse parameters from datatree
        dmajor = self._dtree['major'].to_dataset()
        dminor = self._dtree['minor'].to_dataset()
        dincl = self._dtree['incl'].to_dataset()
        dphase = self._dtree['phase'].to_dataset()
        # calculate currents for each constituent
        dsu = xr.Dataset()
        dsv = xr.Dataset()
        # for each constituent in the major parameter
        for c in dmajor.tmd.constituents:
            # calculate ellipse parameters
            u, v = inverse(dmajor[c].values, dminor[c].values,
                dincl[c].values, dphase[c].values)
            # create xarray DataArray for ellipse parameters
            dsu[c] = xr.DataArray(u, dims=dmajor[c].dims, coords=dmajor.coords)
            dsv[c] = xr.DataArray(v, dims=dmajor[c].dims, coords=dmajor.coords)
            # add attributes to each variable
            dsu[c].attrs['units'] = dmajor[c].attrs.get('units', '')
            dsv[c].attrs['units'] = dmajor[c].attrs.get('units', '')
            if (dmajor[c].tmd.group == 'current'):
                ukey, vkey = 'u', 'v'
            elif (dmajor[c].tmd.group == 'transport'):
                ukey, vkey = 'U', 'V'
        # create output datatree
        dtree = xr.DataTree()
        # add datasets to output datatree
        dtree[ukey] = dsu
        dtree[vkey] = dsv
        # return the datatree
        return dtree

    @property
    def crs(self):
        """Coordinate reference system of the ``DataTree``
        """
        # inherit CRS from one of the datasets
        for key, ds in self._dtree.items():
            ds = ds.to_dataset()
            return ds.tmd.crs

@xr.register_dataset_accessor('tmd')
class Dataset:
    """Accessor for extending an ``xarray.Dataset`` for tidal model data
    """
    def __init__(self, ds):
        # initialize Dataset
        self._ds = ds

    def to_dataarray(self, **kwargs):
        """
        Converts ``Dataset`` to a ``DataArray`` with constituents as a dimension
        """
        kwargs.setdefault('constituents', self.constituents)
        # reduce dataset to constituents and convert to dataarray
        da = self._ds[kwargs['constituents']].to_dataarray(dim='constituent')
        # stack constituents as the last dimension
        da = da.transpose(*da.dims[1:], da.dims[0])
        da = da.assign_coords(constituent=kwargs['constituents'])
        return da
    
    def coords_as(self, 
            x: np.ndarray,
            y: np.ndarray,
            crs: str | int | dict = 4326,
            **kwargs
        ):
        """
        Transform coordinates into DataArrays in the Dataset
        coordinate reference system
        
        Parameters
        ----------
        x: np.ndarray
            Input x-coordinates
        y: np.ndarray
            Input y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of input coordinates

        Returns
        -------
        X: xarray.DataArray
            Transformed x-coordinates
        Y: xarray.DataArray
            Transformed y-coordinates
        """
        # convert coordinate reference system to that of the dataset
        # and format as xarray DataArray with appropriate dimensions
        X, Y = _coords(x, y,
            source_crs=crs,
            target_crs=self.crs,
            **kwargs
        )
        # return the transformed coordinates
        return X, Y

    def crop(self,
            bounds: list | tuple,
            buffer: int | float = 0
        ):
        """
        Crop ``Dataset`` to input bounding box

        Parameters
        ----------
        bounds: list, tuple
            bounding box [min_x, max_x, min_y, max_y]
        buffer: int or float, default 0
            buffer to add to bounds for cropping
        """
        # number of points to pad for global grids
        n = int(180//(self._x[1] - self._x[0]))
        # pad global grids along x-dimension (if necessary)
        lon_wrap = self.crs.to_dict().get('lon_wrap', 0)
        if self.is_global and (lon_wrap == 180) and (np.min(bounds[:2]) < 0):
            ds = self.pad(n=(n,0))
        elif self.is_global and (lon_wrap == 0) and (np.max(bounds[:2]) > 180):
            ds = self.pad(n=(0,n))
        else:
            ds = self._ds.copy()
        # unpack bounds and buffer
        xmin = bounds[0] - buffer
        xmax = bounds[1] + buffer
        ymin = bounds[2] - buffer
        ymax = bounds[3] + buffer
        # crop dataset to bounding box
        ds = ds.where((ds.x >= xmin) & (ds.x <= xmax) &
            (ds.y >= ymin) & (ds.y <= ymax), drop=True)
        # return the cropped dataset
        return ds

    def infer(self, t: float | np.ndarray, **kwargs):
        """
        Infer minor tides from ``Dataset`` at times

        Parameters
        ----------
        t: float or np.ndarray
            days relative to 1992-01-01T00:00:00 UTC
        kwargs: keyword arguments
            additional keyword arguments

        Returns
        -------
        darr: xarray.DataArray
            predicted tides
        """
        from pyTMD.predict import infer_minor
        # infer minor tides at times
        darr = infer_minor(t, self._ds, **kwargs)
        # return the inferred tides
        return darr

    def inpaint(self, **kwargs):
        """
        Inpaint over missing data in ``Dataset``

        Parameters
        ----------
        kwargs: keyword arguments
            keyword arguments for ``pyTMD.interpolate.inpaint``

        Returns
        -------
        ds: xarray.Dataset
            interpolated xarray Dataset
        """
        # import inpaint function
        from pyTMD.interpolate import inpaint
        # create copy of dataset
        ds = self._ds.copy()
        # inpaint each variable in the dataset
        for v in ds.data_vars.keys():
            ds[v].values = inpaint(self._x, self._y,
                self._ds[v].values, **kwargs
            )
        # return the dataset
        return ds

    def interp(self, 
            x: np.ndarray,
            y: np.ndarray, 
            method='linear',
            **kwargs
        ):
        """
        Interpolate ``Dataset`` to input coordinates
        
        Parameters
        ----------
        x: np.ndarray
            input x-coordinates
        y: np.ndarray
            input y-coordinates
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
        # import extrapolate function
        from pyTMD.interpolate import extrapolate
        # set default keyword arguments
        kwargs.setdefault('extrapolate', False)
        kwargs.setdefault('cutoff', np.inf)
        # pad global grids along x-dimension (if necessary)
        if self.is_global:
            self._ds = self.pad(n=1)
        # verify longitudinal convention for geographic models
        if self.crs.is_geographic:
            # grid spacing in x-direction
            dx = self._x[1] - self._x[0]
            # adjust input longitudes to be consistent with model
            if (np.min(x) < 0.0) & (self._x.max() > (180.0 + dx)):
                # input points convention (-180:180)
                # tide model convention (0:360)
                x = xr.where(x < 0, x + 360, x)
            elif (np.max(x) > 180.0) & (self._x.min() < (0.0 - dx)):
                # input points convention (0:360)
                # tide model convention (-180:180)
                x = xr.where(x > 180, x - 360, x)
        # interpolate dataset
        ds = self._ds.interp(x=x, y=y, method=method)
        # extrapolate missing values using nearest-neighbors
        if kwargs['extrapolate']:
            for v in ds.data_vars.keys():
                # check for missing values
                invalid = ds[v].isnull()
                if not invalid.any():
                    # no missing values
                    continue
                elif (ds[v].ndim == 0):
                    # single point extrapolation
                    ds[v].values, = extrapolate(
                        self._x, self._y, self._ds[v].values, x, y,
                        is_geographic=self.crs.is_geographic,
                        **kwargs
                    )
                else:
                    # only extrapolate invalid points
                    ds[v].values[invalid] = extrapolate(
                        self._x, self._y, self._ds[v].values,
                        x.values[invalid], y.values[invalid],
                        is_geographic=self.crs.is_geographic,
                        **kwargs
                    )
        # return xarray dataset
        return ds
    
    def node_equilibrium(self):
        """
        Compute the equilibrium amplitude and phase of the 18.6 year
        node tide :cite:p:`Cartwright:1971iz,Cartwright:1973em`
        """
        # copy dataset
        ds = self._ds.copy()
        # Cartwright and Edden potential amplitude
        amajor = 0.027929 # node
        # Love numbers for long-period tides (Wahr, 1981)
        k2 = 0.299
        h2 = 0.606
        # tilt factor: response with respect to the solid earth
        gamma_2 = (1.0 + k2 - h2)
        # check dimensions
        if (ds.x.ndim == 1) and (ds.y.ndim == 1):
            # 2D grid of coordinates
            x, y = np.meshgrid(self._x, self._y)
        else:
            x, y = ds.x.values, ds.y.values
        # transform model coordinates to lat/lon coordinates
        lon, lat = _transform(x, y,
            source_crs=self.crs,
            target_crs=4326,
            direction='FORWARD'
        )
        # colatitude in radians
        th = np.radians(90.0 - lat)
        # 2nd degree Legendre polynomials
        P20 = 0.5*(3.0*np.cos(th)**2 - 1.0)
        # normalization for spherical harmonics
        dfactor = np.sqrt((4.0 + 1.0)/(4.0*np.pi))
        # calculate equilibrium node constants
        hc = dfactor*P20*gamma_2*amajor*np.exp(-1j*np.pi)
        ds['node'] = xr.DataArray(hc, dims=ds.dims, coords=ds.coords)
        ds['node'].attrs['units'] = 'm'
        # return xarray dataset
        return ds

    def pad(self, n: int = 1, chunks=None):
        """
        Pad ``Dataset`` by repeating edge values in the x-direction

        Parameters
        ----------
        n: int, default 1
            number of padding values to add on each side

        Returns
        -------
        ds: xarray.Dataset
            padded xarray Dataset
        """
        # (possibly) unchunk x-coordinates and pad to wrap at meridian
        x = xr.DataArray(self._x, dims='x').pad(
            x=n, mode="reflect", reflect_type="odd"
        )
        # pad dataset and re-assign x-coordinates
        ds = self._ds.copy()
        ds = ds.pad(x=n, mode="wrap").assign_coords(x=x)
        # rechunk dataset (if specified)
        if chunks is not None:
            ds = ds.chunk(chunks)
        # return the dataset
        return ds
    
    def predict(self, t: float | np.ndarray, **kwargs):
        """
        Predict tides from ``Dataset`` at times

        Parameters
        ----------
        t: float or np.ndarray
            days relative to 1992-01-01T00:00:00 UTC
        kwargs: keyword arguments
            additional keyword arguments

        Returns
        -------
        darr: xarray.DataArray
            predicted tides
        """
        from pyTMD.predict import time_series
        # predict tides at times
        darr = time_series(t, self._ds, **kwargs)
        # return the predicted tides
        return darr

    def subset(self, c: str | list):
        """
        Reduce to a subset of constituents

        Parameters
        ----------
        c: str or list
            List of constituents names
        """
        # create copy of dataset
        ds = self._ds.copy()
        # if no constituents are specified, return self
        # else return reduced dataset
        if c is None:
            return ds
        elif isinstance(c, str):
            return ds[[c]]
        else:
            return ds[c]

    def transform_as(self,
            x: np.ndarray,
            y: np.ndarray,
            crs: str | int | dict = 4326,
            **kwargs
        ):
        """
        Transform coordinates to/from the dataset coordinate reference system

        Parameters
        ----------
        x: np.ndarray
            Input x-coordinates
        y: np.ndarray
            Input y-coordinates
        crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
            Coordinate reference system of input coordinates
        direction: str, default 'FORWARD'
            Direction of transformation

            - ``'FORWARD'``: from input crs to model crs
            - ``'INVERSE'``: from model crs to input crs

        Returns
        -------
        X: np.ndarray
            Transformed x-coordinates
        Y: np.ndarray
            Transformed y-coordinates
        """
        # convert coordinate reference system to that of the dataset
        X, Y = _transform(x, y,
            source_crs=crs,
            target_crs=self.crs,
            **kwargs
        )
        # return the transformed coordinates
        return (X, Y)

    def to_units(self, units: str, value: float = 1.0):
        """Convert ``Dataset`` to specified tide units

        Parameters
        ----------
        units: str
            output units
        value: float, default 1.0
            scaling factor to apply
        """
        # create copy of dataset
        ds = self._ds.copy()
        # convert each constituent in the dataset
        for c in self.constituents:
            ds[c] = ds[c].tmd.to_units(units, value=value)
        # return the dataset
        return ds

    def to_base_units(self):
        """Convert ``Dataset`` to base units
        """
        # create copy of dataset
        ds = self._ds.copy()
        # convert each constituent in the dataset
        for c in self.constituents:
            ds[c] = ds[c].tmd.to_base_units()
        # return the dataset
        return ds

    def to_default_units(self):
        """Convert ``Dataset`` to default tide units
        """
        # create copy of dataset
        ds = self._ds.copy()
        # convert each constituent in the dataset
        for c in self.constituents:
            ds[c] = ds[c].tmd.to_default_units()
        # return the dataset
        return ds

    @property
    def constituents(self):
        """List of tidal constituent names in the ``Dataset``
        """
        # import constituents parser
        from pyTMD.constituents import _parse_name
        # output list of tidal constituents
        cons = []
        # parse list of model constituents
        for i,c in enumerate(self._ds.data_vars.keys()):
            try:
                cons.append(_parse_name(c))
            except ValueError:
                pass
        # return list of constituents
        return cons

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
        dx = self._x[1] - self._x[0]
        # check if global grid
        cyclic = np.isclose(self._x[-1] - self._x[0], 360.0 - dx)
        return self.crs.is_geographic and cyclic

    @property
    def area_of_use(self) -> str | None:
        """Area of use from the dataset CRS
        """
        if self.crs.area_of_use is not None:
            return self.crs.area_of_use.name.replace('.','').lower()
        
    @property
    def _x(self):
        """x-coordinates of the ``Dataset``
        """
        return self._ds.x.values
    
    @property
    def _y(self):
        """y-coordinates of the ``Dataset``
        """
        return self._ds.y.values

@xr.register_dataarray_accessor('tmd')
class DataArray:
    """Accessor for extending an ``xarray.DataArray`` for tidal model data
    """
    def __init__(self, da):
        # initialize DataArray
        self._da = da

    @property
    def amplitude(self):
        """
        Calculate the amplitude of a tide model constituent

        Returns
        -------
        amp: xarray.DataArray
            Tide model constituent amplitude
        """
        # calculate constituent amplitude
        amp = np.sqrt(self._da.real**2 + self._da.imag**2)
        return amp

    @property
    def phase(self):
        """
        Calculate the phase of a tide model constituent

        Returns
        -------
        ph: xarray.DataArray
            Tide model constituent phase (degrees)
        """
        # calculate constituent phase and convert to degrees
        ph = np.degrees(np.arctan2(-self._da.imag, self._da.real))
        ph = ph.where(ph >= 0, ph + 360.0, drop=False)
        return ph

    def to_units(self, units: str, value: float = 1.0):
        """Convert ``DataArray`` to specified tide units

        Parameters
        ----------
        units: str
            output units
        value: float, default 1.0
            scaling factor to apply
        """
        # convert to specified units
        conversion = value*self.quantity.to(units)
        da = self._da*conversion.magnitude
        da.attrs['units'] = str(conversion.units)
        return da

    def to_base_units(self, value=1.0):
        """Convert ``DataArray`` to base units

        Parameters
        ----------
        value: float, default 1.0
            scaling factor to apply
        """
        # convert to base units
        conversion = value*self.quantity.to_base_units()
        da = self._da*conversion.magnitude
        da.attrs['units'] = str(conversion.units)
        return da

    def to_default_units(self, value=1.0):
        """Convert ``DataArray`` to default tide units

        Parameters
        ----------
        value: float, default 1.0
            scaling factor to apply
        """
        # convert to default units
        default_units = _default_units.get(self.group, self.units)
        da = self.to_units(default_units, value=value)
        return da

    @property
    def units(self):
        """Units of the ``DataArray``
        """
        return __ureg__.parse_units(self._da.attrs.get('units', ''))
    
    @property
    def quantity(self):
        """``Pint`` Quantity of the ``DataArray``
        """
        return 1.0*self.units

    @property
    def group(self):
        """Variable group of the ``DataArray``
        """
        if self.units.is_compatible_with('m'):
            return 'elevation'
        elif self.units.is_compatible_with('m/s'):
            return 'current'
        elif self.units.is_compatible_with('m^2/s'):
            return 'transport'
        else:
            raise ValueError(f'Unknown unit group: {self.units}')

def _transform(
        i1: np.ndarray,
        i2: np.ndarray,
        source_crs: str | int | dict = 4326,
        target_crs: str | int | dict = None,
        **kwargs
    ):
    """
    Transform coordinates to/from the dataset coordinate reference system

    Parameters
    ----------
    i1: np.ndarray
        Input x-coordinates
    i2: np.ndarray
        Input y-coordinates
    source_crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
        Coordinate reference system of input coordinates
    target_crs: str, int, or dict, default None
        Coordinate reference system of output coordinates
    direction: str, default 'FORWARD'
        Direction of transformation

        - ``'FORWARD'``: from input crs to model crs
        - ``'INVERSE'``: from model crs to input crs

    Returns
    -------
    o1: np.ndarray
        Transformed x-coordinates
    o2: np.ndarray
        Transformed y-coordinates
    """
    # set the direction of the transformation
    kwargs.setdefault('direction', 'FORWARD')
    assert kwargs['direction'] in ('FORWARD','INVERSE','IDENT')
    # get the coordinate reference system and transform
    source_crs = pyproj.CRS.from_user_input(source_crs)
    transformer = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True)
    # convert coordinate reference system
    o1, o2 = transformer.transform(i1, i2, **kwargs)
    # return the transformed coordinates
    return (o1, o2)

def _coords(
        x: np.ndarray,
        y: np.ndarray,
        source_crs: str | int | dict = 4326,
        target_crs: str | int | dict = None,
        **kwargs
    ):
    """
    Transform coordinates into DataArrays in a new
    coordinate reference system
    
    Parameters
    ----------
    x: np.ndarray
        Input x-coordinates
    y: np.ndarray
        Input y-coordinates
    source_crs: str, int, or dict, default 4326 (WGS84 Latitude/Longitude)
        Coordinate reference system of input coordinates
    target_crs: str, int, or dict, default None
        Coordinate reference system of output coordinates
    type: str or None, default None
        Coordinate data type

        If not provided: must specify ``time`` parameter to auto-detect

        - ``None``: determined from input variable dimensions
        - ``'drift'``: drift buoys or satellite/airborne altimetry
        - ``'grid'``: spatial grids or images
        - ``'time series'``: time series at a single point
    time: np.ndarray or None, default None
        Time variable for determining coordinate data type

    Returns
    -------
    X: xarray.DataArray
        Transformed x-coordinates
    Y: xarray.DataArray
        Transformed y-coordinates
    """
    from pyTMD.spatial import data_type
    # set default keyword arguments
    kwargs.setdefault('type', None)
    kwargs.setdefault('time', None)
    # determine coordinate data type if possible
    if kwargs['type'] is None:
        # must provide time variable to determine data type
        assert kwargs['time'] is not None, \
            'Must provide time parameter when type is not specified'
        coord_type = data_type(x, y, np.ravel(kwargs['time']))
    else:
        # use provided coordinate data type
        # and verify that it is lowercase
        coord_type = kwargs.get('type').lower()
    # convert coordinates to a new coordinate reference system
    if (coord_type == 'grid') and (np.size(x) != np.size(y)):
        gridx, gridy = np.meshgrid(x, y)
        mx, my = _transform(gridx, gridy,
            source_crs=source_crs,
            target_crs=target_crs,
            direction='FORWARD')
    else:
        mx, my = _transform(x, y,
            source_crs=source_crs,
            target_crs=target_crs,
            direction='FORWARD')
    # convert to xarray DataArray with appropriate dimensions
    if (np.ndim(x) == 0) and (np.ndim(y) == 0):
        X = xr.DataArray(mx)
        Y = xr.DataArray(my)
    elif (coord_type == 'grid'):
        X = xr.DataArray(mx, dims=('y','x'))
        Y = xr.DataArray(my, dims=('y','x'))
    elif (coord_type == 'drift'):
        X = xr.DataArray(mx, dims=('time'))
        Y = xr.DataArray(my, dims=('time'))
    elif (coord_type == 'time series'):
        X = xr.DataArray(mx, dims=('station'))
        Y = xr.DataArray(my, dims=('station'))
    else:
        raise ValueError(f'Unknown coordinate data type: {coord_type}')
    # return the transformed coordinates
    return (X, Y)
