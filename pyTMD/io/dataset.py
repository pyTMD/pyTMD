#!/usr/bin/env python
u"""
dataset.py
Written by Tyler Sutterley (11/2025)
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
    Updated 11/2025: get crs directly using pyTMD.CRS.from_user_input
        set variable name to constituent for to_dataarray method
        added is_global property for models covering a global domain
        added pad function to pad global datasets along boundaries
        added inpaint function to fill missing data in datasets
    Updated 09/2025: added argument to limit the list of constituents
        when converting to an xarray DataArray
    Written 08/2025
"""
import numpy as np
from pyTMD.utilities import import_dependency
# attempt imports
xr = import_dependency('xarray')
pyproj = import_dependency('pyproj')
pint = import_dependency('pint')

__all__ = [
    'Dataset',
    'DataArray'
]

# pint unit registry
__ureg__ = pint.UnitRegistry()
# default units for pyTMD outputs
_default_units = {
    'elevation': 'm',
    'current': 'cm/s',
    'transport': 'm^2/s',
}

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
        da = self._ds[kwargs['constituents']].to_dataarray(dim='constituent').T
        da = da.assign_coords(constituent=kwargs['constituents'])
        return da

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
            ds[v].values = inpaint(
                self._ds.x.values, self._ds.y.values,
                self._ds[v].values,
                **kwargs
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
            dx = self._ds.x[1] - self._ds.x[0]
            # adjust input longitudes to be consistent with model
            if (np.min(x) < 0.0) & (self._ds.x.max() > (180.0 + dx)):
                # input points convention (-180:180)
                # tide model convention (0:360)
                x = xr.where(x < 0, x + 360, x)
            elif (np.max(x) > 180.0) & (self._ds.x.min() < (0.0 - dx)):
                # input points convention (0:360)
                # tide model convention (-180:180)
                x = xr.where(x > 180, x - 360, x)
        # interpolate dataset
        ds = self._ds.interp(x=x, y=y, method=method,
            kwargs={"fill_value": None})
        # extrapolate missing values using nearest-neighbors
        if kwargs['extrapolate']:
            for v in ds.data_vars.keys():
                # check for missing values
                invalid = ds[v].isnull()
                if not invalid.any():
                    continue
                # only extrapolate invalid points
                ds[v].values[invalid] = extrapolate(
                    self._ds.x.values, self._ds.y.values,
                    self._ds[v].values, x[invalid], y[invalid],
                    is_geographic=self.crs.is_geographic,
                    **kwargs
                )
        # return xarray dataset
        return ds

    def pad(self, n: int = 1):
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
        x = self._ds.x.pad(x=n, mode="reflect", reflect_type="odd")
        ds = self._ds.pad(x=n, mode="wrap").assign_coords(x=x)
        # return the dataset
        return ds

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

    def transform(self,
            i1: np.ndarray,
            i2: np.ndarray,
            crs: str | int | dict = 4326,
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
        # import constituents class
        from pyTMD.io import constituents
        # output list of tidal constituents
        cons = []
        # parse list of model constituents
        for i,c in enumerate(self._ds.data_vars.keys()):
            try:
                cons.append(constituents.parse(c))
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
        dx = self._ds.x[1] - self._ds.x[0]
        # check if global grid
        cyclic = np.isclose(self._ds.x[-1] - self._ds.x[0], 360.0 - dx)
        return self.crs.is_geographic and cyclic

    @property
    def area_of_use(self) -> str | None:
        """Area of use from the dataset CRS
        """
        if self.crs.area_of_use is not None:
            return self.crs.area_of_use.name.replace('.','').lower()

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
        default_units = _default_units.get(self.type, self.units)
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
    def type(self):
        """Variable type of the ``DataArray``
        """
        if self.units.is_compatible_with('m'):
            return 'elevation'
        elif self.units.is_compatible_with('m/s'):
            return 'current'
        elif self.units.is_compatible_with('m^2/s'):
            return 'transport'
        else:
            raise ValueError(f'Unknown unit type: {self.units}')
