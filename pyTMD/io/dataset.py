#!/usr/bin/env python
u"""
dataset.py
Written by Tyler Sutterley (11/2025)
An xarray.Dataset extension for tidal model data

PYTHON DEPENDENCIES:
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 11/2025: get crs directly using pyTMD.CRS.from_user_input
    Updated 09/2025: added argument to limit the list of constituents
        when converting to an xarray DataArray
    Written 08/2025
"""
from pyTMD.utilities import import_dependency
# attempt imports
xr = import_dependency('xarray')
pyproj = import_dependency('pyproj')

__all__ = [
    'dataset',
]

@xr.register_dataset_accessor('tmd')
class dataset:
    """Accessor for extending an ``xarray.Dataset`` for tidal model data
    """
    def __init__(self, ds):
        # initialize dataset
        self._ds = ds

    def to_dataarray(self, **kwargs):
        """
        Converts ``Dataset`` to a ``DataArray`` with constituents as a dimension
        """
        kwargs.setdefault('constituents', self.constituents)
        # reduce dataset to constituents and convert to dataarray
        da = self._ds[kwargs['constituents']].to_dataarray().assign_coords(
            variable=kwargs['constituents']).T
        return da

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
