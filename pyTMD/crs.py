#!/usr/bin/env python
u"""
crs.py
Written by Tyler Sutterley (09/2024)
Coordinates Reference System (CRS) class

CALLING SEQUENCE:
    x, y = pyTMD.crs().convert(lon, lat, PROJ, 'F')
    lon, lat = pyTMD.crs().convert(x, y, PROJ, 'B')

INPUTS:
    i1: longitude ('F') or projection easting x ('B')
    i2: latitude ('F') or projection northing y ('B')
    PROJ: spatial reference system for coordinate transformations
    BF: backwards ('B') or forward ('F') translations

OPTIONS:
    EPSG: spatial reference system code for input (F) and output (B) coordinates

OUTPUTS:
    o1: projection easting x ('F') or longitude ('B')
    o2: projection northing y ('F') or latitude ('B')

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/

UPDATE HISTORY:
    Updated 09/2024: added function for idealized Arctic Azimuthal projection
        complete refactor to use JSON dictionary format for model projections
    Updated 07/2024: added function to get the CRS transform
    Updated 05/2024: make subscriptable and allow item assignment
    Updated 04/2024: use wrapper to importlib for optional dependencies
    Updated 02/2024: changed class name for ellipsoid parameters to datum
    Updated 12/2023: converted conversion functions to class
    Updated 03/2023: add basic variable typing to function inputs
        renamed coordinate reference system conversion functions
    Updated 02/2023: use named exception before passing to custom
    Updated 11/2022: place some imports within try/except statements
        use f-strings for formatting verbose or ascii output
    Updated 04/2022: updated docstrings to numpy documentation format
    Updated 09/2021: added function for using custom projections
    Updated 06/2021: added 3413 for new 1km Greenland model from ESR
    Updated 08/2020: using conversion protocols following pyproj-2 updates
        https://pyproj4.github.io/pyproj/stable/gotchas.html
    Updated 07/2020: added function docstrings. changed function name
    Updated 03/2020: remove commented coordinate conversion functions
    Updated 11/2019: using pyproj for coordinate conversions
    Written 09/2017
"""

from __future__ import annotations

import logging
import numpy as np
from pyTMD.utilities import import_dependency
# attempt imports
pyproj = import_dependency('pyproj')

__all__ = [
    'crs',
]

class crs:
    """Coordinate Reference System transformations for tide models

    Attributes
    ----------
    name: str
        Projection name
    transformer: obj
        ``pyproj`` transformer for changing coordinate reference system
    """
    def __init__(self):
        self.name = None
        self.transformer = None
        self._direction = None

    def convert(self,
            i1: np.ndarray,
            i2: np.ndarray,
            PROJ: str | dict,
            BF: str,
            EPSG: int | str = 4326
        ):
        """
        Converts points to and from Coordinates Reference Systems (CRS)

        Parameters
        ----------
        i1: np.ndarray
            Input x-coordinates
        i2: np.ndarray
            Input y-coordinates
        PROJ: str or dict
            Spatial reference system for coordinate transformations
        BF: str
            Direction of transformation

                - ``'B'``: backwards
                - ``'F'``: forwards
        EPSG: int or str, default 4326 (WGS84 Latitude/Longitude)
            input (``'F'``) or output (``'B'``) coordinate system

        Returns
        -------
        o1: np.ndarray
            Output transformed x-coordinates
        o2: np.ndarray
            Output transformed y-coordinates
        """
        # name of the projection
        self.name = PROJ
        # get the CRS and transform direction
        self.get(PROJ)
        self._direction = BF[0].upper()
        # run conversion program and return values
        return self.transform(i1, i2, EPSG=EPSG)

    # PURPOSE: try to get the projection information
    def get(self, PROJ: str | dict):
        """
        Tries to get the coordinate reference system

        Parameters
        ----------
        PROJ: str or dict
            Spatial reference system for coordinate transformations
        """
        # get the coordinate reference system
        try:
            self.crs = self.from_input(PROJ)
            self.name = self.crs.name
        except Exception as exc:
            pass
        else:
            return self
        # projection not found or available
        raise pyproj.exceptions.CRSError

    def transform(self,
            i1: np.ndarray,
            i2: np.ndarray,
            EPSG: int | str = 4326,
            **kwargs):
        """
        Performs Coordinates Reference System (CRS) transformations

        Parameters
        ----------
        i1: np.ndarray
            Input x-coordinates
        i2: np.ndarray
            Input y-coordinates
        EPSG: int or str, default 4326 (WGS84 Latitude/Longitude)
            input (``'F'``) or output (``'B'``) coordinate system
        **kwargs: dict
            Keyword arguments for the transformation

        Returns
        -------
        o1: np.ndarray
            Output transformed x-coordinates
        o2: np.ndarray
            Output transformed y-coordinates
        """
        # set the direction of the transformation
        kwargs.setdefault('direction', self.direction)
        # get the coordinate reference system and transform
        source_crs = self.from_input(EPSG)
        self.transformer = pyproj.Transformer.from_crs(
            source_crs, self.crs, always_xy=True)
        # convert coordinate reference system
        o1, o2 = self.transformer.transform(i1, i2, **kwargs)
        # return the transformed coordinates
        return (o1, o2)

    # PURPOSE: try to get the projection information
    def from_input(self, PROJECTION: int | str | dict):
        """
        Attempt to retrieve the Coordinate Reference System

        Parameters
        ----------
        PROJECTION: int, str or dict
            Coordinate Reference System
        """
        # coordinate reference system dictionary
        try:
            CRS = pyproj.CRS.from_user_input(PROJECTION)
        except (ValueError, pyproj.exceptions.CRSError):
            pass
        else:
            return CRS
        # EPSG projection code
        try:
            CRS = pyproj.CRS.from_epsg(int(PROJECTION))
        except (ValueError, pyproj.exceptions.CRSError):
            pass
        else:
            return CRS
        # coordinate reference system string
        try:
            CRS = pyproj.CRS.from_string(PROJECTION)
        except (ValueError, pyproj.exceptions.CRSError):
            pass
        else:
            return CRS
        # no projection can be made
        raise pyproj.exceptions.CRSError

    @property
    def direction(self):
        """
        ``pyproj`` direction of the coordinate transform
        """
        # convert from input coordinates to model coordinates
        if (self._direction is None) or (self._direction.upper() == 'F'):
            return pyproj.enums.TransformDirection.FORWARD
        # convert from model coordinates to coordinates
        elif (self._direction.upper() == 'B'):
            return pyproj.enums.TransformDirection.INVERSE

    @property
    def is_geographic(self):
        """
        Check if the coordinate reference system is geographic
        """
        return self.crs.is_geographic

    def __str__(self):
        """String representation of the ``crs`` object
        """
        properties = ['pyTMD.crs']
        properties.append(f"    name: {self.name}")
        properties.append(f"    direction: {self.direction.name}")
        return '\n'.join(properties)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
