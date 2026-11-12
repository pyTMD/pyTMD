#!/usr/bin/env python
"""
model.py
Written by Tyler Sutterley (05/2024)
Base class for estimating tidal constituents using hydrodynamic modeling
on an Arakawa-C grid

    +-- v --+
    |       |
    u   h   u
    |       |
    +-- v --+

REFERENCES:
    G. D. Egbert and S. Erofeeva, "Efficient Inverse Modeling of Barotropic
        Ocean Tides", Journal of Atmospheric and Oceanic Technology, (2002).

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/docmatrix/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/

PROGRAM DEPENDENCIES:
    astro.py: computes the basic astronomical mean longitudes
    constituents.py: calculates constituent parameters and nodal arguments
    spatial.py: utilities for working with geospatial data
    utilities.py: download and management utilities for files

UPDATE HISTORY:
    Updated 03/2025: use refactored datum class in spatial module
    Updated 05/2024: make subscriptable and allow item assignment
    Written 02/2024
"""

from __future__ import annotations

import pyproj
import numpy as np
import scipy.ndimage
import pyTMD.math
import pyTMD.constituents
from pyTMD.earth import EarthDatum
from pyTMD.spatial import scale_factors


class CGrid:
    """
    Class for setting up finite difference grids for tidal modeling
    on an Arakawa-C grid

    Parameters
    ----------
    extent: list, tuple or np.ndarray
        spatial extent of the grid
    spacing: list, tuple or np.ndarray
        grid spacing
    crs: int, str, pyproj.CRS or NoneType, default None
        Coordinate Reference System definition
    a_axis: float, default 6378137.0
        Semi-major axis of the Earth's ellipsoid [m]
    flat: float, default 1.0/298.257223563
        Flattening of the Earth's ellipsoid
    GM: float, default 3.986004418e14
        Geocentric Gravitational Constant [m^3/s^2]
    gamma: float, default 9.80665
        standard gravitational acceleration [m/s^2]
    omega: float, default 7.292115e-5
        average angular rotation rate of the Earth [rad/s]
    """

    # 360 degrees
    turndeg = 360.0

    def __init__(self, **kwargs):
        # set initial attributes
        self.extent = [None, None, None, None]
        self.spacing = [None, None]
        # set the spatial projection reference information
        crs = kwargs.get("crs", 4326)
        self.crs = pyproj.CRS.from_user_input(crs)
        self.a_axis = self.crs.get_geod().a
        self.flat = self.crs.get_geod().f
        # model parameters and constants
        # Geocentric Gravitational Constant [m^3/s^2]
        self.GM = 3.986004418e14
        # standard gravitational acceleration [m/s^2]
        # (World Meteorological Organization)
        self.gamma = 9.80665
        # average angular rotation rate of the Earth [rad/s]
        self.omega = 7292115e-11
        # set optional fields or redefine constants
        for key, val in kwargs.items():
            setattr(self, key, val)
        # validate inputs
        self.__validate__()

    def get_latlon(self, x: np.ndarray, y: np.ndarray):
        """
        Get the latitude and longitude of grid cells

        Parameters
        ----------
        x: np.ndarray
            x-coordinates of grid cells
        y: np.ndarray
            y-coordinates of grid cells

        Returns
        -------
        longitude: np.ndarray
            longitude coordinates of grid cells
        latitude: np.ndarray
            latitude coordinates of grid cells
        """
        # target spatial reference (WGS84 latitude and longitude)
        target_crs = pyproj.CRS.from_user_input(4326)
        # create transformation
        transformer = pyproj.Transformer.from_crs(
            self.crs, target_crs, always_xy=True
        )
        # create meshgrid of points in original projection
        xgrid, ygrid = np.meshgrid(x, y)
        # convert coordinates to latitude and longitude
        longitude, latitude = transformer.transform(xgrid, ygrid)
        # return the latitude and longitude of the grid cells
        return longitude, latitude

    def get_step_size(self, x: np.ndarray, y: np.ndarray, scale: float = 1.0):
        """Calculate the step size of the grid cells in meters

        Parameters
        ----------
        x: np.ndarray
            x-coordinates of grid cells
        y: np.ndarray
            y-coordinates of grid cells
        scale: float, default 1.0
            scaling factor to convert the step size to meters

        Returns
        -------
        dx: np.ndarray
            step size in the x-direction [m]
        dy: np.ndarray
            step size in the y-direction [m]
        """
        # Get the latitude and longitude of grid cells
        longitude, latitude = self.get_latlon(x, y)
        if self.is_geographic:
            # calculate the step sizes in radians
            dphi = np.deg2rad(np.abs(x[1] - x[0]))
            dth = np.deg2rad(np.abs(y[1] - y[0]))
            # convert the step sizes to meters
            dx = self.rad_e * dphi * np.cos(np.deg2rad(latitude))
            dy = self.rad_e * dth * np.ones_like(latitude)
            # assume a spherical Earth
        elif self.is_stereographic:
            # stereographic scaling factors for areal distortion
            reference_latitude = self._cf["standard_parallel"]
            ps_scale = scale_factors(
                latitude,
                flat=self.flat,
                reference_latitude=reference_latitude,
                metric="distance",
            )
            # calculate the step sizes in meters
            # taking into account the mapping distortion
            dx = scale * np.abs(x[1] - x[0]) * ps_scale
            dy = scale * np.abs(y[1] - y[0]) * ps_scale
        else:
            # calculate the step sizes in meters
            dx = scale * np.abs(x[1] - x[0]) * np.ones_like(latitude)
            dy = scale * np.abs(y[1] - y[0]) * np.ones_like(latitude)
        # return the grid cell step sizes
        return (dx, dy)

    # PURPOSE: construct masks for zeta nodes
    def _mask_nodes(self, hz: np.ndarray):
        """
        Construct valid (wet) masks for zeta nodes on a C-grid

        Parameters
        ----------
        hz: np.ndarray
            bathymetry at grid centers
        """
        # check shape of input bathymetry
        assert hz.shape == self.shape, "Input data does not match grid shape"
        # find where bathymetry is greater than 0
        self.mask_z = (hz > 0).astype(int)
        # return the masks
        return self

    def _simplify_mask(self, mask: np.ndarray):
        """
        Simplify a mask by removing isolated points

        Parameters
        ----------
        mask: np.ndarray
            input mask to be simplified

        Returns
        -------
        isolated: np.ndarray
            simplified mask
        """
        ny, nx = mask.shape
        laplacian = -4.0 * np.copy(mask)
        laplacian += mask * np.roll(mask, 1, axis=1)
        laplacian += mask * np.roll(mask, -1, axis=1)
        temp = np.roll(mask, 1, axis=0)
        temp[0, :] = mask[1, :]
        laplacian += mask * temp
        temp = np.roll(mask, -1, axis=0)
        temp[ny - 1, :] = mask[ny - 2, :]
        laplacian += mask * temp
        # create mask of isolated points
        isolated = np.where(np.abs(laplacian) >= 3, 1, 0)
        return isolated

    def _smooth_mask(
        self, mask: np.ndarray, sigma: float = 0.15, cval: float = 1
    ):
        """
        Simplify a mask using a Gaussian filter

        Parameters
        ----------
        mask: np.ndarray
            input mask to be simplified
        sigma: float, default 0.15
            standard deviation of the Gaussian filter
        cval: float, default 1
            constant value to fill the edges of the mask

        Returns
        -------
        smoothed: np.ndarray
            simplified mask
        """
        # wrap data if global
        mode = "wrap" if self.is_global else "constant"
        # use a Gaussian filter to smooth mask
        smoothed = scipy.ndimage.gaussian_filter(
            mask, sigma, mode=("constant", mode), cval=cval
        ).astype(int)
        return smoothed

    # grid dimensions
    @property
    def dimensions(self) -> list:
        """Dimensions of the zeta grid"""
        dims = [None, None]
        # calculate y dimensions with extents
        dims[0] = np.int64((self.extent[3] - self.extent[2]) / self.spacing[1])
        # calculate x dimensions with extents
        dims[1] = np.int64((self.extent[1] - self.extent[0]) / self.spacing[0])
        return dims

    @property
    def shape(self) -> tuple:
        """Shape of the zeta grid"""
        return (
            self.dimensions[0],
            self.dimensions[1],
        )

    @property
    def ndim(self) -> int:
        """Number of dimensions in the grid"""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of nodes in the zeta grid"""
        return np.prod(self.shape)

    @property
    def is_geographic(self) -> bool:
        """Defines if the grid is geographic"""
        return self.crs.is_geographic

    @property
    def is_projected(self) -> bool:
        """Defines if the grid is projected"""
        return self.crs.is_projected

    @property
    def is_stereographic(self) -> bool:
        """Defines if the grid is in a stereographic projection"""
        return self.is_projected and ("stereo" in self._cf["grid_mapping_name"])

    @property
    def is_global(self) -> bool:
        """Defines if the grid is global in terms of longitude"""
        return self.crs.is_geographic and np.isclose(
            self.x_z[-1] - self.x_z[0], self.turndeg - self.spacing[0]
        )

    def f(self, theta: np.ndarray):
        """Coriolis parameter at colatitudes

        Parameters
        ----------
        theta: np.ndarray
            Colatitude [radians]
        """
        return 2.0 * self.omega * np.cos(theta)

    def rossby(self, theta: np.ndarray):
        r"""Rossby (\ |beta|\) parameter at colatitudes

        .. |beta|    unicode:: U+03B2 .. GREEK SMALL LETTER BETA

        Parameters
        ----------
        theta: np.ndarray
            Colatitude [radians]
        """
        return 2.0 * self.omega * np.sin(theta) / self.rad_e

    def gamma_0(self, theta: np.ndarray):
        """Normal gravity at colatitudes

        Parameters
        ----------
        theta: np.ndarray
            Colatitude [radians]
        """
        return self._ellipsoid.gamma_0(theta)

    @property
    def rad_e(self) -> float:
        """Radius of the Earth with same volume as ellipsoid [m]"""
        return self._ellipsoid.rad_e

    @property
    def _ellipsoid(self):
        """Ellipsoidal parameters based on user definitions"""
        return EarthDatum(
            ellipsoid=None,
            a_axis=self.a_axis,
            flat=self.flat,
            GM=self.GM,
            omega=self.omega,
        )

    @property
    def _cf(self) -> dict:
        """CF Convention metadata for the coordinate reference system"""
        return self.crs.to_cf()

    @property
    def x_z(self):
        """x-coordinates for zeta nodes on an Arakawa C-grid"""
        return (
            self.extent[0]
            + self.spacing[0] * np.arange(self.dimensions[1])
            + self.spacing[0] / 2.0
        )

    @property
    def y_z(self):
        """y-coordinates for zeta nodes on an Arakawa C-grid"""
        return (
            self.extent[2]
            + self.spacing[1] * np.arange(self.dimensions[0])
            + self.spacing[1] / 2.0
        )

    @property
    def x_u(self):
        """x-coordinates for U nodes on an Arakawa C-grid"""
        return self.extent[0] + self.spacing[0] * np.arange(self.dimensions[1])

    @property
    def y_u(self):
        """y-coordinates for U nodes on an Arakawa C-grid"""
        return (
            self.extent[2]
            + self.spacing[1] * np.arange(self.dimensions[0])
            + self.spacing[1] / 2.0
        )

    @property
    def x_v(self):
        """x-coordinates for V nodes on an Arakawa C-grid"""
        return (
            self.extent[0]
            + self.spacing[0] * np.arange(self.dimensions[1])
            + self.spacing[0] / 2.0
        )

    @property
    def y_v(self):
        """y-coordinates for V nodes on an Arakawa C-grid"""
        return self.extent[2] + self.spacing[1] * np.arange(self.dimensions[0])

    def __validate__(self):
        """Check if class inputs are appropriate"""
        assert isinstance(self.extent, (list, tuple, np.ndarray))
        assert len(self.extent) == 4
        assert isinstance(self.spacing, (list, tuple, np.ndarray))
        assert len(self.spacing) == 2
        assert isinstance(self.crs, pyproj.CRS)
        assert isinstance(self.a_axis, (int, float))
        assert isinstance(self.flat, (int, float))
        assert isinstance(self.GM, (int, float))
        assert isinstance(self.omega, (int, float))

    def __str__(self):
        """String representation of the ``CGrid`` object"""
        properties = ["pyTMD.solve.CGrid"]
        properties.append(f"    crs: {self.crs.name}")
        extent = ", ".join(map(str, self.extent))
        properties.append(f"    extent: {extent}")
        spacing = ", ".join(map(str, self.spacing))
        properties.append(f"    spacing: {spacing}")
        shape = ", ".join(map(str, self.shape))
        properties.append(f"    shape: {shape}")
        return "\n".join(properties)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class SWEModel:
    """
    Base class for estimating tidal constituents using
    hydrodynamic modeling :cite:`Egbert:2002ge`

    Parameters
    ----------
    grid: pyTMD.solve.CGrid
        Finite difference grid
    beta: float, default 0.10
        scaling factor for load tides and self attraction effects
    rho_w: float, default 1035.0
        average density of sea water [kg/m^3]
    c_drag: float, default 0.003
        non-dimensional quadratic bottom drag coefficient
    nu: float, default 100.0
        Eddy viscosity [m^2/s^2]
    Rd: float or NoneType, default None
        Rossby radius of deformation
    h2: float, default 0.609
        Love number of vertical displacement
    k2: float, default 0.302
        Love number of gravitational potential
    dt: float, default 30.0
        time step in seconds
    """

    def __init__(self, grid: CGrid, constituents: list = [], **kwargs):
        # set initial attributes
        # set the finite difference grid
        assert isinstance(grid, CGrid)
        self.grid = grid
        # set the tidal constituents
        assert isinstance(constituents, (list, np.ndarray))
        self.constituents = constituents
        # model parameters and constants
        # load tide and self attraction scaling factor
        self.beta = 0.10
        # average density of sea water [kg/m^3]
        self.rho_w = 1035.0
        # non-dimensional quadratic bottom drag coefficient
        # ranges from 0.0025 to 0.003 [Bowden and Fairburn, 1952]
        self.c_drag = 0.003
        # eddy viscosity [m^2/s^2]
        self.nu = 100.0
        # Rossby radius of deformation
        self.Rd = None
        # Love numbers appropriate for body tides [Wahr et al., (1985)]
        self.h2 = 0.609
        self.k2 = 0.302
        # time step
        self.dt = 30.0
        # set optional fields or redefine constants
        for key, val in kwargs.items():
            setattr(self, key, val)
        # validate inputs
        self.__validate__()

    # PURPOSE: load constituent parameters
    def constituent_parameters(self, c: str):
        """
        Loads parameters for a given tidal constituent

        Parameters
        ----------
        c: str
            tidal constituent ID

        Returns
        -------
        amplitude: float
            amplitude of equilibrium tide for tidal constituent (meters)
        phase: float
            phase of tidal constituent (radians)
        omega: float
            angular frequency of constituent (radians)
        alpha: float
            load love number of tidal constituent
        species: float
            spherical harmonic dependence of quadrupole potential
        """
        return pyTMD.constituents._constituent_parameters(c)

    # PURPOSE: calculate the astronomical tide generating force
    def generating_force(self, c: str):
        """
        Computes the astronomical tide generating force (the horizontal gradient
        of the tide generating potential) for a tidal constituent

        Parameters
        ----------
        c: str
            tidal constituent ID
        """
        # longitudes and latitudes for u and v transports
        lon_u, lat_u = self.grid.get_latlon(self.grid.x_u, self.grid.y_u)
        lon_v, lat_v = self.grid.get_latlon(self.grid.x_v, self.grid.y_v)
        # longitudes and colatitudes for u transports in radians
        phi_u = np.deg2rad(lon_u)
        th_u = np.deg2rad(90.0 - lat_u)
        # longitudes and colatitudes for v transports in radians
        phi_v = np.deg2rad(lon_v)
        th_v = np.deg2rad(90.0 - lat_v)
        # gravitational acceleration at each colatitude [m/s^2]
        gamma_u = self.grid.gamma_0(th_u)
        gamma_v = self.grid.gamma_0(th_v)

        # load parameters for each constituent
        amp, ph, omega, alpha, species = self.constituent_parameters(c)
        # no time dependence of astronomical forcing
        ph = 0.0

        # calculate forcing for constituent
        Fu = (
            alpha
            * amp
            * gamma_u
            * np.exp(1j * phi_u * species + 1j * ph)
            / self.grid.rad_e
        )
        Fv = (
            alpha
            * amp
            * gamma_v
            * np.exp(1j * phi_v * species + 1j * ph)
            / self.grid.rad_e
        )

        # calculate latitudinal dependence of forcing
        # for a given spherical harmonic dependence
        if species == 1:
            # diurnal species
            Fu *= 2j * np.cos(th_u)
            Fv *= 2.0 * (2.0 * np.sin(th_v) ** 2 - 1.0)
        elif species == 2:
            # semidiurnal species
            Fu *= 2j * np.sin(th_u)
            Fv *= -2.0 * (np.sin(th_v) * np.cos(th_v))
        else:
            # long-period species
            Fu *= 0.0 + 0j
            Fv *= -3.0 * (np.sin(th_v) * np.cos(th_v))

        # return the generating forces
        return (Fu, Fv)

    # PURPOSE: calculate the astronomical tide generating potential
    # converted to an equilibrium tide elevation
    def generating_potential(self, c: str):
        """
        Computes the astronomical tide generating potential for a tidal
        constituent converted to an equilibrium tide elevation

        Parameters
        ----------
        c: str
            tidal constituent ID
        """
        # longitudes and latitudes for zeta nodes
        lon_z, lat_z = self.grid.get_latlon(self.grid.x_z, self.grid.y_z)
        # longitudes and colatitudes for zeta nodes in radians
        phi_z = np.deg2rad(lon_z)
        th_z = np.deg2rad(90.0 - lat_z)

        # load parameters for each constituent
        amp, ph, omega, alpha, species = self.constituent_parameters(c)
        # no time dependence of astronomical forcing
        ph = 0.0

        # calculate potential for constituent
        zeta = alpha * amp * np.exp(1j * phi_z * species + 1j * ph)

        # calculate latitudinal dependence of potential
        # for a given spherical harmonic dependence
        if species == 1:
            # diurnal species
            zeta *= 2.0 * np.sin(th_z) * np.cos(th_z)
        elif species == 2:
            # semidiurnal species
            zeta *= np.sin(th_z) ** 2
        else:
            # long-period species
            zeta *= -1.0 * (3.0 / 2.0 * np.cos(th_z) ** 2 - 1.0 / 2.0)

        # return the generating potential and the angular frequency
        return (zeta, omega)

    @property
    def shape(self) -> tuple:
        """Shape of the grid"""
        return self.grid.shape

    @property
    def size(self) -> int:
        """Total number of nodes"""
        return self.grid.size

    @property
    def nx(self) -> int:
        """Number of nodes in the x-direction"""
        return self.grid.shape[1]

    @property
    def ny(self) -> int:
        """Number of nodes in the y-direction"""
        return self.grid.shape[0]

    def __validate__(self):
        """Check if class inputs are appropriate"""
        NoneType = type(None)
        assert isinstance(self.beta, (int, float))
        assert isinstance(self.rho_w, (int, float))
        assert isinstance(self.c_drag, (int, float))
        assert isinstance(self.nu, (int, float))
        assert isinstance(self.Rd, (int, float, NoneType))
        assert isinstance(self.h2, (int, float))
        assert isinstance(self.k2, (int, float))
        assert isinstance(self.dt, (int, float))

    def __str__(self):
        """String representation of the ``SWEModel`` object"""
        properties = ["pyTMD.solve.SWEModel"]
        shape = ", ".join(map(str, self.shape))
        properties.append(f"    shape: {shape}")
        return "\n".join(properties)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
