#!/usr/bin/env python
"""
model.py
Written by Tyler Sutterley (11/2026)
Base classes for estimating tidal constituents using hydrodynamic
modeling on Arakawa-C grids

Following the OTIS convention, the u nodes are located on the western
faces and the v nodes are located on the southern faces of the zeta cells

    +-- v --+
    |       |
    u   h   u
    |       |
    +-- v --+

Solves the linearized shallow water equations for volume transports
(:math:`U`, :math:`V`) and elevations (:math:`\\zeta`)

    dU/dt - fV + gH(1-beta) dz/dx + kappa U - nu del2 U = gH dz_eq/dx
    dV/dt + fU + gH(1-beta) dz/dy + kappa V - nu del2 V = gH dz_eq/dy
    dz/dt + dU/dx + dV/dy = 0

REFERENCES:
    G. D. Egbert and S. Erofeeva, "Efficient Inverse Modeling of Barotropic
        Ocean Tides", Journal of Atmospheric and Oceanic Technology, (2002).

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
        https://pyproj4.github.io/pyproj/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

PROGRAM DEPENDENCIES:
    constituents.py: calculates constituent parameters and nodal arguments
    earth.py: calculates Earth parameters and Body Tide Love numbers
    spatial.py: utilities for working with geospatial data

UPDATE HISTORY:
    Updated 11/2026: merge c-grid class into this module
        allow spatially variable quadratic drag coefficients
        add bathymetry and masks for u and v nodes
        build linear operators for transports
        output solutions as xarray DataTrees
    Updated 10/2025: use refactored EarthDatum class in spatial module
    Updated 05/2024: make subscriptable and allow item assignment
    Written 02/2024
"""

from __future__ import annotations

import pyproj
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import xarray as xr
import pyTMD.constituents
from pyTMD.earth import EarthDatum
from pyTMD.utilities import reify
from pyTMD.spatial import scale_factors

__all__ = [
    "CGrid",
    "SWEModel",
]


class CGrid:
    r"""
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
        Semi-major axis of the Earth's ellipsoid (m)
    flat: float, default 1.0/298.257223563
        Flattening of the Earth's ellipsoid
    GM: float, default 3.986004418e14
        Geocentric Gravitational Constant (m\ :sup:`3`/s\ :sup:`2`)
    gamma: float, default 9.80665
        standard gravitational acceleration (m/s\ :sup:`2`)
    omega: float, default 7.292115e-5
        average angular rotation rate of the Earth (radians/s)
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
            # coordinate reference system has already been set
            if key == "crs":
                continue
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

    def get_step_size(self, x: np.ndarray, y: np.ndarray):
        """Calculate the step size of the grid cells in meters

        Parameters
        ----------
        x: np.ndarray
            x-coordinates of grid cells
        y: np.ndarray
            y-coordinates of grid cells

        Returns
        -------
        dx: np.ndarray
            step size in the x-direction (m)
        dy: np.ndarray
            step size in the y-direction (m)
        """
        # Get the latitude and longitude of grid cells
        longitude, latitude = self.get_latlon(x, y)
        if self.is_geographic:
            # calculate the step sizes in radians
            dphi = np.radians(np.abs(x[1] - x[0]))
            dth = np.radians(np.abs(y[1] - y[0]))
            # convert the step sizes to meters
            dx = self.rad_e * dphi * np.cos(np.radians(latitude))
            dy = self.rad_e * dth * np.ones_like(latitude)
            # assume a spherical Earth
        elif self.is_stereographic:
            # stereographic scaling factors for areal distortion
            reference_latitude = self._cf.get(
                "standard_parallel",
                self._cf.get("latitude_of_projection_origin", 90.0),
            )
            ps_scale = scale_factors(
                latitude,
                flat=self.flat,
                reference_latitude=reference_latitude,
                metric="distance",
            )
            # calculate the step sizes in meters
            # taking into account the mapping distortion
            dx = np.abs(x[1] - x[0]) * ps_scale
            dy = np.abs(y[1] - y[0]) * ps_scale
        else:
            # calculate the step sizes in meters
            dx = np.abs(x[1] - x[0]) * np.ones_like(latitude)
            dy = np.abs(y[1] - y[0]) * np.ones_like(latitude)
        # return the grid cell step sizes
        return (dx, dy)

    def set_bathymetry(self, hz: np.ndarray, mask: np.ndarray | None = None):
        """
        Set the bathymetry (water column thickness) and the
        valid (wet) mask for zeta nodes on a C-grid

        Parameters
        ----------
        hz: np.ndarray
            bathymetry at grid centers (m)
        mask: np.ndarray or None, default None
            valid (wet) mask at grid centers

            Default is where the bathymetry is greater than 0
        """
        # check shape of input bathymetry
        hz = np.asarray(hz, dtype=np.float64)
        assert hz.shape == self.shape, "Input data does not match grid shape"
        # replace invalid values
        self.hz = np.where(np.isfinite(hz), hz, 0.0)
        # construct the mask for zeta nodes
        if mask is None:
            self._mask_nodes(self.hz)
        else:
            assert np.shape(mask) == self.shape
            self.mask_z = (np.asarray(mask) & (self.hz > 0)).astype(int)
        return self

    # PURPOSE: construct masks for zeta nodes
    def _mask_nodes(self, hz: np.ndarray):
        """
        Construct valid (wet) masks for zeta nodes on a C-grid

        Parameters
        ----------
        hz: np.ndarray
            bathymetry at grid centers (m)
        """
        # check shape of input bathymetry
        assert hz.shape == self.shape, "Input data does not match grid shape"
        # find where bathymetry is greater than 0
        self.mask_z = (hz > 0).astype(int)
        # return the masks
        return self

    def _west(self, field: np.ndarray, fill: float = 0.0):
        """Shift a field on the zeta grid to get the western neighbor"""
        shifted = np.roll(field, 1, axis=1)
        if not self.is_global:
            shifted[:, 0] = fill
        return shifted

    def _south(self, field: np.ndarray, fill: float = 0.0):
        """Shift a field on the zeta grid to get the southern neighbor"""
        shifted = np.roll(field, 1, axis=0)
        shifted[0, :] = fill
        return shifted

    @property
    def mask_u(self) -> np.ndarray:
        """Valid (wet) mask for u nodes (western faces)"""
        return self.mask_z * self._west(self.mask_z)

    @property
    def mask_v(self) -> np.ndarray:
        """Valid (wet) mask for v nodes (southern faces)"""
        return self.mask_z * self._south(self.mask_z)

    @property
    def hu(self) -> np.ndarray:
        """Bathymetry at u nodes (m)"""
        return self.mask_u * 0.5 * (self.hz + self._west(self.hz))

    @property
    def hv(self) -> np.ndarray:
        """Bathymetry at v nodes (m)"""
        return self.mask_v * 0.5 * (self.hz + self._south(self.hz))

    # grid dimensions
    @property
    def dimensions(self) -> list:
        """Dimensions of the zeta grid"""
        dims = [None, None]
        # calculate y dimensions with extents
        ny = (self.extent[3] - self.extent[2]) / np.abs(self.spacing[1])
        dims[0] = np.int64(np.rint(ny))
        # calculate x dimensions with extents
        nx = (self.extent[1] - self.extent[0]) / np.abs(self.spacing[0])
        dims[1] = np.int64(np.rint(nx))
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
        grid_mapping_name = self._cf.get("grid_mapping_name", "")
        return self.is_projected and ("stereo" in grid_mapping_name)

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
            Colatitude (radians)
        """
        return 2.0 * self.omega * np.cos(theta)

    def rossby(self, theta: np.ndarray):
        r"""Rossby (\ |beta|\) parameter at colatitudes

        .. |beta|    unicode:: U+03B2 .. GREEK SMALL LETTER BETA

        Parameters
        ----------
        theta: np.ndarray
            Colatitude (radians)
        """
        return 2.0 * self.omega * np.sin(theta) / self.rad_e

    def gamma_0(self, theta: np.ndarray):
        """Normal gravity at colatitudes

        Parameters
        ----------
        theta: np.ndarray
            Colatitude (radians)
        """
        return self._ellipsoid.gamma_0(theta)

    @property
    def rad_e(self) -> float:
        """Radius of the Earth with same volume as ellipsoid (m)"""
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
        """String representation of the ``solve.CGrid`` object"""
        properties = ["pyTMD.solve.CGrid"]
        properties.append(f"    crs: {self.crs.name}")
        extent = ", ".join(map(str, self.extent))
        properties.append(f"    extent: {extent}")
        spacing = ", ".join(map(str, self.spacing))
        properties.append(f"    spacing: {spacing}")
        shape = ", ".join(map(str, self.shape))
        properties.append(f"    shape: {shape}")
        return "\n".join(properties)

    def __repr__(self):
        """Representation of the ``solve.CGrid`` object"""
        return self.__str__()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class SWEModel:
    r"""
    Base class for using hydrodynamic modeling with the shallow
    water equations :cite:p:`Egbert:2002ge,Primeau:2007ie`

    Parameters
    ----------
    grid: pyTMD.solve.CGrid
        Finite difference grid with bathymetry
    constituents: list, default []
        Tidal constituent IDs
    beta: float, default 0.10
        scaling factor for load tides and self attraction effects
    c_drag: float or np.ndarray, default 0.003
        non-dimensional quadratic bottom drag coefficient
        :cite:p:`Bowden:1952jm`
    u_rms: float, default 1.0
        characteristic velocity for linearizing the bottom drag (m/s)
    kappa: np.ndarray or NoneType, default None
        linear drag coefficient at zeta nodes (1/s)

        Default is ``c_drag*u_rms/H``
    nu: float, default 100.0
        horizontal eddy viscosity (m\ :sup:`2`/s)
    Rd: float or NoneType, default None
        Rossby radius of deformation for reduced-gravity models (m)
    min_depth: float, default 1.0
        minimum water column thickness for valid nodes (m)
    coriolis: bool, default True
        include Coriolis effects
    astronomical: bool, default True
        include the astronomical tide generating force
    ob: np.ndarray or NoneType, default None
        open boundary mask at zeta nodes
    dt: float, default 30.0
        time step in seconds
    """

    def __init__(self, grid: CGrid, constituents: list = [], **kwargs):
        # set initial attributes
        # set the finite difference Arakawa c-grid
        assert isinstance(grid, CGrid)
        assert hasattr(grid, "hz"), "Grid bathymetry has not been set"
        self.grid = grid
        # set the tidal constituents
        if isinstance(constituents, str):
            constituents = [constituents]
        assert isinstance(constituents, (list, tuple, np.ndarray))
        self.constituents = list(constituents)
        # model parameters and constants
        # load tide and self attraction scaling factor
        self.beta = 0.10
        # non-dimensional quadratic bottom drag coefficient
        # ranges from 0.0025 to 0.003 [Bowden and Fairbairn, 1952]
        self.c_drag = 0.003
        # characteristic velocity for linearized bottom drag [m/s]
        self.u_rms = 1.0
        # linear drag coefficient [1/s]
        self.kappa = None
        # eddy viscosity [m^2/s]
        self.nu = 100.0
        # Rossby radius of deformation
        self.Rd = None
        # minimum water column thickness (m)
        self.min_depth = 1.0
        # include Coriolis effects
        self.coriolis = True
        # include astronomical forcing
        self.astronomical = True
        # open boundary mask
        self.ob = None
        # time step
        self.dt = 30.0
        # set optional fields or redefine constants
        for key, val in kwargs.items():
            setattr(self, key, val)
        # set default open boundary mask
        if self.ob is None:
            self.ob = np.zeros(self.shape, dtype=bool)
        # fix open boundary mask with grid elevation mask
        mask_z = self.grid.mask_z.astype(bool)
        self.ob = np.asarray(self.ob, dtype=bool) & mask_z
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

        Returns
        -------
        zeta: np.ndarray
            complex equilibrium tide elevation at zeta nodes (m)
        omega: float
            angular frequency of constituent (radians/s)
        """
        # longitudes and latitudes for zeta nodes
        lon_z, lat_z = self.grid.get_latlon(self.grid.x_z, self.grid.y_z)
        # longitudes and colatitudes for zeta nodes in radians
        phi_z = np.radians(lon_z)
        th_z = np.radians(90.0 - lat_z)

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

    def d0(self, M: np.ndarray):
        """Converts a vector or matrix to a sparse diagonal matrix

        Parameters
        ----------
        M: np.ndarray
            input field to be converted
        """
        M = np.ravel(M)
        return scipy.sparse.diags_array(
            M, offsets=0, shape=(M.size, M.size), dtype=None
        )

    @reify
    def _shifts(self) -> dict:
        """Sparse shift operators for neighboring zeta cells"""
        Ix = scipy.sparse.eye_array(self.nx, format="csr")
        Iy = scipy.sparse.eye_array(self.ny, format="csr")
        # boundary conditions for the x directions
        if self.grid.is_global:
            # periodic boundaries
            iix = np.arange(self.nx)
            iie = Ix[np.roll(iix, -1), :]
            iiw = Ix[np.roll(iix, 1), :]
        else:
            # closed boundaries
            iie = scipy.sparse.eye_array(self.nx, k=1)
            iiw = scipy.sparse.eye_array(self.nx, k=-1)
        # closed boundaries for the y directions
        iin = scipy.sparse.eye_array(self.ny, k=1)
        iis = scipy.sparse.eye_array(self.ny, k=-1)
        # full shift operators
        shifts = {}
        shifts["E"] = scipy.sparse.kron(Iy, iie, format="csr")
        shifts["W"] = scipy.sparse.kron(Iy, iiw, format="csr")
        shifts["N"] = scipy.sparse.kron(iin, Ix, format="csr")
        shifts["S"] = scipy.sparse.kron(iis, Ix, format="csr")
        shifts["I"] = scipy.sparse.eye_array(self.size, format="csr")
        return shifts

    @reify
    def _metrics(self) -> dict:
        """Grid metrics for the finite difference operators (m)"""
        # step sizes at zeta, u and v nodes
        dx_z, dy_z = self.grid.get_step_size(self.grid.x_z, self.grid.y_z)
        dx_u, dy_u = self.grid.get_step_size(self.grid.x_u, self.grid.y_u)
        dx_v, dy_v = self.grid.get_step_size(self.grid.x_v, self.grid.y_v)
        metrics = {}
        metrics["dx_u"] = dx_u.flatten()
        metrics["dy_u"] = dy_u.flatten()
        metrics["dx_v"] = dx_v.flatten()
        metrics["dy_v"] = dy_v.flatten()
        metrics["area"] = (dx_z * dy_z).flatten()
        return metrics

    @reify
    def _colatitudes(self) -> dict:
        """Colatitudes of the zeta, u and v nodes (radians)"""
        _, lat_z = self.grid.get_latlon(self.grid.x_z, self.grid.y_z)
        _, lat_u = self.grid.get_latlon(self.grid.x_u, self.grid.y_u)
        _, lat_v = self.grid.get_latlon(self.grid.x_v, self.grid.y_v)
        th = {}
        th["z"] = np.radians(90.0 - lat_z).flatten()
        th["u"] = np.radians(90.0 - lat_u).flatten()
        th["v"] = np.radians(90.0 - lat_v).flatten()
        return th

    @staticmethod
    def _inverse(x: np.ndarray):
        """Calculate the inverse of an array replacing invalid values"""
        with np.errstate(divide="ignore", invalid="ignore"):
            y = 1.0 / x
        return np.where(np.isfinite(y), y, 0.0)

    @reify
    def _operators(self) -> dict:
        """Finite difference operators on the full grid"""
        S = self._shifts
        I = S["I"]
        m = self._metrics
        ops = {}
        # gradient of zeta at u nodes (western faces)
        ops["Gx"] = self.d0(self._inverse(m["dx_u"])) @ (I - S["W"])
        # gradient of zeta at v nodes (southern faces)
        ops["Gy"] = self.d0(self._inverse(m["dy_v"])) @ (I - S["S"])
        # divergence of transports at zeta nodes (finite volume)
        inv_area = self.d0(self._inverse(m["area"]))
        ops["Dx"] = inv_area @ (S["E"] - I) @ self.d0(m["dy_u"])
        ops["Dy"] = inv_area @ (S["N"] - I) @ self.d0(m["dx_v"])
        # averaging operator for calculating V at u nodes
        ops["Avu"] = 0.25 * (I + S["W"] + S["N"] + S["N"] @ S["W"])
        # averaging operator for calculating U at v nodes
        ops["Auv"] = 0.25 * (I + S["E"] + S["S"] + S["S"] @ S["E"])
        # laplacian operators for the u and v nodes
        ops["Lu"] = self.d0(self._inverse(m["dx_u"] ** 2)) @ (
            S["E"] + S["W"] - 2.0 * I
        ) + self.d0(self._inverse(m["dy_u"] ** 2)) @ (S["N"] + S["S"] - 2.0 * I)
        ops["Lv"] = self.d0(self._inverse(m["dx_v"] ** 2)) @ (
            S["E"] + S["W"] - 2.0 * I
        ) + self.d0(self._inverse(m["dy_v"] ** 2)) @ (S["N"] + S["S"] - 2.0 * I)
        return ops

    @property
    def H(self) -> np.ndarray:
        """Water column thickness at zeta nodes (m)"""
        return self.grid.mask_z * np.maximum(self.grid.hz, self.min_depth)

    @property
    def Hu(self) -> np.ndarray:
        """Water column thickness at u nodes (m)"""
        return self.grid.mask_u * np.maximum(self.grid.hu, self.min_depth)

    @property
    def Hv(self) -> np.ndarray:
        """Water column thickness at v nodes (m)"""
        return self.grid.mask_v * np.maximum(self.grid.hv, self.min_depth)

    def _gravity(self, node: str = "z") -> np.ndarray:
        r"""Normal gravity at the nodes (m/s\ :sup:`2`)"""
        return self.grid.gamma_0(self._colatitudes[node])

    def _wave_speed(self, node: str = "u") -> np.ndarray:
        r"""
        Square of the (reduced) gravity wave speed at nodes
        (m\ :sup:`2`/s\ :sup:`2`)
        """
        H = dict(u=self.Hu, v=self.Hv)[node].flatten()
        if self.Rd is None:
            # standard shallow water model
            c2 = self._gravity(node) * H
        else:
            # reduced gravity model with a deformation radius
            f = self.grid.f(self._colatitudes[node])
            c2 = (f * self.Rd) ** 2 * (H > 0)
        # account for load tides and self-attraction
        return (1.0 - self.beta) * c2

    def _linear_drag(self) -> tuple:
        """Linear drag coefficients at u and v nodes [1/s]"""
        if self.kappa is None:
            # drag linearized with a characteristic velocity
            c_drag = np.broadcast_to(self.c_drag, self.shape)
            kappa = c_drag * self.u_rms * self._inverse(self.H)
        else:
            kappa = np.broadcast_to(self.kappa, self.shape)
        kappa = np.where(self.grid.mask_z, kappa, 0.0)
        # average the drag coefficients to u and v nodes
        ku = self.grid.mask_u * 0.5 * (kappa + self.grid._west(kappa))
        kv = self.grid.mask_v * 0.5 * (kappa + self.grid._south(kappa))
        return ku.flatten(), kv.flatten()

    def _quadratic_drag(self) -> tuple:
        """Quadratic drag coefficients at u and v nodes"""
        c_drag = np.broadcast_to(self.c_drag, self.shape)
        c_drag = np.where(self.grid.mask_z, c_drag, 0.0)
        # average the drag coefficients to u and v nodes
        cu = self.grid.mask_u * 0.5 * (c_drag + self.grid._west(c_drag))
        cv = self.grid.mask_v * 0.5 * (c_drag + self.grid._south(c_drag))
        return cu.flatten(), cv.flatten()

    def _linear_operators(self, drag: bool = True):
        """
        Create the operator for the linearized shallow water equations

        Parameters
        ----------
        drag: bool, default True
            include the linear bottom drag terms

        Returns
        -------
        L: scipy.sparse.csr_array
            operator for the complete state vector where
            ``ds/dt + L s = F``
        """
        ops = self._operators
        n = self.size
        # Coriolis parameters at each node
        if self.coriolis:
            f_u = self.grid.f(self._colatitudes["u"])
            f_v = self.grid.f(self._colatitudes["v"])
        else:
            f_u = np.zeros(n)
            f_v = np.zeros(n)
        # masks for u and v nodes
        mu = self.d0(self.grid.mask_u)
        mv = self.d0(self.grid.mask_v)
        # antisymmetric (energy conserving) Coriolis operators
        Cuv = -0.5 * (self.d0(f_u) @ ops["Avu"] + ops["Avu"] @ self.d0(f_v))
        Cvu = 0.5 * (self.d0(f_v) @ ops["Auv"] + ops["Auv"] @ self.d0(f_u))
        # linear drag and viscous terms
        if drag:
            ku, kv = self._linear_drag()
        else:
            ku, kv = np.zeros(n), np.zeros(n)
        Kuu = self.d0(ku) - self.nu * ops["Lu"]
        Kvv = self.d0(kv) - self.nu * ops["Lv"]
        # pressure gradient terms
        Puz = self.d0(self._wave_speed("u")) @ ops["Gx"]
        Pvz = self.d0(self._wave_speed("v")) @ ops["Gy"]
        # operator for the linear shallow water model dynamics
        # with viscous terms and coriolis effects
        #     __u__ __v__ __z__
        #  u |_____|_____|_____|
        #  v |_____|_____|_____|
        #  z |_____|_____|_____|
        Z = None
        L = scipy.sparse.block_array(
            [
                [mu @ Kuu @ mu, mu @ Cuv @ mv, mu @ Puz],
                [mv @ Cvu @ mu, mv @ Kvv @ mv, mv @ Pvz],
                [ops["Dx"] @ mu, ops["Dy"] @ mv, Z],
            ],
            format="csr",
        )
        return L

    def _forcing(self, c: str) -> np.ndarray:
        """
        Calculate the astronomical forcing for a tidal constituent
        using the discrete gradient of the equilibrium tide

        Parameters
        ----------
        c: str
            tidal constituent ID

        Returns
        -------
        F: np.ndarray
            complex forcing for the complete state vector
        """
        ops = self._operators
        zeta, omega = self.generating_potential(c)
        zeta = zeta.flatten()
        # gravity wave speed without load tide and self-attraction
        Fu = self._wave_speed("u") / (1.0 - self.beta) * (ops["Gx"] @ zeta)
        Fv = self._wave_speed("v") / (1.0 - self.beta) * (ops["Gy"] @ zeta)
        Fz = np.zeros(self.size, dtype=np.complex128)
        return np.r_[Fu, Fv, Fz]

    def _system(self, omega: float, L=None):
        """
        Create the frequency-domain system of equations for the valid
        (wet) nodes with open boundary conditions

        Parameters
        ----------
        omega: float
            angular frequency [rad/s]
        L: scipy.sparse.csr_array or None, default None
            linear operator for the complete state vector

        Returns
        -------
        A: scipy.sparse.csc_array
            system of equations for the valid nodes
        """
        if L is None:
            L = self._linear_operators()
        n = 3 * self.size
        # time derivatives in the frequency domain
        I = scipy.sparse.eye_array(n, format="csr")
        A = (1j * omega * I + L)[self.imask, :][:, self.imask]
        # replace the continuity equations at open boundary nodes
        # with the prescribed elevations
        if self.has_ob:
            A = A.tolil()
            for i in self._iob_state:
                A.rows[i] = [i]
                A.data[i] = [1.0 + 0j]
            A = A.tocsr()
        return A.tocsc()

    def _rhs(self, c: str, ob: np.ndarray | None = None) -> np.ndarray:
        """
        Create the right-hand side of the frequency-domain system

        Parameters
        ----------
        c: str
            tidal constituent ID
        ob: np.ndarray or None, default None
            complex elevations at open boundary nodes

        Returns
        -------
        b: np.ndarray
            right-hand side for the valid nodes
        """
        if self.astronomical:
            b = self._forcing(c)[self.imask]
        else:
            b = np.zeros(len(self.imask), dtype=np.complex128)
        if self.has_ob:
            ob = np.zeros(self.shape) if ob is None else np.asarray(ob)
            b[self._iob_state] = ob.flatten()[self._iob]
        return b

    def solve(self, c: str | list | None = None, ob: dict | None = None):
        """
        Calculate frequency-domain solutions of the linearized shallow
        water equations for tidal constituents

        Parameters
        ----------
        c: str, list or None, default None
            tidal constituent ID(s)

            Default is all model constituents
        ob: dict or None, default None
            complex elevations at open boundary nodes for each constituent

        Returns
        -------
        dtree: xr.DataTree
            complex elevations and transports for each constituent
        """
        if c is None:
            c = self.constituents
        elif isinstance(c, str):
            c = [c]
        ob = {} if ob is None else ob
        L = self._linear_operators()
        states = {}
        for con in c:
            *_, omega, _, _ = self.constituent_parameters(con)
            A = self._system(omega, L=L)
            b = self._rhs(con, ob.get(con))
            states[con] = self._expand(scipy.sparse.linalg.spsolve(A, b))
        return self.to_datatree(states)

    def linearized_drag(self, dtree: xr.DataTree, power: float = 1.0):
        """
        Update the linear drag coefficients using the root-mean-square
        velocities from model solutions

        Parameters
        ----------
        dtree: xr.DataTree
            complex elevations and transports for each constituent
        power: float, default 1.0
            relaxation factor for updating the drag coefficients

        Returns
        -------
        kappa: np.ndarray
            linear drag coefficients at zeta nodes [1/s]
        """
        # mean square velocities at zeta nodes
        u2 = np.zeros(self.shape)
        for c in dtree["z"].ds.data_vars:
            if c in ("bathymetry", "mask"):
                continue
            U = np.nan_to_num(dtree["u"][c].values)
            V = np.nan_to_num(dtree["v"][c].values)
            # average transports to zeta nodes
            mode = "wrap" if self.grid.is_global else "constant"
            Ue = np.pad(U, ((0, 0), (0, 1)), mode=mode)
            Vn = np.pad(V, ((0, 1), (0, 0)), mode="constant")
            Uz = 0.5 * (Ue[:, :-1] + Ue[:, 1:])
            Vz = 0.5 * (Vn[:-1, :] + Vn[1:, :])
            # mean square of the harmonic velocities
            u2 += (
                0.5
                * (np.abs(Uz) ** 2 + np.abs(Vz) ** 2)
                * self._inverse(self.H**2)
            )
        u_rms = np.sqrt(u2)
        c_drag = np.broadcast_to(self.c_drag, self.shape)
        kappa = c_drag * u_rms * self._inverse(self.H)
        # update the drag coefficients with relaxation
        if self.kappa is None:
            self.kappa = power * kappa
        else:
            self.kappa = power * kappa + (1.0 - power) * self.kappa
        return self.kappa

    def _expand(self, s: np.ndarray) -> np.ndarray:
        """Expand a state vector at valid nodes to the full grid"""
        state = np.zeros(3 * self.size, dtype=s.dtype)
        state[self.imask] = s
        return state

    def to_datatree(self, states: dict) -> xr.DataTree:
        """
        Convert complex state vectors into an xarray DataTree

        Parameters
        ----------
        states: dict
            complex state vectors on the full grid for each constituent

        Returns
        -------
        dtree: xr.DataTree
            complex elevations and transports for each constituent
        """
        n = self.size
        crs = self.grid.crs.to_dict()
        nodes = dict(
            z=(self.grid.x_z, self.grid.y_z, self.grid.mask_z, self.H, "m"),
            u=(
                self.grid.x_u,
                self.grid.y_u,
                self.grid.mask_u,
                self.Hu,
                "m**2/s",
            ),
            v=(
                self.grid.x_v,
                self.grid.y_v,
                self.grid.mask_v,
                self.Hv,
                "m**2/s",
            ),
        )
        groups = dict(z="z", u="U", v="V")
        # offsets of each variable in the complete state vector
        offsets = dict(u=0, v=1, z=2)
        ds = {}
        for key, (x, y, mask, H, units) in nodes.items():
            i = offsets[key]
            ds[key] = xr.Dataset(coords=dict(x=x, y=y))
            for c, state in states.items():
                field = state[i * n : (i + 1) * n].reshape(self.shape)
                field = np.where(mask, field, np.nan + 1j * np.nan)
                ds[key][c] = (("y", "x"), field)
                ds[key][c].attrs["units"] = units
            ds[key]["bathymetry"] = (("y", "x"), H)
            ds[key]["bathymetry"].attrs["units"] = "m"
            ds[key]["mask"] = (("y", "x"), mask.astype(np.int8))
            ds[key].attrs["crs"] = crs
            ds[key].attrs["group"] = groups[key]
        return xr.DataTree.from_dict(ds)

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

    @reify
    def imask(self) -> np.ndarray:
        """Indices of valid (wet) nodes in the complete state vector"""
        return np.flatnonzero(
            np.r_[
                self.grid.mask_u.flatten(),
                self.grid.mask_v.flatten(),
                self.grid.mask_z.flatten(),
            ]
        )

    @reify
    def _iob(self) -> np.ndarray:
        """Indices of open boundary nodes on the zeta grid"""
        return np.flatnonzero(self.ob)

    @reify
    def _iob_state(self) -> np.ndarray:
        """Indices of open boundary nodes in the reduced state vector"""
        return np.searchsorted(self.imask, 2 * self.size + self._iob)

    @property
    def has_ob(self) -> bool:
        """Defines if the model has open boundary conditions"""
        return bool(np.any(self.ob))

    def __validate__(self):
        """Check if class inputs are appropriate"""
        NoneType = type(None)
        assert isinstance(self.beta, (int, float))
        assert isinstance(self.c_drag, (int, float, np.ndarray))
        if isinstance(self.c_drag, np.ndarray):
            assert np.shape(self.c_drag) == self.shape
        assert isinstance(self.u_rms, (int, float))
        assert isinstance(self.kappa, (int, float, np.ndarray, NoneType))
        assert isinstance(self.nu, (int, float))
        assert isinstance(self.Rd, (int, float, NoneType))
        assert isinstance(self.min_depth, (int, float))
        assert isinstance(self.dt, (int, float))
        assert np.shape(self.ob) == self.shape

    def __str__(self):
        """String representation of the ``solve.SWEModel`` object"""
        properties = ["pyTMD.solve.SWEModel"]
        shape = ", ".join(map(str, self.shape))
        properties.append(f"    shape: {shape}")
        constituents = ", ".join(self.constituents)
        properties.append(f"    constituents: {constituents}")
        return "\n".join(properties)

    def __repr__(self):
        """Representation of the ``solve.SWEModel`` object"""
        return self.__str__()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
