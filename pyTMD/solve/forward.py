#!/usr/bin/env python
"""
forward.py
Written by Tyler Sutterley (11/2026)
Class for estimating non-linear parameters with a finite difference model
using a Crank-Nicolson scheme

REFERENCES:
    G. D. Egbert and S. Erofeeva, "Efficient Inverse Modeling of Barotropic
        Ocean Tides", Journal of Atmospheric and Oceanic Technology, (2002).
    F. W. Primeau and D. Newman, "Bifurcation structure of a wind-driven
        shallow water model with layer-outcropping", Ocean Modelling, (2007).

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
    model.py: base class for hydrodynamic tide modeling

UPDATE HISTORY:
    Updated 11/2026: move linear operators to the base model class
        fix the Crank-Nicolson scheme and add astronomical forcing
        add open boundary conditions for multiple constituents
        add harmonic analysis of the model state during runs
        allow spatially variable quadratic drag coefficients
    Updated 05/2024: make subscriptable and allow item assignment
    Written 03/2024
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import xarray as xr
from pyTMD.solve.model import CGrid, SWEModel

__all__ = ["ForwardModel"]


class ForwardModel(SWEModel):
    """
    Class for estimating non-linear parameters using a finite
    difference model with a Crank-Nicolson scheme
    :cite:p:`Egbert:2002ge,Primeau:2007ie`

    The model state is propagated in time following

    .. math::
        \\left(I + \\frac{\\Delta t}{2}L\\right)s^{n+1} =
        \\left(I - \\frac{\\Delta t}{2}L\\right)s^{n} +
        \\frac{\\Delta t}{2}\\left(F^{n} + F^{n+1}\\right) +
        \\Delta t N^{n}

    where :math:`L` is the linearized shallow water operator,
    :math:`F` is the astronomical forcing and :math:`N` are the
    (explicit) quadratic bottom drag terms.

    Parameters
    ----------
    grid: pyTMD.solve.CGrid
        Finite difference grid with bathymetry
    constituents: list, default []
        Tidal constituent IDs
    nonlinear: bool, default True
        use quadratic bottom drag rather than linear drag
    ramp: float, default 0.0
        time scale for linearly ramping the forcing [s]
    **kwargs: dict
        keyword arguments for the base ``SWEModel`` class
    """

    # inherit SWEModel class
    def __init__(self, grid: CGrid, constituents: list = [], **kwargs):
        # default keyword arguments
        self.nonlinear = kwargs.pop("nonlinear", True)
        self.ramp = kwargs.pop("ramp", 0.0)
        # set initial model
        super().__init__(grid, constituents, **kwargs)
        # create iterator and stop index
        self.__index__ = 0
        self.__stop__ = 0

    def initialize(self, **kwargs):
        r"""Set the initial conditions for the model

        Parameters
        ----------
        t0: float, default 0.0
            initial time in seconds
        u0: np.ndarray, default np.zeros((ny, nx))
            initial transports in the x-direction (m\ :sup:`2`/s)
        v0: np.ndarray, default np.zeros((ny, nx))
            initial transports in the y-direction (m\ :sup:`2`/s)
        z0: np.ndarray, default np.zeros((ny, nx))
            initial height field (m)
        ob: dict, default {}
            complex elevations at open boundary nodes for each constituent
        """
        # default initial conditions
        kwargs.setdefault("t0", 0.0)
        kwargs.setdefault("u0", np.zeros((self.ny, self.nx)))
        kwargs.setdefault("v0", np.zeros((self.ny, self.nx)))
        kwargs.setdefault("z0", np.zeros((self.ny, self.nx)))
        kwargs.setdefault("ob", {})
        # time variable
        self.t = np.float64(kwargs["t0"])
        # complete state vector
        state = np.r_[
            np.ravel(kwargs["u0"]),
            np.ravel(kwargs["v0"]),
            np.ravel(kwargs["z0"]),
        ].astype(np.float64)
        # state vector at valid points
        self._s = state[self.imask]
        # angular frequencies of each constituent [rad/s]
        self._omega = np.zeros(len(self.constituents))
        # complex forcing and open boundary conditions
        self._F = np.zeros((len(self._s), len(self.constituents)), complex)
        self._ob = np.zeros((len(self._iob), len(self.constituents)), complex)
        for j, c in enumerate(self.constituents):
            *_, omega, _, _ = self.constituent_parameters(c)
            self._omega[j] = omega
            if self.astronomical:
                self._F[:, j] = self._forcing(c)[self.imask]
            if self.has_ob and (c in kwargs["ob"]):
                ob = np.asarray(kwargs["ob"][c]).flatten()
                self._ob[:, j] = ob[self._iob]
        # reset harmonic analysis
        self._reset_harmonics()
        # setup the model operators and the model
        self._factorize()
        return self

    def _factorize(self):
        """Initialize the linear system of equations"""
        # linear operator (with linear drag if not using quadratic drag)
        L = self._linear_operators(drag=not self.nonlinear)
        L = L[self.imask, :][:, self.imask]
        n = len(self.imask)
        I = scipy.sparse.eye_array(n, format="csr")
        # using a Crank-Nicolson scheme for the time step
        A = (I + (self.dt / 2.0) * L).tolil()
        B = (I - (self.dt / 2.0) * L).tolil()
        # replace equations at open boundary nodes
        for i in self._iob_state:
            A.rows[i] = [i]
            A.data[i] = [1.0]
            B.rows[i] = []
            B.data[i] = []
        # prefactor operators for inversion (LU decomposition)
        # function for solving the sparse linear system
        self.solver = scipy.sparse.linalg.factorized(A.tocsc())
        self.rhs = B.tocsr()
        # averaging operators for the quadratic drag
        if self.nonlinear:
            ops = self._operators
            mask_u = self.grid.mask_u.flatten()
            mask_v = self.grid.mask_v.flatten()
            self._Avu = (self.d0(mask_u) @ ops["Avu"] @ self.d0(mask_v)).tocsr()
            self._Auv = (self.d0(mask_v) @ ops["Auv"] @ self.d0(mask_u)).tocsr()
            # quadratic drag coefficients at u and v nodes
            self._cu, self._cv = self._quadratic_drag()

    def _ramp(self, t: float) -> float:
        """Ramp function for the forcing"""
        if self.ramp > 0:
            return np.clip(t / self.ramp, 0.0, 1.0)
        return 1.0

    def forcing(self, t: float) -> np.ndarray:
        """Astronomical forcing at a given time

        Parameters
        ----------
        t: float
            time (seconds)
        """
        if len(self.constituents) == 0:
            return np.zeros(len(self.imask))
        phase = np.exp(1j * self._omega * t)
        return self._ramp(t) * np.real(self._F @ phase)

    def boundary(self, t: float) -> np.ndarray:
        """Elevations at open boundary nodes at a given time

        Parameters
        ----------
        t: float
            time (seconds)
        """
        if len(self.constituents) == 0:
            return np.zeros(len(self._iob))
        phase = np.exp(1j * self._omega * t)
        return self._ramp(t) * np.real(self._ob @ phase)

    def _nonlinear_terms(self) -> np.ndarray:
        """Calculate the quadratic bottom drag terms"""
        state = self.state
        n = self.size
        U, V = state[:n], state[n : 2 * n]
        Hu = self.Hu.flatten()
        Hv = self.Hv.flatten()
        # transports at u and v nodes
        Vu = self._Avu @ V
        Uv = self._Auv @ U
        # quadratic drag on transports: c_d |u| U / H
        tau_x = -self._cu * np.hypot(U, Vu) * U * self._inverse(Hu**2)
        tau_y = -self._cv * np.hypot(Uv, V) * V * self._inverse(Hv**2)
        N = np.r_[tau_x, tau_y, np.zeros(n)]
        return N[self.imask]

    def _reset_harmonics(self):
        """Reset the accumulators for the harmonic analysis"""
        nb = 1 + 2 * len(self.constituents)
        self._G = np.zeros((nb, nb))
        self._R = np.zeros((nb, len(self._s)))
        self._nh = 0

    def _accumulate(self):
        """Add the current state to the harmonic analysis"""
        th = self._omega * self.t
        basis = np.r_[1.0, np.cos(th), np.sin(th)]
        self._G += np.outer(basis, basis)
        self._R += np.outer(basis, self._s)
        self._nh += 1

    def harmonic_analysis(self) -> xr.DataTree:
        """
        Calculate the harmonic constants for each constituent from
        the accumulated model states

        Returns
        -------
        dtree: xr.DataTree
            complex elevations and transports for each constituent
        """
        if self._nh == 0:
            raise ValueError("No model states have been accumulated")
        nc = len(self.constituents)
        coeffs = np.linalg.lstsq(self._G, self._R, rcond=None)[0]
        states = {}
        for j, c in enumerate(self.constituents):
            # s = a cos(wt) + b sin(wt) = Re[(a - ib) exp(iwt)]
            hc = coeffs[1 + j, :] - 1j * coeffs[1 + nc + j, :]
            states[c] = self._expand(hc)
        return self.to_datatree(states)

    def step(self):
        """Calculate the fields at the next time step"""
        t0, t1 = self.t, self.t + self.dt
        # right-hand side using values at previous time step
        b = self.rhs @ self._s
        b += (self.dt / 2.0) * (self.forcing(t0) + self.forcing(t1))
        # calculate the non-linear (drag) terms at time step
        if self.nonlinear:
            b += self.dt * self._nonlinear_terms()
        # enforce the boundary conditions at the next time step
        if self.has_ob:
            b[self._iob_state] = self.boundary(t1)
        # solve for the state at the next time step
        self._s = self.solver(b)
        # add time step
        self.t = t1
        return self

    def run(self, N: int, record: bool = False):
        """Run the model for a number of time steps

        Parameters
        ----------
        N: int
            number of time steps to run the model
        record: bool, default False
            accumulate states for harmonic analysis
        """
        self.__stop__ = self.__index__ + N
        for _ in self:
            if record:
                self._accumulate()
        return self

    @property
    def state(self) -> np.ndarray:
        """Complete state vector on the full grid"""
        return self._expand(self._s)

    @property
    def u(self) -> np.ndarray:
        r"""Transports in the x-direction (m\ :sup:`2`/s)"""
        return self.state[: self.size].reshape(self.shape)

    @property
    def v(self) -> np.ndarray:
        r"""Transports in the y-direction (m\ :sup:`2`/s)"""
        return self.state[self.size : 2 * self.size].reshape(self.shape)

    @property
    def z(self) -> np.ndarray:
        """Elevations (m)"""
        return self.state[2 * self.size :].reshape(self.shape)

    def __str__(self):
        """String representation of the ``solve.ForwardModel`` object"""
        properties = ["pyTMD.solve.ForwardModel"]
        shape = ", ".join(map(str, self.shape))
        properties.append(f"    shape: {shape}")
        constituents = ", ".join(self.constituents)
        properties.append(f"    constituents: {constituents}")
        return "\n".join(properties)

    def __repr__(self):
        """Representation of the ``solve.ForwardModel`` object"""
        return self.__str__()

    def __iter__(self):
        """Iterate over time steps"""
        return self

    def __next__(self):
        """Calculate the fields at the next time step"""
        # halt run if the iterator is greater than the stop index
        if self.__index__ >= self.__stop__:
            raise StopIteration
        self.step()
        # add to counter
        self.__index__ += 1
        return self
