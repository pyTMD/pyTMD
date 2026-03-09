#!/usr/bin/env python
"""
interpolate.py
Written by Tyler Sutterley (03/2026)
Interpolators for spatial data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 03/2026: break up extrapolation into separate functions to allow
        for caching of the kd-tree when interpolating multiple variables
    Updated 02/2026: output data from extrapolate as an xarray DataArray
        where there are no valid points within the cutoff distance
    Updated 01/2026: output data from extrapolate as an xarray DataArray
    Updated 11/2025: calculate lambda function after nearest-neighbors
        set default data type for interpolation functions as input data type
        generalize vectorized 1D linear interpolation for more cases of fp
        allow iterating over variable with a recursion in interp1d
        drop prior interpolation functions to prefer xarray internals
    Updated 08/2025: added vectorized 1D linear interpolation function
        improve performance of bilinear interpolation and allow extrapolation
        added a penalized least square inpainting function to gap fill data
        standardized most variable names between interpolation functions
    Updated 09/2024: deprecation fix case where an array is output to scalars
    Updated 07/2024: changed projection flag in extrapolation to is_geographic
    Written 12/2022
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import scipy.fftpack
import scipy.spatial
import pyTMD.spatial

__all__ = [
    "interp1d",
    "inpaint",
    "extrapolate",
]


# PURPOSE: 1-dimensional linear interpolation on arrays
def interp1d(x: float | np.ndarray, xp: np.ndarray, fp: np.ndarray, **kwargs):
    """
    Vectorized one-dimensional linear interpolation

    Parameters
    ----------
    x: float | np.ndarray
        x-coordinate(s) of the interpolated values
    xp: np.ndarray
        x-coordinates of the data points
    fp: np.ndarray
        y-coordinates of the data points
    extrapolate: str, default = 'linear'
        Method of extrapolation

            - ``'linear'``
            - ``'nearest'``

    Returns
    -------
    f: np.ndarray
        Interpolated values at x
    """
    # get extrapolation method
    extrapolate = kwargs.get("extrapolate", "linear").lower()
    if extrapolate not in ("linear", "nearest"):
        raise ValueError(f"Invalid extrapolate method: {extrapolate}")
    # recursion for multiple x values
    if isinstance(x, np.ndarray) and (x.ndim > 0):
        # allocate output array
        f = np.zeros((*fp.shape[:-1], len(x)), dtype=fp.dtype)
        for i, val in enumerate(x):
            f[..., i] = interp1d(val, xp, fp, **kwargs)
        # return the array of interpolated values
        return f
    # clip coordinates to handle nearest-neighbor extrapolation
    if extrapolate == "nearest":
        x = np.clip(x, a_min=xp.min(), a_max=xp.max())
    # find indice where x could be inserted into xp
    j = np.searchsorted(xp, x) - 1
    # clip indices to handle linear extrapolation
    if extrapolate == "linear":
        j = np.clip(j, a_min=0, a_max=len(xp) - 2)
    # fractional distance between points
    d = np.divide(x - xp[j], xp[j + 1] - xp[j])
    # calculate interpolated values
    f = (1.0 - d) * fp[..., j] + d * fp[..., j + 1]
    # return the interpolated values
    return f


def inpaint(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    N: int = 0,
    s0: int = 3,
    power: int = 2,
    epsilon: float = 2.0,
    **kwargs,
):
    """
    Inpaint over missing data in a two-dimensional array using a
    penalized least square method based on discrete cosine transforms
    :cite:p:`Garcia:2010hn,Wang:2012ei`

    Parameters
    ----------
    xs: np.ndarray
        input x-coordinates
    ys: np.ndarray
        input y-coordinates
    zs: np.ndarray
        input data
    N: int, default 0
        Number of iterations (0 for nearest neighbors)
    s0: int, default 3
        Smoothing
    power: int, default 2
        power for lambda function
    epsilon: float, default 2.0
        relaxation factor
    """
    # find masked values
    if isinstance(zs, np.ma.MaskedArray):
        W = np.logical_not(zs.mask)
    else:
        W = np.isfinite(zs)
    # no valid values can be found
    if not np.any(W):
        raise ValueError("No valid values found")

    # dimensions of input grid
    ny, nx = np.shape(zs)

    # calculate initial values using nearest neighbors
    # computation of distance Matrix
    # use scipy spatial KDTree routines
    xgrid, ygrid = np.meshgrid(xs, ys)
    tree = scipy.spatial.cKDTree(np.c_[xgrid[W], ygrid[W]])
    # find nearest neighbors
    masked = np.logical_not(W)
    _, ii = tree.query(np.c_[xgrid[masked], ygrid[masked]], k=1)
    # copy valid original values
    z0 = np.zeros((ny, nx), dtype=zs.dtype)
    z0[W] = np.copy(zs[W])
    # copy nearest neighbors
    z0[masked] = zs[W][ii]
    # return nearest neighbors interpolation
    if N == 0:
        return z0

    # copy data to new array with 0 values for mask
    ZI = np.zeros((ny, nx), dtype=zs.dtype)
    ZI[W] = np.copy(z0[W])

    # calculate lambda function
    L = np.zeros((ny, nx))
    L += np.broadcast_to(np.cos(np.pi * np.arange(ny) / ny)[:, None], (ny, nx))
    L += np.broadcast_to(np.cos(np.pi * np.arange(nx) / nx)[None, :], (ny, nx))
    LAMBDA = np.power(2.0 * (2.0 - L), power)

    # smoothness parameters
    s = np.logspace(s0, -6, N)
    for i in range(N):
        # calculate discrete cosine transform
        GAMMA = 1.0 / (1.0 + s[i] * LAMBDA)
        DISCOS = GAMMA * scipy.fftpack.dctn(W * (ZI - z0) + z0, norm="ortho")
        # update interpolated grid
        z0 = (
            epsilon * scipy.fftpack.idctn(DISCOS, norm="ortho")
            + (1.0 - epsilon) * z0
        )

    # reset original values
    z0[W] = np.copy(zs[W])
    # return the inpainted grid
    return z0


# PURPOSE: Nearest-neighbor extrapolation of valid data to output data
def extrapolate(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    fill_value: float = None,
    cutoff: int | float = np.inf,
    is_geographic: bool = True,
    **kwargs,
):
    """
    Nearest-neighbor (NN) extrapolation of valid model data using `KD-trees
    <https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.spatial.cKDTree.html>`_

    Parameters
    ----------
    xs: np.ndarray
        x-coordinates of tidal model
    ys: np.ndarray
        y-coordinates of tidal model
    zs: np.ndarray
        tide model data
    X: np.ndarray
        output x-coordinates
    Y: np.ndarray
        output y-coordinates
    fill_value: float, default np.nan
        invalid value
    dtype: np.dtype, default np.float64
        output data type
    cutoff: float, default np.inf
        return only neighbors within distance [km]

        Set to ``np.inf`` to extrapolate for all points
    is_geographic: bool, default True
        input grid is in geographic coordinates

    Returns
    -------
    DATA: np.ndarray
        interpolated data
    """
    # set geographic flag if using old EPSG projection keyword
    if hasattr(kwargs, "EPSG") and (kwargs["EPSG"] == "4326"):
        is_geographic = True
    # calculate meshgrid of model coordinates
    gridx, gridy = np.meshgrid(xs, ys)
    # find valid values
    if isinstance(zs, np.ma.MaskedArray):
        indy, indx = np.nonzero(np.logical_not(zs.mask))
    else:
        indy, indx = np.nonzero(np.isfinite(zs))
    # reduce to valid original values
    x0 = gridx[indy, indx]
    y0 = gridy[indy, indx]
    z0 = zs[indy, indx]
    # verify output dimensions
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    # extrapolate valid data values to data
    npts = len(X)
    # return none if no invalid points
    if npts == 0:
        return
    # calculate coordinates for nearest-neighbors
    p_in = _to_cartesian(x0, y0, is_geographic=is_geographic)
    p_out = _to_cartesian(X, Y, is_geographic=is_geographic)
    # create KD-tree of valid points
    tree = _build_tree(p_in)
    # query output data points and find nearest neighbor within cutoff
    data = _nearest_neighbors(
        tree, p_out, z0, cutoff=cutoff, fill_value=fill_value, **kwargs
    )
    # return the extrapolated data
    return data


def _to_cartesian(
    x: np.ndarray,
    y: np.ndarray,
    is_geographic: bool = True,
):
    """
    Convert input coordinates to an array of points in a
    Cartesian coordinate system

    Parameters
    ----------
    x: np.ndarray
        x-coordinates to be converted
    y: np.ndarray
        y-coordinates to be converted
    is_geographic: bool, default True
        coordinates are geographic

    Returns
    -------
    points: np.ndarray
        output points in Cartesian coordinates
    """
    # verify output dimensions
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    # calculate coordinates for nearest-neighbors
    if is_geographic:
        # global or regional equirectangular model
        # ellipsoidal major axis in kilometers
        a_axis = 6378.137
        # calculate Cartesian coordinates of input grid
        xi, yi, zi = pyTMD.spatial.to_cartesian(x, y, a_axis=a_axis)
        # calculate Cartesian coordinates of output coordinates
        points = np.c_[xi, yi, zi]
    else:
        points = np.c_[x, y]
    # return the output points in Cartesian coordinates
    return points


def _build_tree(points: np.ndarray, **kwargs):
    """
    Build a KD-tree to search for the nearest-neighbors (NN)

    Parameters
    ----------
    points: np.ndarray
        input points in Cartesian coordinates
    kwargs: dict
        additional keyword arguments for scipy.spatial.cKDTree

    Returns  
    -------  
    tree: scipy.spatial.cKDTree  
        KD-tree from input points  
    """
    # create KD-tree of points for nearest-neighbor extrapolation
    tree = scipy.spatial.cKDTree(points, **kwargs)
    return tree


def _nearest_neighbors(
    tree: scipy.spatial.cKDTree,
    points: np.ndarray,
    flattened: np.ndarray,
    cutoff: int | float = np.inf,
    fill_value: float = None,
    **kwargs,
):
    """
    Nearest-neighbor (NN) extrapolation of valid model data using KD-trees

    Parameters
    ----------
    tree: scipy.spatial.cKDTree
        KD-tree of valid points for nearest-neighbor extrapolation
    points: np.ndarray
        output points in Cartesian coordinates
    flattened: np.ndarray
        valid data array to be extrapolated
    cutoff: float, default np.inf
        return only neighbors within distance [km]
    fill_value: float, default None
        invalid value
    dtype: np.dtype, default np.float64
        output data type
    """
    # set default data type
    dtype = kwargs.get("dtype", flattened.dtype)
    # number of data points
    npts, _ = points.shape
    # query output data points and find nearest neighbor within cutoff
    dd, ii = tree.query(points, k=1, distance_upper_bound=cutoff)
    # allocate to output extrapolate data array
    data = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    data.mask = np.ones((npts), dtype=bool)
    # initially set all data to fill value
    data.data[:] = data.fill_value
    # spatially extrapolate using nearest neighbors
    if np.any(np.isfinite(dd)):
        (ind,) = np.nonzero(np.isfinite(dd))
        data.data[ind] = flattened[ii[ind]]
        data.mask[ind] = False
    # return extrapolated values
    return xr.DataArray(data)
