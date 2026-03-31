#!/usr/bin/env python
"""
interpolate.py
Written by Tyler Sutterley (04/2026)
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
    Updated 04/2026: add 1st and 2nd order barycentric interpolation function
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
    "_to_cartesian",
    "_build_tree",
    "_nearest_neighbors",
    "barycentric",
    "_to_barycentric",
    "_inside_triangle",
    "_shape_functions",
]


# PURPOSE: 1-dimensional linear interpolation on arrays
def interp1d(
    x: float | np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    **kwargs,
):
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
        Interpolated values at new coordinates
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
    penalized least-squares method based on discrete cosine transforms
    :cite:p:`Garcia:2010hn,Wang:2012ei`

    Parameters
    ----------
    xs: np.ndarray
        x-coordinates
    ys: np.ndarray
        y-coordinates
    zs: np.ndarray
        Data with masked values
    N: int, default 0
        Number of iterations (0 for nearest neighbors)
    s0: int, default 3
        Smoothing factor
    power: int, default 2
        Power for lambda function
    epsilon: float, default 2.0
        Relaxation factor

    Returns
    -------
    z0: np.ndarray
        Data with inpainted (filled) values
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
        Tide model data
    X: np.ndarray
        Output x-coordinates
    Y: np.ndarray
        Output y-coordinates
    fill_value: float, default np.nan
        Invalid value
    dtype: np.dtype, default np.float64
        Output data type
    cutoff: float, default np.inf
        Return only neighbors within distance (kilometers)

        Set to ``np.inf`` to extrapolate for all points
    is_geographic: bool, default True
        Input grid is in geographic coordinates

    Returns
    -------
    data: xr.DataArray
        Interpolated data
    """
    # set geographic flag if using old EPSG projection keyword
    if hasattr(kwargs, "EPSG") and (kwargs["EPSG"] == "4326"):
        is_geographic = True
    # calculate meshgrid of model coordinates
    gridx, gridy = np.meshgrid(xs, ys)
    # find valid values
    if isinstance(zs, np.ma.MaskedArray):
        indy, indx = np.nonzero(np.logical_not(zs.mask) & np.isfinite(zs.data))
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
        Coordinates are geographic

    Returns
    -------
    points: np.ndarray
        Output points in Cartesian coordinates
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
        Input points in Cartesian coordinates
    kwargs: dict
        Additional keyword arguments for ``scipy.spatial.cKDTree``

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
        Output points in Cartesian coordinates
    flattened: np.ndarray
        Valid data array to be extrapolated
    cutoff: float, default np.inf
        Return only neighbors within distance (kilometers)
    fill_value: float, default None
        Invalid value
    dtype: np.dtype, default from input data
        Output data type

    Returns
    -------
    data: xr.DataArray
        Extrapolated data
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


def barycentric(
    xv: np.ndarray,
    yv: np.ndarray,
    ze: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    order: int = 1,
    **kwargs,
):
    """
    Interpolation of unstructured model data using a barycentric
    method

    Parameters
    ----------
    xv: np.ndarray
        x-coordinates of triangle vertices
    yv: np.ndarray
        y-coordinates of triangle vertices
    ze: np.ndarray
        Unstructured model data at elements
    x: np.ndarray
        Output x-coordinates
    y: np.ndarray
        Output y-coordinates
    order: int, default 1
        Polynomial order of the triangular elements

        - ``1``: linear
        - ``2``: quadratic

    Returns
    -------
    data: xr.DataArray
        Interpolated data
    """
    # set default data type
    dtype = kwargs.get("dtype", ze.dtype)
    # convert to barycentric coordinates
    xi, eta = _to_barycentric(xv, yv, x, y)
    # check if inside polygon
    valid = _inside_triangle(xi, eta)
    # get shape functions for order
    N = _shape_functions(xi, eta, order)
    # allocate to output extrapolate data array
    data = np.zeros_like(x, dtype=dtype)
    for p, sf in enumerate(N):
        data += sf * valid * ze[..., p]
    # return the interpolated value
    return xr.DataArray(data)


def _to_barycentric(
    xv: np.ndarray,
    yv: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
):
    """
    Convert coordinates to barycentric space

    Parameters
    ----------
    xv: np.ndarray
        x-coordinates of triangle vertices
    yv: np.ndarray
        y-coordinates of triangle vertices
    x: np.ndarray
        Output x-coordinates
    y: np.ndarray
        Output y-coordinates

    Returns
    -------
    xi: np.ndarray
        Normalized barycentric (areal) xi-coordinates
    eta: np.ndarray
        Normalized barycentric (areal) eta-coordinates
    """
    # calculate triangle area
    A = 0.5 * (
        xv[0] * (yv[1] - yv[2])
        + xv[1] * (yv[2] - yv[0])
        + xv[2] * (yv[0] - yv[1])
    )
    # calculate Jacobian
    # ignore divide by zero and invalid value warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        J = 1.0 / (2.0 * A)
    # mapping into barycentric coordinates
    xi = J * (
        (xv[1] * yv[2] - xv[2] * yv[1])
        + (yv[1] - yv[2]) * x
        + (xv[2] - xv[1]) * y
    )
    eta = J * (
        (xv[2] * yv[0] - xv[0] * yv[2])
        + (yv[2] - yv[0]) * x
        + (xv[0] - xv[2]) * y
    )
    # return the barycentric coordinates
    return xi, eta


def _inside_triangle(
    xi: np.ndarray,
    eta: np.ndarray,
    atol: float = 1e-8,
):
    """
    Check if point is within the triangular area

    Parameters
    ----------
    xi: np.ndarray
        Normalized barycentric (areal) xi-coordinates
    eta: np.ndarray
        Normalized barycentric (areal) eta-coordinates
    atol: float = 1e-8
        Absolute tolerance parameter

    Returns
    -------
    valid: np.ndarray
        Mask for coordinates
    """
    # simple check to see if areas are valid
    la = 1.0 - eta - xi
    # all barycentric coordinates should be within 0 to 1
    # and have valid Jacobians (not dividing by 0)
    valid = (
        (np.isfinite(xi) & np.isfinite(eta))
        & (la >= (0.0 - atol))
        & (la <= (1.0 + atol))
        & (xi >= (0.0 - atol))
        & (xi <= (1.0 + atol))
        & (eta >= (0.0 - atol))
        & (eta <= (1.0 + atol))
    )
    return valid


def _shape_functions(xi: np.ndarray, eta: np.ndarray, order: int):
    """
    Get the interpolating shape functions for a polynomial order

    Parameters
    ----------
    xi: np.ndarray
        Normalized barycentric (areal) xi-coordinates
    eta: np.ndarray
        Normalized barycentric (areal) eta-coordinates
    order: int
        Polynomial order of the triangular elements

        - ``1``: linear
        - ``2``: quadratic

    Returns
    -------
    N: list
        Shape functions in barycentric space
    """
    # shape functions in barycentric space
    N = [None] * (3 * order)
    if order == 1:
        # 1st order terms: linear triangular elements
        N[0] = xi
        N[1] = eta
        N[2] = 1.0 - eta - xi
    elif order == 2:
        # 2nd order terms: quadratic triangular elements
        N[0] = xi * (2.0 * xi - 1.0)
        N[1] = 4.0 * xi * eta
        N[2] = eta * (2.0 * eta - 1.0)
        N[3] = 4.0 * eta * (1.0 - xi - eta)
        N[4] = (1.0 - xi - eta) * (1.0 - 2.0 * xi - 2.0 * eta)
        N[5] = 4.0 * xi * (1.0 - xi - eta)
    else:
        raise ValueError(f"Unsupported polynomial order {order}")
    # return the shape functions
    return N
