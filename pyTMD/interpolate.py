#!/usr/bin/env python
u"""
interpolate.py
Written by Tyler Sutterley (11/2025)
Interpolators for spatial data

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/

UPDATE HISTORY:
    Updated 11/2025: calculate lambda function after nearest-neighbors
        set default data type for interpolation functions as input data type
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
import scipy.fftpack
import scipy.spatial
import scipy.interpolate
import pyTMD.spatial

__all__ = [
    "interp1d",
    "inpaint",
    "bilinear",
    "spline",
    "regulargrid",
    "extrapolate",
    "_distance"
]

# PURPOSE: 1-dimensional linear interpolation on arrays
def interp1d(
        x: float,
        xp: np.ndarray,
        fp: np.ndarray,
        extrapolate: str = 'linear',
        **kwargs
    ):
    """
    Vectorized one-dimensional linear interpolation

    Parameters
    ----------
    x: np.ndarray
        x-coordinates of the interpolated values
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
    # clip coordinates to handle nearest-neighbor extrapolation
    if (extrapolate == 'nearest'):
        x = np.clip(x, a_min=xp.min(), a_max=xp.max())
    # find indice where x could be inserted into xp
    j = np.searchsorted(xp, x) - 1
    # clip indices to handle linear extrapolation
    if (extrapolate == 'linear'):
        j = np.clip(j, a_min=0, a_max=len(xp) - 2)
    # fractional distance between points
    d = np.divide(x - xp[j], xp[j+1] - xp[j])
    # calculate interpolated values
    f = (1.0 - d)*fp[:,j] + d*fp[:,j+1]
    return f

def inpaint(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        N: int = 0,
        s0: int = 3,
        power: int = 2,
        epsilon: float = 2.0,
        **kwargs
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
        raise ValueError('No valid values found')

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
    if (N == 0):
        return z0

    # copy data to new array with 0 values for mask
    ZI = np.zeros((ny, nx), dtype=zs.dtype)
    ZI[W] = np.copy(z0[W])

    # calculate lambda function
    L = np.zeros((ny, nx))
    L += np.broadcast_to(np.cos(np.pi*np.arange(ny)/ny)[:, None], (ny, nx))
    L += np.broadcast_to(np.cos(np.pi*np.arange(nx)/nx)[None, :], (ny, nx))
    LAMBDA = np.power(2.0*(2.0 - L), power)

    # smoothness parameters
    s = np.logspace(s0, -6, N)
    for i in range(N):
        # calculate discrete cosine transform
        GAMMA = 1.0/(1.0 + s[i]*LAMBDA)
        DISCOS = GAMMA*scipy.fftpack.dctn(W*(ZI - z0) + z0, norm='ortho')
        # update interpolated grid
        z0 = epsilon*scipy.fftpack.idctn(DISCOS, norm='ortho') + \
            (1.0 - epsilon)*z0

    # reset original values
    z0[W] = np.copy(zs[W])
    # return the inpainted grid
    return z0

# PURPOSE: bilinear interpolation of input data to output data
def bilinear(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        fill_value: float = np.nan,
        extrapolate: bool = False,
        **kwargs
    ):
    """
    Bilinear interpolation of input data to output coordinates

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
    extrapolate: bool, default False
        allow linear extrapolation of points
    dtype: np.dtype, default np.float64
        output data type

    Returns
    -------
    data: np.ndarray
        interpolated data
    """
    # set default data type
    dtype = kwargs.get('dtype', zs.dtype)
    # verify that input data is masked array
    if not isinstance(zs, np.ma.MaskedArray):
        zs = np.ma.array(zs)
        zs.mask = np.zeros_like(zs, dtype=bool)
    # interpolate gridded data values to data
    npts = len(X)
    # allocate to output interpolated data array
    data = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    data.mask = np.ones((npts), dtype=bool)
    # initially set all data to fill value
    data.data[:] = data.fill_value
    # for each point
    for i in range(npts):
        # calculating the indices for the original grid
        ix = np.searchsorted(xs, X[i]) - 1
        iy = np.searchsorted(ys, Y[i]) - 1
        # check that all points are within valid bounds
        bounds = (ix >= 0) & (iy >= 0) & (ix < len(xs)) & (iy < len(ys))
        if not (extrapolate or bounds):
            continue
        # clip to handle extrapolation
        ix = np.clip(ix, a_min=0, a_max=len(xs) - 2)
        iy = np.clip(iy, a_min=0, a_max=len(ys) - 2)
        # corner data values for adjacent grid cells
        IM = np.ma.zeros((4), fill_value=fill_value, dtype=dtype)
        IM.mask = np.ones((4), dtype=bool)
        # corner weight values for adjacent grid cells
        WM = np.zeros((4))
        # build data and weight arrays
        for j,XI,YI in zip([0,1,2,3],[ix,ix+1,ix,ix+1],[iy,iy,iy+1,iy+1]):
            IM.data[j] = zs.data[YI,XI].astype(dtype)
            IM.mask[j] = zs.mask[YI,XI]
            WM[3-j] = np.abs(X[i]-xs[XI])*np.abs(Y[i]-ys[YI])
        # if on corner value: use exact
        if (np.isclose(Y[i],ys[iy]) & np.isclose(X[i],xs[ix])):
            data.data[i] = zs.data[iy,ix].astype(dtype)
            data.mask[i] = zs.mask[iy,ix]
        elif (np.isclose(Y[i],ys[iy+1]) & np.isclose(X[i],xs[ix])):
            data.data[i] = zs.data[iy+1,ix].astype(dtype)
            data.mask[i] = zs.mask[iy+1,ix]
        elif (np.isclose(Y[i],ys[iy]) & np.isclose(X[i],xs[ix+1])):
            data.data[i] = zs.data[iy,ix+1].astype(dtype)
            data.mask[i] = zs.mask[iy,ix+1]
        elif (np.isclose(Y[i],ys[iy+1]) & np.isclose(X[i],xs[ix+1])):
            data.data[i] = zs.data[iy+1,ix+1].astype(dtype)
            data.mask[i] = zs.mask[iy+1,ix+1]
        elif np.any(np.isfinite(IM) & (~IM.mask)):
            # find valid indices for data summation and weight matrix
            ii, = np.nonzero(np.isfinite(IM) & (~IM.mask))
            # calculate interpolated value for i
            data.data[i] = np.sum(WM[ii]*IM[ii])/np.sum(WM[ii])
            data.mask[i] = np.all(IM.mask[ii])
    # return interpolated values
    return data

def spline(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        fill_value: float = None,
        reducer=np.ceil,
        **kwargs
    ):
    """
    `Bivariate spline interpolation
    <https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.interpolate.RectBivariateSpline.html>`_
    of input data to output coordinates

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
    fill_value: float or NoneType, default None
        invalid value
    dtype: np.dtype, default np.float64
        output data type
    reducer: obj, default np.ceil
        operation for converting mask to boolean
    kx: int, default 1
        degree of the bivariate spline in the x-dimension
    ky: int, default 1
        degree of the bivariate spline in the y-dimension
    **kwargs: dict
        additional arguments for ``scipy.interpolate.RectBivariateSpline``

    Returns
    -------
    data: np.ndarray
        interpolated data
    """
    # set default keyword arguments
    kwargs.setdefault('kx', 1)
    kwargs.setdefault('ky', 1)
    # set default data type
    dtype = kwargs.get('dtype', zs.dtype)
    # verify that input data is masked array
    if not isinstance(zs, np.ma.MaskedArray):
        zs = np.ma.array(zs)
        zs.mask = np.zeros_like(zs, dtype=bool)
    # interpolate gridded data values to data
    npts = len(X)
    # allocate to output interpolated data array
    data = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    data.mask = np.ones((npts), dtype=bool)
    # construct splines for input data and mask
    if np.iscomplexobj(zs):
        s1 = scipy.interpolate.RectBivariateSpline(xs, ys,
            zs.data.real.T, **kwargs)
        s2 = scipy.interpolate.RectBivariateSpline(xs, ys,
            zs.data.imag.T, **kwargs)
        s3 = scipy.interpolate.RectBivariateSpline(xs, ys,
            zs.mask.T, **kwargs)
        # evaluate the spline at input coordinates
        data.data.real[:] = s1.ev(X, Y)
        data.data.imag[:] = s2.ev(X, Y)
        data.mask[:] = reducer(s3.ev(X, Y)).astype(bool)
    else:
        s1 = scipy.interpolate.RectBivariateSpline(xs, ys,
            zs.data.T, **kwargs)
        s2 = scipy.interpolate.RectBivariateSpline(xs, ys,
            zs.mask.T, **kwargs)
        # evaluate the spline at input coordinates
        data.data[:] = s1.ev(X, Y).astype(dtype)
        data.mask[:] = reducer(s2.ev(X, Y)).astype(bool)
    # return interpolated values
    return data

def regulargrid(
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        fill_value: float = None,
        reducer=np.ceil,
        **kwargs
    ):
    """
    `Regular grid interpolation
    <https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.interpolate.RegularGridInterpolator.html>`_
    of input data to output coordinates

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
    fill_value: float or NoneType, default None
        invalid value
    dtype: np.dtype, default np.float64
        output data type
    reducer: obj, default np.ceil
        operation for converting mask to boolean
    bounds_error: bool, default False
        raise Exception when values are requested outside domain
    method: str, default 'linear'
        Method of interpolation

            - ``'linear'``
            - ``'nearest'``
            - ``'slinear'``
            - ``'cubic'``
            - ``'quintic'``
    **kwargs: dict
        additional arguments for ``scipy.interpolate.RegularGridInterpolator``

    Returns
    -------
    data: np.ndarray
        interpolated data
    """
    # set default keyword arguments
    kwargs.setdefault('bounds_error', False)
    kwargs.setdefault('method', 'linear')
    # set default data type
    dtype = kwargs.get('dtype', zs.dtype)
    # verify that input data is masked array
    if not isinstance(zs, np.ma.MaskedArray):
        zs = np.ma.array(zs)
        zs.mask = np.zeros_like(zs, dtype=bool)
    # interpolate gridded data values to data
    npts = len(X)
    # allocate to output interpolated data array
    data = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    data.mask = np.ones((npts), dtype=bool)
    # use scipy regular grid to interpolate values for a given method
    r1 = scipy.interpolate.RegularGridInterpolator((ys, xs),
        zs.data, fill_value=fill_value, **kwargs)
    r2 = scipy.interpolate.RegularGridInterpolator((ys, xs),
        zs.mask, fill_value=1, **kwargs)
    # evaluate the interpolator at input coordinates
    data.data[:] = r1.__call__(np.c_[Y, X])
    data.mask[:] = reducer(r2.__call__(np.c_[Y, X])).astype(bool)
    # return interpolated values
    return data

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
        **kwargs
    ):
    """
    Nearest-neighbor (`NN`) extrapolation of valid model data using `kd-trees
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
    # set default data type
    dtype = kwargs.get('dtype', zs.dtype)
    # verify that input data is masked array
    if not isinstance(zs, np.ma.MaskedArray):
        zs = np.ma.array(zs, dtype=zs.dtype,
            fill_value=np.ma.default_fill_value(zs.dtype))
        zs.mask = np.isnan(zs)
    # set geographic flag if using old EPSG projection keyword
    if hasattr(kwargs, 'EPSG') and (kwargs['EPSG'] == '4326'):
        is_geographic = True
    # verify output dimensions
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    # extrapolate valid data values to data
    npts = len(X)
    # return none if no invalid points
    if (npts == 0):
        return

    # allocate to output extrapolate data array
    data = np.ma.zeros((npts), dtype=dtype, fill_value=fill_value)
    data.mask = np.ones((npts), dtype=bool)
    # initially set all data to fill value
    data.data[:] = zs.fill_value

    # create combined valid mask
    valid_mask = (~zs.mask) & np.isfinite(zs.data)
    # reduce to model points within bounds of input points
    valid_bounds = np.ones_like(zs.mask, dtype=bool)

    # calculate coordinates for nearest-neighbors
    if is_geographic:
        # global or regional equirectangular model
        # calculate meshgrid of model coordinates
        gridlon, gridlat = np.meshgrid(xs, ys)
        # ellipsoidal major axis in kilometers
        a_axis = 6378.137
        # calculate Cartesian coordinates of input grid
        gridx, gridy, gridz = pyTMD.spatial.to_cartesian(
            gridlon, gridlat, a_axis=a_axis)
        # calculate Cartesian coordinates of output coordinates
        XI, YI, ZI = pyTMD.spatial.to_cartesian(
            X, Y, a_axis=a_axis)
        # range of output points in cartesian coordinates
        xmin, xmax = (np.min(XI), np.max(XI))
        ymin, ymax = (np.min(YI), np.max(YI))
        zmin, zmax = (np.min(ZI), np.max(ZI))
        # reduce to model points within bounds of input points
        valid_bounds = np.ones_like(zs.mask, dtype=bool)
        valid_bounds &= (gridx >= (xmin - 2.0*cutoff))
        valid_bounds &= (gridx <= (xmax + 2.0*cutoff))
        valid_bounds &= (gridy >= (ymin - 2.0*cutoff))
        valid_bounds &= (gridy <= (ymax + 2.0*cutoff))
        valid_bounds &= (gridz >= (zmin - 2.0*cutoff))
        valid_bounds &= (gridz <= (zmax + 2.0*cutoff))
        # check if there are any valid points within the input bounds
        if not np.any(valid_mask & valid_bounds):
            # return filled masked array
            return data
        # find where input grid is valid and close to output points
        indy, indx = np.nonzero(valid_mask & valid_bounds)
        # create KD-tree of valid points
        tree = scipy.spatial.cKDTree(np.c_[gridx[indy, indx],
            gridy[indy, indx], gridz[indy, indx]])
        # flattened valid data array
        flattened = zs.data[indy, indx]
        # output coordinates
        points = np.c_[XI, YI, ZI]
    else:
        # projected model
        # calculate meshgrid of model coordinates
        gridx, gridy = np.meshgrid(xs, ys)
        # range of output points
        xmin, xmax = (np.min(X), np.max(X))
        ymin, ymax = (np.min(Y), np.max(Y))
        # reduce to model points within bounds of input points
        valid_bounds = np.ones_like(zs.mask, dtype=bool)
        valid_bounds &= (gridx >= (xmin - 2.0*cutoff))
        valid_bounds &= (gridx <= (xmax + 2.0*cutoff))
        valid_bounds &= (gridy >= (ymin - 2.0*cutoff))
        valid_bounds &= (gridy <= (ymax + 2.0*cutoff))
        # check if there are any valid points within the input bounds
        if not np.any(valid_mask & valid_bounds):
            # return filled masked array
            return data
        # find where input grid is valid and close to output points
        indy, indx = np.nonzero(valid_mask & valid_bounds)
        # flattened model coordinates
        tree = scipy.spatial.cKDTree(np.c_[gridx[indy, indx],
            gridy[indy, indx]])
        # flattened valid data array
        flattened = zs.data[indy, indx]
        # output coordinates
        points = np.c_[X, Y]

    # query output data points and find nearest neighbor within cutoff
    dd, ii = tree.query(points, k=1, distance_upper_bound=cutoff)
    # spatially extrapolate using nearest neighbors
    if np.any(np.isfinite(dd)):
        ind, = np.nonzero(np.isfinite(dd))
        data.data[ind] = flattened[ii[ind]]
        data.mask[ind] = False
    # return extrapolated values
    return data

# PURPOSE: calculate Euclidean distances between points
def _distance(c1: np.ndarray, c2: np.ndarray):
    """
    Calculate Euclidean distances between points

    Parameters
    ----------
    c1: np.ndarray
        first set of coordinates
    c2: np.ndarray
        second set of coordinates

    Returns
    -------
    c: np.ndarray
        Euclidean distance
    """
    # decompose Euclidean distance: (x-y)^2 = x^2 - 2xy + y^2
    dx2 = np.sum(c1**2)
    dxy = np.dot(c1[np.newaxis,:], c2.T)
    dy2 = np.sum(c2**2, axis=1)
    # calculate Euclidean distance
    D, = np.sqrt(dx2 - 2.0*dxy + dy2)
    return D
