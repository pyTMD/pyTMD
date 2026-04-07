"""
test_triangle.py
Tests the interpolation of triangular elements
"""

import pytest
import pyTMD
import numpy as np


# PURPOSE: test first and second order shape functions
@pytest.mark.parametrize("order", [1, 2], ids=["linear", "quadratic"])
def test_shape_functions(order):
    # vertices of an equilateral triangle
    xv = np.array([-0.5, 0.5, 0.0])
    yv = np.array([0.0, 0.0, np.sqrt(3.0 / 4.0)])
    # check that shape functions match expected
    nodes = 3 * order
    N = [None] * (nodes)
    for i in range(nodes):
        # calculate node coordinates
        if order == 1:
            # linear shape functions
            # nodes are vertices
            x = np.copy(xv[i])
            y = np.copy(yv[i])
        elif order == 2:
            # quadratic shape functions
            if np.mod(i, 2) == 0:
                # even nodes are vertices
                j = int(i // 2)
                x = np.copy(xv[j])
                y = np.copy(yv[j])
            elif np.mod(i, 2) == 1:
                # odd nodes are midpoints of edges
                j1 = np.mod(int((i - 1) // 2), 3)
                j2 = np.mod(int((i + 1) // 2), 3)
                x = np.mean([xv[j1], xv[j2]])
                y = np.mean([yv[j1], yv[j2]])
        # calculate barycentric coordinates
        xi, eta = pyTMD.interpolate._to_barycentric(xv, yv, x, y)
        # coordinate should be in the triangle
        assert pyTMD.interpolate._inside_triangle(xi, eta)
        # get shape functions for polynomial order
        N = pyTMD.interpolate._shape_functions(xi, eta, order)
        # sum of shape functions should be 1
        assert np.isclose(np.sum(N), 1.0, atol=1e-10)
        # shape function at the node should be 1
        assert np.isclose(N[i], 1.0, atol=1e-10)
        # shape functions at other nodes should be 0
        comp = [N[o] for o in range(nodes) if o != i]
        assert np.allclose(comp, 0.0, atol=1e-10)


# PURPOSE: test barycentric interpolation of a simple triangle
@pytest.mark.parametrize("order", [1, 2], ids=["linear", "quadratic"])
def test_simple_barycentric(order):
    # vertices of an equilateral triangle
    xv = np.array([-0.5, 0.5, 0.0])
    yv = np.array([0.0, 0.0, np.sqrt(3.0 / 4.0)])
    # random values at the nodes
    nodes = 3 * order
    ze = -1.0 + 2.0 * np.random.rand(nodes)
    # test point at the triangle centroid
    x, y = 0.0, np.sqrt(1.0 / 12.0)
    # calculate value using linear barycentric interpolation
    z = pyTMD.interpolate.barycentric(xv, yv, ze, x, y, order=order)
    # calculate weights
    weights = np.zeros((nodes))
    if order == 1:
        # linear weights for a point at the centroid
        weights[:] = 1.0
    elif order == 2:
        # quadratic weights for a point at the centroid
        weights[0::2] = -1.0
        weights[1::2] = 4.0
    # calculate expected values
    exp = np.sum(weights * ze) / np.sum(weights)
    # check that interpolated value matches expected
    assert np.isclose(z.values, exp, atol=1e-10)


# PURPOSE: check the winding number of triangles
def test_winding_number():
    # vertices of an equilateral triangle
    # counter-clockwise direction (winding is positive)
    xv = np.array([-0.5, 0.5, 0.0])
    yv = np.array([0.0, 0.0, np.sqrt(3.0 / 4.0)])
    wind = pyTMD.interpolate._winding_number(xv, yv)
    assert wind >= 0
    # vertices of an equilateral triangle
    # clockwise direction (winding is negative)
    xv = np.array([0.5, -0.5, 0.0])
    yv = np.array([0.0, 0.0, np.sqrt(3.0 / 4.0)])
    wind = pyTMD.interpolate._winding_number(xv, yv)
    assert wind < 0
    # fix vertices to cross "meridian"
    xv[xv < 0] += 1.0
    wind = pyTMD.interpolate._winding_number(xv, yv)
    assert wind >= 0
