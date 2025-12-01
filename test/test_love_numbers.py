#!/usr/bin/env python
u"""
test_love_numbers.py (03/2025)
Verify Body Tide Love numbers for different constituents

UPDATE HISTORY:
    Updated 03/2025: added ratio check for different models
    Updated 11/2024: moved love number calculator to arguments
    Written 09/2024
"""
import pytest
import numpy as np
import pyTMD.constituents

def test_love_numbers():
    """
    Tests the calculation of body tide Love numbers compared
    with the 1066A values from Wahr et al. (1981)
    """
    # expected values
    exp = {}
    # diurnal species
    exp['2q1'] = (0.604, 0.298, 0.0841)
    exp['sigma1'] = (0.604, 0.298, 0.0841)
    exp['q1'] = (0.603, 0.298, 0.0841)
    exp['rho1'] = (0.603, 0.298, 0.0841)
    exp['o1'] = (0.603, 0.298, 0.0841)
    exp['tau1'] = (0.603, 0.298, 0.0842)
    exp['m1'] = (0.600, 0.297, 0.0842)
    exp['chi1'] = (0.600, 0.296, 0.0843)
    exp['pi1'] = (0.587, 0.290, 0.0847)
    exp['p1'] = (0.581, 0.287, 0.0849)
    exp['s1'] = (0.568, 0.280, 0.0853)
    exp['k1'] = (0.520, 0.256, 0.0868)
    exp['psi1'] = (0.937, 0.466, 0.0736)
    exp['phi1'] = (0.662, 0.328, 0.0823)
    exp['theta1'] = (0.612, 0.302, 0.0839)
    exp['j1'] = (0.611, 0.302, 0.0839)
    exp['so1'] = (0.608, 0.301, 0.0840)
    exp['oo1'] = (0.608, 0.301, 0.0840)
    exp['ups1'] = (0.607, 0.300, 0.0840)
    # semi-diurnal species
    exp['m2'] = (0.609, 0.302, 0.0852)
    # long-period species
    exp['mm'] = (0.606, 0.299, 0.0840)
    # for each tidal constituent
    for c, v in exp.items():
        # calculate Love numbers
        omega, = pyTMD.constituents.frequency(c)
        h2, k2, l2 = pyTMD.constituents._love_numbers(
            omega, model='1066A')
        # check Love numbers
        assert np.isclose(h2, v[0], atol=15e-4)
        assert np.isclose(k2, v[1], atol=15e-4)
        assert np.isclose(l2, v[2], atol=15e-4)

def test_complex_love_numbers():
    """
    Tests the calculation of complex body tide Love numbers
    for long-period constituents compared with the values from
    Mathews et al. (1997)
    """
    # Doodson coefficients
    coefficients = {}
    # diurnal species
    coefficients['q1'] = [1.0, -2.0, 0.0, 1.0, 0.0, 0.0, -1.0]
    coefficients['o1'] = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    coefficients['pi1'] = [1.0, 1.0, -3.0, 0.0, 0.0, 1.0, -1.0]
    coefficients['p1'] = [1.0, 1.0, -2.0, 0.0, 0.0, 0.0, -1.0]
    # semi-diurnal species
    coefficients['m2'] = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # long-period species
    coefficients['055.565'] = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
    coefficients['ssa'] = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    coefficients['mm'] = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]
    coefficients['mf'] = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    coefficients['075.565'] = [0.0, 2.0, 0.0, 0.0, -1.0, 0.0, 0.0]
    # expected values
    exp = {}
    # diurnal species
    exp['q1'] = (0.6036 - 0.0026j, 0.29785 - 0.00139j, 0.0846 - 0.0008j)
    exp['o1'] = (0.6028 - 0.0025j, 0.29747 - 0.00137j, 0.0846 - 0.0008j)
    exp['pi1'] = (0.5878 - 0.0015j, 0.28996 - 0.00086j, 0.0847 - 0.0007j)
    exp['p1'] = (0.5817 - 0.0011j, 0.28692 - 0.00067j, 0.0853 - 0.0007j)
    # semi-diurnal species
    exp['m2'] = (0.6078 - 0.0022j, 0.30102 - 0.0013j, 0.0847 - 0.0007j)
    # long-period species
    exp['055.565'] = (0.6344 - 0.0093j, 0.31537 - 0.00541j, 0.0936 - 0.0028j)
    exp['ssa'] = (0.6182 - 0.0054j, 0.30593 - 0.00315j, 0.0886 - 0.0016j)
    exp['mm'] = (0.6126 - 0.0041j, 0.30270 - 0.00237j, 0.0870 - 0.0012j)
    exp['mf'] = (0.6109 - 0.0037j, 0.30171 - 0.00213j, 0.0864 - 0.0011j)
    exp['075.565'] = (0.6109 - 0.0037j, 0.30171 - 0.00213j, 0.0864 - 0.0011j)
    # for each tidal constituent
    for c, v in exp.items():
        # calculate Love numbers
        omega = pyTMD.constituents._frequency(coefficients[c])
        h2, k2, l2 = pyTMD.constituents._complex_love_numbers(omega)
        # check Love numbers
        assert np.isclose(h2, v[0], atol=15e-4)
        assert np.isclose(k2, v[1], atol=15e-4)
        assert np.isclose(l2, v[2], atol=15e-4)

@pytest.mark.parametrize("model", ['1066A-N', 'PEM-C', 'C2'])
def test_love_number_ratios(model):
    """
    Tests the calculation of body tide Love numbers compared
    with the values from J. Wahr (1979)
    """
    # expected values for each model
    exp = {'1066A-N': {}, 'PEM-C': {}, 'C2': {}}
    # expected values (1066A Neutral)
    exp['1066A-N']['m1'] = (0.995, 0.997, 1.001)
    exp['1066A-N']['p1'] = (0.964, 0.963, 1.010)
    exp['1066A-N']['k1'] = (0.862, 0.859, 1.032)
    exp['1066A-N']['psi1'] = (1.554, 1.564, 0.875)
    exp['1066A-N']['phi1'] = (1.098, 1.101, 0.979)
    exp['1066A-N']['j1'] = (1.013, 1.013, 0.998)
    # expected values (PEM-C)
    exp['PEM-C']['m1'] = (0.995, 0.997, 1.001)
    exp['PEM-C']['p1'] = (0.964, 0.963, 1.008)
    exp['PEM-C']['k1'] = (0.862, 0.859, 1.031)
    exp['PEM-C']['psi1'] = (1.557, 1.567, 0.876)
    exp['PEM-C']['phi1'] = (1.096, 1.097, 0.979)
    exp['PEM-C']['j1'] = (1.013, 1.013, 0.998)
    # expected values (C2)
    exp['C2']['m1'] = (0.997, 0.997, 1.001)
    exp['C2']['p1'] = (0.965, 0.963, 1.008)
    exp['C2']['k1'] = (0.865, 0.862, 1.031)
    exp['C2']['psi1'] = (1.565, 1.574, 0.877)
    exp['C2']['phi1'] = (1.098, 1.101, 0.979)
    exp['C2']['j1'] = (1.013, 1.013, 0.998)
    # frequency of the o1 tidal constituent
    omega = pyTMD.constituents.frequency('o1')
    # calculate Love numbers for o1
    ho1, ko1, lo1 = pyTMD.constituents._love_numbers(omega, model=model)
    # for each tidal constituent
    for c, v in exp[model].items():
        # calculate Love numbers
        omega, = pyTMD.constituents.frequency(c)
        h2, k2, l2 = pyTMD.constituents._love_numbers(omega, model=model)
        # check Love numbers
        assert np.isclose(h2/ho1, v[0], atol=25e-4)
        assert np.isclose(k2/ko1, v[1], atol=25e-4)
        assert np.isclose(l2/lo1, v[2], atol=25e-4)
