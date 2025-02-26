"""
test_constituents.py (02/2025)
Tests parsing model constituents from strings

UPDATE HISTORY:
    Updated 02/2025: try parsing the entire Doodson table
    Written 02/2025
"""
import json
import pytest
import pyTMD.io
from pyTMD.utilities import get_data_path

def test_constituents():
    """
    Test parsing of model constituents
    """
    cindex = ['sa','ssa','mm','msf','mt','mf','alpha1','2q1','sigma1',
        'q1','rho1','o1','tau1','m1','chi1','pi1','p1','s1','k1','psi1',
        'phi1','beta1','theta1','j1','oo1','2n2','mu2','n2','nu2','m2',
        'm2a','m2b','lambda2','l2','t2','s2','alpha2','beta2','delta2',
        'gamma2','r2','k2','eta2','mns2','2sm2','m3','mk3','s3','mn4',
        'm4','ms4','mk4','so1','s4','s5','m6','s6','s7','s8','m8','mks2',
        'msqm','mtm','n4','eps2','ups1','z0','node']
    for c in cindex:
        # test standard case
        assert (pyTMD.io.constituents.parse(c) == c)
        # test with uppercase
        assert (pyTMD.io.constituents.parse(c.upper()) == c)
        # test with additional characters
        assert (pyTMD.io.constituents.parse(f'_{c}_') == c)
        assert (pyTMD.io.constituents.parse(f'{c:10}') == c)

def test_remapping():
    """
    Test remapping of model constituents
    """
    mapping = [('2n','2n2'), ('alp1', 'alpha1'), ('alp2', 'alpha2'),
        ('bet1', 'beta1'), ('bet2', 'beta2'), ('del2', 'delta2'),
        ('e2','eps2'), ('ep2','eps2'), ('gam2', 'gamma2'),
        ('la2','lambda2'), ('lam2','lambda2'), ('lm2','lambda2'),
        ('msq', 'msqm'), ('omega0', 'node'), ('om0', 'node'),
        ('rho', 'rho1'), ('sig1','sigma1'),
        ('the', 'theta1'), ('the1', 'theta1')]
    for m in mapping:
        # test standard case
        assert (pyTMD.io.constituents.parse(m[0]) == m[1])
        # test with uppercase
        assert (pyTMD.io.constituents.parse(m[0].upper()) == m[1])
        # test with additional characters
        assert (pyTMD.io.constituents.parse(f'_{m[0]}_') == m[1])
        assert (pyTMD.io.constituents.parse(f'{m[0]:10}') == m[1])

def test_doodson_table():
    """
    Test parsing table of Doodson coefficients
    """
    # JSON file of Doodson coefficients
    table = get_data_path(['data','doodson.json'])
    # modified Doodson coefficients for constituents
    with table.open(mode='r', encoding='utf8') as fid:
        coefficients = json.load(fid)
    # test parsing of Doodson coefficients
    for key, val in coefficients.items():
        assert (pyTMD.io.constituents.parse(key) == key)
