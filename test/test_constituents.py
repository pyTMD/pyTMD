"""
test_constituents.py (02/2025)
Tests parsing model constituents from strings

UPDATE HISTORY:
    Updated 02/2025: try parsing the entire Doodson table
    Written 02/2025
"""
import json
import numpy as np
import pyTMD.constituents
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
        assert (pyTMD.constituents._parse_name(c) == c)
        # test with uppercase
        assert (pyTMD.constituents._parse_name(c.upper()) == c)
        # test with additional characters
        assert (pyTMD.constituents._parse_name(f'_{c}_') == c)
        assert (pyTMD.constituents._parse_name(f'{c:10}') == c)

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
        assert (pyTMD.constituents._parse_name(m[0]) == m[1])
        # test with uppercase
        assert (pyTMD.constituents._parse_name(m[0].upper()) == m[1])
        # test with additional characters
        assert (pyTMD.constituents._parse_name(f'_{m[0]}_') == m[1])
        assert (pyTMD.constituents._parse_name(f'{m[0]:10}') == m[1])

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
        assert (pyTMD.constituents._parse_name(key) == key)

def test_doodson():
    """
    Tests the calculation of Doodson numbers
    """
    # expected values
    exp = {}
    # semi-diurnal species
    exp['m2'] = 255.555
    exp['s2'] = 273.555
    exp['n2'] = 245.655
    exp['nu2'] = 247.455
    exp['mu2'] = 237.555
    exp['2n2'] = 235.755
    exp['lambda2'] = 263.655
    exp['l2'] = 265.455
    exp['k2'] = 275.555
    # diurnal species
    exp['m1'] = 155.555
    exp['s1'] = 164.555
    exp['o1'] = 145.555
    exp['oo1'] = 185.555
    exp['k1'] = 165.555
    exp['q1'] = 135.655
    exp['2q1'] = 125.755
    exp['p1'] = 163.555
    # long-period species
    exp['mm'] = 065.455
    exp['ssa'] = 057.555
    exp['msf'] = 073.555
    exp['mf'] = 075.555
    exp['msqm'] = 093.555
    exp['mtm'] = 085.455
    exp['node'] = 055.565
    # short-period species
    exp['m3'] = 355.555
    exp['m4'] = 455.555
    exp['m6'] = 655.555
    exp['m8'] = 855.555
    # shallow water species
    exp['2so3'] = '3X1.555'
    exp['2jp3'] = '3X3.355'
    exp['kso3'] = '3X3.555'
    exp['2jk3'] = '3X4.355'
    exp['2ko3'] = '3X5.555'
    # 3rd degree terms
    exp["2q1'"] = 125.655
    exp["q1'"] = 135.555
    exp["o1'"] = 145.655
    exp["m1'"] = 155.555
    exp["k1'"] = 165.455
    exp["j1'"] = 175.555
    exp["2n2'"] = 235.655
    exp["n2'"] = 245.555
    exp["m2'"] = 255.655
    exp["l2'"] = 265.555
    exp["m3'"] = 355.555
    exp['lambda3'] = 363.655
    exp["l3"] = 365.455
    exp["l3b"] = 365.655
    exp["f3"] = 375.555
    exp["j3"] = 375.555
    exp["s3'"] = 382.555
    # get observed values for constituents
    obs = pyTMD.constituents.doodson_number(exp.keys())
    cartwright = pyTMD.constituents.doodson_number(exp.keys(),
        formalism='Cartwright')
    # check values
    for key,val in exp.items():
        assert val == obs[key]
        # check values when entered as string
        test = pyTMD.constituents.doodson_number(key)
        assert val == test
        # check conversion to Doodson numbers
        doodson = pyTMD.constituents._to_doodson_number(cartwright[key])
        # check values when entered as Cartwright
        assert val == doodson
        # check values when entered as Doodson
        coefficients = pyTMD.constituents._from_doodson_number(val)
        assert np.all(cartwright[key] == coefficients)

def test_extended():
    """
    Tests the calculation of UKHO Extended Doodson numbers
    """
    # expected values
    exp = {}
    # semi-diurnal species
    exp['m2'] = 'BZZZZZZ'
    exp['s2'] = 'BBXZZZZ'
    exp['n2'] = 'BYZAZZZ'
    exp['nu2'] = 'BYBYZZZ'
    exp['mu2'] = 'BXBZZZZ'
    exp['2n2'] = 'BXZBZZZ'
    exp['lambda2'] = 'BAXAZZB'
    exp['l2'] = 'BAZYZZB'
    exp['k2'] = 'BBZZZZZ'
    # diurnal species
    exp['m1'] = 'AZZZZZA'
    exp['s1'] = 'AAYZZZA'
    exp['o1'] = 'AYZZZZY'
    exp['oo1'] = 'ACZZZZA'
    exp['k1'] = 'AAZZZZA'
    exp['q1'] = 'AXZAZZY'
    exp['2q1'] = 'AWZBZZY'
    exp['p1'] = 'AAXZZZY'
    # long-period species
    exp['mm'] = 'ZAZYZZZ'
    exp['ssa'] = 'ZZBZZZZ'
    exp['msf'] = 'ZBXZZZZ'
    exp['mf'] = 'ZBZZZZZ'
    exp['msqm'] = 'ZDXZZZZ'
    exp['mtm'] = 'ZCZYZZZ'
    exp['node'] = 'ZZZZAZB'
    # short-period species
    exp['m3'] = 'CZZZZZZ'
    exp['m4'] = 'DZZZZZZ'
    exp['n4'] = 'DXZBZZZ'
    exp['m6'] = 'FZZZZZZ'
    exp['n6'] = 'FWZCZZZ'
    exp['m8'] = 'HZZZZZZ'
    exp['m10'] = 'JZZZZZZ'
    exp['m12'] = 'LZZZZZZ'
    # shallow water species
    exp['2so3'] = 'CEVZZZA'
    exp['2jp3'] = 'CEXXZZA'
    exp['kso3'] = 'CEXZZZA'
    exp['2jk3'] = 'CEYXZZZ'
    exp['2ko3'] = 'CEZZZZA'
    # get observed values for constituents
    obs = pyTMD.constituents.doodson_number(exp.keys(),
        formalism='Extended')
    # check values
    for key,val in exp.items():
        assert val == obs[key]
        # check constituent IDs from coefficients
        coefficients = pyTMD.constituents._from_extended_doodson(val)
        c = pyTMD.constituents._to_constituent_id(coefficients,
            corrections='GOT', arguments=6)
        assert c == key

def test_constituent_id():
    """
    Tests the conversion of Doodson number to constituent ID
    """
    # constituents array (not all are included in tidal program)
    cindex = ['sa', 'ssa', 'mm', 'msf', 'mf', 'mtm', 'alpha1', '2q1', 'sigma1',
        'q1', 'rho1', 'o1', 'tau1', 'm1', 'chi1', 'pi1', 'p1', 's1', 'k1',
        'psi1', 'phi1', 'theta1', 'j1', 'oo1', '2n2', 'mu2', 'n2', 'nu2', 'm2a',
        'm2', 'm2b', 'lambda2', 'l2', 't2', 's2', 'r2', 'k2', 'eta2', 'eps2',
        '2sm2', 'm3', 'mk3', 's3', 'mn4', 'm4', 'ms4', 'mk4', 's4', 's5', 'm6',
        's6', 's7', 's8', 'm8', 'mks2', 'msqm', 'n4', 'z0']
    # test conversion of conversion to constituent ID
    for i, exp in enumerate(cindex):
        # get observed values for constituents
        coef = pyTMD.constituents.coefficients_table(exp, corrections='GOT')    
        c = pyTMD.constituents._to_constituent_id(coef[:,0], corrections='GOT')
        assert (c == exp)