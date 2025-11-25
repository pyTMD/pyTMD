"""
test_model.py (11/2025)
Tests the reading of model definition files

UPDATE HISTORY:
    Updated 11/2025: use new z, u, v database and JSON format
    Updated 07/2025: added GOT4.10_SAL subset of constituents
    Updated 06/2025: added function to check extra databases
    Updated 02/2025: added function to try to parse bathymetry files
    Updated 09/2024: drop support for the ascii definition file format
        fix parsing of TPXO8-atlas-nc constituents
        using new JSON dictionary format for model projections
    Updated 08/2024: add automatic detection of definition file format
    Updated 07/2024: add new JSON format definition file format
    Written 04/2024
"""
from __future__ import annotations

import io
import json
import pytest
import shutil
import inspect
import pathlib
import pyTMD.io

# current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
filepath = pathlib.Path(filename).absolute().parent

def test_definition_CATS2008():
    """Tests the reading of the CATS2008 model definition file
    """
    # read definition file
    definition_file = 'model_CATS2008.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    # test read variables
    assert m.format == 'OTIS'
    assert m.name == 'CATS2008'
    assert m.z.grid_file == pathlib.Path('CATS2008/grid_CATS2008')
    assert m.z.model_file == pathlib.Path('CATS2008/hf.CATS2008.out')
    assert m.z.variable == 'tide_ocean'
    assert m.projection == {'datum': 'WGS84', 'lat_0': -90, 'lat_ts': -71,
        'lon_0': -70, 'proj': 'stere', 'type': 'crs', 'units': 'km',
        'x_0': 0, 'y_0': 0}

def test_definition_FES():
    """Tests the reading of the FES2014 model definition file
    """
    # read definition file
    definition_file = 'model_FES2014.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    m.parse_constituents(type='z')
    # model files and constituents
    model_files = ['fes2014/ocean_tide/2n2.nc.gz',
        'fes2014/ocean_tide/eps2.nc.gz', 'fes2014/ocean_tide/j1.nc.gz',
        'fes2014/ocean_tide/k1.nc.gz', 'fes2014/ocean_tide/k2.nc.gz',
        'fes2014/ocean_tide/l2.nc.gz', 'fes2014/ocean_tide/la2.nc.gz',
        'fes2014/ocean_tide/m2.nc.gz', 'fes2014/ocean_tide/m3.nc.gz',
        'fes2014/ocean_tide/m4.nc.gz', 'fes2014/ocean_tide/m6.nc.gz',
        'fes2014/ocean_tide/m8.nc.gz', 'fes2014/ocean_tide/mf.nc.gz',
        'fes2014/ocean_tide/mks2.nc.gz', 'fes2014/ocean_tide/mm.nc.gz',
        'fes2014/ocean_tide/mn4.nc.gz', 'fes2014/ocean_tide/ms4.nc.gz',
        'fes2014/ocean_tide/msf.nc.gz', 'fes2014/ocean_tide/msqm.nc.gz',
        'fes2014/ocean_tide/mtm.nc.gz', 'fes2014/ocean_tide/mu2.nc.gz',
        'fes2014/ocean_tide/n2.nc.gz', 'fes2014/ocean_tide/n4.nc.gz',
        'fes2014/ocean_tide/nu2.nc.gz', 'fes2014/ocean_tide/o1.nc.gz',
        'fes2014/ocean_tide/p1.nc.gz', 'fes2014/ocean_tide/q1.nc.gz',
        'fes2014/ocean_tide/r2.nc.gz', 'fes2014/ocean_tide/s1.nc.gz',
        'fes2014/ocean_tide/s2.nc.gz', 'fes2014/ocean_tide/s4.nc.gz',
        'fes2014/ocean_tide/sa.nc.gz', 'fes2014/ocean_tide/ssa.nc.gz',
        'fes2014/ocean_tide/t2.nc.gz']
    constituents = ['2n2','eps2','j1','k1','k2','l2',
                'lambda2','m2','m3','m4','m6','m8','mf','mks2','mm',
                'mn4','ms4','msf','msqm','mtm','mu2','n2','n4','nu2',
                'o1','p1','q1','r2','s1','s2','s4','sa','ssa','t2']
    # test read variables
    assert m.format == 'FES-netcdf'
    assert m.name == 'FES2014'
    # assert that all model files are in the model definition
    for f in model_files:
        assert pathlib.Path(f) in m['z'].model_file
    assert m.z.units == 'cm'
    assert m.z.variable == 'tide_ocean'
    assert m.compressed is True
    # check validity of parsed constituents
    parsed_constituents = [pyTMD.io.model.parse_file(f) for f in model_files]
    assert parsed_constituents == constituents

# PURPOSE: test glob file functionality
def test_definition_FES_glob():
    """Tests the reading of the FES2014 model definition file
    with glob file searching
    """
    # read model definition file
    definition_file = 'model_FES2014.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    m.parse_constituents(type='z')
    # model files
    model_files = ['fes2014/ocean_tide/2n2.nc.gz',
        'fes2014/ocean_tide/eps2.nc.gz', 'fes2014/ocean_tide/j1.nc.gz',
        'fes2014/ocean_tide/k1.nc.gz', 'fes2014/ocean_tide/k2.nc.gz',
        'fes2014/ocean_tide/l2.nc.gz', 'fes2014/ocean_tide/la2.nc.gz',
        'fes2014/ocean_tide/m2.nc.gz', 'fes2014/ocean_tide/m3.nc.gz',
        'fes2014/ocean_tide/m4.nc.gz', 'fes2014/ocean_tide/m6.nc.gz',
        'fes2014/ocean_tide/m8.nc.gz', 'fes2014/ocean_tide/mf.nc.gz',
        'fes2014/ocean_tide/mks2.nc.gz', 'fes2014/ocean_tide/mm.nc.gz',
        'fes2014/ocean_tide/mn4.nc.gz', 'fes2014/ocean_tide/ms4.nc.gz',
        'fes2014/ocean_tide/msf.nc.gz', 'fes2014/ocean_tide/msqm.nc.gz',
        'fes2014/ocean_tide/mtm.nc.gz', 'fes2014/ocean_tide/mu2.nc.gz',
        'fes2014/ocean_tide/n2.nc.gz', 'fes2014/ocean_tide/n4.nc.gz',
        'fes2014/ocean_tide/nu2.nc.gz', 'fes2014/ocean_tide/o1.nc.gz',
        'fes2014/ocean_tide/p1.nc.gz', 'fes2014/ocean_tide/q1.nc.gz',
        'fes2014/ocean_tide/r2.nc.gz', 'fes2014/ocean_tide/s1.nc.gz',
        'fes2014/ocean_tide/s2.nc.gz', 'fes2014/ocean_tide/s4.nc.gz',
        'fes2014/ocean_tide/sa.nc.gz', 'fes2014/ocean_tide/ssa.nc.gz',
        'fes2014/ocean_tide/t2.nc.gz']
    # create temporary files for testing glob functionality
    for model_file in model_files:
        local = filepath.joinpath(model_file)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.touch(exist_ok=True)
    # create model definition file
    fid = io.StringIO()
    glob_string = r'fes2014/ocean_tide/*.nc.gz'
    attrs = ['name','format','compressed','version']
    # create JSON definition file
    d = {attr:getattr(m,attr) for attr in attrs}
    d['z'] = m['z'].__dict__
    d['z']['model_file'] = glob_string
    json.dump(d, fid)
    # rewind the glob definition file
    fid.seek(0)
    # use model definition file as input
    model = pyTMD.io.model(directory=filepath).from_file(fid)
    model.parse_constituents(type='z')
    for attr in attrs:
        assert getattr(model,attr) == getattr(m,attr)
    # verify that the model files and constituents match
    assert (len(model['z'].model_file) == len(model_files))
    for f in model_files:
        assert pathlib.Path(filepath).joinpath(f) in model['z'].model_file
    for c in m.constituents:
        assert c in model.constituents
    # close the glob definition file
    fid.close()
    # clean up model
    shutil.rmtree(filepath.joinpath('fes2014'))

def test_definition_FES_currents():
    """Tests the reading of the FES2014 model definition file for currents
    """
    # read model definition file
    definition_file = 'model_FES2014.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    # model files and constituents
    model_files = {}
    model_files['u'] = ['fes2014/eastward_velocity/2n2.nc.gz',
        'fes2014/eastward_velocity/eps2.nc.gz',
        'fes2014/eastward_velocity/j1.nc.gz',
        'fes2014/eastward_velocity/k1.nc.gz',
        'fes2014/eastward_velocity/k2.nc.gz',
        'fes2014/eastward_velocity/l2.nc.gz',
        'fes2014/eastward_velocity/la2.nc.gz',
        'fes2014/eastward_velocity/m2.nc.gz',
        'fes2014/eastward_velocity/m3.nc.gz',
        'fes2014/eastward_velocity/m4.nc.gz',
        'fes2014/eastward_velocity/m6.nc.gz',
        'fes2014/eastward_velocity/m8.nc.gz',
        'fes2014/eastward_velocity/mf.nc.gz',
        'fes2014/eastward_velocity/mks2.nc.gz',
        'fes2014/eastward_velocity/mm.nc.gz',
        'fes2014/eastward_velocity/mn4.nc.gz',
        'fes2014/eastward_velocity/ms4.nc.gz',
        'fes2014/eastward_velocity/msf.nc.gz',
        'fes2014/eastward_velocity/msqm.nc.gz',
        'fes2014/eastward_velocity/mtm.nc.gz',
        'fes2014/eastward_velocity/mu2.nc.gz',
        'fes2014/eastward_velocity/n2.nc.gz',
        'fes2014/eastward_velocity/n4.nc.gz',
        'fes2014/eastward_velocity/nu2.nc.gz',
        'fes2014/eastward_velocity/o1.nc.gz',
        'fes2014/eastward_velocity/p1.nc.gz',
        'fes2014/eastward_velocity/q1.nc.gz',
        'fes2014/eastward_velocity/r2.nc.gz',
        'fes2014/eastward_velocity/s1.nc.gz',
        'fes2014/eastward_velocity/s2.nc.gz',
        'fes2014/eastward_velocity/s4.nc.gz',
        'fes2014/eastward_velocity/sa.nc.gz',
        'fes2014/eastward_velocity/ssa.nc.gz',
        'fes2014/eastward_velocity/t2.nc.gz']
    model_files['v'] = ['fes2014/northward_velocity/2n2.nc.gz',
        'fes2014/northward_velocity/eps2.nc.gz',
        'fes2014/northward_velocity/j1.nc.gz',
        'fes2014/northward_velocity/k1.nc.gz',
        'fes2014/northward_velocity/k2.nc.gz',
        'fes2014/northward_velocity/l2.nc.gz',
        'fes2014/northward_velocity/la2.nc.gz',
        'fes2014/northward_velocity/m2.nc.gz',
        'fes2014/northward_velocity/m3.nc.gz',
        'fes2014/northward_velocity/m4.nc.gz',
        'fes2014/northward_velocity/m6.nc.gz',
        'fes2014/northward_velocity/m8.nc.gz',
        'fes2014/northward_velocity/mf.nc.gz',
        'fes2014/northward_velocity/mks2.nc.gz',
        'fes2014/northward_velocity/mm.nc.gz',
        'fes2014/northward_velocity/mn4.nc.gz',
        'fes2014/northward_velocity/ms4.nc.gz',
        'fes2014/northward_velocity/msf.nc.gz',
        'fes2014/northward_velocity/msqm.nc.gz',
        'fes2014/northward_velocity/mtm.nc.gz',
        'fes2014/northward_velocity/mu2.nc.gz',
        'fes2014/northward_velocity/n2.nc.gz',
        'fes2014/northward_velocity/n4.nc.gz',
        'fes2014/northward_velocity/nu2.nc.gz',
        'fes2014/northward_velocity/o1.nc.gz',
        'fes2014/northward_velocity/p1.nc.gz',
        'fes2014/northward_velocity/q1.nc.gz',
        'fes2014/northward_velocity/r2.nc.gz',
        'fes2014/northward_velocity/s1.nc.gz',
        'fes2014/northward_velocity/s2.nc.gz',
        'fes2014/northward_velocity/s4.nc.gz',
        'fes2014/northward_velocity/sa.nc.gz',
        'fes2014/northward_velocity/ssa.nc.gz',
        'fes2014/northward_velocity/t2.nc.gz']
    constituents = ['2n2','eps2','j1','k1','k2','l2',
                'lambda2','m2','m3','m4','m6','m8','mf','mks2','mm',
                'mn4','ms4','msf','msqm','mtm','mu2','n2','n4','nu2',
                'o1','p1','q1','r2','s1','s2','s4','sa','ssa','t2']
    # test read variables
    assert m.format == 'FES-netcdf'
    assert m.name == 'FES2014'
    # assert that all model files are in the model definition
    for t in ['u','v']:
        for f in model_files[t]:
            assert pyTMD.utilities.Path(f) in m[t].model_file
        assert m[t].units == 'cm/s'
    assert m.compressed is True
    # check validity of parsed constituents
    parsed_constituents = \
        [pyTMD.io.model.parse_file(f) for f in model_files['u']]
    assert parsed_constituents == constituents
    assert m['u'].variable == 'zonal_tidal_current'
    assert m['v'].variable == 'meridional_tidal_current'

# PURPOSE: test glob file functionality
def test_definition_FES_currents_glob():
    """Tests the reading of the FES2014 model definition file
    with glob file searching for currents
    """
    # read model definition file
    definition_file = 'model_FES2014.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    # model files for each component
    model_files = {}
    model_files['u'] = ['fes2014/eastward_velocity/2n2.nc.gz',
        'fes2014/eastward_velocity/eps2.nc.gz',
        'fes2014/eastward_velocity/j1.nc.gz',
        'fes2014/eastward_velocity/k1.nc.gz',
        'fes2014/eastward_velocity/k2.nc.gz',
        'fes2014/eastward_velocity/l2.nc.gz',
        'fes2014/eastward_velocity/la2.nc.gz',
        'fes2014/eastward_velocity/m2.nc.gz',
        'fes2014/eastward_velocity/m3.nc.gz',
        'fes2014/eastward_velocity/m4.nc.gz',
        'fes2014/eastward_velocity/m6.nc.gz',
        'fes2014/eastward_velocity/m8.nc.gz',
        'fes2014/eastward_velocity/mf.nc.gz',
        'fes2014/eastward_velocity/mks2.nc.gz',
        'fes2014/eastward_velocity/mm.nc.gz',
        'fes2014/eastward_velocity/mn4.nc.gz',
        'fes2014/eastward_velocity/ms4.nc.gz',
        'fes2014/eastward_velocity/msf.nc.gz',
        'fes2014/eastward_velocity/msqm.nc.gz',
        'fes2014/eastward_velocity/mtm.nc.gz',
        'fes2014/eastward_velocity/mu2.nc.gz',
        'fes2014/eastward_velocity/n2.nc.gz',
        'fes2014/eastward_velocity/n4.nc.gz',
        'fes2014/eastward_velocity/nu2.nc.gz',
        'fes2014/eastward_velocity/o1.nc.gz',
        'fes2014/eastward_velocity/p1.nc.gz',
        'fes2014/eastward_velocity/q1.nc.gz',
        'fes2014/eastward_velocity/r2.nc.gz',
        'fes2014/eastward_velocity/s1.nc.gz',
        'fes2014/eastward_velocity/s2.nc.gz',
        'fes2014/eastward_velocity/s4.nc.gz',
        'fes2014/eastward_velocity/sa.nc.gz',
        'fes2014/eastward_velocity/ssa.nc.gz',
        'fes2014/eastward_velocity/t2.nc.gz']
    model_files['v'] = ['fes2014/northward_velocity/2n2.nc.gz',
        'fes2014/northward_velocity/eps2.nc.gz',
        'fes2014/northward_velocity/j1.nc.gz',
        'fes2014/northward_velocity/k1.nc.gz',
        'fes2014/northward_velocity/k2.nc.gz',
        'fes2014/northward_velocity/l2.nc.gz',
        'fes2014/northward_velocity/la2.nc.gz',
        'fes2014/northward_velocity/m2.nc.gz',
        'fes2014/northward_velocity/m3.nc.gz',
        'fes2014/northward_velocity/m4.nc.gz',
        'fes2014/northward_velocity/m6.nc.gz',
        'fes2014/northward_velocity/m8.nc.gz',
        'fes2014/northward_velocity/mf.nc.gz',
        'fes2014/northward_velocity/mks2.nc.gz',
        'fes2014/northward_velocity/mm.nc.gz',
        'fes2014/northward_velocity/mn4.nc.gz',
        'fes2014/northward_velocity/ms4.nc.gz',
        'fes2014/northward_velocity/msf.nc.gz',
        'fes2014/northward_velocity/msqm.nc.gz',
        'fes2014/northward_velocity/mtm.nc.gz',
        'fes2014/northward_velocity/mu2.nc.gz',
        'fes2014/northward_velocity/n2.nc.gz',
        'fes2014/northward_velocity/n4.nc.gz',
        'fes2014/northward_velocity/nu2.nc.gz',
        'fes2014/northward_velocity/o1.nc.gz',
        'fes2014/northward_velocity/p1.nc.gz',
        'fes2014/northward_velocity/q1.nc.gz',
        'fes2014/northward_velocity/r2.nc.gz',
        'fes2014/northward_velocity/s1.nc.gz',
        'fes2014/northward_velocity/s2.nc.gz',
        'fes2014/northward_velocity/s4.nc.gz',
        'fes2014/northward_velocity/sa.nc.gz',
        'fes2014/northward_velocity/ssa.nc.gz',
        'fes2014/northward_velocity/t2.nc.gz']
    # create temporary files for testing glob functionality
    for t in ['u','v']:
        for model_file in model_files[t]:
            local = filepath.joinpath(model_file)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.touch(exist_ok=True)
    # create model definition file
    fid = io.StringIO()
    attrs = ['name','format','compressed','version']
    glob_string = {}
    glob_string['u'] = r'fes2014/eastward_velocity/*.nc.gz'
    glob_string['v'] = r'fes2014/northward_velocity/*.nc.gz'
    # create JSON definition file
    d = {attr:getattr(m,attr) for attr in attrs}
    for t in ['u','v']:
        d[t] = m[t].__dict__
        d[t]['model_file'] = glob_string[t]
    json.dump(d, fid)
    # rewind the glob definition file
    fid.seek(0)
    # use model definition file as input
    model = pyTMD.io.model(directory=filepath).from_file(fid)
    for attr in attrs:
        assert getattr(model,attr) == getattr(m,attr)
    # verify that the model files and constituents match
    for t in ['u','v']:
        assert (len(model[t].model_file) == len(model_files[t]))
        for f in model_files[t]:
            assert pyTMD.utilities.Path(filepath).joinpath(f) in model[t].model_file
    # check validity of parsed constituents
    parsed_constituents = \
        [pyTMD.io.model.parse_file(f) for f in model_files['u']]
    model.parse_constituents(type='u')
    for c in parsed_constituents:
        assert c in model.constituents
    # close the glob definition file
    fid.close()
    # clean up model
    shutil.rmtree(filepath.joinpath('fes2014'))

def test_definition_GOT():
    """Tests the reading of the GOT4.10 model definition file
    """
    # read model definition file
    definition_file = 'model_GOT4.10.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    # model files
    model_files = ['GOT4.10c/grids_loadtide/k1load.d.gz',
        'GOT4.10c/grids_loadtide/k2load.d.gz',
        'GOT4.10c/grids_loadtide/m2load.d.gz',
        'GOT4.10c/grids_loadtide/m4load.d.gz',
        'GOT4.10c/grids_loadtide/n2load.d.gz',
        'GOT4.10c/grids_loadtide/o1load.d.gz',
        'GOT4.10c/grids_loadtide/p1load.d.gz',
        'GOT4.10c/grids_loadtide/q1load.d.gz',
        'GOT4.10c/grids_loadtide/s1load.d.gz',
        'GOT4.10c/grids_loadtide/s2load.d.gz']
    # test read variables
    assert m.format == 'GOT-ascii'
    assert m.name == 'GOT4.10'
    # assert that all model files are in the model definition
    for f in model_files:
        assert pyTMD.utilities.Path(f) in m['z'].model_file
    assert m['z'].units == 'mm'
    assert m['z'].variable == 'tide_load'
    assert m.compressed is True

# PURPOSE: test glob file functionality
def test_definition_GOT_glob():
    """Tests the reading of the GOT4.10 model definition file
    with glob file searching
    """
    # read model definition file
    definition_file = 'model_GOT4.10.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    # model files
    model_files = ['GOT4.10c/grids_loadtide/k1load.d.gz',
        'GOT4.10c/grids_loadtide/k2load.d.gz',
        'GOT4.10c/grids_loadtide/m2load.d.gz',
        'GOT4.10c/grids_loadtide/m4load.d.gz',
        'GOT4.10c/grids_loadtide/n2load.d.gz',
        'GOT4.10c/grids_loadtide/o1load.d.gz',
        'GOT4.10c/grids_loadtide/p1load.d.gz',
        'GOT4.10c/grids_loadtide/q1load.d.gz',
        'GOT4.10c/grids_loadtide/s1load.d.gz',
        'GOT4.10c/grids_loadtide/s2load.d.gz']
    # create temporary files for testing glob functionality
    for model_file in model_files:
        local = filepath.joinpath(model_file)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.touch(exist_ok=True)
    # create model definition file
    fid = io.StringIO()
    attrs = ['name','format','compressed']
    glob_string = r'GOT4.10c/grids_loadtide/*.d.gz'
    # create JSON definition file
    d = {attr:getattr(m,attr) for attr in attrs}
    d['z'] = m['z'].__dict__
    d['z']['model_file'] = glob_string
    json.dump(d, fid)
    # rewind the glob definition file
    fid.seek(0)
    # use model definition file as input
    model = pyTMD.io.model(directory=filepath).from_file(fid)
    for attr in attrs:
        assert getattr(model,attr) == getattr(m,attr)
    # verify that the model files match
    assert (len(model['z'].model_file) == len(model_files))
    for f in model_files:
        assert pyTMD.utilities.Path(filepath).joinpath(f) in model['z'].model_file
    # close the glob definition file
    fid.close()
    # clean up model
    shutil.rmtree(filepath.joinpath('GOT4.10c'))

def test_definition_TPXO9():
    """Tests the reading of the TPXO9-atlas-v5 model definition file
    """
    # read model definition file
    definition_file = 'model_TPXO9-atlas-v5.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    # model files
    model_files = ['TPXO9_atlas_v5/h_2n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_k1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_k2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_m2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_m4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_mf_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_mm_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_mn4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_ms4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_o1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_p1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_q1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_s1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_s2_tpxo9_atlas_30_v5.nc']
    grid_file = pathlib.Path('TPXO9_atlas_v5/grid_tpxo9_atlas_30_v5.nc')
    # test read variables
    assert m.format == 'ATLAS-netcdf'
    assert m.name == 'TPXO9-atlas-v5'
    assert m['z'].grid_file == grid_file
    # assert that all model files are in the model definition
    for f in model_files:
        assert pathlib.Path(f) in m['z'].model_file
    assert m['z'].units == 'cm'
    assert m['z'].variable == 'tide_ocean'
    assert m.compressed is False

# PURPOSE: test glob file functionality
def test_definition_TPXO9_glob():
    """Tests the reading of the TPXO9-atlas-v5 model definition file
    with glob file searching
    """
    # read model definition file
    definition_file = 'model_TPXO9-atlas-v5.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    # model files
    model_files = ['TPXO9_atlas_v5/h_2n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_k1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_k2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_m2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_m4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_mf_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_mm_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_mn4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_ms4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_o1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_p1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_q1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_s1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/h_s2_tpxo9_atlas_30_v5.nc']
    grid_file = pathlib.Path('TPXO9_atlas_v5/grid_tpxo9_atlas_30_v5.nc')
    # create temporary files for testing glob functionality
    for model_file in model_files:
        local = filepath.joinpath(model_file)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.touch(exist_ok=True)
    # create temporary grid file
    local = filepath.joinpath(grid_file)
    local.touch(exist_ok=True)
    # test read variables
    assert m.format == 'ATLAS-netcdf'
    assert m.name == 'TPXO9-atlas-v5'
    # create model definition file
    fid = io.StringIO()
    attrs = ['name','format','compressed']
    glob_string = r'TPXO9_atlas_v5/h*.nc'
    # create JSON definition file
    d = {attr:getattr(m,attr) for attr in attrs}
    d['z'] = m['z'].__dict__
    d['z']['model_file'] = glob_string
    d['z']['grid_file'] = str(grid_file)
    json.dump(d, fid)
    # rewind the glob definition file
    fid.seek(0)
    # use model definition file as input
    model = pyTMD.io.model(directory=filepath).from_file(fid)
    for attr in attrs:
        assert getattr(model,attr) == getattr(m,attr)
    # verify that the model files match
    assert (len(model['z'].model_file) == len(model_files))
    for f in model_files:
        assert pathlib.Path(filepath).joinpath(f) in model['z'].model_file
    # close the glob definition file
    fid.close()
    # clean up model
    shutil.rmtree(filepath.joinpath('TPXO9_atlas_v5'))

def test_definition_TPXO9_currents():
    """Tests the reading of the TPXO9-atlas-v5 model definition file
    for currents
    """
    # read model definition file
    definition_file = 'model_TPXO9-atlas-v5.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    # model files for each component
    model_files = {}
    model_files['u'] = ['TPXO9_atlas_v5/u_2n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_k1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_k2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_m2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_m4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mf_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mm_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mn4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_ms4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_o1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_p1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_q1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_s1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_s2_tpxo9_atlas_30_v5.nc']
    model_files['v'] = ['TPXO9_atlas_v5/u_2n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_k1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_k2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_m2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_m4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mf_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mm_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mn4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_ms4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_o1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_p1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_q1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_s1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_s2_tpxo9_atlas_30_v5.nc']
    grid_file = pathlib.Path('TPXO9_atlas_v5/grid_tpxo9_atlas_30_v5.nc')
    # test read variables
    assert m.format == 'ATLAS-netcdf'
    assert m.name == 'TPXO9-atlas-v5'
    for t in ['u','v']:
        assert sorted(m[t].model_file) == \
            [pathlib.Path(f) for f in model_files[t]]
        assert m[t].grid_file == grid_file
        assert m[t].units == 'm^2/s'
    assert m.compressed is False
    # test derived properties
    assert m['u'].variable == 'zonal_tidal_current'
    assert m['v'].variable == 'meridional_tidal_current'

# PURPOSE: test glob file functionality
def test_definition_TPXO9_currents_glob():
    """Tests the reading of the TPXO9-atlas-v5 model definition file for
    currents with glob file searching
    """
    # read model definition file
    definition_file = 'model_TPXO9-atlas-v5.json'
    m = pyTMD.io.model().from_file(filepath.joinpath(definition_file))
    # model files for each component
    model_files = {}
    model_files['u'] = ['TPXO9_atlas_v5/u_2n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_k1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_k2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_m2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_m4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mf_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mm_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mn4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_ms4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_o1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_p1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_q1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_s1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_s2_tpxo9_atlas_30_v5.nc']
    model_files['v'] = ['TPXO9_atlas_v5/u_2n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_k1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_k2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_m2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_m4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mf_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mm_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_mn4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_ms4_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_n2_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_o1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_p1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_q1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_s1_tpxo9_atlas_30_v5.nc',
        'TPXO9_atlas_v5/u_s2_tpxo9_atlas_30_v5.nc']
    grid_file = pathlib.Path('TPXO9_atlas_v5/grid_tpxo9_atlas_30_v5.nc')
    # create temporary files for testing glob functionality
    for t in ['u','v']:
        for model_file in model_files[t]:
            local = filepath.joinpath(model_file)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.touch(exist_ok=True)
    # create temporary grid file
    local = filepath.joinpath(grid_file)
    local.touch(exist_ok=True)
    # create model definition file
    fid = io.StringIO()
    attrs = ['name','format','compressed']
    glob_string = {}
    glob_string['u'] = r'TPXO9_atlas_v5/u*.nc'
    glob_string['v'] = r'TPXO9_atlas_v5/u*.nc'
    # create JSON definition file
    d = {attr:getattr(m,attr) for attr in attrs}
    for t in ['u','v']:
        d[t] = m[t].__dict__
        d[t]['model_file'] = glob_string[t]
        d[t]['grid_file'] = str(grid_file)
    json.dump(d, fid)
    # rewind the glob definition file
    fid.seek(0)
    # use model definition file as input
    model = pyTMD.io.model(directory=filepath).from_file(fid)
    for attr in attrs:
        assert getattr(model,attr) == getattr(m,attr)
    # verify that the model files match
    for t in ['u','v']:
        assert (len(model[t].model_file) == len(model_files[t]))
        for f in model_files[t]:
            assert pathlib.Path(filepath).joinpath(f) in model[t].model_file
    # close the glob definition file
    fid.close()
    # clean up model
    shutil.rmtree(filepath.joinpath('TPXO9_atlas_v5'))

# parameterize model
@pytest.mark.parametrize("MODEL", pyTMD.io.model.FES())
def test_parse_FES_elevation(MODEL):
    """Tests the parsing of FES-type elevation model files
    """
    m = pyTMD.io.model(verify=False).from_database(MODEL, type='z')
    m.parse_constituents(type='z')
    constituents = [pyTMD.io.model.parse_file(f) for f in m['z'].model_file]
    assert (m.constituents == constituents)

# parameterize model
current_models = set(pyTMD.io.model.FES()) & set(pyTMD.io.model.ocean_current())
@pytest.mark.parametrize("MODEL", sorted(current_models))
def test_parse_FES_currents(MODEL):
    """Tests the parsing of FES-type current model files
    """
    # test ocean current constituents
    m = pyTMD.io.model(verify=False).from_database(MODEL, type=['u','v'])
    m.parse_constituents(type='u')
    constituents = [pyTMD.io.model.parse_file(f) for f in m['u'].model_file]
    assert (m.constituents == constituents)
    m.parse_constituents(type='v')
    constituents = [pyTMD.io.model.parse_file(f) for f in m['v'].model_file]
    assert (m.constituents == constituents)

# parameterize model bathymetry files
@pytest.mark.parametrize("FILE", ["ba.HAMTIDE.nc","grid_tpxo10atlas_v2.nc"])
def test_parse_bathymetry(FILE):
    """Verifies that model bathymetry files are parsed correctly
    """
    constituents = pyTMD.io.model.parse_file(FILE)
    assert constituents is None

# parameterize model
@pytest.mark.parametrize("MODEL", pyTMD.io.model.GOT())
def test_parse_GOT_elevation(MODEL):
    """Tests the parsing of GOT-type elevation model files
    """
    m = pyTMD.io.model(verify=False).from_database(MODEL, type='z')
    m.constituents = [pyTMD.io.model.parse_file(f) for f in m['z'].model_file]
    # constituents for long-period and short-period tides
    if MODEL in ('RE14',):
        constituents = ['mf','mm','mt','node','sa','ssa']
    elif MODEL in ('GOT4.10_SAL'):
        constituents = ['o1','p1','k1','q1','n2','m2','s2','k2']
    else:
        constituents = ['q1','o1','p1','k1','n2','m2','s2','k2','s1','m4']
    # extend list with third degree constituents
    if MODEL in ('GOT5.6','GOT5.6_extrapolated'):
        constituents.extend(["l2'", "m1'", 'm3', "n2'"])
    # verify that all constituents exist
    assert all(c in m.constituents for c in constituents)

# parameterize model
@pytest.mark.parametrize("MODEL", pyTMD.io.model.ATLAS())
def test_parse_TPXO9_elevation(MODEL):
    """Tests the parsing of ATLAS-type elevation model files
    """
    m = pyTMD.io.model(verify=False).from_database(MODEL, type='z')
    m.parse_constituents(type='z')
    constituents = ['q1','o1','p1','k1','n2','m2','s2','k2','m4']
    assert all(c in m.constituents for c in constituents)
    # test additional constituents found in newer models
    if m.name in ('TPXO9-atlas','TPXO9-atlas-v2','TPXO9-atlas-v3',
            'TPXO9-atlas-v4','TPXO9-atlas-v5'):
        assert all(c in m.constituents for c in ['2n2','mn4','ms4'])
    if m.name in ('TPXO9-atlas-v3','TPXO9-atlas-v4','TPXO9-atlas-v5'):
        assert all(c in m.constituents for c in ['mf','mm'])
    if m.name in ('TPXO9-atlas-v5',):
        assert all(c in m.constituents for c in ['s1',])

# parameterize model
current_models = set(pyTMD.io.model.ATLAS()) & \
    set(pyTMD.io.model.ocean_current())
@pytest.mark.parametrize("MODEL", sorted(current_models))
def test_parse_TPXO9_currents(MODEL):
    """Tests the parsing of ATLAS-type current model files
    """
    m = pyTMD.io.model(verify=False).from_database(MODEL, type=['u','v'])
    m.parse_constituents(type='u')
    constituents = ['q1','o1','p1','k1','n2','m2','s2','k2',
        'm4','ms4','mn4','2n2']
    assert all(c in m.constituents for c in constituents)
    # test additional constituents found in newer models
    if m.name in ('TPXO9-atlas-v3','TPXO9-atlas-v4','TPXO9-atlas-v5'):
        assert all(c in m.constituents for c in ['mf','mm'])
    if m.name in ('TPXO9-atlas-v5',):
        assert all(c in m.constituents for c in ['s1',])

# PURPOSE: test reading of model database
def test_read_database():
    """Tests the reading of the pyTMD model database
    """
    database = pyTMD.io.load_database()
    # for each database entry
    for key, val in database.items():
        assert isinstance(key, str)
        assert val == pyTMD.models[key]
    # assert that models are accessible
    assert pyTMD.models.get('CATS2008') is not None
    assert pyTMD.models.get('FES2014') is not None

# custom database from a JSON file
_extra_database = filepath.joinpath("extra_database.json")
# custom database from a dictionary
_custom_database = {
    "EOT20_custom": {
        "format": "FES-netcdf",
        "name": "EOT20_custom",
        "reference": "https://doi.org/10.17882/79489",
        "version": "EOT20",
        "z": {
            "model_file": [
                "EOT20/ocean_tides/2N2_ocean_eot20.nc",
                "EOT20/ocean_tides/J1_ocean_eot20.nc",
                "EOT20/ocean_tides/K1_ocean_eot20.nc",
                "EOT20/ocean_tides/K2_ocean_eot20.nc",
                "EOT20/ocean_tides/M2_ocean_eot20.nc",
                "EOT20/ocean_tides/M4_ocean_eot20.nc",
                "EOT20/ocean_tides/MF_ocean_eot20.nc",
                "EOT20/ocean_tides/MM_ocean_eot20.nc",
                "EOT20/ocean_tides/N2_ocean_eot20.nc",
                "EOT20/ocean_tides/O1_ocean_eot20.nc",
                "EOT20/ocean_tides/P1_ocean_eot20.nc",
                "EOT20/ocean_tides/Q1_ocean_eot20.nc",
                "EOT20/ocean_tides/S1_ocean_eot20.nc",
                "EOT20/ocean_tides/S2_ocean_eot20.nc",
                "EOT20/ocean_tides/SA_ocean_eot20.nc",
                "EOT20/ocean_tides/SSA_ocean_eot20.nc",
                "EOT20/ocean_tides/T2_ocean_eot20.nc",
            ],
            "units": "cm",
            "variable": "tide_ocean",
        },
    }
}

# PURPOSE: test reading extra model databases in file and dict format
@pytest.mark.parametrize("extra_databases", [_extra_database, _custom_database])
def test_read_extra_database(extra_databases):
    """Tests that extra model databases can be read in file and dict format
    """
    # load default db, and default + extra db
    db_default = pyTMD.io.load_database()
    db_extra = pyTMD.io.load_database(extra_databases=extra_databases)
    # verify that custom model exists in db
    assert 'EOT20_custom' not in db_default.keys()
    assert 'EOT20_custom' in db_extra.keys()
    # verify default db is a subset of default + extra db
    assert db_default.items() <= db_extra.items()
