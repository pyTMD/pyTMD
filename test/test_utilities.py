#!/usr/bin/env python
u"""
test_utilities.py (12/2020)
Verify file utility functions
"""
import io
import gzip
import pytest
import pathlib
import pyTMD.utilities

def test_hash():
    # get hash of compressed file
    ocean_pole_tide_file = pyTMD.utilities.get_data_path(['data',
        'opoleloadcoefcmcor.txt.gz'])
    TEST = pyTMD.utilities.get_hash(ocean_pole_tide_file)
    assert (TEST == '9c66edc2d0fbf627e7ae1cb923a9f0e5')
    # get hash of uncompressed file
    with gzip.open(ocean_pole_tide_file) as fid:
        TEST = pyTMD.utilities.get_hash(io.BytesIO(fid.read()))
        assert (TEST == 'cea08f83d613ed8e1a81f3b3a9453721')

_default_directory = pyTMD.utilities.get_cache_path()
def test_valid_url():
    # test over some valid urls
    URLS = [
        'https://arcticdata.io/',
        'http://www.esr.org/research/polar-tide-models',
        's3://pytmd-scratch/CATS2008.zarr'
    ]
    for URL in URLS:
        assert pyTMD.utilities.is_valid_url(URL)
    # test over some file paths
    PATHS = [
        pathlib.PurePosixPath('/home/user/data/CATS2008/grid_CATS2008'),
        pathlib.PureWindowsPath('C://Users/user/data/CATS2008/grid_CATS2008'),
        _default_directory.joinpath('CATS2008','grid_CATS2008')
    ]
    for PATH in PATHS:
        assert not pyTMD.utilities.is_valid_url(PATH)
