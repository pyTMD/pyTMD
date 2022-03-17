#!/usr/bin/env python
u"""
compute_tides_ICESat2_ATL12.py
Written by Tyler Sutterley (03/2022)
Calculates tidal elevations for correcting ICESat-2 ocean surface height data

Uses OTIS format tidal solutions provided by Ohio State University and ESR
    http://volkov.oce.orst.edu/tides/region.html
    https://www.esr.org/research/polar-tide-models/list-of-polar-tide-models/
    ftp://ftp.esr.org/pub/datasets/tmd/
Global Tide Model (GOT) solutions provided by Richard Ray at GSFC
or Finite Element Solution (FES) models provided by AVISO

COMMAND LINE OPTIONS:
    -D X, --directory X: Working data directory
    -T X, --tide X: Tide model to use in correction
    -I X, --interpolate X: Interpolation method
        spline
        linear
        nearest
        bilinear
    -E X, --extrapolate X: Extrapolate with nearest-neighbors
    -c X, --cutoff X: Extrapolation cutoff in kilometers
        set to inf to extrapolate for all points
    -M X, --mode X: Permission mode of directories and files created
    -V, --verbose: Output information about each created file

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/

PROGRAM DEPENDENCIES:
    read_ICESat2_ATL12.py: reads ICESat-2 ocean surface height data files
    time.py: utilities for calculating time operations
    model.py: retrieves tide model parameters for named tide models
    utilities.py: download and management utilities for syncing files
    calc_astrol_longitudes.py: computes the basic astronomical mean longitudes
    calc_delta_time.py: calculates difference between universal and dynamic time
    convert_ll_xy.py: convert lat/lon points to and from projected coordinates
    infer_minor_corrections.py: return corrections for minor constituents
    load_constituent.py: loads parameters for a given tidal constituent
    load_nodal_corrections.py: load the nodal corrections for tidal constituents
    read_tide_model.py: extract tidal harmonic constants from OTIS tide models
    read_netcdf_model.py: extract tidal harmonic constants from netcdf models
    read_GOT_model.py: extract tidal harmonic constants from GSFC GOT models
    read_FES_model.py: extract tidal harmonic constants from FES tide models
    bilinear_interp.py: bilinear interpolation of data to coordinates
    nearest_extrap.py: nearest-neighbor extrapolation of data to coordinates
    predict_tide_drift.py: predict tidal elevations using harmonic constants

UPDATE HISTORY:
    Updated 03/2022: using static decorators to define available models
    Updated 02/2022: added Arctic 2km model (Arc2kmTM) to list of models
    Updated 12/2021: added TPXO9-atlas-v5 to list of available tide models
    Updated 10/2021: using python logging for handling verbose output
    Updated 09/2021: refactor to use model class for files and attributes
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 06/2021: added new Gr1km-v2 1km Greenland model from ESR
    Updated 05/2021: added option for extrapolation cutoff in kilometers
    Updated 04/2021: can use a generically named ATL12 file as input
    Updated 03/2021: added TPXO9-atlas-v4 in binary OTIS format
        simplified netcdf inputs to be similar to binary OTIS read program
        replaced numpy bool/int to prevent deprecation warnings
    Updated 12/2020: H5py deprecation warning change to use make_scale
        added valid data extrapolation with nearest_extrap
        merged time conversion routines into module
    Updated 11/2020: added model constituents from TPXO9-atlas-v3
    Updated 10/2020: using argparse to set command line parameters
    Updated 08/2020: using builtin time operations.  python3 regular expressions
    Updated 07/2020: added FES2014 and FES2014_load.  use merged delta times
    Updated 06/2020: added version 2 of TPXO9-atlas (TPXO9-atlas-v2)
    Updated 03/2020: use read_ICESat2_ATL12.py from read-ICESat-2 repository
    Updated 02/2020: changed CATS2008 grid to match version on U.S. Antarctic
        Program Data Center http://www.usap-dc.org/view/dataset/601235
    Forked 12/2019 from compute_tides_ICESat2_ATL07.py
    Updated 11/2019: added AOTIM-5-2018 tide model (2018 update to 2004 model)
    Forked 11/2019 from compute_tides_ICESat2_atl06.py
    Updated 10/2019: external read functions.  adjust regex for processed files
        changing Y/N flags to True/False
    Updated 09/2019: using date functions paralleling public repository
        add option for TPXO9-atlas.  add OTIS netcdf tide option
    Updated 05/2019: check if beam exists in a try except else clause
    Updated 04/2019: check if subsetted beam contains ocean surface data
    Written 04/2019
"""
from __future__ import print_function

import sys
import os
import re
import h5py
import logging
import argparse
import datetime
import numpy as np
import pyTMD.time
import pyTMD.model
import pyTMD.utilities
from pyTMD.calc_delta_time import calc_delta_time
from pyTMD.read_tide_model import extract_tidal_constants
from pyTMD.read_netcdf_model import extract_netcdf_constants
from pyTMD.read_GOT_model import extract_GOT_constants
from pyTMD.read_FES_model import extract_FES_constants
from pyTMD.infer_minor_corrections import infer_minor_corrections
from pyTMD.predict_tide_drift import predict_tide_drift
from icesat2_toolkit.read_ICESat2_ATL12 import read_HDF5_ATL12

#-- PURPOSE: read ICESat-2 ocean surface height (ATL12) from NSIDC
#-- compute tides at points and times using tidal model driver algorithms
def compute_tides_ICESat2(tide_dir, INPUT_FILE, TIDE_MODEL=None,
    ATLAS_FORMAT=None, GZIP=True, DEFINITION_FILE=None, METHOD='spline',
    EXTRAPOLATE=False, CUTOFF=None, VERBOSE=False, MODE=0o775):

    #-- create logger for verbosity level
    loglevel = logging.INFO if VERBOSE else logging.CRITICAL
    logger = pyTMD.utilities.build_logger('pytmd',level=loglevel)

    #-- get parameters for tide model
    if DEFINITION_FILE is not None:
        model = pyTMD.model(tide_dir).from_file(DEFINITION_FILE)
    else:
        model = pyTMD.model(tide_dir, format=ATLAS_FORMAT,
            compressed=GZIP).elevation(TIDE_MODEL)

    #-- read data from input file
    logger.info('{0} -->'.format(INPUT_FILE))
    IS2_atl12_mds,IS2_atl12_attrs,IS2_atl12_beams = read_HDF5_ATL12(INPUT_FILE,
        ATTRIBUTES=True)
    DIRECTORY = os.path.dirname(INPUT_FILE)
    #-- extract parameters from ICESat-2 ATLAS HDF5 ocean surface file name
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
        r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    try:
        SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = rx.findall(INPUT_FILE).pop()
    except:
        #-- output tide HDF5 file (generic)
        fileBasename,fileExtension = os.path.splitext(INPUT_FILE)
        args = (fileBasename,model.name,fileExtension)
        OUTPUT_FILE = '{0}_{1}_TIDES{2}'.format(*args)
    else:
        #-- output tide HDF5 file for ASAS/NSIDC granules
        args = (PRD,model.name,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX)
        file_format = '{0}_{1}_TIDES_{2}{3}{4}{5}{6}{7}_{8}{9}{10}_{11}_{12}{13}.h5'
        OUTPUT_FILE = file_format.format(*args)

    #-- number of GPS seconds between the GPS epoch
    #-- and ATLAS Standard Data Product (SDP) epoch
    atlas_sdp_gps_epoch = IS2_atl12_mds['ancillary_data']['atlas_sdp_gps_epoch']
    #-- delta time (TT - UT1) file
    delta_file = pyTMD.utilities.get_data_path(['data','merged_deltat.data'])

    #-- copy variables for outputting to HDF5 file
    IS2_atl12_tide = {}
    IS2_atl12_fill = {}
    IS2_atl12_dims = {}
    IS2_atl12_tide_attrs = {}
    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    #-- Add this value to delta time parameters to compute full gps_seconds
    IS2_atl12_tide['ancillary_data'] = {}
    IS2_atl12_tide_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        #-- get each HDF5 variable
        IS2_atl12_tide['ancillary_data'][key] = IS2_atl12_mds['ancillary_data'][key]
        #-- Getting attributes of group and included variables
        IS2_atl12_tide_attrs['ancillary_data'][key] = {}
        for att_name,att_val in IS2_atl12_attrs['ancillary_data'][key].items():
            IS2_atl12_tide_attrs['ancillary_data'][key][att_name] = att_val

    #-- for each input beam within the file
    for gtx in sorted(IS2_atl12_beams):
        #-- output data dictionaries for beam
        IS2_atl12_tide[gtx] = dict(ssh_segments={})
        IS2_atl12_fill[gtx] = dict(ssh_segments={})
        IS2_atl12_dims[gtx] = dict(ssh_segments={})
        IS2_atl12_tide_attrs[gtx] = dict(ssh_segments={})

        #-- number of segments
        val = IS2_atl12_mds[gtx]['ssh_segments']
        n_seg = len(val['delt_seg'])

        #-- convert time from ATLAS SDP to days relative to Jan 1, 1992
        gps_seconds = atlas_sdp_gps_epoch + val['delta_time']
        leap_seconds = pyTMD.time.count_leap_seconds(gps_seconds)
        tide_time = pyTMD.time.convert_delta_time(gps_seconds-leap_seconds,
            epoch1=(1980,1,6,0,0,0), epoch2=(1992,1,1,0,0,0), scale=1.0/86400.0)
        #-- read tidal constants and interpolate to grid points
        if model.format in ('OTIS','ATLAS'):
            amp,ph,D,c = extract_tidal_constants(val['longitude'],
                val['latitude'], model.grid_file, model.model_file,
                model.projection, TYPE=model.type, METHOD=METHOD,
                EXTRAPOLATE=EXTRAPOLATE, CUTOFF=CUTOFF, GRID=model.format)
            deltat = np.zeros_like(tide_time)
        elif (model.format == 'netcdf'):
            amp,ph,D,c = extract_netcdf_constants(val['longitude'],
                val['latitude'], model.grid_file, model.model_file,
                TYPE=model.type, METHOD=METHOD, EXTRAPOLATE=EXTRAPOLATE,
                CUTOFF=CUTOFF, SCALE=model.scale, GZIP=model.compressed)
            deltat = np.zeros_like(tide_time)
        elif (model.format == 'GOT'):
            amp,ph,c = extract_GOT_constants(val['longitude'], val['latitude'],
                model.model_file, METHOD=METHOD, EXTRAPOLATE=EXTRAPOLATE,
                CUTOFF=CUTOFF, SCALE=model.scale, GZIP=model.compressed)
            #-- interpolate delta times from calendar dates to tide time
            deltat = calc_delta_time(delta_file, tide_time)
        elif (model.format == 'FES'):
            amp,ph = extract_FES_constants(val['longitude'], val['latitude'],
                model.model_file, TYPE=model.type, VERSION=model.version,
                METHOD=METHOD, EXTRAPOLATE=EXTRAPOLATE, CUTOFF=CUTOFF,
                SCALE=model.scale, GZIP=model.compressed)
            #-- available model constituents
            c = model.constituents
            #-- interpolate delta times from calendar dates to tide time
            deltat = calc_delta_time(delta_file, tide_time)

        #-- calculate complex phase in radians for Euler's
        cph = -1j*ph*np.pi/180.0
        #-- calculate constituent oscillation
        hc = amp*np.exp(cph)

        #-- predict tidal elevations at time and infer minor corrections
        tide = np.ma.empty((n_seg))
        tide.mask = np.any(hc.mask,axis=1)
        tide.data[:] = predict_tide_drift(tide_time, hc, c,
            DELTAT=deltat, CORRECTIONS=model.format)
        minor = infer_minor_corrections(tide_time, hc, c,
            DELTAT=deltat, CORRECTIONS=model.format)
        tide.data[:] += minor.data[:]
        #-- replace masked and nan values with fill value
        invalid, = np.nonzero(np.isnan(tide.data) | tide.mask)
        tide.data[invalid] = tide.fill_value
        tide.mask[invalid] = True

        #-- group attributes for beam
        IS2_atl12_tide_attrs[gtx]['Description'] = IS2_atl12_attrs[gtx]['Description']
        IS2_atl12_tide_attrs[gtx]['atlas_pce'] = IS2_atl12_attrs[gtx]['atlas_pce']
        IS2_atl12_tide_attrs[gtx]['atlas_beam_type'] = IS2_atl12_attrs[gtx]['atlas_beam_type']
        IS2_atl12_tide_attrs[gtx]['groundtrack_id'] = IS2_atl12_attrs[gtx]['groundtrack_id']
        IS2_atl12_tide_attrs[gtx]['atmosphere_profile'] = IS2_atl12_attrs[gtx]['atmosphere_profile']
        IS2_atl12_tide_attrs[gtx]['atlas_spot_number'] = IS2_atl12_attrs[gtx]['atlas_spot_number']
        IS2_atl12_tide_attrs[gtx]['sc_orientation'] = IS2_atl12_attrs[gtx]['sc_orientation']
        #-- group attributes for ssh_segments
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['Description'] = ("Contains "
            "parameters relating to the calculated surface height.")
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['data_rate'] = ("Data within "
            "this group are stored at the variable ocean processing segment rate.")

        #-- geolocation, time and segment ID
        #-- delta time
        IS2_atl12_tide[gtx]['ssh_segments']['delta_time'] = val['delta_time'].copy()
        IS2_atl12_fill[gtx]['ssh_segments']['delta_time'] = None
        IS2_atl12_dims[gtx]['ssh_segments']['delta_time'] = None
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delta_time'] = {}
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delta_time']['units'] = "seconds since 2018-01-01"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delta_time']['long_name'] = "Elapsed GPS seconds"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delta_time']['standard_name'] = "time"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delta_time']['source'] = "telemetry"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delta_time']['calendar'] = "standard"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delta_time']['description'] = ("Number of "
            "GPS seconds since the ATLAS SDP epoch. The ATLAS Standard Data Products (SDP) epoch "
            "offset is defined within /ancillary_data/atlas_sdp_gps_epoch as the number of GPS "
            "seconds between the GPS epoch (1980-01-06T00:00:00.000000Z UTC) and the ATLAS SDP "
            "epoch. By adding the offset contained within atlas_sdp_gps_epoch to delta time "
            "parameters, the time in gps_seconds relative to the GPS epoch can be computed.")
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delta_time']['coordinates'] = \
            "latitude longitude"
        #-- latitude
        IS2_atl12_tide[gtx]['ssh_segments']['latitude'] = val['latitude'].copy()
        IS2_atl12_fill[gtx]['ssh_segments']['latitude'] = None
        IS2_atl12_dims[gtx]['ssh_segments']['latitude'] = ['delta_time']
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['latitude'] = {}
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['latitude']['units'] = "degrees_north"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['latitude']['contentType'] = "physicalMeasurement"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['latitude']['long_name'] = "Latitude"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['latitude']['standard_name'] = "latitude"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['latitude']['description'] = ("Latitude of "
            "segment center")
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['latitude']['valid_min'] = -90.0
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['latitude']['valid_max'] = 90.0
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['latitude']['coordinates'] = \
            "delta_time longitude"
        #-- longitude
        IS2_atl12_tide[gtx]['ssh_segments']['longitude'] = val['longitude'].copy()
        IS2_atl12_fill[gtx]['ssh_segments']['longitude'] = None
        IS2_atl12_dims[gtx]['ssh_segments']['longitude'] = ['delta_time']
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['longitude'] = {}
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['longitude']['units'] = "degrees_east"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['longitude']['contentType'] = "physicalMeasurement"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['longitude']['long_name'] = "Longitude"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['longitude']['standard_name'] = "longitude"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['longitude']['description'] = ("Longitude of "
            "segment center")
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['longitude']['valid_min'] = -180.0
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['longitude']['valid_max'] = 180.0
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['longitude']['coordinates'] = \
            "delta_time latitude"
        #-- Ocean Segment Duration
        IS2_atl12_tide[gtx]['ssh_segments']['delt_seg'] = val['delt_seg']
        IS2_atl12_fill[gtx]['ssh_segments']['delt_seg'] = None
        IS2_atl12_dims[gtx]['ssh_segments']['delt_seg'] = ['delta_time']
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delt_seg'] = {}
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delt_seg']['units'] = "seconds"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delt_seg']['contentType'] = \
            "referenceInformation"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delt_seg']['long_name'] = \
            "Ocean Segment Duration"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delt_seg']['description'] = \
            "Time duration segment"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['delt_seg']['coordinates'] = \
            "delta_time latitude longitude"

        #-- stats variables
        IS2_atl12_tide[gtx]['ssh_segments']['stats'] = {}
        IS2_atl12_fill[gtx]['ssh_segments']['stats'] = {}
        IS2_atl12_dims[gtx]['ssh_segments']['stats'] = {}
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats'] = {}
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats']['Description'] = ("Contains parameters "
            "related to quality and corrections on the sea surface height parameters.")
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats']['data_rate'] = ("Data within this group "
            "are stored at the variable ocean processing segment rate.")

        #-- computed tide
        IS2_atl12_tide[gtx]['ssh_segments']['stats'][model.atl12] = tide.copy()
        IS2_atl12_fill[gtx]['ssh_segments']['stats'][model.atl12] = tide.fill_value
        IS2_atl12_dims[gtx]['ssh_segments']['stats'][model.atl12] = ['delta_time']
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats'][model.atl12] = {}
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats'][model.atl12]['units'] = "meters"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats'][model.atl12]['contentType'] = "referenceInformation"
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats'][model.atl12]['long_name'] = model.long_name
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats'][model.atl12]['description'] = model.description
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats'][model.atl12]['source'] = model.name
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats'][model.atl12]['reference'] = model.reference
        IS2_atl12_tide_attrs[gtx]['ssh_segments']['stats'][model.atl12]['coordinates'] = \
            "../delta_time ../latitude ../longitude"

    #-- print file information
    logger.info('\t{0}'.format(OUTPUT_FILE))
    HDF5_ATL12_tide_write(IS2_atl12_tide, IS2_atl12_tide_attrs,
        CLOBBER=True, INPUT=os.path.basename(INPUT_FILE),
        FILL_VALUE=IS2_atl12_fill, DIMENSIONS=IS2_atl12_dims,
        FILENAME=os.path.join(DIRECTORY,OUTPUT_FILE))
    #-- change the permissions mode
    os.chmod(os.path.join(DIRECTORY,OUTPUT_FILE), MODE)

#-- PURPOSE: outputting the tide values for ICESat-2 data to HDF5
def HDF5_ATL12_tide_write(IS2_atl12_tide, IS2_atl12_attrs, INPUT=None,
    FILENAME='', FILL_VALUE=None, DIMENSIONS=None, CLOBBER=False):
    #-- setting HDF5 clobber attribute
    if CLOBBER:
        clobber = 'w'
    else:
        clobber = 'w-'

    #-- open output HDF5 file
    fileID = h5py.File(os.path.expanduser(FILENAME), clobber)

    #-- create HDF5 records
    h5 = {}

    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01T00:00:00Z UTC)
    h5['ancillary_data'] = {}
    for k,v in IS2_atl12_tide['ancillary_data'].items():
        #-- Defining the HDF5 dataset variables
        val = 'ancillary_data/{0}'.format(k)
        h5['ancillary_data'][k] = fileID.create_dataset(val, np.shape(v), data=v,
            dtype=v.dtype, compression='gzip')
        #-- add HDF5 variable attributes
        for att_name,att_val in IS2_atl12_attrs['ancillary_data'][k].items():
            h5['ancillary_data'][k].attrs[att_name] = att_val

    #-- write each output beam
    beams = [k for k in IS2_atl12_tide.keys() if bool(re.match(r'gt\d[lr]',k))]
    for gtx in beams:
        fileID.create_group(gtx)
        #-- add HDF5 group attributes for beam
        for att_name in ['Description','atlas_pce','atlas_beam_type',
            'groundtrack_id','atmosphere_profile','atlas_spot_number',
            'sc_orientation']:
            fileID[gtx].attrs[att_name] = IS2_atl12_attrs[gtx][att_name]
        #-- create ssh_segments group
        fileID[gtx].create_group('ssh_segments')
        h5[gtx] = dict(ssh_segments={})
        for att_name in ['Description','data_rate']:
            att_val = IS2_atl12_attrs[gtx]['ssh_segments'][att_name]
            fileID[gtx]['ssh_segments'].attrs[att_name] = att_val

        #-- delta_time, geolocation and segment description variables
        for k in ['delta_time','latitude','longitude','delt_seg']:
            #-- values and attributes
            v = IS2_atl12_tide[gtx]['ssh_segments'][k]
            attrs = IS2_atl12_attrs[gtx]['ssh_segments'][k]
            fillvalue = FILL_VALUE[gtx]['ssh_segments'][k]
            #-- Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}'.format(gtx,'ssh_segments',k)
            if fillvalue:
                h5[gtx]['ssh_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, fillvalue=fillvalue,
                    compression='gzip')
            else:
                h5[gtx]['ssh_segments'][k] = fileID.create_dataset(val,
                    np.shape(v), data=v, dtype=v.dtype, compression='gzip')
            #-- create or attach dimensions for HDF5 variable
            if DIMENSIONS[gtx]['ssh_segments'][k]:
                #-- attach dimensions
                for i,dim in enumerate(DIMENSIONS[gtx]['ssh_segments'][k]):
                    h5[gtx]['ssh_segments'][k].dims[i].attach_scale(
                        h5[gtx]['ssh_segments'][dim])
            else:
                #-- make dimension
                h5[gtx]['ssh_segments'][k].make_scale(k)
            #-- add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['ssh_segments'][k].attrs[att_name] = att_val

        #-- add to stats variables
        key = 'stats'
        fileID[gtx]['ssh_segments'].create_group(key)
        h5[gtx]['ssh_segments'][key] = {}
        for att_name in ['Description','data_rate']:
            att_val=IS2_atl12_attrs[gtx]['ssh_segments'][key][att_name]
            fileID[gtx]['ssh_segments'][key].attrs[att_name] = att_val
        for k,v in IS2_atl12_tide[gtx]['ssh_segments'][key].items():
            #-- attributes
            attrs = IS2_atl12_attrs[gtx]['ssh_segments'][key][k]
            fillvalue = FILL_VALUE[gtx]['ssh_segments'][key][k]
            #-- Defining the HDF5 dataset variables
            val = '{0}/{1}/{2}/{3}'.format(gtx,'ssh_segments',key,k)
            if fillvalue:
                h5[gtx]['ssh_segments'][key][k] = \
                    fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, fillvalue=fillvalue, compression='gzip')
            else:
                h5[gtx]['ssh_segments'][key][k] = \
                    fileID.create_dataset(val, np.shape(v), data=v,
                    dtype=v.dtype, compression='gzip')
            #-- attach dimensions
            for i,dim in enumerate(DIMENSIONS[gtx]['ssh_segments'][key][k]):
                h5[gtx]['ssh_segments'][key][k].dims[i].attach_scale(
                    h5[gtx]['ssh_segments'][dim])
            #-- add HDF5 variable attributes
            for att_name,att_val in attrs.items():
                h5[gtx]['ssh_segments'][key][k].attrs[att_name] = att_val

    #-- HDF5 file title
    fileID.attrs['featureType'] = 'trajectory'
    fileID.attrs['title'] = 'ATLAS/ICESat-2 L3A Ocean Surface Height'
    fileID.attrs['summary'] = ('Estimates of the ocean surface tidal parameters '
        'needed to interpret and assess the quality of ocean height estimates.')
    fileID.attrs['description'] = ('Sea Surface Height (SSH) of the global '
        'open ocean including the ice-free seasonal ice zone (SIZ) and '
        'near-coast regions.')
    date_created = datetime.datetime.today()
    fileID.attrs['date_created'] = date_created.isoformat()
    project = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = project
    platform = 'ICESat-2 > Ice, Cloud, and land Elevation Satellite-2'
    fileID.attrs['project'] = platform
    #-- add attribute for elevation instrument and designated processing level
    instrument = 'ATLAS > Advanced Topographic Laser Altimeter System'
    fileID.attrs['instrument'] = instrument
    fileID.attrs['source'] = 'Spacecraft'
    fileID.attrs['references'] = 'https://nsidc.org/data/icesat-2'
    fileID.attrs['processing_level'] = '4'
    #-- add attributes for input ATL12 file
    fileID.attrs['input_files'] = os.path.basename(INPUT)
    #-- find geospatial and temporal ranges
    lnmn,lnmx,ltmn,ltmx,tmn,tmx = (np.inf,-np.inf,np.inf,-np.inf,np.inf,-np.inf)
    for gtx in beams:
        lon = IS2_atl12_tide[gtx]['ssh_segments']['longitude']
        lat = IS2_atl12_tide[gtx]['ssh_segments']['latitude']
        delta_time = IS2_atl12_tide[gtx]['ssh_segments']['delta_time']
        #-- setting the geospatial and temporal ranges
        lnmn = lon.min() if (lon.min() < lnmn) else lnmn
        lnmx = lon.max() if (lon.max() > lnmx) else lnmx
        ltmn = lat.min() if (lat.min() < ltmn) else ltmn
        ltmx = lat.max() if (lat.max() > ltmx) else ltmx
        tmn = delta_time.min() if (delta_time.min() < tmn) else tmn
        tmx = delta_time.max() if (delta_time.max() > tmx) else tmx
    #-- add geospatial and temporal attributes
    fileID.attrs['geospatial_lat_min'] = ltmn
    fileID.attrs['geospatial_lat_max'] = ltmx
    fileID.attrs['geospatial_lon_min'] = lnmn
    fileID.attrs['geospatial_lon_max'] = lnmx
    fileID.attrs['geospatial_lat_units'] = "degrees_north"
    fileID.attrs['geospatial_lon_units'] = "degrees_east"
    fileID.attrs['geospatial_ellipsoid'] = "WGS84"
    fileID.attrs['date_type'] = 'UTC'
    fileID.attrs['time_type'] = 'CCSDS UTC-A'
    #-- convert start and end time from ATLAS SDP seconds into GPS seconds
    atlas_sdp_gps_epoch=IS2_atl12_tide['ancillary_data']['atlas_sdp_gps_epoch']
    gps_seconds = atlas_sdp_gps_epoch + np.array([tmn,tmx])
    #-- calculate leap seconds
    leaps = pyTMD.time.count_leap_seconds(gps_seconds)
    #-- convert from seconds since 1980-01-06T00:00:00 to Julian days
    time_julian = 2400000.5 + pyTMD.time.convert_delta_time(gps_seconds - leaps,
        epoch1=(1980,1,6,0,0,0), epoch2=(1858,11,17,0,0,0), scale=1.0/86400.0)
    #-- convert to calendar date
    YY,MM,DD,HH,MN,SS = pyTMD.time.convert_julian(time_julian,FORMAT='tuple')
    #-- add attributes with measurement date start, end and duration
    tcs = datetime.datetime(int(YY[0]), int(MM[0]), int(DD[0]),
        int(HH[0]), int(MN[0]), int(SS[0]), int(1e6*(SS[0] % 1)))
    fileID.attrs['time_coverage_start'] = tcs.isoformat()
    tce = datetime.datetime(int(YY[1]), int(MM[1]), int(DD[1]),
        int(HH[1]), int(MN[1]), int(SS[1]), int(1e6*(SS[1] % 1)))
    fileID.attrs['time_coverage_end'] = tce.isoformat()
    fileID.attrs['time_coverage_duration'] = '{0:0.0f}'.format(tmx-tmn)
    #-- Closing the HDF5 file
    fileID.close()

#-- Main program that calls compute_tides_ICESat2()
def main():
    #-- Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Calculates tidal elevations for correcting ICESat-2 ATL12
            ocean surface height data
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    #-- command line parameters
    group = parser.add_mutually_exclusive_group(required=True)
    #-- input ICESat-2 ocean surface height files
    parser.add_argument('infile',
        type=lambda p: os.path.abspath(os.path.expanduser(p)), nargs='+',
        help='ICESat-2 ATL12 file to run')
    #-- directory with tide data
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    #-- tide model to use
    choices = sorted(pyTMD.model.ocean_elevation() + pyTMD.model.load_elevation())
    group.add_argument('--tide','-T',
        metavar='TIDE', type=str,
        choices=choices,
        help='Tide model to use in correction')
    parser.add_argument('--atlas-format',
        type=str, choices=('OTIS','netcdf'), default='netcdf',
        help='ATLAS tide model format')
    parser.add_argument('--gzip','-G',
        default=False, action='store_true',
        help='Tide model files are gzip compressed')
    #-- tide model definition file to set an undefined model
    group.add_argument('--definition-file',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        help='Tide model definition file for use as correction')
    #-- interpolation method
    parser.add_argument('--interpolate','-I',
        metavar='METHOD', type=str, default='spline',
        choices=('spline','linear','nearest','bilinear'),
        help='Spatial interpolation method')
    #-- extrapolate with nearest-neighbors
    parser.add_argument('--extrapolate','-E',
        default=False, action='store_true',
        help='Extrapolate with nearest-neighbors')
    #-- extrapolation cutoff in kilometers
    #-- set to inf to extrapolate over all points
    parser.add_argument('--cutoff','-c',
        type=np.float64, default=10.0,
        help='Extrapolation cutoff in kilometers')
    #-- verbosity settings
    #-- verbose will output information about each output file
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Output information about each created file')
    #-- permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files created')
    args,_ = parser.parse_known_args()

    #-- run for each input ATL12 file
    for FILE in args.infile:
        compute_tides_ICESat2(args.directory, FILE, TIDE_MODEL=args.tide,
            ATLAS_FORMAT=args.atlas_format, GZIP=args.gzip,
            DEFINITION_FILE=args.definition_file, METHOD=args.interpolate,
            EXTRAPOLATE=args.extrapolate, CUTOFF=args.cutoff,
            VERBOSE=args.verbose, MODE=args.mode)

#-- run main program
if __name__ == '__main__':
    main()
