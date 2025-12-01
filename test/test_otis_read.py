#!/usr/bin/env python
u"""
test_otis_read.py (11/2025)
Tests for OTIS-formatted tide model data

Tests that constituents are being extracted
Tests that interpolated results are comparable to Matlab TMD program
    https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    Oct2Py: Python to GNU Octave Bridge
        https://oct2py.readthedocs.io/en/latest/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

UPDATE HISTORY:
    Updated 11/2025: using new xarray interface for tidal model data
    Updated 10/2025: split directories between validation and model data
        fetch data from pyTMD developers test data repository
    Updated 09/2025: added check if running on GitHub Actions or locally
        renamed test_download_and_read.py to test_otis_read.py
    Updated 08/2025: added xarray tests to verify implementation
    Updated 06/2025: subset to specific constituents when reading model
    Updated 12/2024: create test files from matlab program for comparison
    Updated 09/2024: drop support for the ascii definition file format
        use model class attributes for file format and corrections
        using new JSON dictionary format for model projections
    Updated 07/2024: add parametrize over cropping the model fields
    Updated 04/2024: use timescale for temporal operations
    Updated 01/2024: refactored compute functions into compute.py
    Updated 04/2023: using pathlib to define and expand paths
    Updated 12/2022: add check for read and interpolate constants
    Updated 11/2022: added encoding for writing ascii files
        use f-strings for formatting verbose or ascii output
    Updated 10/2022: added encoding for reading ascii files
    Updated 09/2021: added test for model definition files
    Updated 07/2021: download CATS2008 and AntTG from S3 to bypass USAP captcha
    Updated 05/2021: added test for check point program
    Updated 03/2021: use pytest fixture to setup and teardown model data
        use TMD tmd_tide_pred_plus to calculate OB time series
        refactor program into two classes for CATS2008 and AOTIM-5-2018
        replaced numpy bool/int to prevent deprecation warnings
    Updated 01/2021: download CATS2008 and AOTIM-5-2018 to subdirectories
    Updated 08/2020: Download Antarctic tide gauge database and compare with RMS
        directly call Matlab program (octave+oct2py) and compare outputs
        compare outputs for both Antarctic (CATS2008) and Arctic (AOTIM-5-2018)
        will install octave and oct2py in development requirements
    Written 08/2020
"""
import io
import copy
import gzip
import json
import pytest
import inspect
import pathlib
import numpy as np
import xarray as xr
import pyTMD
import timescale

# attempt imports
pd = pyTMD.utilities.import_dependency('pandas')
oct2py = pyTMD.utilities.import_dependency('oct2py')

# current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
filepath = pathlib.Path(filename).absolute().parent

# PURPOSE: calculate the matlab serial date from calendar date
# http://scienceworld.wolfram.com/astronomy/JulianDate.html
def convert_calendar_serial(year, month, day, hour=0.0, minute=0.0, second=0.0):
    # return the date in days since serial epoch 0000-01-01T00:00:00
    sd = 367.0*year - np.floor(7.0*(year + np.floor((month+9.0)/12.0))/4.0) - \
        np.floor(3.0*(np.floor((year + (month - 9.0)/7.0)/100.0) + 1.0)/4.0) + \
        np.floor(275.0*month/9.0) + day + hour/24.0 + minute/1440.0 + \
        second/86400.0 - 30.0
    return sd

# PURPOSE: Test and Verify CATS2008 model read and prediction programs
class Test_CATS2008:
    @pytest.fixture(autouse=True)
    def init(self, directory):
        self.directory = pathlib.Path(directory).expanduser().absolute()

    # PURPOSE: create verification from Matlab program
    @pytest.fixture(scope="class", autouse=False)
    def update_verify_CATS2008(self, directory):
        # compute validation data from Matlab TMD program using octave
        # https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5
        octave = copy.copy(oct2py.octave)
        TMDpath = directory.joinpath('..','TMD_Matlab_Toolbox','TMD').absolute()
        octave.addpath(octave.genpath(str(TMDpath)))
        octave.addpath(str(directory))
        # turn off octave warnings
        octave.warning('off', 'all')
        # model parameters for CATS2008
        model = pyTMD.io.model(directory).from_database('CATS2008')
        # iterate over type: heights versus currents
        for TYPE in ['z', 'U', 'V']:
            # path to tide model files
            modelpath = model[TYPE].grid_file.parent
            octave.addpath(str(modelpath))
            # input control file for model
            CFname = directory.joinpath('Model_CATS2008')
            assert CFname.exists()

            # open Antarctic Tide Gauge (AntTG) database
            AntTG = directory.joinpath('AntTG_ocean_height_v1.txt')
            with AntTG.open(mode='r', encoding='utf8') as f:
                file_contents = f.read().splitlines()
            # counts the number of lines in the header
            count = 0
            HEADER = True
            # Reading over header text
            while HEADER:
                # check if file line at count starts with matlab comment string
                HEADER = file_contents[count].startswith('%')
                # add 1 to counter
                count += 1
            # rewind 1 line
            count -= 1
            # iterate over number of stations
            antarctic_stations = (len(file_contents) - count)//10
            stations = [None]*antarctic_stations
            shortname = [None]*antarctic_stations
            station_type = [None]*antarctic_stations
            station_lon = np.zeros((antarctic_stations))
            station_lat = np.zeros((antarctic_stations))
            for s in range(antarctic_stations):
                i = count + s*10
                stations[s] = file_contents[i + 1].strip()
                shortname[s] = file_contents[i + 3].strip()
                lat,lon,_,_ = file_contents[i + 4].split()
                station_type[s] = file_contents[i + 6].strip()
                station_lon[s] = np.float64(lon)
                station_lat[s] = np.float64(lat)

            # calculate daily results for a time period
            # serial dates for matlab program (days since 0000-01-01T00:00:00)
            SDtime = np.arange(convert_calendar_serial(2000,1,1),
                convert_calendar_serial(2000,12,31)+1)

            # run Matlab TMD program with octave
            # MODE: OB time series
            validation,_ = octave.tmd_tide_pred_plus(str(CFname), SDtime,
                station_lat, station_lon,
                TYPE, nout=2)
            
            # create dataframe for validation data
            df = pd.DataFrame(data=validation, index=SDtime, columns=shortname)

            # add attributes for each valid station
            for i,s in enumerate(shortname):
                df[s].attrs['station'] = stations[i]
                df[s].attrs['type'] = station_type[i]
                df[s].attrs['latitude'] = station_lat[i]
                df[s].attrs['longitude'] = station_lon[i]

            # save to (gzipped) csv
            output_file = filepath.joinpath(f'TMDv2.5_CATS2008_{TYPE}.csv.gz')
            with gzip.open(output_file, 'wb') as f:
                df.to_csv(f, index_label='time')

    # PURPOSE: create ellipse verification from Matlab program
    @pytest.fixture(scope="class", autouse=False)
    def update_tidal_ellipse(self, directory):
        # model parameters for CATS2008
        model = pyTMD.io.model(directory).from_database('CATS2008')
        modelpath = model['u'].grid_file.parent
        TYPES = ['U','V']

        # compute validation data from Matlab TMD program using octave
        # https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5
        octave = copy.copy(oct2py.octave)
        TMDpath = directory.joinpath('..','TMD_Matlab_Toolbox','TMD').absolute()
        octave.addpath(octave.genpath(str(TMDpath)))
        octave.addpath(str(directory))
        octave.addpath(str(modelpath))
        # turn off octave warnings
        octave.warning('off', 'all')
        # input control file for model
        CFname = directory.joinpath('Model_CATS2008')
        assert CFname.exists()

        # open Antarctic Tide Gauge (AntTG) database
        AntTG = directory.joinpath('AntTG_ocean_height_v1.txt')
        with AntTG.open(mode='r', encoding='utf8') as f:
            file_contents = f.read().splitlines()
        # counts the number of lines in the header
        count = 0
        HEADER = True
        # Reading over header text
        while HEADER:
            # check if file line at count starts with matlab comment string
            HEADER = file_contents[count].startswith('%')
            # add 1 to counter
            count += 1
        # rewind 1 line
        count -= 1
        # iterate over number of stations
        antarctic_stations = (len(file_contents) - count)//10
        stations = [None]*antarctic_stations
        shortname = [None]*antarctic_stations
        station_type = [None]*antarctic_stations
        station_lon = np.zeros((antarctic_stations))
        station_lat = np.zeros((antarctic_stations))
        for s in range(antarctic_stations):
            i = count + s*10
            stations[s] = file_contents[i + 1].strip()
            shortname[s] = file_contents[i + 3].strip()
            lat,lon,_,_ = file_contents[i + 4].split()
            station_type[s] = file_contents[i + 6].strip()
            station_lon[s] = np.float64(lon)
            station_lat[s] = np.float64(lat)

        # save complex amplitude for each current
        hc = {}
        # iterate over zonal and meridional currents
        for TYPE in TYPES:
            # extract tidal harmonic constants out of a tidal model
            amp,ph,D,cons = octave.tmd_extract_HC(str(CFname),
                station_lat, station_lon, TYPE, nout=4)
            # calculate complex phase in radians for Euler's
            cph = -1j*ph*np.pi/180.0
            # calculate constituent oscillation for station
            hc[TYPE] = amp*np.exp(cph)

        # compute tidal ellipse parameters for TMD matlab program
        umajor,uminor,uincl,uphase = octave.TideEl(hc['U'],hc['V'],nout=4)
        # build matrix of ellipse parameters
        ellipse = np.r_[umajor,uminor,uincl,uphase]
        # build index for dataframe
        index = []
        for i,c in enumerate(cons):
            c = c.strip()
            cindex = [f'{c}_umajor',f'{c}_uminor',f'{c}_uincl',f'{c}_uphase']
            index.extend(cindex)

        # create dataframe for validation data
        df = pd.DataFrame(data=ellipse, index=index, columns=shortname)

        # add attributes for each valid station
        for i,s in enumerate(shortname):
            df[s].attrs['station'] = stations[i]
            df[s].attrs['type'] = station_type[i]
            df[s].attrs['latitude'] = station_lat[i]
            df[s].attrs['longitude'] = station_lon[i]

        # save to (gzipped) csv
        output_file = filepath.joinpath(f'TMDv2.5_CATS2008_ellipse.csv.gz')
        with gzip.open(output_file, 'wb') as f:
            df.to_csv(f, index_label='ellipse')

    # PURPOSE: Tests check point program
    def test_check_CATS2008(self):
        lons = np.zeros((10)) + 178.0
        lats = -45.0 - np.arange(10)*5.0
        obs = pyTMD.compute.tide_masks(lons, lats, directory=self.directory,
            model='CATS2008', crs=4326)
        exp = np.array([False, False, False, False, True,
            True, True, True, False, False])
        assert np.all(obs == exp)

    # PURPOSE: Tests that interpolated results are comparable to AntTG database
    @pytest.mark.parametrize("use_mmap", [False, True])
    def test_compare_CATS2008(self, use_mmap):
        # model parameters for CATS2008
        model = pyTMD.io.model(self.directory).from_database('CATS2008')
        # open model dataset
        ds = model.open_dataset(type='z', use_mmap=use_mmap)
        # convert dataset to cm
        ds = ds.tmd.to_units('cm')

        # open Antarctic Tide Gauge (AntTG) database
        AntTG = self.directory.joinpath('AntTG_ocean_height_v1.txt')
        with AntTG.open(mode='r', encoding='utf8') as f:
            file_contents = f.read().splitlines()
        # counts the number of lines in the header
        count = 0
        HEADER = True
        # Reading over header text
        while HEADER:
            # check if file line at count starts with matlab comment string
            HEADER = file_contents[count].startswith('%')
            # add 1 to counter
            count += 1
        # rewind 1 line
        count -= 1
        # iterate over number of stations
        constituents = ['q1','o1','p1','k1','n2','m2','s2','k2']
        antarctic_stations = (len(file_contents) - count)//10
        stations = [None]*antarctic_stations
        shortname = [None]*antarctic_stations
        station_lon = np.zeros((antarctic_stations))
        station_lat = np.zeros((antarctic_stations))
        station_amp = np.ma.zeros((antarctic_stations,len(constituents)))
        station_ph = np.ma.zeros((antarctic_stations,len(constituents)))
        for s in range(antarctic_stations):
            i = count + s*10
            stations[s] = file_contents[i + 1].strip()
            shortname[s] = file_contents[i + 3].strip()
            lat,lon,_,_ = file_contents[i + 4].split()
            station_lon[s] = np.float64(lon)
            station_lat[s] = np.float64(lat)
            amp = file_contents[i + 7].split()
            ph = file_contents[i + 8].split()
            station_amp.data[s,:] = np.array(amp,dtype=np.float64)
            station_ph.data[s,:] = np.array(ph,dtype=np.float64)
        # update masks where NaN
        station_amp.mask = np.isnan(station_amp.data) | \
            (station_amp.data == 0.0)
        station_ph.mask = np.isnan(station_ph.data)
        # replace nans with fill values
        station_amp.data[station_amp.mask] = station_amp.fill_value
        station_ph.data[station_ph.mask] = station_ph.fill_value
        # calculate complex constituent oscillations
        station_z = station_amp*np.exp(-1j*station_ph*np.pi/180.0)

        # convert data to coordinate reference system of model
        X, Y = ds.tmd.transform(station_lon, station_lat, crs=4326)
        # convert to data arrays
        X = xr.DataArray(X, dims=('station'))
        Y = xr.DataArray(Y, dims=('station'))
        # extract amplitude and phase from tide model
        local = ds.tmd.interp(X, Y, extrapolate=True)

        # find stations with all constituents valid
        valid_stations = np.all(np.logical_not(station_z.mask), axis=1)
        for i,c in enumerate(constituents):
            valid_stations &= np.isfinite(local[c].values)
        # valid stations for all constituents
        invalid_list = ['Ablation Lake','Amery','Bahia Esperanza','Beaver Lake',
            'Cape Roberts','Casey','Doake Ice Rumples','EE4A','EE4B',
            'Eklund Islands','Gerlache C','Groussac','Gurrachaga',
            'Half Moon Is.','Heard Island','Hobbs Pool','Mawson','McMurdo',
            'Mikkelsen','Palmer','Primavera','Rutford GL','Rutford GPS',
            'Rothera','Scott Base','Seymour Is','Terra Nova Bay']
        # remove coastal stations from the list
        invalid_stations = [i for i,s in enumerate(shortname)
            if s in invalid_list]
        valid_stations[invalid_stations] = False
        nv = len(valid_stations)
        # find valid stations for constituents
        valid, = np.nonzero(valid_stations)
        # compare with RMS values from King et al. (2011)
        # https://doi.org/10.1029/2011JC006949
        RMS = np.array([1.4,2.7,1.7,3.5,2.9,7.3,5.0,1.7])
        rms = np.zeros((len(constituents)))
        for i,c in enumerate(constituents):
            # reduce to valid stations
            model_z = local[c].values[valid]
            # calculate difference and rms
            difference = np.abs(station_z[valid,i] - model_z)
            variance = np.sum(difference**2)/(2.0*nv)
            # round to precision of King et al. (2011)
            rms[i] = np.round(np.sqrt(variance), decimals=1)
        # test RMS differences
        assert np.all(rms <= RMS)

    # parameterize type: heights versus currents
    @pytest.mark.parametrize("use_mmap", [False, True])
    # PURPOSE: Tests that interpolated results are comparable to Matlab program
    def test_verify_CATS2008(self, use_mmap):
        # model parameters for CATS2008
        model = pyTMD.io.model(self.directory).from_database('CATS2008')
        # open datatree for model
        TYPES = ['z','U','V']
        dtree = model.open_datatree(type=TYPES, use_mmap=use_mmap)

        # open Antarctic Tide Gauge (AntTG) database
        AntTG = self.directory.joinpath('AntTG_ocean_height_v1.txt')
        with AntTG.open(mode='r', encoding='utf8') as f:
            file_contents = f.read().splitlines()
        # counts the number of lines in the header
        count = 0
        HEADER = True
        # Reading over header text
        while HEADER:
            # check if file line at count starts with matlab comment string
            HEADER = file_contents[count].startswith('%')
            # add 1 to counter
            count += 1
        # rewind 1 line
        count -= 1
        # iterate over number of stations
        antarctic_stations = (len(file_contents) - count)//10
        stations = [None]*antarctic_stations
        shortname = [None]*antarctic_stations
        station_type = [None]*antarctic_stations
        station_lon = np.zeros((antarctic_stations))
        station_lat = np.zeros((antarctic_stations))
        for s in range(antarctic_stations):
            i = count + s*10
            stations[s] = file_contents[i + 1].strip()
            shortname[s] = file_contents[i + 3].strip()
            lat,lon,_,_ = file_contents[i + 4].split()
            station_type[s] = file_contents[i + 6].strip()
            station_lon[s] = np.float64(lon)
            station_lat[s] = np.float64(lat)

        # convert data to coordinate reference system of model
        X, Y = dtree.tmd.transform(station_lon, station_lat, crs=4326)
        # convert to data arrays
        X = xr.DataArray(X, dims=('station'))
        Y = xr.DataArray(Y, dims=('station'))
        # extract amplitude and phase from tide model
        local = dtree.tmd.interp(X, Y, extrapolate=True)

        # compare daily outputs at each station point
        invalid_list = ['Ablation Lake','Amery','Bahia Esperanza','Beaver Lake',
            'Cape Roberts','Casey','Doake Ice Rumples','EE4A','EE4B',
            'Eklund Islands','Gerlache C','Groussac','Gurrachaga',
            'Half Moon Is.','Heard Island','Hobbs Pool','Mawson','McMurdo',
            'Mikkelsen','Palmer','Primavera','Rutford GL','Rutford GPS',
            'Rothera','Scott Base','Seymour Is','Terra Nova Bay']
        # remove coastal stations from the list
        valid_stations=[i for i,s in enumerate(shortname)
            if s not in invalid_list]
        # will verify differences between model outputs are within tolerance
        eps = np.finfo(np.float16).eps

        # for each type of data (z, u, v)
        for TYPE, ds in local.items():
            # convert to dataset
            ds = ds.to_dataset()
            # read validation data from Matlab TMD program
            # https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5
            validation_file = f'TMDv2.5_CATS2008_{TYPE}.csv.gz'
            df = pd.read_csv(filepath.joinpath(validation_file))
            # calculate daily results for a time period
            # convert time to days since 1992-01-01T00:00:00
            ts = timescale.from_julian(df.time.values + 1721058.5)
            # not converting times to dynamic times for model comparisons
            deltat = np.zeros_like(ts.tt_ut1)

            # predict tides and infer minor corrections
            tide = ds.tmd.predict(ts.tide, deltat=deltat,
                corrections=model['corrections'])
            tide += ds.tmd.infer(ts.tide, deltat=deltat,
                corrections=model['corrections'])

            # for each valid station
            for i,s in enumerate(valid_stations):
                # get station name
                station = shortname[s]
                # calculate differences between matlab and python version
                difference = tide.isel(station=s) - df[station].values
                difference = difference.where(~np.isnan(difference), other=0.0)
                assert np.all(np.abs(difference) < eps)

    # PURPOSE: Tests that tidal ellipse results are comparable to Matlab program
    @pytest.mark.parametrize("use_mmap", [False, True])
    def test_tidal_ellipse(self, use_mmap):
        # model parameters for CATS2008
        model = pyTMD.io.model(self.directory).from_database('CATS2008')
        # open datatree for model
        TYPES = ['U','V']
        dtree = model.open_datatree(type=TYPES, use_mmap=use_mmap)

        # open Antarctic Tide Gauge (AntTG) database
        AntTG = self.directory.joinpath('AntTG_ocean_height_v1.txt')
        with AntTG.open(mode='r', encoding='utf8') as f:
            file_contents = f.read().splitlines()
        # counts the number of lines in the header
        count = 0
        HEADER = True
        # Reading over header text
        while HEADER:
            # check if file line at count starts with matlab comment string
            HEADER = file_contents[count].startswith('%')
            # add 1 to counter
            count += 1
        # rewind 1 line
        count -= 1
        # iterate over number of stations
        antarctic_stations = (len(file_contents) - count)//10
        stations = [None]*antarctic_stations
        shortname = [None]*antarctic_stations
        station_type = [None]*antarctic_stations
        station_lon = np.zeros((antarctic_stations))
        station_lat = np.zeros((antarctic_stations))
        for s in range(antarctic_stations):
            i = count + s*10
            stations[s] = file_contents[i + 1].strip()
            shortname[s] = file_contents[i + 3].strip()
            lat,lon,_,_ = file_contents[i + 4].split()
            station_type[s] = file_contents[i + 6].strip()
            station_lon[s] = np.float64(lon)
            station_lat[s] = np.float64(lat)

        # compare outputs at each station point
        invalid_list = ['Ablation Lake','Amery','Bahia Esperanza','Beaver Lake',
            'Cape Roberts','Casey','Doake Ice Rumples','EE4A','EE4B',
            'Eklund Islands','Gerlache C','Groussac','Gurrachaga',
            'Half Moon Is.','Heard Island','Hobbs Pool','Mawson','McMurdo',
            'Mikkelsen','Palmer','Primavera','Rutford GL','Rutford GPS',
            'Rothera','Scott Base','Seymour Is','Terra Nova Bay']
        # remove coastal stations from the list
        valid_stations=[i for i,s in enumerate(shortname)
            if s not in invalid_list]
        # will verify differences between model outputs are within tolerance
        eps = np.finfo(np.float16).eps

        # read validation data from Matlab TMD program
        # https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5
        validation_file = 'TMDv2.5_CATS2008_ellipse.csv.gz'
        df = pd.read_csv(filepath.joinpath(validation_file),
            index_col='ellipse')
        # number of constituents
        nc = int(df.shape[0]//4)

        # convert data to coordinate reference system of model
        X, Y = dtree.tmd.transform(station_lon, station_lat, crs=4326)
        # convert to data arrays
        X = xr.DataArray(X, dims=('station'))
        Y = xr.DataArray(Y, dims=('station'))
        # extract amplitude and phase from tide model
        local = dtree.tmd.interp(X, Y, extrapolate=True)

        # compute tidal ellipse parameters for python program
        test = local.tmd.to_ellipse()
        dmajor = test['major'].to_dataset()
        dminor = test['minor'].to_dataset()
        dincl = test['incl'].to_dataset()
        dphase = test['phase'].to_dataset()

        # for each valid station
        for i,s in enumerate(valid_stations):
            station = shortname[s]
            # extract ellipse parameters from dataframe
            umajor,uminor,uincl,uphase = df[station].values.reshape(4,nc)
            # calculate differences between matlab and python version
            difference = dmajor.isel(station=s).tmd.to_dataarray() - umajor
            difference = difference.where(~np.isnan(difference), other=0.0)
            assert np.all(np.abs(difference) < eps)
            difference = dminor.isel(station=s).tmd.to_dataarray() - uminor
            difference = difference.where(~np.isnan(difference), other=0.0)
            assert np.all(np.abs(difference) < eps)
            difference = dincl.isel(station=s).tmd.to_dataarray() - uincl
            difference = difference.where(~np.isnan(difference), other=0.0)
            assert np.all(np.abs(difference) < eps)
            difference = dphase.isel(station=s).tmd.to_dataarray() - uphase
            difference = difference.where(~np.isnan(difference), other=0.0)
            assert np.all(np.abs(difference) < eps)

        # calculate currents using tidal ellipse inverse
        inverse = test.tmd.from_ellipse()
        # calculate differences between forward and inverse functions
        for key in ['U', 'V']:
            ds = (local[key] - inverse[key]).to_dataset()
            difference = ds.tmd.to_dataarray()
            difference = difference.where(~np.isnan(difference), other=0.0)
            assert np.all(np.abs(difference) < eps)

    # PURPOSE: Tests solving for harmonic constants
    @pytest.mark.parametrize("SOLVER", ['lstsq','gelsy','gelss','gelsd','bvls'])
    def test_solve(self, SOLVER):
        # get model parameters
        model = pyTMD.io.model(self.directory).from_database('CATS2008')
        # open dataset
        ds = model.open_dataset(type='z')

        # calculate a forecast every minute
        minutes = np.arange(366*1440)
        # convert time to days relative to Jan 1, 1992 (48622 MJD)
        year, month, day = 2000, 1, 1
        ts = timescale.from_calendar(year, month, day, minute=minutes)
        DELTAT = np.zeros_like(ts.tide)

        # interpolate constants to a coordinate
        LAT, LON = (-76.0, -40.0)
        # convert data to coordinate reference system of model
        X, Y = ds.tmd.transform(LON, LAT, crs=4326)
        # extract amplitude and phase from tide model
        local = ds.tmd.interp(X, Y, extrapolate=True)
        # model constituents
        c = local.tmd.constituents

        # predict tidal elevations at times
        TIDE = local.tmd.predict(ts.tide, deltat=DELTAT,
            corrections=model.corrections)
        # solve for amplitude and phase
        ds = pyTMD.solve.constants(ts.tide, TIDE, c,
            solver=SOLVER)
        # verify differences are within tolerance
        eps = 5e-3
        for k,cons in enumerate(c):
            assert np.isclose(local[cons], ds[cons], rtol=eps, atol=eps)

    # PURPOSE: test the tide correction wrapper function
    def test_Ross_Ice_Shelf(self):
        # create a drift track along the Ross Ice Shelf
        xlimits = np.array([-740000,520000])
        ylimits = np.array([-1430000,-300000])
        # limits of x and y coordinates for region
        xrange = xlimits[1] - xlimits[0]
        yrange = ylimits[1] - ylimits[0]
        # x and y coordinates
        x = xlimits[0] + xrange*np.random.random((100))
        y = ylimits[0] + yrange*np.random.random((100))
        # time dimension
        delta_time = np.random.random((100))*86400
        # calculate tide drift corrections
        tide = pyTMD.compute.tide_elevations(x, y, delta_time,
            directory=self.directory, model='CATS2008',
            epoch=timescale.time._j2000_epoch, type='drift', standard='UTC',
            crs=3031, extrapolate=True)
        assert np.any(tide)

    # PURPOSE: test the tide currents wrapper function
    def test_Ross_Ice_Shelf_currents(self):
        # create a drift track along the Ross Ice Shelf
        xlimits = np.array([-740000,520000])
        ylimits = np.array([-1430000,-300000])
        # limits of x and y coordinates for region
        xrange = xlimits[1] - xlimits[0]
        yrange = ylimits[1] - ylimits[0]
        # x and y coordinates
        x = xlimits[0] + xrange*np.random.random((100))
        y = ylimits[0] + yrange*np.random.random((100))
        # time dimension
        delta_time = np.random.random((100))*86400
        # calculate tide drift corrections
        tide = pyTMD.compute.tide_currents(x, y, delta_time,
            directory=self.directory, model='CATS2008',
            epoch=timescale.time._j2000_epoch, type='drift', standard='UTC',
            crs=3031, extrapolate=True)
        # iterate over zonal and meridional currents
        for key,val in tide.items():
            assert np.any(val)

    # PURPOSE: test definition file functionality
    @pytest.mark.parametrize("MODEL", ['CATS2008'])
    def test_definition_file(self, MODEL):
        # get model parameters
        model = pyTMD.io.model(verify=False).from_database(MODEL)
        # create model definition file
        fid = io.StringIO()
        d = model.to_dict(serialize=True)
        json.dump(d, fid)
        fid.seek(0)
        # use model definition file as input
        m = pyTMD.io.model().from_file(fid)
        # check that (serialized) attributes are the same
        assert m.__parameters__ == model.__parameters__

# PURPOSE: Test and Verify AOTIM-5-2018 model read and prediction programs
class Test_AOTIM5_2018:
    @pytest.fixture(autouse=True)
    def init(self, directory):
        self.directory = pathlib.Path(directory).expanduser().absolute()

    # PURPOSE: create verification from Matlab program
    @pytest.fixture(scope="class", autouse=False)
    def update_AOTIM5_2018(self, directory):
        # compute validation data from Matlab TMD program using octave
        # https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5
        octave = copy.copy(oct2py.octave)
        TMDpath = directory.joinpath('..','TMD_Matlab_Toolbox','TMD').absolute()
        octave.addpath(octave.genpath(str(TMDpath)))
        octave.addpath(str(directory))
        # turn off octave warnings
        octave.warning('off', 'all')
        # model parameters for AOTIM-5-2018
        model = pyTMD.io.model(directory).from_database('AOTIM-5-2018')
        # iterate over type: heights versus currents
        for TYPE in ['z', 'U', 'V']:
            # path to tide model files
            modelpath = model[TYPE].grid_file.parent
            octave.addpath(str(modelpath))
            # input control file for model
            CFname = directory.joinpath('Model_Arc5km2018')
            assert CFname.exists()

            # open Arctic Tidal Current Atlas list of records
            ATLAS = directory.joinpath('List_of_records.txt')
            with ATLAS.open(mode='r', encoding='utf8') as f:
                file_contents = f.read().splitlines()
            # skip 2 header rows
            count = 2
            # iterate over number of stations
            arctic_stations = len(file_contents) - count
            stations = [None]*arctic_stations
            shortname = [None]*arctic_stations
            station_lon = np.zeros((arctic_stations))
            station_lat = np.zeros((arctic_stations))
            for s in range(arctic_stations):
                line_contents = file_contents[count+s].split()
                stations[s] = line_contents[1]
                shortname[s] = line_contents[2]
                station_lat[s] = np.float64(line_contents[10])
                station_lon[s] = np.float64(line_contents[11])

            # serial dates for matlab program (days since 0000-01-01T00:00:00)
            SDtime = np.arange(convert_calendar_serial(2000,1,1),
                convert_calendar_serial(2000,12,31)+1)

            # run Matlab TMD program with octave
            # MODE: OB time series
            validation,_ = octave.tmd_tide_pred_plus(str(CFname), SDtime,
                station_lat, station_lon, TYPE, nout=2)

            # create dataframe for validation data
            df = pd.DataFrame(data=validation, index=SDtime, columns=shortname)

            # add attributes for each valid station
            for i,s in enumerate(shortname):
                df[s].attrs['station'] = stations[i]
                df[s].attrs['latitude'] = station_lat[i]
                df[s].attrs['longitude'] = station_lon[i]

            # save to (gzipped) csv
            output_file = filepath.joinpath(f'TMDv2.5_Arc5km2018_{TYPE}.csv.gz')
            with gzip.open(output_file, 'wb') as f:
                df.to_csv(f, index_label='time')

    # PURPOSE: Tests that interpolated results are comparable to Matlab program
    @pytest.mark.parametrize("use_mmap", [False, True])
    def test_verify_AOTIM5_2018(self, use_mmap):
        # model parameters for AOTIM-5-2018
        model = pyTMD.io.model(self.directory).from_database('AOTIM-5-2018')
        # open datatree for model
        TYPES = ['z','U','V']
        dtree = model.open_datatree(type=TYPES, use_mmap=use_mmap)

        # open Arctic Tidal Current Atlas list of records
        ATLAS = self.directory.joinpath('List_of_records.txt')
        with ATLAS.open(mode='r', encoding='utf8') as f:
            file_contents = f.read().splitlines()
        # skip 2 header rows
        count = 2
        # iterate over number of stations
        arctic_stations = len(file_contents) - count
        stations = [None]*arctic_stations
        shortname = [None]*arctic_stations
        station_lon = np.zeros((arctic_stations))
        station_lat = np.zeros((arctic_stations))
        for s in range(arctic_stations):
            line_contents = file_contents[count+s].split()
            stations[s] = line_contents[1]
            shortname[s] = line_contents[2]
            station_lat[s] = np.float64(line_contents[10])
            station_lon[s] = np.float64(line_contents[11])

        # convert data to coordinate reference system of model
        X, Y = dtree.tmd.transform(station_lon, station_lat, crs=4326)
        # convert to data arrays
        X = xr.DataArray(X, dims=('station'))
        Y = xr.DataArray(Y, dims=('station'))
        # extract amplitude and phase from tide model
        local = dtree.tmd.interp(X, Y, extrapolate=True)

        # will verify differences between model outputs are within tolerance
        eps = np.finfo(np.float16).eps

        # compare daily outputs at each station point
        invalid_list = ['BC1','KS02','KS12','KS14','BI3','BI4']
        # remove coastal stations from the list
        valid_stations=[i for i,s in enumerate(shortname)
            if s not in invalid_list]
        
        # for each type of data (z, u, v)
        for TYPE, ds in local.items():
            # convert to dataset
            ds = ds.to_dataset()

            # read validation data from Matlab TMD program
            # https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5
            validation_file = f'TMDv2.5_Arc5km2018_{TYPE}.csv.gz'
            df = pd.read_csv(filepath.joinpath(validation_file))
            # calculate daily results for a time period
            # convert time to days since 1992-01-01T00:00:00
            ts = timescale.from_julian(df.time.values + 1721058.5)
            # presently not converting times to dynamic times for model comparisons
            deltat = np.zeros_like(ts.tt_ut1)

            # predict tides and infer minor corrections
            tide = ds.tmd.predict(ts.tide, deltat=deltat,
                corrections=model['corrections'])
            tide += ds.tmd.infer(ts.tide, deltat=deltat,
                corrections=model['corrections'])

            # for each valid station
            for i,s in enumerate(valid_stations):
                # non-unique station names (use dataframe columns)
                station = df.columns[s+1]
                # calculate differences between matlab and python version
                difference = tide.isel(station=s) - df[station].values
                difference = difference.where(~np.isnan(difference), other=0.0)
                assert np.all(np.abs(difference) < eps)

    # PURPOSE: test the tide correction wrapper function
    def test_Arctic_Ocean(self):
        # create an image around the Arctic Ocean
        # use NSIDC Polar Stereographic definitions
        # https://nsidc.org/data/polar-stereo/ps_grids.html
        xlimits = [-3850000,3750000]
        ylimits = [-5350000,5850000]
        spacing = [50e3,-50e3]
        # x and y coordinates
        x = np.arange(xlimits[0],xlimits[1]+spacing[0],spacing[0])
        y = np.arange(ylimits[1],ylimits[0]+spacing[1],spacing[1])
        xgrid,ygrid = np.meshgrid(x,y)
        # time dimension
        delta_time = 0.0
        # calculate tide map
        tide = pyTMD.compute.tide_elevations(xgrid, ygrid, delta_time,
            directory=self.directory, model='AOTIM-5-2018',
            epoch=timescale.time._j2000_epoch, type='grid', standard='UTC',
            crs=3413, extrapolate=True)
        assert np.any(tide)

    # PURPOSE: test definition file functionality
    @pytest.mark.parametrize("MODEL", ['AOTIM-5-2018'])
    def test_definition_file(self, MODEL):
        # get model parameters
        model = pyTMD.io.model(verify=False).from_database(MODEL)
        # create model definition file
        fid = io.StringIO()
        d = model.to_dict(serialize=True)
        json.dump(d, fid)
        fid.seek(0)
        # use model definition file as input
        m = pyTMD.io.model().from_file(fid)
        # check that (serialized) attributes are the same
        assert m.__parameters__ == model.__parameters__
