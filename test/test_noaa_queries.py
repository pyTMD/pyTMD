#!/usr/bin/env python
u"""
test_noaa_queries.py (07/2025)
Verify NOAA webservices API query functions

PYTHON DEPENDENCIES:
    pandas: Python Data Analysis Library
        https://pandas.pydata.org

UPDATE HISTORY:
    Written 07/2025
"""
import pyTMD.io.NOAA
import numpy as np

def test_noaa_stations():
    """Test NOAA station information retrieval
    """
    api = 'tidepredictionstations'
    xpath = pyTMD.io.NOAA._xpaths[api]
    # get list of tide prediction stations
    url, namespaces = pyTMD.io.NOAA.build_query(api)
    stations = pyTMD.io.NOAA.from_xml(url, xpath=xpath,
        namespaces=namespaces)
    # check that station indicator is in list
    assert '9410230' in stations['ID'].values

def test_noaa_harmonic_constituents():
    """Test NOAA harmonic constituents retrieval
    """
    # set query parameters
    station_id = '9410230'
    unit = 0
    timeZone = 0
    # get harmonic constituents for station
    api = 'harmonicconstituents'
    xpath = pyTMD.io.NOAA._xpaths[api]
    # get list of harmonic constituents
    url, namespaces = pyTMD.io.NOAA.build_query(api,
        stationId=station_id, unit=unit, timeZone=timeZone)
    hcons = pyTMD.io.NOAA.from_xml(url, xpath=xpath,
        namespaces=namespaces).set_index('constNum')
    # check if the values match expected
    expected_columns = ['name', 'amplitude', 'phase', 'speed']
    assert hcons.columns.tolist() == expected_columns
    assert 'M2' in hcons['name'].values
    # get dataframe using wrapper function
    df = pyTMD.io.NOAA.harmonic_constituents(stationId=station_id)
    # check if the values match expected
    assert 'm2' in df['constituent'].values
    # check if the values match between queries
    for i, row in df.iterrows():
        assert row['amplitude'] == hcons.loc[i, 'amplitude']
        assert row['phase'] == hcons.loc[i, 'phase']
        assert row['speed'] == hcons.loc[i, 'speed']

def test_noaa_water_level():
    """Test NOAA water level data retrieval
    """
    # set query parameters
    station_id = '9410230'
    unit = 0
    timeZone = 0
    startdate = '20200101'
    enddate = '20200101'
    datum = 'MSL'
    # get water levels for station and date range
    api = 'waterlevelverifiedhourly'
    xpath = pyTMD.io.NOAA._xpaths[api]
    url, namespaces = pyTMD.io.NOAA.build_query(api,
        stationId=station_id, unit=unit, timeZone=timeZone,
        beginDate=startdate, endDate=enddate, datum=datum)
    wlevel = pyTMD.io.NOAA.from_xml(url, xpath=xpath,
        namespaces=namespaces, parse_dates=['timeStamp'])
    expected_columns = ['timeStamp', 'WL', 'sigma', 'I', 'L']
    expected_WL = np.array([-0.2, -0.438, -0.571, -0.65, -0.589,
        -0.447, -0.278, -0.026, 0.159, 0.28, 0.341, 0.299, 0.246,
        0.162, 0.078, 0.06, 0.08, 0.147, 0.231, 0.305, 0.354,
        0.379, 0.298, 0.154])
    # check if the values match expected
    assert wlevel.columns.tolist() == expected_columns
    assert wlevel['timeStamp'][0] == np.datetime64('2020-01-01')
    assert np.allclose(wlevel['WL'].values, expected_WL)
    # get dataframe using wrapper function
    df = pyTMD.io.NOAA.water_level(api, stationId=station_id,
        beginDate=startdate, endDate=enddate)
    # check if the values match expected
    assert df.columns.tolist() == expected_columns
    assert df['timeStamp'][0] == np.datetime64('2020-01-01')
    assert np.allclose(df['WL'].values, expected_WL)
    # check if the values match between queries
    for i, row in df.iterrows():
        assert row['timeStamp'] == wlevel.loc[i, 'timeStamp']
        assert row['WL'] == wlevel.loc[i, 'WL']
        assert row['sigma'] == wlevel.loc[i, 'sigma']
        assert row['I'] == wlevel.loc[i, 'I']
        assert row['L'] == wlevel.loc[i, 'L']
