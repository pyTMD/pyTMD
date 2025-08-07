#!/usr/bin/env python
u"""
NOAA.py
Written by Tyler Sutterley (08/2025)
Query and parsing functions for NOAA webservices API

PYTHON DEPENDENCIES:
    pandas: Python Data Analysis Library
        https://pandas.pydata.org

UPDATE HISTORY:
    Updated 08/2025: replace invalid water level values with NaN
        convert all station names to title case (some are upper)
    Written 07/2025: extracted from Compare NOAA Tides notebook
"""
from __future__ import annotations

import logging
import traceback
import numpy as np
import pyTMD.io.constituents
from pyTMD.utilities import import_dependency

# attempt imports
pd = import_dependency('pandas')

__all__ = [
    "build_query",
    "from_xml",
    "prediction_stations",
    "harmonic_constituents",
    "water_level"
]

_apis = [
    'currentpredictionstations',
    'tidepredictionstations',
    'harmonicconstituents',
    'waterlevelrawonemin',
    'waterlevelrawsixmin',
    'waterlevelverifiedsixmin',
    'waterlevelverifiedhourly',
    'waterlevelverifieddaily',
    'waterlevelverifiedmonthly',
]

_xpaths = {
    'currentpredictionstations': '//wsdl:station',
    'tidepredictionstations': '//wsdl:station',
    'harmonicconstituents': '//wsdl:item',
    'waterlevelrawonemin': '//wsdl:item',
    'waterlevelrawsixmin': '//wsdl:item',
    'waterlevelverifiedsixmin': '//wsdl:item',
    'waterlevelverifiedhourly': '//wsdl:item',
    'waterlevelverifieddaily': '//wsdl:item',
    'waterlevelverifiedmonthly': '//wsdl:item'
}

def build_query(api, **kwargs):
    """
    Build a query for the NOAA webservices API
    
    Parameters
    ----------
    api: str
        NOAA webservices API endpoint to query
    **kwargs: dict
        Additional query parameters to include in the request
    
    Returns
    -------
    url: str
        The complete URL for the API request
    namespaces: dict
        A dictionary of namespaces for parsing XML responses
    """
    # NOAA webservices hosts
    HOST = 'https://tidesandcurrents.noaa.gov/axis/webservices'
    OPENDAP = 'https://opendap.co-ops.nos.noaa.gov/axis/webservices'
    # NOAA webservices query arguments
    arguments = '?format=xml'
    for key, value in kwargs.items():
        arguments += f'&{key}={value}'
    arguments += '&Submit=Submit'
    # NOAA API query url
    url = f'{HOST}/{api}/response.jsp{arguments}'
    # lxml namespaces for parsing
    namespaces = {}
    namespaces['wsdl'] = f'{OPENDAP}/{api}/wsdl'
    return (url, namespaces)

def from_xml(url, **kwargs):
    """
    Query the NOAA webservices API and return as a ``DataFrame``
    
    Parameters
    ----------
    url: str
        The complete URL for the API request
    **kwargs: dict
        Additional keyword arguments to pass to ``pandas.read_xml``
    
    Returns
    -------
    df: pandas.DataFrame
        The ``DataFrame`` containing the parsed XML data
    """
    # query the NOAA webservices API
    try:
        logging.debug(url)
        df = pd.read_xml(url, **kwargs)
    except ValueError:
        logging.error(traceback.format_exc())
    # return the dataframe
    else:
        return df

def prediction_stations(
        api: str = 'tidepredictionstations',
        **kwargs
    ):
    """
    Retrieve a list of tide prediction stations
    
    Parameters
    ----------
    api: str
        NOAA webservices API endpoint to query
    **kwargs: dict
        Additional query parameters to include in the request

    Returns
    -------
    df: pandas.DataFrame
        A ``DataFrame`` containing the station information
    """
    # get list of tide prediction stations
    xpath = _xpaths[api]
    url, namespaces = build_query(api, **kwargs)
    df = from_xml(url, xpath=xpath, namespaces=namespaces)
    # convert station names to title case
    df['name'] = df['name'].str.title()
    # set the index to the station name
    df = df.set_index('name')
    # sort the index and drop metadata column
    df = df.sort_index().drop(columns=['metadata'])
    # return the dataframe
    return df

def harmonic_constituents(
        api: str = 'harmonicconstituents',
        **kwargs
    ):
    """
    Retrieve a list of harmonic constituents for a specified station

    Parameters
    ----------
    api: str
        NOAA webservices API endpoint to query
    **kwargs: dict
        Additional query parameters to include in the request

    Returns
    -------
    df: pandas.DataFrame
        ``DataFrame`` containing the harmonic constituent information
    """
    # set default query parameters
    kwargs.setdefault('unit', 0)
    kwargs.setdefault('timeZone', 0)
    # get list of harmonic constituents
    xpath = _xpaths[api]
    url, namespaces = build_query(api, **kwargs)
    df = from_xml(url, xpath=xpath, namespaces=namespaces)
    # set the index to the constituent number
    df = df.set_index('constNum')
    # parse harmonic constituents
    c = [pyTMD.io.constituents.parse(row['name']) for i, row in df.iterrows()]
    df['constituent'] = c
    # return the dataframe
    return df

def water_level(
        api: str = 'waterlevelrawsixmin',
        **kwargs
    ):
    """
    Retrieve water level data for a specified station and date range

    Parameters
    ----------
    api: str
        NOAA webservices API endpoint to query
    **kwargs: dict
        Additional query parameters to include in the request

    Returns
    -------
    df: pandas.DataFrame
        ``DataFrame`` containing the water level data
    """
    # set default query parameters
    kwargs.setdefault('unit', 0)
    kwargs.setdefault('timeZone', 0)
    kwargs.setdefault('datum', 'MSL')
    # get water levels for station and date range
    xpath = _xpaths[api]
    url, namespaces = build_query(api, **kwargs)
    df = from_xml(url, xpath=xpath, namespaces=namespaces,
        parse_dates=['timeStamp'])
    # replace invalid water level values with NaN
    df = df.replace(to_replace=[-999], value=np.nan)
    # return the dataframe
    return df
