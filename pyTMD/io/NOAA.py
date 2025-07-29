#!/usr/bin/env python
u"""
NOAA.py
Written by Tyler Sutterley (07/2025)
Query and parsing functions for NOAA webservices API

PYTHON DEPENDENCIES:
    pandas: Python Data Analysis Library
        https://pandas.pydata.org

UPDATE HISTORY:
    Written 07/2025: extracted from Compare NOAA Tides notebook
"""
from __future__ import annotations

import logging
import traceback
import pyTMD.utilities
# attempt imports
pd = pyTMD.utilities.import_dependency('pandas')

__all__ = [
    "build_query",
    "from_xml"
]

_apis = [
    'currentpredictionstations',
    'tidepredictionstations',
    'harmonicconstituents',
    'waterlevelrawsixmin'
]

_xpaths = {
    'currentpredictionstations': '//wsdl:station',
    'tidepredictionstations': '//wsdl:station',
    'harmonicconstituents': '//wsdl:item',
    'waterlevelrawsixmin': '//wsdl:item'
}

def build_query(api, **kwargs):
    """
    Build a query for the NOAA webservices API
    
    Parameters
    ----------
    api: str
        The API endpoint to query
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
