#!/usr/bin/env python
"""
test_sidereal_time.py (07/2025)
Verify sidereal times against the US Naval Observatory (USNO)

UPDATE HISTORY:
    Written 07/2026
"""

import pytest
import numpy as np
import timescale
import pyTMD.utilities

# internal Equation of the Equinox methods
_methods = ["IERS", "Meeus", "USNO", "approximate"]


# parametrize over eqeq methods
@pytest.mark.parametrize("method", _methods)
def test_usno_sidereal_time(method):
    """Test against USNO sidereal times"""
    # USNO Astronomical Applications API
    HOST = pyTMD.utilities.URL("https://aa.usno.navy.mil/api")
    # API url for service
    service = "siderealtime"
    url = HOST.joinpath(f"{service}?")
    # number of API queries
    iterations = 366
    # latitude and longitude
    lat, lon = 47.6062, -122.3321
    # parameters for API query
    parameters = {}
    parameters["date"] = "2000-01-01"
    parameters["time"] = "12:00:00"
    parameters["coords"] = f"{lat:0.4f},{lon:0.4f}"
    parameters["reps"] = iterations
    parameters["intv_mag"] = 1
    parameters["intv_unit"] = "day"
    # build query
    for i, key in enumerate(parameters.keys()):
        joiner = "" if (i == 0) else "&"
        url += f"{joiner}{key}={parameters[key]}"

    # get data from API
    try:
        results = url.load()
    except pyTMD.utilities.urllib2.HTTPError as exc:
        pytest.xfail(exc.reason)

    # allocate for output validation data
    validation = {}
    validation["ut1time"] = np.zeros(iterations, dtype="datetime64[s]")
    for key in ["gmst", "gast", "lmst", "last", "eqofeq"]:
        validation[key] = np.zeros(iterations)
    # get data from JSON response
    for i, data in enumerate(results["properties"]["data"]):
        ut1time = "{year:4d}-{month:02d}-{day:02d}T{ut1time}".format(**data)
        validation["ut1time"][i] = ut1time
        # convert gmst and gast into fractions of day
        for key in ["gmst", "gast", "lmst", "last"]:
            HH, MM, SS = np.array(data[key].split(":"), dtype="f8")
            validation[key][i] = HH / 24.0 + MM / 1440.0 + SS / 86400.0
        # extract equation of the equinoxes and convert to fraction
        validation["eqofeq"][i] = np.float64(data["eqofeq"]) / 86400.0

    # build timescale from ut1times
    ts = timescale.from_datetime(validation["ut1time"])
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (ts.MJD - pyTMD.astro._mjd_j2000) / pyTMD.astro._century
    # allocate for output data
    output = {}
    # calculate GMST using equinox method
    output["gmst"] = ts.st
    # calculate equation of the equinoxes and convert to fraction
    eqofeq = pyTMD.astro.eqeq(T, method=method)
    output["eqofeq"] = eqofeq / (2.0 * np.pi)
    # calculate GAST using selected method
    output["gast"] = pyTMD.astro.gast(T, method=method)
    # rotate by longitudes for local sidereal times
    output["lmst"] = np.mod(output["gmst"] + lon / 360.0, 1.0)
    output["last"] = np.mod(output["gast"] + lon / 360.0, 1.0)
    # validate against USNO data
    for key, val in output.items():
        # make sure calculations are within half a second
        assert np.allclose(val, validation[key], atol=0.5 / 86400.0)
    # validate that equation of the equinoxes makes sense
    assert np.allclose(output["eqofeq"], output["gast"] - output["gmst"])
