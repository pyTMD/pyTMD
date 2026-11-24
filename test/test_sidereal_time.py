#!/usr/bin/env python
"""
test_sidereal_time.py (07/2025)
Verify sidereal times against the US Naval Observatory (USNO)

UPDATE HISTORY:
    Updated 09/2026: added validation checks from the NAOJ
    Written 07/2026
"""

import pytest
import timescale
import numpy as np
import numpy.lib.recfunctions
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
    except pyTMD.utilities.urllib2.URLError as exc:
        pytest.xfail(exc.reason)
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


def test_naoj_sidereal_time():
    """Test against sidereal times from the National Astronomical
    Observatory of Japan (NAOJ)"""
    names = ("date", "GAST", "GMST", "eqofeq")
    formats = ('datetime64[s]', "<U12", "<U12", 'f8')
    validation = np.array(
        [
            ("2009-01-01T00:00:00", "06:43:07.139", "06:43:06.320", 0.819),
            ("2010-01-01T00:00:00", "06:42:10.036", "06:42:09.030", 1.006),
            ("2011-01-01T00:00:00", "06:41:12.809", "06:41:11.739", 1.070),
            ("2012-01-01T00:00:00", "06:40:15.486", "06:40:14.448", 1.038),
            ("2013-01-01T00:00:00", "06:43:14.609", "06:43:13.713", 0.896),
            ("2014-01-01T00:00:00", "06:42:17.058", "06:42:16.422", 0.636),
            ("2015-01-01T00:00:00", "06:41:19.430", "06:41:19.132", 0.298),
            ("2016-01-01T00:00:00", "06:40:21.789", "06:40:21.841", -0.052),
            ("2017-01-01T00:00:00", "06:43:20.711", "06:43:21.106", -0.395),
            ("2018-01-01T00:00:00", "06:42:23.108", "06:42:23.815", -0.707),
            ("2019-01-01T00:00:00", "06:41:25.602", "06:41:26.525", -0.923),
            ("2020-01-01T00:00:00", "06:40:28.226", "06:40:29.234", -1.009),
            ("2021-01-01T00:00:00", "06:43:27.511", "06:43:28.499", -0.988),
            ("2022-01-01T00:00:00", "06:42:30.331", "06:42:31.209", -0.877),
            ("2023-01-01T00:00:00", "06:41:33.272", "06:41:33.918", -0.646),
            ("2024-01-01T00:00:00", "06:40:36.300", "06:40:36.628", -0.328),
            ("2025-01-01T00:00:00", "06:43:35.905", "06:43:35.893", 0.012),
            ("2026-01-01T00:00:00", "06:42:38.934", "06:42:38.602", 0.332),
            ("2027-01-01T00:00:00", "06:41:41.957", "06:41:41.312", 0.645),
        ],
        dtype=dict(names=names, formats=formats)
    )
    # number of rows
    rows = len(validation)
    # append gmst and gast (fraction) to array
    validation = numpy.lib.recfunctions.append_fields(
        validation,
        ('gmst', 'gast'),
        (np.zeros((rows)), np.zeros((rows)))
    )
    # convert gmst and gast into fraction of day
    for i, row in enumerate(validation):
        for var in ('GMST', 'GAST'):
            key = var.lower()
            HH, MM, SS = np.array(row[var].split(":"), dtype="f8")
            validation[key][i] = HH / 24.0 + MM / 1440.0 + SS / 86400.0
    # convert equation of the equinoxes to fraction
    validation["eqofeq"] /= 86400.0
    # convert dates to timescale object
    ts = timescale.from_datetime(validation['date'])
    # convert from MJD to centuries relative to 2000-01-01T12:00:00
    T = (ts.MJD - pyTMD.astro._mjd_j2000) / pyTMD.astro._century
    # allocate for output data
    output = {}
    # calculate GMST
    output["gmst"] = ts.gmst
    # calculate equation of the equinoxes and convert to fraction
    eqofeq = pyTMD.astro.eqeq(T, method="IERS") / (2.0 * np.pi)
    output["eqofeq"] = eqofeq
    # calculate GAST from GMST and equation of the equinoxes
    output["gast"] = np.mod(ts.gmst + eqofeq, 1.0)
    # validate against NAOJ data
    for key, val in output.items():
        assert np.allclose(val, validation[key], atol=5e-8)
