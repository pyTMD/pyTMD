#!/usr/bin/env python
"""
AntTG.py
Written by Tyler Sutterley (11/2026)

Reads constants from the Antarctic Tide Gauge (AntTG) database

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Written 11/2026
"""

from __future__ import annotations

import pathlib
import numpy as np
import xarray as xr
import pyTMD.utilities

__all__ = ["open_dataset"]


def open_dataset(input_file, chunks: int | dict | str | None = None):
    """
    Open Antarctic Tide Gauge (AntTG) database

    Parameters
    ----------
    input_file: str or pathlib.Path
        Input AntTG file
    chunks: int, dict, str, or None, default None
        Variable chunk sizes for dask [see ``xarray.open_dataset``]

    Returns
    -------
    ds: xarray.Dataset
        AntTG tide model data
    """
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if isinstance(input_file, pathlib.Path) and not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    # open the Antarctic Tide Gauge database file
    with input_file.open(mode="r", encoding="utf8") as f:
        file_contents = f.read().splitlines()
    # counts the number of lines in the header
    count = 0
    header = True
    # Reading over header text
    while header:
        # check if file line at count starts with matlab comment string
        header = file_contents[count].startswith("%")
        # add 1 to counter
        count += 1
    # rewind 1 line
    count -= 1
    # number of lines per station
    nlines = 10
    # iterate over number of stations
    constituents = ["q1", "o1", "p1", "k1", "n2", "m2", "s2", "k2"]
    antarctic_stations = (len(file_contents) - count) // nlines
    stations = [None] * antarctic_stations
    shortname = [None] * antarctic_stations
    sensor_type = [None] * antarctic_stations
    duration = np.zeros((antarctic_stations))
    sampling = np.zeros((antarctic_stations))
    longitude = np.zeros((antarctic_stations))
    latitude = np.zeros((antarctic_stations))
    amp = np.ma.zeros((antarctic_stations, len(constituents)))
    ph = np.ma.zeros((antarctic_stations, len(constituents)))
    for s in range(antarctic_stations):
        i = count + s * nlines
        stations[s] = file_contents[i + 1].strip()
        shortname[s] = file_contents[i + 3].strip()
        lat, lon, *_ = file_contents[i + 4].split()
        record, sample, *_ = file_contents[i + 5].split()
        sensor_type[s] = file_contents[i + 6].strip()
        longitude[s] = np.float64(lon)
        latitude[s] = np.float64(lat)
        duration[s] = np.float64(record)
        sampling[s] = np.float64(sample)
        amp.data[s, :] = np.array(file_contents[i + 7].split(), dtype="f8")
        ph.data[s, :] = np.array(file_contents[i + 8].split(), dtype="f8")
    # update masks where NaN
    amp.mask = np.isnan(amp.data) | (amp.data == 0.0)
    ph.mask = np.isnan(ph.data)
    # convert amplitude and phase to complex form
    hc = xr.DataArray(
        amp * np.exp(-1j * np.radians(ph)),
        dims=("station", "constituent"),
        coords=dict(station=stations, constituent=constituents),
        attrs=dict(units="cm"),
    )
    # convert to dataset
    ds = hc.to_dataset(dim="constituent", promote_attrs=True)
    # add coordinates
    ds["x"] = xr.DataArray(
        longitude,
        coords=dict(station=stations),
        attrs=dict(units="degrees_east", long_name="longitude"),
    )
    ds["y"] = xr.DataArray(
        latitude,
        coords=dict(station=stations),
        attrs=dict(units="degrees_north", long_name="latitude"),
    )
    # add station metadata
    ds["sensor"] = xr.DataArray(
        sensor_type,
        coords=dict(station=stations),
    )
    ds["shortname"] = xr.DataArray(
        shortname,
        coords=dict(station=stations),
    )
    ds["duration"] = xr.DataArray(
        duration,
        coords=dict(station=stations),
        attrs=dict(units="days"),
    )
    ds["sampling"] = xr.DataArray(
        sampling,
        coords=dict(station=stations),
        attrs=dict(units="seconds"),
    )
    # coerce to specified chunks
    if chunks is not None:
        ds = ds.chunk(chunks)
    # add attributes
    ds.attrs = {}
    ds.attrs["lineage"] = pathlib.Path(input_file).name
    ds.attrs["crs"] = "EPSG:4326"
    ds.attrs["constituents"] = constituents
    # add attributes for each constituent
    for c in constituents:
        ds[c].attrs.update(hc.attrs)
    # return the dataset
    return ds
