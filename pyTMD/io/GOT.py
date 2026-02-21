#!/usr/bin/env python
"""
GOT.py
Written by Tyler Sutterley (02/2026)

Reads ascii and netCDF4 files from Richard Ray's Goddard Ocean Tide (GOT) model
    https://earth.gsfc.nasa.gov/geo/data/ocean-tide-models

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    xarray: N-D labeled arrays and datasets in Python
        https://docs.xarray.dev/en/stable/

UPDATE HISTORY:
    Updated 02/2026: make dataset accessor for GOT be a subaccessor from dataset
        some models have units in the second line of the header text
    Updated 12/2025: no longer subclassing pathlib.Path for working directories
        added function to write to output GOT-formatted ascii files
        fixed writing of output constituents to match GOT attribute format
    Updated 11/2025: near-complete rewrite of program to use xarray
    Updated 10/2025: simplify ascii read function to use masked_equal
    Updated 08/2025: use numpy degree to radian conversions
        added option to gap fill when reading constituent grids
    Updated 11/2024: expose buffer distance for cropping tide model data
    Updated 10/2024: fix error when using default bounds in extract_constants
    Updated 07/2024: added crop and bounds keywords for trimming model data
        use parse function from constituents class to extract names
    Updated 04/2023: fix repeated longitudinal convention adjustment
        using pathlib to define and expand tide model paths
    Updated 03/2023: add basic variable typing to function inputs
    Updated 12/2022: refactor tide read programs under io
        new functions to read and interpolate from constituents class
        new functions to read and write GOT netCDF4 files
        refactored interpolation routines into new module
    Updated 11/2022: use f-strings for formatting verbose or ascii output
    Updated 05/2022: reformat arguments to extract_GOT_constants definition
        changed keyword arguments to camel case
    Updated 04/2022: updated docstrings to numpy documentation format
        include utf-8 encoding in reads to be windows compliant
    Updated 12/2021: adjust longitude convention based on model longitude
    Updated 07/2021: added check that tide model files are accessible
    Updated 06/2021: add warning for tide models being entered as string
    Updated 05/2021: added option for extrapolation cutoff in kilometers
    Updated 03/2021: add extrapolation check where there are no invalid points
        prevent ComplexWarning for fill values when calculating amplitudes
        simplified inputs to be similar to binary OTIS read program
        replaced numpy bool/int to prevent deprecation warnings
    Updated 02/2021: set invalid values to nan in extrapolation
        replaced numpy bool to prevent deprecation warning
    Updated 12/2020: added valid data extrapolation with nearest_extrap
    Updated 09/2020: set bounds error to false for regular grid interpolations
        adjust dimensions of input coordinates to be iterable
    Updated 08/2020: replaced griddata with scipy regular grid interpolators
    Updated 07/2020: added function docstrings. separate bilinear interpolation
        update griddata interpolation. add option for compression
    Updated 06/2020: use argmin and argmax in bilinear interpolation
    Updated 11/2019: find invalid mask points for each constituent
    Updated 09/2019: output as numpy masked arrays instead of nan-filled arrays
    Updated 07/2019: interpolate fill value mask with bivariate splines
    Updated 12/2018: python3 compatibility updates for division and zip
    Updated 10/2018: added scale as load tides are in mm and ocean are in cm
    Updated 08/2018: added multivariate spline interpolation option
    Written 07/2018
"""

from __future__ import division, annotations

import re
import gzip
import pathlib
import datetime
import numpy as np
import xarray as xr
import pyTMD.version
import pyTMD.constituents
import pyTMD.utilities
from .dataset import register_dataset_subaccessor

# attempt imports
dask = pyTMD.utilities.import_dependency("dask")
dask_available = pyTMD.utilities.dependency_available("dask")

__all__ = [
    "open_mfdataset",
    "open_got_dataset",
    "open_got_ascii",
    "open_got_netcdf",
    "GOTDataset",
]


# PURPOSE: read a list of GOT ASCII or netCDF4 files
def open_mfdataset(
    model_files: list[str] | list[pathlib.Path],
    parallel: bool = False,
    **kwargs,
):
    """
    Open multiple GOT model files

    Parameters
    ----------
    model_files: list of str or pathlib.Path
        list of OTIS model files
    parallel: bool, default False
        Open files in parallel using ``dask.delayed``
    **kwargs: dict
        additional keyword arguments for opening GOT files

    Returns
    -------
    ds: xarray.Dataset
        combined GOT model data
    """
    # merge multiple granules
    if parallel and dask_available:
        opener = dask.delayed(open_got_dataset)
    else:
        opener = open_got_dataset
    # read each file as xarray dataset and append to list
    d = [opener(f, **kwargs) for f in model_files]
    # read datasets as dask arrays
    if parallel and dask_available:
        (d,) = dask.compute(d)
    # merge datasets
    ds = xr.merge(d, compat="override")
    # return xarray dataset
    return ds


# PURPOSE: reads a GOT ASCII or netCDF4 file
def open_got_dataset(input_file: str | pathlib.Path, **kwargs):
    """
    Open GOT-formatted model files

    Parameters
    ----------
    input_file: str or pathlib.Path
        input transport file
    format: str, default 'netcdf'
        Model format

            - ``'ascii'``: traditional GOT ascii format
            - ``'netcdf'``: GOT netCDF4 format
    **kwargs: dict
        additional keyword arguments for opening GOT files

    Returns
    -------
    ds: xarray.Dataset
        GOT tide model data
    """
    # detect file format if not provided
    if kwargs.get("format", None) is None:
        kwargs["format"] = pyTMD.utilities.detect_format(input_file)
    # detect if file is compressed if not provided
    if kwargs.get("compressed", None) is None:
        kwargs["compressed"] = pyTMD.utilities.detect_compression(input_file)
    # read constituent from elevation file
    if kwargs["format"] == "ascii":
        # GOT ascii constituent files
        ds = open_got_ascii(input_file, **kwargs)
    elif kwargs["format"] == "netcdf":
        # GOT netCDF4 constituent files
        ds = open_got_netcdf(input_file, **kwargs)
    else:
        raise ValueError(f"Unrecognized file format: {kwargs['format']}")
    # return dataset
    return ds


# PURPOSE: read GOT ASCII files
def open_got_ascii(
    input_file: str | pathlib.Path,
    chunks: int | dict | str | None = None,
    **kwargs,
):
    """
    Open GOT-formatted ASCII files

    Parameters
    ----------
    input_file: str or pathlib.Path
        Model file
    chunks: int, dict, str, or None, default None
        coerce output to specified chunks
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    ds: xarray.Dataset
        GOT tide model data
    """
    # set default keyword arguments
    kwargs.setdefault("compressed", False)
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if isinstance(input_file, pathlib.Path) and not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    # read the ASCII-format tide elevation file
    if kwargs["compressed"]:
        # read gzipped ascii file
        with gzip.open(input_file, "rb") as f:
            file_contents = f.read().decode("utf8").splitlines()
    else:
        with open(input_file, mode="r", encoding="utf8") as f:
            file_contents = f.read().splitlines()
    # parse header text
    # constituent identifier
    cons = pyTMD.constituents._parse_name(file_contents[0])
    # get units from header if available
    rx = re.compile(r"\((\w+m)\)", re.IGNORECASE)
    # GOT headers from Richard Ray have units on the first line
    # some other models have units on the second line
    if rx.search(file_contents[0]):
        units = rx.findall(file_contents[0], re.IGNORECASE)
    elif rx.search(file_contents[1]):
        units = rx.findall(file_contents[1], re.IGNORECASE)
    else:
        units = None
    # grid dimensions
    nlat, nlon = np.array(file_contents[2].split(), dtype=int)
    # longitude range
    ilat = np.array(file_contents[3].split(), dtype=np.float64)
    # latitude range
    ilon = np.array(file_contents[4].split(), dtype=np.float64)
    # mask fill value
    masked_values = file_contents[5].split()
    fill_value = np.array(masked_values[0], dtype=np.float32)
    # number of columns in ascii file
    ncol = 11
    # create latitude and longitude arrays
    lat = np.linspace(ilat[0], ilat[1], nlat)
    lon = np.linspace(ilon[0], ilon[1], nlon)
    # data dictionary
    var = dict(dims=("y", "x"), coords={}, data_vars={})
    var["coords"]["y"] = dict(data=lat.copy(), dims="y")
    var["coords"]["x"] = dict(data=lon.copy(), dims="x")
    # input amplitude and phase
    amp = np.zeros((nlat, nlon), dtype=np.float32)
    ph = np.zeros((nlat, nlon), dtype=np.float32)
    # starting lines to fill amplitude and phase variables
    l1 = 7
    l2 = 14 + int(nlon // ncol) * nlat + nlat
    # for each latitude
    for i in range(nlat):
        for j in range(nlon // ncol):
            j1 = j * ncol
            # amplitude and phase are on two separate rows
            amp[i, j1 : j1 + ncol] = np.array(file_contents[l1].split())
            ph[i, j1 : j1 + ncol] = np.array(file_contents[l2].split())
            l1 += 1
            l2 += 1
        # add last row of tidal variables
        j1 = (j + 1) * ncol
        j2 = nlon % ncol
        # amplitude and phase are on two separate rows
        amp[i, j1 : j1 + j2] = np.array(file_contents[l1].split())
        ph[i, j1 : j1 + j2] = np.array(file_contents[l2].split())
        l1 += 1
        l2 += 1
    # convert to masked arrays
    amp = np.ma.masked_equal(amp, fill_value)
    ph = np.ma.masked_equal(ph, fill_value)
    # store the data variables
    var["data_vars"][cons] = {}
    var["data_vars"][cons]["dims"] = ("y", "x")
    var["data_vars"][cons]["data"] = amp * np.exp(-1j * ph * np.pi / 180.0)
    # convert to xarray Dataset from the data dictionary
    ds = xr.Dataset.from_dict(var)
    # coerce to specified chunks
    if chunks is not None:
        ds = ds.chunk(chunks)
    # add attributes
    ds.attrs["group"] = "z"
    if units:
        ds[cons].attrs["units"] = units[0].lower()
    # return xarray dataset
    return ds


# PURPOSE: read GOT netCDF4 files
def open_got_netcdf(
    input_file: str | pathlib.Path,
    chunks: int | dict | str | None = None,
    **kwargs,
):
    """
    Open GOT-formatted netCDF4 files

    Parameters
    ----------
    input_file: str or pathlib.Path
        model file
    chunks: int, dict, str, or None, default None
        variable chunk sizes for dask (see ``xarray.open_dataset``)
    compressed: bool, default False
        Input file is gzip compressed

    Returns
    -------
    ds: xarray.Dataset
        GOT tide model data
    """
    # set default keyword arguments
    kwargs.setdefault("compressed", False)
    # tilde-expand input file
    input_file = pyTMD.utilities.Path(input_file).resolve()
    if isinstance(input_file, pathlib.Path) and not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    # read the netCDF4-format tide elevation file
    if kwargs["compressed"]:
        # read gzipped netCDF4 file
        f = gzip.open(input_file, "rb")
        tmp = xr.open_dataset(
            f, mask_and_scale=True, decode_coords=False, chunks=chunks
        )
    else:
        tmp = xr.open_dataset(
            input_file, mask_and_scale=True, decode_coords=False, chunks=chunks
        )
    # extract constituent from attribute
    cons = pyTMD.constituents._parse_name(tmp.attrs["Constituent"])
    # create output xarray dataset for file
    ds = xr.Dataset()
    # assign coordinates
    ds.coords["x"] = tmp.longitude
    ds.coords["y"] = tmp.latitude
    # calculate complex form of constituent oscillation
    ds[cons] = tmp.amplitude * np.exp(-1j * tmp.phase * np.pi / 180.0)
    # rename dimensions
    mapping_coords = dict(lon="x", lat="y")
    ds = ds.rename(mapping_coords)
    # add attributes
    ds.attrs["group"] = "z"
    ds[cons].attrs["units"] = tmp["amplitude"].attrs.get("units")
    # return xarray dataset
    return ds


# PURPOSE: GOT utilities for xarray Datasets
@register_dataset_subaccessor("got")
class GOTDataset:
    """``xarray.Dataset`` utilities for GOT tidal models"""

    def __init__(self, ds):
        self._ds = ds

    def to_ascii(
        self,
        path: str | pathlib.Path,
        fill_value: float = 999.0,
        mode: str = "w",
        **kwargs,
    ):
        """
        Writes tidal constituents to ASCII files in GOT format

        Parameters
        ----------
        path: str | pathlib.Path
            output directory for ASCII files
        fill_value: float, default -999.0
            fill value for missing data
        mode: str, default 'w'
            file mode
        **kwargs: dict
            additional keyword arguments for ASCII writer
        """
        # tilde-expand output path
        path = pyTMD.utilities.Path(path).resolve()
        # for each variable
        for v in self._ds.data_vars.keys():
            # output dataset
            ds = xr.Dataset()
            # calculate amplitude and phase with fill values
            ds["amplitude"] = self._ds[v].tmd.amplitude.fillna(fill_value)
            ds["phase"] = self._ds[v].tmd.phase.fillna(fill_value)
            # get min and max of coordinates
            nlat, nlon = self._ds[v].shape
            ymin, ymax = ds["y"].values.min(), ds["y"].values.max()
            xmin, xmax = ds["x"].values.min(), ds["x"].values.max()
            # number of columns in ascii file
            ncol = 11
            # write GOT ASCII file
            FILE = path.joinpath(f"{v}.d")
            with open(FILE, mode=mode, encoding="utf8") as f:
                # write header information for amplitude
                units = ds["amplitude"].attrs.get("units", "")
                f.write(f"{v.upper()} tide amplitude ({units})\n")
                f.write(f"Generated by pyTMD {pyTMD.version.full_version}\n")
                f.write(f"{nlat:20d} {nlon:20d}\n")
                f.write(f"{ymin:20.4f} {ymax:20.4f}\n")
                f.write(f"{xmin:20.4f} {xmax:20.4f}\n")
                f.write(f"{fill_value:20.2f} {fill_value:20.2f}\n")
                f.write("\n")
                # write amplitude  data
                for i in range(nlat):
                    amp = ds["amplitude"].isel(y=i).values
                    fmt = [f"{v:7.2f}" for v in amp]
                    for j in range(nlon // ncol + 1):
                        j1 = j * ncol
                        f.write("".join(fmt[j1 : j1 + ncol]) + "\n")
                # write header information for phase
                units = ds["phase"].attrs.get("units", "")
                f.write(f"{v.upper()} tide phase lags ({units})\n")
                f.write(f"Generated by pyTMD {pyTMD.version.full_version}\n")
                f.write(f"{nlat:20d} {nlon:20d}\n")
                f.write(f"{ymin:20.4f} {ymax:20.4f}\n")
                f.write(f"{xmin:20.4f} {xmax:20.4f}\n")
                f.write(f"{fill_value:20.2f} {fill_value:20.2f}\n")
                f.write("\n")
                # write phase data
                for i in range(nlat):
                    ph = ds["phase"].isel(y=i).values
                    fmt = [f"{v:7.2f}" for v in ph]
                    for j in range(nlon // ncol + 1):
                        j1 = j * ncol
                        f.write("".join(fmt[j1 : j1 + ncol]) + "\n")

    # PURPOSE: output tidal constituent file in GOT netCDF format
    def to_netcdf(
        self,
        path: str | pathlib.Path,
        mode: str = "w",
        encoding: dict = {"zlib": True, "complevel": 9},
        **kwargs,
    ):
        """
        Writes tidal constituents to netCDF4 files in GOT format

        Parameters
        ----------
        path: str | pathlib.Path
            output directory for netCDF4 files
        mode: str, default 'w'
            netCDF4 file mode
        encoding: dict, default {"zlib": True, "complevel": 9}
            netCDF4 variable compression settings
        **kwargs: dict
            additional keyword arguments for xarray netCDF4 writer
        """
        # tilde-expand output path
        path = pyTMD.utilities.Path(path).resolve()
        # set default encoding
        kwargs.setdefault("encoding", dict(amplitude=encoding, phase=encoding))
        # coordinate remapping
        mapping_coords = dict(x="longitude", y="latitude")
        attrs = dict(longitude={}, latitude={})
        attrs["longitude"] = dict(units="degrees_east", long_name="longitude")
        attrs["latitude"] = dict(units="degrees_north", long_name="latitude")
        # get longitude and latitude arrays
        # for each variable
        for v in self._ds.data_vars.keys():
            ds = xr.Dataset()
            # calculate amplitude and phase
            ds["amplitude"] = self._ds[v].tmd.amplitude
            ds["phase"] = self._ds[v].tmd.phase
            ds["amplitude"].attrs["units"] = self._ds[v].attrs.get("units", "")
            ds["phase"].attrs["units"] = "degrees"
            ds["amplitude"].attrs["long_name"] = f"Tide amplitude"
            ds["phase"].attrs["long_name"] = f"Greenwich tide phase lag"
            # rename dimensions
            ds = ds.swap_dims(dict(x="lon", y="lat"))
            # remap coordinates to GOT convention
            ds = ds.rename(mapping_coords)
            # update variable attributes
            for att_name, att_val in attrs.items():
                ds[att_name].attrs.update(att_val)
            # add global attributes
            ds.attrs["title"] = "GOT tidal constituent data"
            ds.attrs["authors"] = "Richard Ray"
            ds.attrs["institution"] = "NASA Goddard Space Flight Center"
            # define and fill constituent ID
            ds.attrs["Constituent"] = v.upper().encode("utf8")
            ds.attrs["date_created"] = datetime.datetime.now().isoformat()
            ds.attrs["software_reference"] = pyTMD.version.project_name
            ds.attrs["software_version"] = pyTMD.version.full_version
            # write GOT netCDF4 file
            FILE = path.joinpath(f"{v}.nc")
            ds.to_netcdf(FILE, mode=mode, **kwargs)
