#!/usr/bin/env python
u"""
reduce_otis.py
Written by Tyler Sutterley (11/2025)
Read OTIS-format tidal files and reduce to a regional subset

COMMAND LINE OPTIONS:
    -D X, --directory X: working data directory
    -T X, --tide X: Tide model to use
    -B X, --bounds X: Grid Bounds (xmin,xmax,ymin,ymax)
    --projection X: spatial projection of bounds as EPSG code or PROJ4 string
        4326: latitude and longitude coordinates on WGS84 reference ellipsoid
    -M X, --mode X: permissions mode of the output files

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    scipy: Scientific Tools for Python
        https://docs.scipy.org/doc/
    pyproj: Python interface to PROJ library
        https://pypi.org/project/pyproj/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    io/model.py: retrieves tide model parameters for named tide models
    io/OTIS.py: extract tidal harmonic constants from OTIS tide models
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 11/2025: use new xarray file access protocols for OTIS files
    Updated 10/2025: change default directory for tide models to cache
    Updated 09/2025: renamed module and function to reduce_otis
        made a callable function and added function docstrings
    Updated 07/2025: add a default directory for tide models
    Updated 11/2024: use "stem" instead of "basename"
    Updated 07/2024: renamed format for ATLAS to ATLAS-compact
    Updated 04/2024: add debug mode printing input arguments
        use wrapper to importlib for optional dependencies
    Updated 12/2023: use new crs class for coordinate reprojection
    Updated 04/2023: using pathlib to define and expand paths
    Updated 03/2023: new function name for coordinate reference systems
    Updated 12/2022: refactored OTIS model input and output
    Updated 11/2022: place some imports within try/except statements
        use f-strings for formatting verbose or ascii output
    Updated 05/2022: updated keyword arguments to read tide model programs
    Updated 04/2022: use argparse descriptions within documentation
    Updated 11/2021: add function for attempting to extract projection
    Updated 09/2021: refactor to use model class for files and attributes
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 10/2020: using argparse to set command line parameters
    Updated 09/2020: can use projected coordinates for output model bounds
        compatibility updates for python3
    Updated 07/2020: renamed coordinate conversion program
    Updated 02/2020: changed CATS2008 grid to match version on U.S. Antarctic
        Program Data Center http://www.usap-dc.org/view/dataset/601235
    Updated 11/2019: added AOTIM-5-2018 tide model (2018 update to 2004 model)
    Written 08/2018
"""
from __future__ import print_function

import sys
import os
import logging
import pathlib
import argparse
import traceback
import xarray as xr
import numpy as np
import pyTMD.io
import pyTMD.utilities
import timescale.time

# default data directory for tide models
_default_directory = pyTMD.utilities.get_cache_path()

# PURPOSE: keep track of threads
def info(args):
    logging.debug(pathlib.Path(sys.argv[0]).name)
    logging.debug(args)
    logging.debug(f'module name: {__name__}')
    if hasattr(os, 'getppid'):
        logging.debug(f'parent process: {os.getppid():d}')
    logging.debug(f'process id: {os.getpid():d}')

# PURPOSE: reads OTIS-format tidal files and reduces to a regional subset
def reduce_otis(MODEL: str,
        directory: str | pathlib.Path | None = _default_directory,
        bounds=4*[None],
        projection='4326',
        mode=0o775
    ):
    """
    Reads OTIS-format tidal files and reduces to a regional subset

    Parameters
    ----------
    MODEL: str
        Tide model to use
    DIRECTORY: str or pathlib.Path
        Working data directory
    BOUNDS: list, default 4*[None]
        Grid bounds for reducing model [xmin,xmax,ymin,ymax]
    PROJECTION: str, default '4326'
        Spatial projection as EPSG code or PROJ4 string
    MODE: oct, default 0o775
        Permission mode of the output files
    """
    # get parameters for tide model grid
    m = pyTMD.io.model(directory=directory).from_database(MODEL)

    # read the OTIS-format tide grid file
    if (m.format == 'ATLAS-compact'):
        # if reading a global solution with localized solutions
        dsg, dtg = pyTMD.io.OTIS.open_atlas_grid(m['z'].grid_file)
        dsz, dtz = pyTMD.io.OTIS.open_atlas_elevation(m['z'].model_file)
        dsu, dsv, dtu, dtv = pyTMD.io.OTIS.open_atlas_transport(m['u'].model_file)
        # combine local solutions with global solution
        dsg = dsg.compact.combine_local(dtg)
        dsz = dsz.compact.combine_local(dtz)
        dsu = dsu.compact.combine_local(dtu)
        dsv = dsv.compact.combine_local(dtv)
    else:
        # if reading a pure global solution
        dsg = pyTMD.io.OTIS.open_otis_grid(m['z'].grid_file)
        dsz = pyTMD.io.OTIS.open_otis_grid(m['z'].model_file)
        dsu, dsv = pyTMD.io.OTIS.open_otis_transport(m['u'].model_file)

    # convert bounds to model coordinates
    # bounds is in the form [xmin,xmax,ymin,ymax]
    x, y = dsg.tmd.transform_as(bounds[:2], bounds[2:], crs=projection)
    # merge bathymetry and elevation datasets
    ds = xr.merge([dsg, dsz], compat='override')
    # crop datasets and create new datatree
    dtree = xr.DataTree()
    dtree['z'] = ds.tmd.crop([x.min(), x.max(), y.min(), y.max()])
    dtree['U'] = dsu.tmd.crop([x.min(), x.max(), y.min(), y.max()])
    dtree['V'] = dsv.tmd.crop([x.min(), x.max(), y.min(), y.max()])

    # create unique filenames for reduced datasets
    new_grid_file = create_unique_filename(m['z'].grid_file)
    new_elevation_file = create_unique_filename(m['z'].model_file)
    new_transport_file = create_unique_filename(m['u'].model_file)
    # output reduced datasets to file
    dtree.otis.to_grid(new_grid_file)
    dtree.otis.to_elevation(new_elevation_file)
    dtree.otis.to_transport(new_transport_file)
    # change the permissions level to mode
    new_grid_file.chmod(mode=mode)
    new_elevation_file.chmod(mode=mode)
    new_transport_file.chmod(mode=mode)

# PURPOSE: create a unique filename adding a numerical instance if existing
def create_unique_filename(filename):
    # split filename into parts
    filename = pathlib.Path(filename)
    stem = filename.stem
    suffix = '' if (filename.suffix in ('.out','.oce')) else filename.suffix
    # replace extension with reduced flag
    filename = filename.with_name(f'{stem}{suffix}.reduced')
    # create counter to add to the end of the filename if existing
    counter = 1
    while counter:
        try:
            # open file descriptor only if the file doesn't exist
            fd = filename.open(mode='xb')
        except OSError:
            pass
        else:
            # close the file descriptor and return the filename
            fd.close()
            return filename
        # new filename adds counter before the file extension
        filename = filename.with_name(f'{stem}{suffix}.reduced_{counter:d}')
        counter += 1

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Read OTIS-format tidal files and reduce to a regional
            subset
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    # command line options
    # set data directory containing the tidal data
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=_default_directory,
        help='Working data directory')
    # tide model to use
    model_choices = ('CATS0201','CATS2008','CATS2008_load','TPXO9-atlas',
        'TPXO9.1','TPXO8-atlas','TPXO7.2','TPXO7.2_load','AODTM-5','AOTIM-5',
        'AOTIM-5-2018')
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, default='TPXO9.1',
        choices=model_choices,
        help='Tide model to use')
    # spatial projection (EPSG code or PROJ4 string)
    parser.add_argument('--projection','-P',
        type=str, default='4326',
        help='Spatial projection as EPSG code or PROJ4 string')
    # bounds for reducing model (xmin,xmax,ymin,ymax)
    parser.add_argument('--bounds','-B',
        metavar=('xmin','xmax','ymin','ymax'), type=float, nargs=4,
        help='Grid bounds for reducing model')
    # verbose output of processing run
    # print information about processing run
    parser.add_argument('--verbose','-V',
        action='count', default=0,
        help='Verbose output of processing run')
    # permissions mode of output reduced files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of the output files')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # create logger
    loglevels = [logging.CRITICAL, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=loglevels[args.verbose])

    # try to run regional program
    try:
        info(args)
        reduce_otis(args.tide,
            directory=args.directory,
            bounds=args.bounds,
            projection=args.projection,
            mode=args.mode
        )
    except Exception as exc:
        # if there has been an error exception
        # print the type, value, and stack trace of the
        # current exception being handled
        logging.critical(f'process id {os.getpid():d} failed')
        logging.error(traceback.format_exc())

# run main program
if __name__ == '__main__':
    main()
