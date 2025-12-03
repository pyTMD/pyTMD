#!/usr/bin/env python
u"""
fetch_gsfc_got.py
Written by Tyler Sutterley (10/2025)
Download Goddard Ocean Tide (GOT) models

CALLING SEQUENCE:
    python fetch_gsfc_got.py --tide=GOT5.6

COMMAND LINE OPTIONS:
    --help: list the command line options
    -D X, --directory X: working data directory
    -T X, --tide X: GOT tide model to download
        GOT4.8
        GOT4.10
        GOT5.5
        GOT5.5D
        GOT5.6
        RE14
    --format: GOT tide model format to download
        ascii
        netCDF
    -G, --gzip: compress output ascii and netCDF4 tide files
    -t X, --timeout X: timeout in seconds for blocking operations
    -M X, --mode X: Local permissions mode of the files downloaded

PYTHON DEPENDENCIES:
    future: Compatibility layer between Python 2 and Python 3
        https://python-future.org/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 10/2025: change default directory for tide models to cache
    Updated 09/2025: renamed module and function to fetch_gsfc_got
        made a callable function and added function docstrings
    Updated 07/2025: add a default directory for tide models
    Updated 01/2025: scrubbed use of pathlib.os to just use os directly
    Updated 09/2024: added Ray and Erofeeva (2014) long-period tide model
    Updated 08/2024: keep prime nomenclature for 3rd degree tides
    Written 07/2024
"""
from __future__ import print_function, annotations

import os
import re
import gzip
import shutil
import logging
import pathlib
import tarfile
import argparse
import posixpath
import pyTMD.utilities

# default data directory for tide models
_default_directory = pyTMD.utilities.get_cache_path()

# PURPOSE: Download Arctic Ocean Tide Models from the NSF ArcticData archive
def fetch_gsfc_got(model: str,
        directory: str | pathlib.Path | None = _default_directory,
        format: str = 'netcdf',
        compressed: bool = False,
        timeout: int | None = None,
        mode: oct = 0o775
    ):
    """
    Download Goddard Ocean Tide (GOT) models from NASA Goddard Space Flight
    Center (GSFC)

    Parameters
    ----------
    model: str
        GOT tide model to download
    directory: str or pathlib.Path
        Working data directory
    format: str, default 'netcdf'
        GOT tide model format to download
    compressed: bool, default False
        Compress output ascii and netCDF4 tide files
    timeout: int, default None
        Timeout in seconds for blocking operations
    mode: oct, default 0o775
        Local permissions mode of the files downloaded
    """

    # create logger for verbosity level
    logger = pyTMD.utilities.build_logger(__name__, level=logging.INFO)
    # if compressing the output files
    opener = gzip.open if compressed else open

    # url for each tide model tarfile
    PATH = {}
    PATH['GOT4.8'] = ['2022-07','got4.8.tar.gz']
    PATH['GOT4.10'] = ['2023-12','got4.10c.tar.gz']
    PATH['GOT5.5'] = ['2024-07','GOT5.5.tar%201.gz']
    PATH['GOT5.5D'] = ['2024-07','GOT5.5D.tar%201.gz']
    PATH['GOT5.6'] = ['2024-07','GOT5.6.tar%201.gz']
    PATH['RE14'] = ['2022-07','re14_longperiodtides_rel.tar']
    # tarfile mode for each tide model
    TAR = {}
    TAR['GOT4.8'] = 'r:gz'
    TAR['GOT4.10'] = 'r:gz'
    TAR['GOT5.5'] = 'r:gz'
    TAR['GOT5.5D'] = 'r:gz'
    TAR['GOT5.6'] = 'r:gz'
    TAR['RE14'] = 'r'

    # recursively create directories if non-existent
    directory = pyTMD.utilities.Path(directory).resolve()
    directory.mkdir(mode=mode, parents=True, exist_ok=True)

    # build host url for model
    url = ['https://earth.gsfc.nasa.gov','sites','default','files',*PATH[model]]
    # download tarfile from host
    logger.info(f'{posixpath.join(*url)} -->\n')
    fileobj = pyTMD.utilities.from_http(url, timeout=timeout)
    # open the tar file
    tar = tarfile.open(name=PATH[model][-1], fileobj=fileobj, mode=TAR[model])
    # read tar file and extract all files
    member_files = [m for m in tar.getmembers() if tarfile.TarInfo.isfile(m)]
    for m in member_files:
        # extract file contents to new file
        base, sfx = posixpath.splitext(m.name)
        # skip files that are not in the desired format
        if (sfx == '.nc') and (format == 'ascii'):
            continue
        elif (sfx == '.d') and (format == 'netcdf'):
            continue
        elif re.match(r'^.DS', posixpath.basename(m.name)):
            continue
        elif re.match(r'^._', posixpath.basename(m.name)):
            continue
        # output file name
        if sfx in ('.d','.nc') and compressed:
            output = f'{m.name}.gz'
        else:
            output = m.name
        # create local file path
        local_file = directory.joinpath(*posixpath.split(output))
        # check if the local file exists
        if local_file.exists() and newer(m.mtime, local_file.stat().st_mtime):
            # check the modification time of the local file
            # if remote file is newer: overwrite the local file
            continue
        # print the file being transferred
        logger.info(f'\t{str(local_file)}')
        # recursively create output directory if non-existent
        local_file.parent.mkdir(mode=mode, parents=True, exist_ok=True)
        # extract file to local directory
        with tar.extractfile(m) as f_in, opener(local_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        # get last modified date of remote file within tar file
        # keep remote modification time of file and local access time
        os.utime(local_file, (local_file.stat().st_atime, m.mtime))
        local_file.chmod(mode=mode)

# PURPOSE: compare the modification time of two files
def newer(t1: int, t2: int) -> bool:
    """
    Compare the modification time of two files
    
    Parameters
    ----------
    t1: int
        Modification time of first file
    t2: int
        Modification time of second file
    """
    return (pyTMD.utilities.even(t1) <= pyTMD.utilities.even(t2))

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Download Goddard Tide models from NASA
            Goddard Space Flight Center (GSFC)
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    # command line parameters
    # working data directory for location of tide models
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=_default_directory,
        help='Working data directory')
    # Goddard Ocean Tide model to download
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, nargs='+', default=['GOT5.5'],
        choices=('GOT4.8','GOT4.10','GOT5.5','GOT5.5D','GOT5.6','RE14'),
        help='Goddard Ocean Tide model to download')
    # Goddard Ocean Tide model format to download
    parser.add_argument('--format',
        type=str, default='netcdf',
        choices=('ascii','netcdf'),
        help='Goddard Ocean Tide model format to download')
    # compress output ascii and netCDF4 tide files with gzip
    parser.add_argument('--gzip','-G',
        default=False, action='store_true',
        help='Compress output ascii and netCDF tide files')
    # connection timeout
    parser.add_argument('--timeout','-t',
        type=int, default=360,
        help='Timeout in seconds for blocking operations')
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permissions mode of the files downloaded')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # check internet connection before attempting to run program
    if pyTMD.utilities.check_connection('https://earth.gsfc.nasa.gov'):
        for m in args.tide:
            fetch_gsfc_got(m,
                directory=args.directory,
                format=args.format,
                compressed=args.gzip,
                timeout=args.timeout,
                mode=args.mode
            )

# run main program
if __name__ == '__main__':
    main()
