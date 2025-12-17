#!/usr/bin/env python
"""
fetch_jpl_ssd.py
Written by Tyler Sutterley (12/2025)
Download planetary ephemeride kernels from the
    JPL Solar System Dynamics server

CALLING SEQUENCE:
    python fetch_jpl_ssd.py --kernel=de440s.bsp

COMMAND LINE OPTIONS:
    --help: list the command line options
    -K X, --kernel X: JPL kernel file to download
    -L X, --local X: Local path to kernel file
    -t X, --timeout X: timeout in seconds for blocking operations
    -M X, --mode X: Local permissions mode of the files downloaded

PYTHON DEPENDENCIES:
    future: Compatibility layer between Python 2 and Python 3
        https://python-future.org/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Written 12/2025
"""
import os
import logging
import pathlib
import argparse
import pyTMD.utilities
from timescale.time import parse

def fetch_jpl_ssd(
    kernel="de440s.bsp",
    local: str | pathlib.Path | None = None,
    timeout: int | None = None,
    mode: oct = 0o775,
):
    """
    Download `planetary ephemeride kernels`__ from the JPL Solar
    System Dynamics server

    .. __: https://ssd.jpl.nasa.gov/planets/eph_export.html

    Parameters
    ----------
    kernel: str
        JPL kernel file to download
    local: str or pathlib.Path or NoneType, default None
        Local path to kernel file
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    mode: oct, default 0o775
        permissions mode of output local file
    """
    # create logger for verbosity level
    logger = pyTMD.utilities.build_logger(__name__, level=logging.INFO)
    # determine which kernel file to download
    if local is None:
        # local path to kernel file
        local = pyTMD.utilities.get_cache_path(kernel)
    elif (kernel is None) and (local is not None):
        # verify inputs for remote http host
        local = pathlib.Path(local).expanduser().absolute()
        kernel = local.name
    # remote host path to kernel file
    url = ["https://ssd.jpl.nasa.gov", "ftp", "eph", "planets", "bsp", kernel]
    URL = pyTMD.utilities.URL.from_parts(url)
    # get kernel file from remote host
    logger.info("Downloading JPL Planetary Ephemeride Kernel File")
    logger.info(f"{URL}")
    URL.get(local=local, timeout=timeout)
    # get last modified time from remote host
    last_modified = URL.headers.get("last-modified", None)
    # keep remote modification time of file and local access time
    if last_modified:
        os.utime(local,
            (local.stat().st_atime, parse(last_modified).timestamp())
        )
    # change the permissions mode
    local.chmod(mode=mode)


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Download planetary ephemeride kernels from the
            JPL Solar System Dynamics server
            """,
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    # command line parameters
    parser.add_argument('--kernel',
        '-K',
        type=str,
        default='de440s.bsp',
        help='JPL kernel file to download')
    # local path to kernel file
    parser.add_argument(
        "--local",
        "-L",
        type=pathlib.Path,
        help="Local path to kernel file",
    )
    # connection timeout
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=360,
        help="Timeout in seconds for blocking operations",
    )
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument(
        "--mode",
        "-M",
        type=lambda x: int(x, base=8),
        default=0o775,
        help="Permissions mode of the files downloaded",
    )
    # return the parser
    return parser


# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args, _ = parser.parse_known_args()

    # check internet connection before attempting to run program
    if pyTMD.utilities.check_connection("https://ssd.jpl.nasa.gov"):
        fetch_jpl_ssd(
            args.kernel,
            local=args.local,
            timeout=args.timeout,
            mode=args.mode,
        )


# run main program
if __name__ == "__main__":
    main()

