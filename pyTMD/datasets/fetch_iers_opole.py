#!/usr/bin/env python
"""
fetch_iers_opole.py
Written by Tyler Sutterley (12/2025)
Download ocean pole tide map file from the IERS server

CALLING SEQUENCE:
    python fetch_iers_opole.py

COMMAND LINE OPTIONS:
    --help: list the command line options
    -t X, --timeout X: timeout in seconds for blocking operations
    -M X, --mode X: Local permissions mode of the files downloaded

PYTHON DEPENDENCIES:
    future: Compatibility layer between Python 2 and Python 3
        https://python-future.org/
    timescale: Python tools for time and astronomical calculations
        https://pypi.org/project/timescale/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

REFERENCES:
    S. Desai, "Observing the pole tide with satellite altimetry", Journal of
        Geophysical Research: Oceans, 107(C11), 2002. doi: 10.1029/2001JC001224
    S. Desai, J. Wahr and B. Beckley "Revisiting the pole tide for and from
        satellite altimetry", Journal of Geodesy, 89(12), p1233-1243, 2015.
        doi: 10.1007/s00190-015-0848-7

UPDATE HISTORY:
    Written 12/2025
"""

import os
import logging
import pathlib
import argparse
import pyTMD.utilities
from timescale.time import parse

# default working data directory for tide models
_default_directory = pyTMD.utilities.get_cache_path()


def fetch_iers_opole(
    directory: str | pathlib.Path = _default_directory,
    timeout: int | None = None,
    mode: oct = 0o775,
):
    """
    Download ocean pole tide map file from the IERS server
    :cite:p:`Desai:2002ev,Desai:2015jr`

    Parameters
    ----------
    directory: str or pathlib.Path
        Directory to download the ocean pole tide map file
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    mode: oct, default 0o775
        permissions mode of output local file
    """
    # create logger for verbosity level
    logger = pyTMD.utilities.build_logger(__name__, level=logging.INFO)
    # remote host path to ocean pole file
    url = [
        "https://iers-conventions.obspm.fr",
        "content",
        "chapter7",
        "additional_info",
        "opoleloadcoefcmcor.txt.gz",
    ]
    URL = pyTMD.utilities.URL.from_parts(url)
    # create local directory if it doesn't exist
    directory = pathlib.Path(directory).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True, mode=mode)
    local = directory.joinpath(URL.name)
    # get ocean pole tide file from remote host
    logger.info(URL)
    URL.get(local=local, timeout=timeout)
    # get last modified time from remote host
    last_modified = URL._headers.get("last-modified", None)
    # keep remote modification time of file and local access time
    if last_modified:
        os.utime(
            local, (local.stat().st_atime, parse(last_modified).timestamp())
        )
    # change the permissions mode
    local.chmod(mode=mode)


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Download ocean pole tide map file from the IERS server
            """,
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    # command line parameters
    # working data directory for location of tide models
    parser.add_argument(
        "--directory",
        "-D",
        type=pathlib.Path,
        default=_default_directory,
        help="Working data directory",
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
    if pyTMD.utilities.check_connection("https://iers-conventions.obspm.fr"):
        fetch_iers_opole(
            directory=args.directory,
            timeout=args.timeout,
            mode=args.mode,
        )


# run main program
if __name__ == "__main__":
    main()
