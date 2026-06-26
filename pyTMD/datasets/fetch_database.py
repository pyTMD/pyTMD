#!/usr/bin/env python
"""
fetch_database.py
Written by Tyler Sutterley (06/2026)
Downloads an updated pyTMD model JSON database file from the
project GitHub repository

COMMAND LINE OPTIONS:
    --help: list the command line options
    -t X, --timeout X: timeout in seconds for blocking operations
    -M X, --mode X: Local permissions mode of the files downloaded

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Written 06/2026
"""

import os
import logging
import argparse
import pyTMD.utilities


# PURPOSE: downloads a newer version of a pyTMD database file
def fetch_database(
    username: str = "pyTMD",
    repository: str = "pyTMD",
    branch: str = "main",
    timeout: int | None = 360,
    mode: oct = 0o775,
):
    """
    Downloads an updated pyTMD model JSON database file from
    the project GitHub repository

    Parameters
    ----------
    username: str, default 'pyTMD'
        GitHub username for project repository
    repository: str, default 'pyTMD'
        name of the GitHub repository
    branch: str, default 'main'
        branch of the GitHub repository for downloading files
    timeout: int or None, default 360
        timeout for the HTTP request in seconds
    mode: oct, default 0o775
        permissions mode of output file
    """
    # create logger for verbosity level
    logger = pyTMD.utilities.build_logger(__name__, level=logging.INFO)
    # path to model database
    database = pyTMD.utilities.get_data_path(["data", "database.json"])
    # check if the database file is writable
    if os.access(database, os.R_OK) and not os.access(database, os.W_OK):
        raise PermissionError(f"Database file is not writable: {database}")
    # MD5 hash for comparing with remote
    HASH = pyTMD.utilities.get_hash(database)
    # try downloading from GitHub repository
    HOST = pyTMD.utilities.get_github_url(
        ["pyTMD", "data", database.name],
        username=username,
        repository=repository,
        branch=branch,
    )
    # log message for failed download
    msg = f"Unable to download {database.name} from {username}/{repository}"
    try:
        pyTMD.utilities.from_http(
            HOST,
            local=database,
            hash=HASH,
            timeout=timeout,
            verbose=True,
            mode=mode,
        )
    except pyTMD.utilities.urllib2.HTTPError as exc:
        logger.info(msg)
        logger.debug(exc.code)
    except pyTMD.utilities.urllib2.URLError as exc:
        logger.info(msg)
        logger.debug(exc.reason)


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Downloads an updated pyTMD model JSON database
            """,
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
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

    fetch_database(
        timeout=args.timeout,
        mode=args.mode,
    )


# run main program
if __name__ == "__main__":
    main()
