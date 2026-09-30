#!/usr/bin/env python
"""
fetch_usap_cats.py
Written by Tyler Sutterley (09/2026)
Download Circum-Antarctic Tidal Simulations from the US Antarctic Program

CATS2008: https://www.usap-dc.org/view/dataset/601235
CATS2008-v2023: https://www.usap-dc.org/view/dataset/601772

CALLING SEQUENCE:
    python fetch_usap_cats.py --tide CATS2008 --token <usap-dc-api-token>

COMMAND LINE OPTIONS:
    --help: list the command line options
    -D X, --directory X: working data directory
    -T X, --tide X: Circum-Antarctic tide model to download
        CATS2008
        CATS2008-v2023
    -t X, --timeout X: timeout in seconds for blocking operations
    -M X, --mode X: Local permissions mode of the files downloaded

PYTHON DEPENDENCIES:
    future: Compatibility layer between Python 2 and Python 3
        https://python-future.org/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

REFERENCE:
    https://www.usap-dc.org/api

UPDATE HISTORY:
    Updated 09/2026: new USAP-DC API for querying and fetching datasets
    Updated 04/2023: using pathlib to define and expand paths
    Updated 11/2022: use f-strings for formatting verbose or ascii output
    Updated 04/2022: use argparse descriptions within documentation
    Updated 10/2021: using python logging for handling verbose output
    Updated 08/2021: USAP now requires captchas for dataset downloads
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 10/2020: using argparse to set command line parameters
    Written 08/2020
"""

from __future__ import print_function, annotations

import re
import io
import ssl
import shutil
import logging
import pathlib
import zipfile
import argparse
import posixpath
import timescale
import pyTMD.utilities

# default data directory for tide models
_default_directory = pyTMD.utilities.get_cache_path()
# default ssl context
_default_ssl_context = pyTMD.utilities._default_ssl_context
# USAP-DC API host
_usap_api_url = "https://www.usap-dc.org/api/v2.0"


# PURPOSE: create an opener for USAP-DC with a supplied user access token
def build_opener(
    token: str,
    context: ssl.SSLContext = _default_ssl_context,
    redirect: bool = True,
):
    """
    Build ``urllib`` opener for USAP-DC with supplied user access token

    Parameters
    ----------
    token: str
        USAP-DC user access token
    context: obj, default pyTMD.utilities._default_ssl_context
        ``SSL`` context for ``urllib`` opener object
    redirect: bool, default True
        Create redirect handler object
    """
    # https://docs.python.org/3/howto/urllib2.html#id5
    handler = []
    # create cookie jar for storing cookies for session
    cookie_jar = pyTMD.utilities.CookieJar()
    handler.append(pyTMD.utilities.urllib2.HTTPCookieProcessor(cookie_jar))
    handler.append(pyTMD.utilities.urllib2.HTTPSHandler(context=context))
    # redirect handler
    if redirect:
        handler.append(pyTMD.utilities.urllib2.HTTPRedirectHandler())
    # create "opener" (OpenerDirector instance)
    opener = pyTMD.utilities.urllib2.build_opener(*handler)
    # add Authorization header to opener
    opener.addheaders = [("X-Auth-Token", token)]
    # Now all calls to urllib2.urlopen use our opener.
    pyTMD.utilities.urllib2.install_opener(opener)
    return opener


# PURPOSE: Download Circum-Antarctic Tidal Simulations from USAP
def fetch_usap_cats(
    model: str,
    token: str,
    directory: str | pathlib.Path | None = _default_directory,
    timeout: int | None = None,
    chunk: int = 16384,
    mode: oct = 0o775,
):
    """
    Download Circum-Antarctic Tidal Simulations from the US Antarctic Program

    Parameters
    ----------
    model: str
        Circum-Antarctic tide model to download
    token: str
        USAP-DC user access token
    directory: str or pathlib.Path
        Working data directory
    timeout: int, default None
        Timeout in seconds for blocking operations
    chunk: int, default 16384
        Chunk size for transfer encoding
    mode: oct, default 0o775
        Local permissions mode of the files downloaded
    """

    # create logger for verbosity level
    logger = pyTMD.utilities.build_logger(__name__, level=logging.INFO)

    # USAP-DC dataset identification number
    dataset_uid = {}
    dataset_uid["CATS2008"] = "601235"
    dataset_uid["CATS2008-v2023"] = "601772"
    # remote subdirectories for each model
    dateparts = {}
    dateparts["CATS2008"] = "2019-12-19T23:26:43.6Z"
    dateparts["CATS2008-v2023"] = "2024-03-27T18:10:28.0Z"
    # local subdirectory for each model
    LOCAL = {}
    LOCAL["CATS2008"] = "CATS2008"
    LOCAL["CATS2008-v2023"] = "CATS2008_v2023"

    # recursively create directories if non-existent
    directory = pyTMD.utilities.Path(directory).resolve()
    local_dir = directory.joinpath(LOCAL[model])
    local_dir.mkdir(mode=mode, parents=True, exist_ok=True)

    # build list of API query parameters
    QUERY = {}
    QUERY["dataset_uid"] = dataset_uid[model]

    # query dataset API
    HOST = pyTMD.utilities.URL(_usap_api_url)
    URL = HOST.joinpath(
        "datasets",
        "?" + pyTMD.utilities.urlencode(QUERY),
    )
    # get JSON response
    response = URL.load().pop()

    # for each file
    for f in response["file_names"]:
        # download file or extract files from zip
        URL = HOST.joinpath(
            "datafiles",
            dataset_uid[model],
            dateparts[model],
            f,
        )
        logger.info(f"{URL} -->\n")
        response = URL.urlopen()
        if pathlib.Path(f).suffix == ".zip":
            # copy remote file contents to bytesIO object
            remote_buffer = io.BytesIO()
            shutil.copyfileobj(response, remote_buffer, chunk)
            remote_buffer.seek(0)
            # extract the zip file into the local directory
            with zipfile.ZipFile(remote_buffer) as z:
                # extract each file and set permissions
                for member in z.filelist:
                    # strip directories from member filename
                    member.filename = pathlib.Path(member.filename).name
                    z.extract(path=local_dir, member=member)
                    local_file = local_dir.joinpath(member.filename)
                    logger.info(f"\t{str(local_file)}\n")
                    local_file.chmod(mode=mode)
        else:
            # write the file to the local directory
            local_file = local_dir.joinpath(f)
            logger.info(f"\t{str(local_file)}\n")
            with local_file.open(mode="wb") as f:
                shutil.copyfileobj(response, f, chunk)
            # change the permissions mode
            local_file.chmod(mode=mode)


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Download Circum-Antarctic Tidal Simulations from the
            US Antarctic Program
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
        default=pathlib.Path.cwd(),
        help="Working data directory",
    )
    # USAP-DC user access token
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="User access token for USAP-DC API",
    )
    # Antarctic Ocean tide model to download
    parser.add_argument(
        "--tide",
        "-T",
        metavar="TIDE",
        type=str,
        nargs="+",
        default=["CATS2008"],
        choices=("CATS2008", "CATS2008-v2023"),
        help="Circum-Antarctic tide model to download",
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

    # build an opener for accessing USAP-DC API
    opener = build_opener(args.token)
    # check internet connection before attempting to run program
    if pyTMD.utilities.check_connection("https://www.usap-dc.org"):
        for m in args.tide:
            fetch_usap_cats(
                m,
                args.token,
                directory=args.directory,
                timeout=args.timeout,
                mode=args.mode,
            )


# run main program
if __name__ == "__main__":
    main()
