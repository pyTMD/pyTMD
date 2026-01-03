#!/usr/bin/env python
"""
fetch_box_tpxo.py
Written by Tyler Sutterley (01/2026)
Downloads TPXO ATLAS tide models from the box file sharing service

Need to generate a user token that has sufficient permissions to
access the folder containing the tide model files.
See: https://developer.box.com/guides/

Developer tokens can be created from the Box Developer Console:
https://app.box.com/developers/console

CALLING SEQUENCE:
    python fetch_box_tpxo.py --token <token> --tide TPXO9-atlas-v5
    where <username> is your box api access token

COMMAND LINE OPTIONS:
    --help: list the command line options
    --directory X: working data directory
    -t X, --token X: user access token for box API
    -F X, --folder X: box folder id for model
    --tide X: TPXO ATLAS model to download
    --currents: download tide model current outputs
    -M X, --mode X: Local permissions mode of the files downloaded

PYTHON DEPENDENCIES:
    future: Compatibility layer between Python 2 and Python 3
        https://python-future.org/

REFERENCE:
    https://developer.box.com/guides/

UPDATE HISTORY:
    Updated 01/2026: fixed the box token to allow file downloads
    Updated 12/2025: use URL class to build and operate on URLs
    Updated 11/2025: use from_database to access model parameters
    Updated 10/2025: change default directory for tide models to cache
    Updated 09/2025: made a callable function and added function docstrings
    Updated 07/2025: add a default directory for tide models
    Updated 01/2025: scrubbed use of pathlib.os to just use os directly
    Updated 09/2024: use model class to define output directory
    Updated 04/2023: using pathlib to define and expand paths
    Updated 01/2023: use default context from utilities module
    Updated 11/2022: use f-strings for formatting verbose or ascii output
    Updated 04/2022: use argparse descriptions within documentation
    Updated 12/2021: added TPXO9-atlas-v5 to list of available tide models
    Updated 10/2021: using python logging for handling verbose output
    Updated 07/2021: can use prefix files to define command line arguments
    Written 03/2021
"""

from __future__ import print_function

import os
import re
import ssl
import gzip
import json
import shutil
import logging
import pathlib
import argparse
import pyTMD.utilities

# default data directory for tide models
_default_directory = pyTMD.utilities.get_cache_path()
# default ssl context
_default_ssl_context = pyTMD.utilities._default_ssl_context
# box API host
_box_api_url = "https://api.box.com/2.0"


# PURPOSE: create an opener for box with a supplied user access token
def build_opener(
    token: str,
    context: ssl.SSLContext = _default_ssl_context,
    redirect: bool = True,
):
    """
    Build ``urllib`` opener for box with supplied user access token

    Parameters
    ----------
    token: str
        box user access token
    context: obj, default pyTMD.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    redirect: bool, default True
        create redirect handler object
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
    opener.addheaders = [("Authorization", f"Bearer {token}")]
    # Now all calls to urllib2.urlopen use our opener.
    pyTMD.utilities.urllib2.install_opener(opener)
    return opener


# PURPOSE: fetch TPXO ATLAS files from box server
def fetch_box_tpxo(
    model: str,
    folder_id: str,
    directory: str | pathlib.Path | None = _default_directory,
    currents: bool = False,
    compressed: bool = False,
    timeout: int | None = None,
    chunk: int = 16384,
    mode: oct = 0o775,
    **kwargs,
):
    """
    Download files from box file sharing service

    Parameters
    ----------
    model: str
        TPXO9-atlas tide model to download
    folder_id: str
        box folder id for model
    directory: str or pathlib.Path
        download directory
    currents: bool, default False
        download tidal current files
    compressed: bool, default False
        Compress output binary or netCDF4 tide files
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    chunk: int, default 16384
        chunk size for transfer encoding
    mode: oct, default 0o775
        permissions mode of output local file
    """
    # create logger for verbosity level
    logger = pyTMD.utilities.build_logger(__name__, level=logging.INFO)

    # check if local directory exists and recursively create if not
    m = pyTMD.io.model(directory=directory, verify=False).from_database(model)
    localpath = m["z"].model_file[0].parent
    # create output directory if non-existent
    localpath.mkdir(mode=mode, parents=True, exist_ok=True)
    # if compressing the output file
    opener = gzip.open if compressed else open

    # regular expression pattern for files of interest
    regex_patterns = []
    regex_patterns.append("grid")
    regex_patterns.append("h")
    # append currents
    if currents:
        regex_patterns.append("u")
    # build regular expression object
    rx = re.compile(r"^({0})".format(r"|".join(regex_patterns)), re.VERBOSE)

    # box api url
    URL = pyTMD.utilities.URL(_box_api_url)
    # create and submit request and load JSON response
    folder_url = URL.joinpath("folders", folder_id, "items")
    folder_contents = json.loads(folder_url.read())
    # find files of interest
    file_entries = [
        entry
        for entry in folder_contents["entries"]
        if (entry["type"] == "file") and rx.match(entry["name"])
    ]
    # for each file in the folder
    for entry in file_entries:
        # need to have sufficient permissions for downloading content
        file_url = URL.joinpath("files", entry["id"])
        content_url = file_url.joinpath("content")
        # print remote path
        logger.info(f"{file_url} -->")
        # create and submit request and load JSON response
        file_contents = json.loads(file_url.read())
        remote_mtime = pyTMD.utilities.get_unix_time(
            file_contents["modified_at"], format="%Y-%m-%dT%H:%M:%S%z"
        )
        # local output filename
        output = entry.get("name")
        # append .gz to filename if compressing
        if compressed:
            output += ".gz"
        # print file information
        local = localpath.joinpath(output)
        logger.info(f"\t{str(local)}")
        # extract file to local directory
        with (
            content_url.urlopen(timeout=timeout) as f_in,
            opener(local, "wb") as f_out,
        ):
            shutil.copyfileobj(f_in, f_out, chunk)
        # keep remote modification time of file and local access time
        os.utime(local, (local.stat().st_atime, remote_mtime))
        # change the permissions mode of the local file
        local.chmod(mode=mode)


# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Downloads TPXO ATLAS global
            tide models from the box file sharing service
            """,
        fromfile_prefix_chars="@",
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    # command line parameters
    # working data directory
    parser.add_argument(
        "--directory",
        "-D",
        type=pathlib.Path,
        default=_default_directory,
        help="Working data directory",
    )
    # box user access token
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="User access token for box API",
    )
    # box folder id
    parser.add_argument(
        "--folder", "-F", type=str, default="", help="box folder id for model"
    )
    # TPXO ATLAS tide models
    parser.add_argument(
        "--tide",
        "-T",
        type=str,
        default="TPXO10-atlas-v2",
        help="TPXO ATLAS tide model to download",
    )
    # download tidal currents
    parser.add_argument(
        "--currents",
        default=False,
        action="store_true",
        help="Download tide model current outputs",
    )
    # compress output binary or netCDF4 tide files with gzip
    parser.add_argument(
        "--gzip",
        "-G",
        default=False,
        action="store_true",
        help="Compress output binary or netCDF4 tide files",
    )
    # connection timeout
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=3600,
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

    # build an opener for accessing box folders
    opener = build_opener(args.token)
    # check internet connection before attempting to run program
    if pyTMD.utilities.check_connection("https://app.box.com/"):
        fetch_box_tpxo(
            args.tide,
            args.folder,
            directory=args.directory,
            currents=args.currents,
            compressed=args.gzip,
            timeout=args.timeout,
            mode=args.mode,
        )


# run main program
if __name__ == "__main__":
    main()
