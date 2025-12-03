#!/usr/bin/env python
u"""
fetch_test_data.py
Written by Tyler Sutterley (09/2025)
Download files necessary to run the test suite

CALLING SEQUENCE:
    python fetch_test_data.py

COMMAND LINE OPTIONS:
    --help: list the command line options
    -D X, --directory X: working data directory
    -t X, --timeout X: timeout in seconds for blocking operations
    -M X, --mode X: Local permissions mode of the files downloaded

PYTHON DEPENDENCIES:
    future: Compatibility layer between Python 2 and Python 3
        https://python-future.org/

PROGRAM DEPENDENCIES:
    utilities.py: download and management utilities for syncing files

UPDATE HISTORY:
    Updated 10/2025: change default directory for tide models to cache
    Written 10/2025
"""
import ssl
import json
import shutil
import logging
import pathlib
import zipfile
import argparse
import pyTMD.utilities

# default working data directory for tide models
_default_directory = pyTMD.utilities.get_cache_path()
# default ssl context
_default_ssl_context = pyTMD.utilities._default_ssl_context

def fetch_test_data(
        directory: str | pathlib.Path = _default_directory,
        provider: str = 'figshare',
        mode: oct = 0o775,
        **kwargs
    ):
    """
    Download files necessary to run the test suite

    Parameters
    ----------
    directory: str or pathlib.Path
        download directory
    provider: str, default 'figshare'
        data provider name
    kwargs: dict
        additional keyword arguments for data provider functions
    """
    # create download directory if it doesn't exist
    directory = pyTMD.utilities.Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=mode)
    # create logger for verbosity level
    logger = pyTMD.utilities.build_logger(__name__, level=logging.INFO)
    if (provider == 'figshare'):
        from_figshare(directory=directory, logger=logger, **kwargs)
    else:
        raise ValueError(f'Unknown data provider: {provider}')

# PURPOSE: download data files from figshare
def from_figshare(
        directory: str | pathlib.Path = _default_directory,
        article: str = '30260326',
        timeout: int | None = None,
        context: ssl.SSLContext = _default_ssl_context,
        chunk: int | None = 16384,
        logger: logging.Logger | None = None,
        mode: oct = 0o775,
        **kwargs
    ):
    """
    Download files necessary to run the test suite from figshare

    Parameters
    ----------
    directory: str or pathlib.Path
        download directory
    article: str, default '30260326'
        figshare article number
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default pyTMD.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    hash: str, default ''
        MD5 hash of local file
    chunk: int, default 16384
        chunk size for transfer encoding
    logger: logging.logger object
        Logger for outputting file transfer information
    mode: oct, default 0o775
        permissions mode of output local file
    """
    # figshare host for articles
    HOST = ['https://api.figshare.com', 'v2', 'articles', article]
    # Create and submit request
    response = pyTMD.utilities.from_http(HOST,
        timeout=timeout, context=context)
    resp = json.loads(response.read())
    # for each file in the JSON response
    for f in resp['files']:
        # check if file already exists by matching MD5 checksums
        local_file = directory.joinpath(f['name'])
        original_md5 = local_file.md5_hash
        # skip download if checksums match
        if (original_md5 == f['supplied_md5']):
            continue
        # output file information
        logger.info(f["download_url"])
        # get remote file as a byte-stream
        remote_buffer = pyTMD.utilities.from_http(f['download_url'],
            timeout=timeout, context=context)
        # verify MD5 checksums
        computed_md5 = pyTMD.utilities.get_hash(remote_buffer)
        # raise exception if checksums do not match
        if (computed_md5 != f['supplied_md5']):
            raise Exception(f'Checksum mismatch: {f["download_url"]}')
        # download file or extract files from zip
        if (pathlib.Path(f['name']).suffix == '.zip'):
            # extract the zip file into the local directory
            with zipfile.ZipFile(remote_buffer) as z:
                # extract each file and set permissions
                for member in z.filelist:
                    z.extract(path=directory, member=member)
                    local_file = directory.joinpath(member.filename)
                    local_file.chmod(mode=mode)
        else:
            # write the file to the local directory
            with local_file.open(mode='wb') as f:
                shutil.copyfileobj(remote_buffer, f, chunk)
            # change the permissions mode
            local_file.chmod(mode=mode)

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Download models for running the test suite
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    # command line parameters
    # working data directory for location of tide models
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=_default_directory,
        help='Working data directory')
    # download provider
    parser.add_argument('--provider','-P',
        metavar='PROVIDER', type=str, default='figshare',
        choices=('figshare',),
        help='Data provider')
    # connection timeout
    parser.add_argument('--timeout','-t',
        type=int, default=3600,
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

    # fetch test data
    fetch_test_data(
        directory=args.directory,
        provider=args.provider,
        timeout=args.timeout,
        mode=args.mode
    )

# run main program
if __name__ == '__main__':
    main()
