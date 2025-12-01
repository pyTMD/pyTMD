#!/usr/bin/env python
u"""
fetch_aviso_fes.py
Written by Tyler Sutterley (10/2025)
Downloads the FES (Finite Element Solution) global tide model from AVISO
Decompresses the model tar files into the constituent files and auxiliary files
    https://www.aviso.altimetry.fr/data/products/auxiliary-products/
        global-tide-fes.html
    https://www.aviso.altimetry.fr/en/data/data-access.html

CALLING SEQUENCE:
    python fetch_aviso_fes.py --user <username> --tide FES2014
    where <username> is your AVISO data dissemination server username

COMMAND LINE OPTIONS:
    --help: list the command line options
    --directory X: working data directory
    -U X, --user: username for AVISO FTP servers (email)
    -P X, --password: password for AVISO FTP servers
    -N X, --netrc X: path to .netrc file for authentication
    --tide X: FES tide model to download
        FES1999
        FES2004
        FES2012
        FES2014
        FES2022
    --load: download load tide model outputs
        (FES2014)
    --currents: download tide model current outputs
        (FES2012 and FES2014)
    --extrapolated: Download extrapolated tide model outputs
        (FES2014 and FES2022)
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
        remove printing to log file
    Updated 09/2025: renamed module and function to fetch_aviso_fes
        made a callable function and added function docstrings
    Updated 07/2025: added extrapolation option for FES2014 tide model
        add a default directory for tide models
    Updated 01/2025: new ocean tide directory for latest FES2022 version
        scrubbed use of pathlib.os to just use os directly
    Updated 07/2024: added list and download for FES2022 tide model
        compare modification times with remote to not overwrite files
    Updated 05/2023: added option to change connection timeout
    Updated 04/2023: using pathlib to define and expand paths
        added option to include AVISO FTP password as argument
    Updated 11/2022: added encoding for writing ascii files
        use f-strings for formatting verbose or ascii output
    Updated 04/2022: use argparse descriptions within documentation
    Updated 10/2021: using python logging for handling verbose output
    Updated 07/2021: can use prefix files to define command line arguments
    Updated 05/2021: use try/except for retrieving netrc credentials
    Updated 04/2021: set a default netrc file and check access
    Updated 10/2020: using argparse to set command line parameters
    Updated 07/2020: add gzip option to compress output ascii and netCDF4 files
    Updated 06/2020: added netrc option for alternative authentication
    Updated 05/2019: new authenticated ftp host (changed 2018-05-31)
    Written 09/2017
"""
from __future__ import print_function, annotations

import sys
import os
import io
import re
import gzip
import lzma
import netrc
import shutil
import logging
import tarfile
import getpass
import pathlib
import argparse
import builtins
import posixpath
import calendar, time
import ftplib
import pyTMD.utilities

# default data directory for tide models
_default_directory = pyTMD.utilities.get_cache_path()

# PURPOSE: download local AVISO FES files with ftp server
def fetch_aviso_fes(model: str,
        directory: str | pathlib.Path | None = _default_directory,
        user: str = '',
        password: str = '',
        load: bool = False,
        currents: bool = False,
        extrapolated: bool = False,
        compressed: bool = False,
        timeout: int | None = None,
        mode: oct = 0o775
    ):
    """
    Download AVISO FES global tide models from the AVISO FTP server

    Parameters
    ----------
    model: str
        FES tide model to download
    directory: str or pathlib.Path
        Working data directory
    user: str, default ''
        Username for AVISO Login
    password: str, default ''
        Password for AVISO Login
    load: bool, default False
        Download load tide model outputs
    currents: bool, default False
        Download tide model current outputs
    extrapolated: bool, default False
        Download extrapolated tide model outputs
    compressed: bool, default False
        Compress output ascii and netCDF4 tide files
    timeout: int, default None
        Timeout in seconds for blocking operations
    mode: oct, default 0o775
        Local permissions mode of the files downloaded
    """

    # connect and login to AVISO ftp server
    f = ftplib.FTP('ftp-access.aviso.altimetry.fr', timeout=timeout)
    f.login(user, password)

    # create logger for verbosity level
    logger = pyTMD.utilities.build_logger(__name__, level=logging.INFO)

    # download the FES tide model files
    if model in ('FES1999','FES2004','FES2012','FES2014'):
        aviso_fes_tar(model, f, logger,
            directory=directory,
            load=load,
            currents=currents,
            extrapolated=extrapolated,
            GZIP=compressed,
            MODE=mode
        )
    elif model in ('FES2022',):
        aviso_fes_list(model, f, logger,
            DIRECTORY=directory,
            LOAD=load,
            CURRENTS=currents,
            EXTRAPOLATED=extrapolated,
            GZIP=compressed,
            MODE=mode
        )

    # close the ftp connection
    f.quit()

# PURPOSE: download local AVISO FES files with ftp server
# by downloading tar files and extracting contents
def aviso_fes_tar(MODEL, f, logger,
        DIRECTORY: str | pathlib.Path | None = _default_directory,
        LOAD: bool = False,
        CURRENTS: bool = False,
        EXTRAPOLATED: bool = False,
        GZIP: bool = False,
        MODE: oct = 0o775
    ):
    """
    Download tar-compressed AVISO FES tide models from the AVISO FTP server

    Parameters
    ----------
    MODEL: str
        FES tide model to download
    f: ftplib.FTP object
        Active ftp connection to AVISO server
    logger: logging.logger object
        Logger for outputting file transfer information
    DIRECTORY: str or pathlib.Path
        Working data directory
    LOAD: bool, default False
        Download load tide model outputs
    CURRENTS: bool, default False
        Download tide model current outputs
    EXTRAPOLATED: bool, default False
        Download extrapolated tide model outputs
    GZIP: bool, default False
        Compress output ascii and netCDF4 tide files
    MODE: oct, default 0o775
        Local permissions mode of the files downloaded
    """

    # check if local directory exists and recursively create if not
    localpath = pyTMD.utilities.Path(DIRECTORY).joinpath(MODEL.lower())
    localpath.mkdir(MODE, parents=True, exist_ok=True)

    # path to remote directory for FES
    FES = {}
    # mode for reading tar files
    TAR = {}
    # flatten file structure
    FLATTEN = {}

    # 1999 model
    FES['FES1999']=[]
    FES['FES1999'].append(['fes1999_fes2004','readme_fes1999.html'])
    FES['FES1999'].append(['fes1999_fes2004','fes1999.tar.gz'])
    TAR['FES1999'] = [None,'r:gz']
    FLATTEN['FES1999'] = [None,True]
    # 2004 model
    FES['FES2004']=[]
    FES['FES2004'].append(['fes1999_fes2004','readme_fes2004.html'])
    FES['FES2004'].append(['fes1999_fes2004','fes2004.tar.gz'])
    TAR['FES2004'] = [None,'r:gz']
    FLATTEN['FES2004'] = [None,True]
    # 2012 model
    FES['FES2012']=[]
    FES['FES2012'].append(['fes2012_heights','readme_fes2012_heights_v1.1'])
    FES['FES2012'].append(['fes2012_heights','fes2012_heights_v1.1.tar.lzma'])
    TAR['FES2012'] = []
    TAR['FES2012'].extend([None,'r:xz'])
    FLATTEN['FES2012'] = []
    FLATTEN['FES2012'].extend([None,True])
    if CURRENTS:
        subdir = 'fes2012_currents'
        FES['FES2012'].append([subdir,'readme_fes2012_currents_v1.1'])
        FES['FES2012'].append([subdir,'fes2012_currents_v1.1_block1.tar.lzma'])
        FES['FES2012'].append([subdir,'fes2012_currents_v1.1_block2.tar.lzma'])
        FES['FES2012'].append([subdir,'fes2012_currents_v1.1_block3.tar.lzma'])
        FES['FES2012'].append([subdir,'fes2012_currents_v1.1_block4.tar.lzma'])
        TAR['FES2012'].extend([None,'r:xz','r:xz','r:xz','r:xz'])
        FLATTEN['FES2012'].extend([None,False,False,False,False])
    # 2014 model
    FES['FES2014']=[]
    FES['FES2014'].append(['fes2014_elevations_and_load',
        'readme_fes2014_elevation_and_load_v1.2.txt'])
    FES['FES2014'].append(['fes2014_elevations_and_load',
        'fes2014b_elevations','ocean_tide.tar.xz'])
    TAR['FES2014'] = []
    TAR['FES2014'].extend([None,'r'])
    FLATTEN['FES2014'] = []
    FLATTEN['FES2014'].extend([None,False])
    if LOAD:
        FES['FES2014'].append(['fes2014_elevations_and_load',
            'fes2014a_loadtide','load_tide.tar.xz'])
        TAR['FES2014'].extend(['r'])
        FLATTEN['FES2014'].extend([False])
    if EXTRAPOLATED:
        FES['FES2014'].append(['fes2014_elevations_and_load',
            'fes2014b_elevations_extrapolated',
            'ocean_tide_extrapolated.tar.xz'])
        TAR['FES2014'].extend(['r'])
        FLATTEN['FES2014'].extend([False])
    if CURRENTS:
        subdir = 'fes2014a_currents'
        FES['FES2014'].append([subdir,'readme_fes2014_currents_v1.2.txt'])
        FES['FES2014'].append([subdir,'eastward_velocity.tar.xz'])
        FES['FES2014'].append([subdir,'northward_velocity.tar.xz'])
        TAR['FES2014'].extend(['r'])
        FLATTEN['FES2014'].extend([False])

    # for each file for a model
    for remotepath,tarmode,flatten in zip(FES[MODEL],TAR[MODEL],FLATTEN[MODEL]):
        # download file from ftp and decompress tar files
        ftp_download(logger, f, remotepath, localpath,
            TARMODE=tarmode,
            FLATTEN=flatten,
            GZIP=GZIP,
            MODE=MODE
        )

# PURPOSE: download local AVISO FES files with ftp server
# by downloading individual files
def aviso_fes_list(MODEL, f, logger,
        DIRECTORY: str | pathlib.Path | None = _default_directory,
        LOAD: bool = False,
        CURRENTS: bool = False,
        EXTRAPOLATED: bool = False,
        GZIP: bool = False,
        MODE: oct = 0o775
    ):
    """
    Download AVISO FES2022 tide model files from the AVISO FTP server

    Parameters
    ----------
    MODEL: str
        FES tide model to download
    f: ftplib.FTP object
        Active ftp connection to AVISO server
    logger: logging.logger object
        Logger for outputting file transfer information
    DIRECTORY: str or pathlib.Path
        Working data directory
    LOAD: bool, default False
        Download load tide model outputs
    CURRENTS: bool, default False
        Download tide model current outputs
    EXTRAPOLATED: bool, default False
        Download extrapolated tide model outputs
    GZIP: bool, default False
        Compress output ascii and netCDF4 tide files
    MODE: oct, default 0o775
        Local permissions mode of the files downloaded
    """

    # validate local directory
    DIRECTORY = pyTMD.utilities.Path(DIRECTORY).resolve()

    # path to remote directory for FES
    FES = {}
    # 2022 model
    FES['FES2022'] = []
    # updated directory for ocean tide model
    # latest version fixes the valid_max attribute for longitudes
    FES['FES2022'].append(['fes2022b','ocean_tide_20241025'])
    if LOAD:
        FES['FES2022'].append(['fes2022b','load_tide'])
    if CURRENTS:
        logger.warning('FES2022 does not presently have current outputs')
    if EXTRAPOLATED:
        FES['FES2022'].append(['fes2022b','ocean_tide_extrapolated'])

    # for each model file type
    for subdir in FES[MODEL]:
        local_dir = DIRECTORY.joinpath(*subdir)
        file_list = ftp_list(f, subdir, basename=True, sort=True)
        for fi in file_list:
            remote_path = [*subdir, fi]
            LZMA = fi.endswith('.xz')
            ftp_download(logger, f, remote_path, local_dir,
                LZMA=LZMA,
                GZIP=GZIP,
                CHUNK=32768,
                MODE=MODE
            )

# PURPOSE: List a directory on a ftp host
def ftp_list(f, remote_path,
        basename=False,
        pattern=None,
        sort=False
    ):
    """
    List a directory on a ftp host

    Parameters
    ----------
    f: ftplib.FTP object
        Active ftp connection to AVISO server
    remote_path: list
        Remote path components to directory on ftp server
    basename: bool, default False
        Return only the basenames of the listed items
    pattern: str, default None
        Regular expression pattern to filter listed items
    sort: bool, default False
        Sort the listed items alphabetically
    """
    # list remote path
    output = f.nlst(posixpath.join('auxiliary','tide_model',*remote_path))
    # reduce to basenames
    if basename:
        output = [posixpath.basename(i) for i in output]
    # reduce using regular expression pattern
    if pattern:
        i = [i for i,o in enumerate(output) if re.search(pattern,o)]
        # reduce list of listed items
        output = [output[indice] for indice in i]
    # sort the list
    if sort:
        i = [i for i,o in sorted(enumerate(output), key=lambda i: i[1])]
        # sort list of listed items
        output = [output[indice] for indice in i]
    # return the list of items
    return output

# PURPOSE: pull file from a remote ftp server and decompress if tar file
def ftp_download(logger, f, remote_path, local_dir,
        LZMA=None,
        TARMODE=None,
        FLATTEN=None,
        GZIP=False,
        CHUNK=8192,
        MODE=0o775
    ):
    """
    Pull file from a remote ftp server and decompress if tar file
    
    Parameters
    ----------
    logger: logging.logger object
        Logger for outputting file transfer information
    f: ftplib.FTP object
        Active ftp connection to AVISO server
    remote_path: list
        Remote path components to file on ftp server
    local_dir: str or pathlib.Path
        Local directory to save file
    LZMA: bool, default None
        Decompress lzma-compressed file
    TARMODE: str, default None
        Mode for reading tar-compressed file
    FLATTEN: bool, default None
        Flatten tar file structure when extracting files
    GZIP: bool, default False
        Compress output ascii and netCDF4 tide files
    CHUNK: int, default 8192
        Block size for downloading files from ftp server
    MODE: oct, default 0o775
        Local permissions mode of the files downloaded
    """
    # remote and local directory for data product
    remote_file = posixpath.join('auxiliary','tide_model',*remote_path)
    # if compressing the output file
    opener = gzip.open if GZIP else open

    # Printing files transferred
    remote_ftp_url = posixpath.join('ftp://', f.host, remote_file)
    logger.info(f'{remote_ftp_url} -->')
    if TARMODE:
        # copy remote file contents to bytesIO object
        fileobj = io.BytesIO()
        f.retrbinary(f'RETR {remote_file}', fileobj.write, blocksize=CHUNK)
        fileobj.seek(0)
        # open the tar file
        tar = tarfile.open(name=remote_path[-1], fileobj=fileobj, mode=TARMODE)
        # read tar file and extract all files
        member_files = [m for m in tar.getmembers() if tarfile.TarInfo.isfile(m)]
        for m in member_files:
            member = posixpath.basename(m.name) if FLATTEN else m.name
            base, sfx = posixpath.splitext(m.name)
            # extract file contents to new file
            output = f'{member}.gz' if sfx in ('.asc','.nc') and GZIP else member
            local_file = local_dir.joinpath(*posixpath.split(output))
            # check if the local file exists
            if local_file.exists() and newer(m.mtime, local_file.stat().st_mtime):
                # check the modification time of the local file
                # if remote file is newer: overwrite the local file
                continue
            # print the file being transferred
            logger.info(f'\t{str(local_file)}')
            # recursively create output directory if non-existent
            local_file.parent.mkdir(mode=MODE, parents=True, exist_ok=True)
            # extract file to local directory
            with tar.extractfile(m) as f_in, opener(local_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            # get last modified date of remote file within tar file
            # keep remote modification time of file and local access time
            os.utime(local_file, (local_file.stat().st_atime, m.mtime))
            local_file.chmod(mode=MODE)
    elif LZMA:
        # get last modified date of remote file and convert into unix time
        mdtm = f.sendcmd(f'MDTM {remote_file}')
        mtime = calendar.timegm(time.strptime(mdtm[4:],"%Y%m%d%H%M%S"))
        # output file name for compressed and uncompressed cases
        stem = posixpath.basename(posixpath.splitext(remote_file)[0])
        base, sfx = posixpath.splitext(stem)
        # extract file contents to new file
        output = f'{stem}.gz' if sfx in ('.asc','.nc') and GZIP else stem
        local_file = local_dir.joinpath(output)
        # check if the local file exists
        if local_file.exists() and newer(mtime, local_file.stat().st_mtime):
            # check the modification time of the local file
            # if remote file is newer: overwrite the local file
            return
        # print the file being transferred
        logger.info(f'\t{str(local_file)}')
        # recursively create output directory if non-existent
        local_file.parent.mkdir(mode=MODE, parents=True, exist_ok=True)
        # copy remote file contents to bytesIO object
        fileobj = io.BytesIO()
        f.retrbinary(f'RETR {remote_file}', fileobj.write, blocksize=CHUNK)
        fileobj.seek(0)
        # decompress lzma file and extract contents to local directory
        with lzma.open(fileobj) as f_in, opener(local_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        # get last modified date of remote file within tar file
        # keep remote modification time of file and local access time
        os.utime(local_file, (local_file.stat().st_atime, mtime))
        local_file.chmod(mode=MODE)
    else:
        # copy readme and uncompressed files directly
        stem = posixpath.basename(remote_file)
        base, sfx = posixpath.splitext(stem)
        # output file name for compressed and uncompressed cases
        output = f'{stem}.gz' if sfx in ('.asc','.nc') and GZIP else stem
        local_file = local_dir.joinpath(output)
        # get last modified date of remote file and convert into unix time
        mdtm = f.sendcmd(f'MDTM {remote_file}')
        mtime = calendar.timegm(time.strptime(mdtm[4:],"%Y%m%d%H%M%S"))
        # check if the local file exists
        if local_file.exists() and newer(mtime, local_file.stat().st_mtime):
            # check the modification time of the local file
            # if remote file is newer: overwrite the local file
            return
        # print the file being transferred
        logger.info(f'\t{str(local_file)}\n')
        # recursively create output directory if non-existent
        local_file.parent.mkdir(mode=MODE, parents=True, exist_ok=True)
        # copy remote file contents to local file
        with opener(local_file, 'wb') as f_out:
            f.retrbinary(f'RETR {remote_file}', f_out.write, blocksize=CHUNK)
        # keep remote modification time of file and local access time
        os.utime(local_file, (local_file.stat().st_atime, mtime))
        local_file.chmod(mode=MODE)

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
        description="""Downloads the FES (Finite Element Solution) global tide
            model from AVISO.  Decompresses the model tar files into the
            constituent files and auxiliary files.
            """,
        fromfile_prefix_chars="@"
    )
    parser.convert_arg_line_to_args = pyTMD.utilities.convert_arg_line_to_args
    # command line parameters
    # AVISO FTP credentials
    parser.add_argument('--user','-U',
        type=str, default=os.environ.get('AVISO_USERNAME'),
        help='Username for AVISO Login')
    parser.add_argument('--password','-W',
        type=str, default=os.environ.get('AVISO_PASSWORD'),
        help='Password for AVISO Login')
    parser.add_argument('--netrc','-N',
        type=pathlib.Path, default=pathlib.Path().home().joinpath('.netrc'),
        help='Path to .netrc file for authentication')
    # working data directory
    parser.add_argument('--directory','-D',
        type=pathlib.Path, default=_default_directory,
        help='Working data directory')
    # FES tide models
    choices = ['FES1999','FES2004','FES2012','FES2014','FES2022']
    parser.add_argument('--tide','-T',
        metavar='TIDE', type=str, nargs='+',
        default=['FES2022'], choices=choices,
        help='FES tide model to download')
    # download FES load tides
    parser.add_argument('--load',
        default=False, action='store_true',
        help='Download load tide model outputs')
    # download FES tidal currents
    parser.add_argument('--currents',
        default=False, action='store_true',
        help='Download tide model current outputs')
    # download extrapolate FES tidal data
    parser.add_argument('--extrapolated',
        default=False, action='store_true',
        help='Download extrapolated tide model outputs')
    # compress output ascii and netCDF4 tide files with gzip
    parser.add_argument('--gzip','-G',
        default=False, action='store_true',
        help='Compress output ascii and netCDF4 tide files')
    # connection timeout
    parser.add_argument('--timeout','-t',
        type=int, default=360,
        help='Timeout in seconds for blocking operations')
    # permissions mode of the local directories and files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='Permission mode of directories and files downloaded')
    # return the parser
    return parser

# This is the main part of the program that calls the individual functions
def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # AVISO FTP Server hostname
    HOST = 'ftp-access.aviso.altimetry.fr'
    # get authentication
    if not args.user and not args.netrc.exists():
        # check that AVISO credentials were entered
        args.user = builtins.input(f'Username for {HOST}: ')
        # enter password securely from command-line
        args.password = getpass.getpass(f'Password for {args.user}@{HOST}: ')
    elif args.netrc.exists():
        args.user,_,args.password = netrc.netrc(args.netrc).authenticators(HOST)
    elif args.user and not args.password:
        # enter password securely from command-line
        args.password = getpass.getpass(f'Password for {args.user}@{HOST}: ')

    # check internet connection before attempting to run program
    if pyTMD.utilities.check_ftp_connection(HOST,args.user,args.password):
        for m in args.tide:
            fetch_aviso_fes(m,
                directory=args.directory,
                user=args.user,
                password=args.password,
                load=args.load,
                currents=args.currents,
                extrapolated=args.extrapolated,
                compressed=args.gzip,
                timeout=args.timeout,
                mode=args.mode
            )

# run main program
if __name__ == '__main__':
    main()
