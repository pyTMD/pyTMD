#!/usr/bin/env python
u"""
utilities.py
Written by Tyler Sutterley (11/2025)
Download and management utilities for syncing time and auxiliary files

PYTHON DEPENDENCIES:
    lxml: processing XML and HTML in Python
        https://pypi.python.org/pypi/lxml
    platformdirs: Python module for determining platform-specific directories
        https://pypi.org/project/platformdirs/

UPDATE HISTORY:
    Updated 11/2025: added string check to determine if is a valid URL
        added function to check if a dependency is available
        added detection functions for compression and model format
        create a Path class for handling local and URL paths
    Updated 10/2025: allow additional keyword arguments to http functions
        added get_cache_path function for application cache directories
    Updated 07/2025: removed (now) unused functions that were moved to timescale
        add tilde compression for file paths (squash home directory)
    Updated 01/2025: added function to list a directory from the UHSLC
    Updated 08/2024: generalize hash function to use any available algorithm
    Updated 07/2024: added function to parse JSON responses from https
    Updated 06/2024: make default case for an import exception be a class
    Updated 04/2024: add wrapper to importlib for optional dependencies
    Updated 11/2023: updated ssl context to fix deprecation error
    Updated 06/2023: add functions to retrieve and revoke Earthdata tokens
    Updated 05/2023: add reify decorator for evaluation of properties
        make urs a keyword argument in CCDIS list and download functions
        add case for JPL kernel file download where local path is defined
    Updated 04/2023: using pathlib to define and expand paths
        added function to download ephemeride files from JPL SSD server
    Updated 03/2023: add basic variable typing to function inputs
    Updated 01/2023: updated SSL context to fix some deprecation warnings
    Updated 11/2022: added list program for IERS Bulletin-A https server
        use f-strings for formatting verbose or ascii output
    Updated 04/2022: updated docstrings to numpy documentation format
    Updated 10/2021: build python logging instance for handling verbose output
    Updated 09/2021: added generic list from Apache http server
    Updated 08/2021: added function to open a file path
    Updated 07/2021: add parser for converting file files to arguments
    Updated 03/2021: added sha1 option for retrieving file hashes
    Updated 01/2021: added username and password to ftp functions
        added ftp connection check
    Updated 12/2020: added file object keyword for downloads if verbose
        add url split function for creating url location lists
    Updated 11/2020: normalize source and destination paths in copy
        make context an optional keyword argument in from_http
    Updated 09/2020: copy from http and https to bytesIO object in chunks
        use netrc credentials if not entered from CDDIS functions
        generalize build opener function for different Earthdata instances
    Updated 08/2020: add GSFC CDDIS opener, login and download functions
    Written 08/2020
"""
from __future__ import print_function, division, annotations

import sys
import os
import re
import io
import ssl
import json
import ftplib
import shutil
import socket
import inspect
import hashlib
import logging
import pathlib
import warnings
import importlib
import posixpath
import subprocess
import lxml.etree
import platformdirs
import calendar, time
if sys.version_info[0] == 2:
    from urllib import quote_plus
    from cookielib import CookieJar
    from urlparse import urlparse
    import urllib2
else:
    from urllib.parse import quote_plus, urlparse
    from http.cookiejar import CookieJar
    import urllib.request as urllib2

__all__ = [
    "get_data_path",
    "get_cache_path",
    "import_dependency",
    "dependency_available",
    "is_valid_url",
    "Path",
    "reify",
    "detect_format",
    "detect_compression",
    "get_hash",
    "get_git_revision_hash",
    "get_git_status",
    "url_split",
    "convert_arg_line_to_args",
    "build_logger",
    "get_unix_time",
    "even",
    "copy",
    "check_ftp_connection",
    "ftp_list",
    "from_ftp",
    "_create_default_ssl_context",
    "_create_ssl_context_no_verify",
    "_set_ssl_context_options",
    "check_connection",
    "http_list",
    "from_http",
    "from_json",
    "iers_list",
    "from_jpl_ssd",
    "uhslc_list"
]

# PURPOSE: get absolute path within a package from a relative path
def get_data_path(relpath: list | str | pathlib.Path):
    """
    Get the absolute path within a package from a relative path

    Parameters
    ----------
    relpath: list, str or pathlib.Path
        relative path
    """
    # current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = Path(filename).absolute().parent
    if isinstance(relpath, list):
        # use *splat operator to extract from list
        return filepath.joinpath(*relpath)
    elif isinstance(relpath, (str, pathlib.Path)):
        return filepath.joinpath(relpath)

# PURPOSE: get the path to the user cache directory
def get_cache_path(
        relpath: list | str | pathlib.Path | None = None,
        appname='pytmd'
    ):
    """
    Get the path to the user cache directory for an application

    Parameters
    ----------
    relpath: list, str, pathlib.Path or None
        relative path
    appname: str, default 'pytmd'
        application name
    """
    # get platform-specific cache directory
    filepath = platformdirs.user_cache_path(appname=appname,
        ensure_exists=True)
    if isinstance(relpath, list):
        # use *splat operator to extract from list
        filepath = filepath.joinpath(*relpath)
    elif isinstance(relpath, (str, pathlib.Path)):
        filepath = filepath.joinpath(relpath)
    return Path(filepath)

def import_dependency(
        name: str,
        extra: str = "",
        raise_exception: bool = False
    ):
    """
    Import an optional dependency

    Adapted from ``pandas.compat._optional::import_optional_dependency``

    Parameters
    ----------
    name: str
        Module name
    extra: str, default ""
        Additional text to include in the ``ImportError`` message
    raise_exception: bool, default False
        Raise an ``ImportError`` if the module is not found

    Returns
    -------
    module: obj
        Imported module
    """
    # check if the module name is a string
    msg = f"Invalid module name: '{name}'; must be a string"
    assert isinstance(name, str), msg
    # default error if module cannot be imported
    err = f"Missing optional dependency '{name}'. {extra}"
    module = type('module', (), {})
    # try to import the module
    try:
        module = importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        if raise_exception:
            raise ImportError(err) from exc
        else:
            logging.debug(err)
    # return the module
    return module

def dependency_available(name: str, minversion: str | None = None):
    """
    Checks whether a module is installed without importing it

    Adapted from ``xarray.namedarray.utils.module_available``

    Parameters
    ----------
    name: str
        Module name
    minversion : str, optional
        Minimum version of the module

    Returns
    -------
    available : bool
        Whether the module is installed
    """
    # check if module is available
    if importlib.util.find_spec(name) is None:
        return False
    # check if the version is greater than the minimum required
    if minversion is not None:
        version = importlib.metadata.version(name)
        return (version >= minversion)
    # return if both checks are passed
    return True

def is_valid_url(url: str) -> bool:
    """
    Checks if a string is a valid URL

    Parameters
    ----------
    url: str
        URL to check
    """
    try:
        result = urlparse(str(url))
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False

class Path(pathlib.Path):
    """``Pathlib.Path`` subclass with additional methods and properties
    """

    def __init__(self, filename: str | pathlib.Path, *args, **kwargs):
        if is_valid_url(filename):
            self.filename = str(filename)
            self._raw_paths = list(url_split(self.filename))
        else:
            super().__init__(filename)
            self.filename = pathlib.Path(filename, *args, **kwargs)

    def joinpath(self, *args):
        """Join path components to the existing path
        """
        if self.is_url():
            return Path('/'.join([self.filename, *args]))
        else:
            return Path(self.filename.joinpath(*args))

    # PURPOSE: platform independent file opener
    def openfile(self):
        """Platform independent file opener
        """
        if self.is_url():
            raise RuntimeError('Cannot open URL paths')
        if self.is_local() and (sys.platform == "win32"):
            os.startfile(self.filename, "explore")
        elif  self.is_local() and (sys.platform == "darwin"):
            subprocess.call(["open", self.filename])
        else:
            subprocess.call(["xdg-open", self.filename])

    def compressuser(self):
        """Tilde-compress a file to be relative to the home directory
        """
        if self.is_url():
            raise RuntimeError('Cannot compress URL paths')
        # attempt to compress filename relative to home directory
        filename = pathlib.Path(self.filename).expanduser().absolute()
        try:
            relative_to = filename.relative_to(pathlib.Path().home())
        except (ValueError, AttributeError) as exc:
            return self.filename
        else:
            return Path('~').joinpath(relative_to)
        
    def exists(self):
        """Check if the resolved path exists
        """
        if self.is_url():
            return is_valid_url(self.filename)
        else:
            return self.filename.expanduser().absolute().exists()
        
    def resolve(self):
        """Resolve the path to an absolute path
        """
        if self.is_url():
            return Path('/'.join(url_split(self.filename)))
        else:
            return Path(self.filename.expanduser().absolute())

    def is_file(self):
        """Boolean flag if path is a local file
        """
        return not is_valid_url(self.filename) and \
            self.filename.expanduser().absolute().is_file()

    def is_dir(self):
        """Boolean flag if path is a local directory
        """
        return not is_valid_url(self.filename) and \
            self.filename.expanduser().absolute().is_dir()

    def is_local(self):
        """Boolean flag if path is a local file or directory
        """
        return not is_valid_url(self.filename)

    def is_url(self):
        """Boolean flag if path is a URL
        """
        return is_valid_url(self.filename)

    @property
    def md5_hash(self):
        """MD5 hash value of the file
        """
        return get_hash(self.filename, algorithm='md5')

    def __repr__(self):
        """Representation of the ``Path`` object
        """
        return str(self.filename)

    def __str__(self):
        """String representation of the ``Path`` object
        """
        return str(self.filename)

class reify(object):
    """Class decorator that puts the result of the method it
    decorates into the instance"""
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.__name__ = wrapped.__name__
        self.__doc__ = wrapped.__doc__

    def __get__(self, inst, objtype=None):
        if inst is None:
            return self
        val = self.wrapped(inst)
        setattr(inst, self.wrapped.__name__, val)
        return val

def detect_format(filename: str | pathlib.Path) -> str:
    """
    Detect tide file format based on file extension

    Parameters
    ----------
    filename: str or pathlib.Path
        model file

    Returns
    -------
    format: str
        Model format

            - ``'ascii'``: ascii format
            - ``'netcdf'``: netCDF4 format
    """
    filename = Path(filename    ).resolve()
    if re.search(r'(\.asc|\.d)(\.gz)?$', filename.name, re.IGNORECASE):
        # FES or GOT ascii formats
        return 'ascii'
    elif re.search(r'\.nc(\.gz)?$', filename.name, re.IGNORECASE):
        # FES or GOT netCDF4 formats
        return 'netcdf'
    else:
        raise ValueError(f'Unrecognized FES file format: {filename}')

def detect_compression(filename: str | pathlib.Path) -> bool:
    """
    Detect if file is compressed based on file extension

    Parameters
    ----------
    filename: str or pathlib.Path
        model file

    Returns
    -------
    compressed: bool
        Input file is gzip compressed
    """
    filename = Path(filename).resolve()
    return bool(re.search(r'\.gz$', filename.name, re.IGNORECASE))

# PURPOSE: get the hash value of a file
def get_hash(
        local: str | io.IOBase | pathlib.Path,
        algorithm: str = 'md5'
    ):
    """
    Get the hash value from a local file or ``BytesIO`` object

    Parameters
    ----------
    local: obj, str or pathlib.Path
        BytesIO object or path to file
    algorithm: str, default 'md5'
        hashing algorithm for checksum validation
    """
    # check if open file object or if local file exists
    if isinstance(local, io.IOBase):
        # generate checksum hash for a given type
        if algorithm in hashlib.algorithms_available:
            return hashlib.new(algorithm, local.getvalue()).hexdigest()
        else:
            raise ValueError(f'Invalid hashing algorithm: {algorithm}')
    elif isinstance(local, (str, pathlib.Path)):
        # generate checksum hash for local file
        local = pathlib.Path(local).expanduser()
        # if file currently doesn't exist, return empty string
        if not local.exists():
            return ''
        # open the local_file in binary read mode
        with local.open(mode='rb') as local_buffer:
            # generate checksum hash for a given type
            if algorithm in hashlib.algorithms_available:
                return hashlib.new(algorithm, local_buffer.read()).hexdigest()
            else:
                raise ValueError(f'Invalid hashing algorithm: {algorithm}')
    else:
        return ''

# PURPOSE: get the git hash value
def get_git_revision_hash(
        refname: str = 'HEAD',
        short: bool = False
    ):
    """
    Get the ``git`` hash value for a particular reference

    Parameters
    ----------
    refname: str, default HEAD
        Symbolic reference name
    short: bool, default False
        Return the shorted hash value
    """
    # get path to .git directory from current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    basepath = pathlib.Path(filename).absolute().parent.parent
    gitpath = basepath.joinpath('.git')
    # build command
    cmd = ['git', f'--git-dir={gitpath}', 'rev-parse']
    cmd.append('--short') if short else None
    cmd.append(refname)
    # get output
    with warnings.catch_warnings():
        return str(subprocess.check_output(cmd), encoding='utf8').strip()

# PURPOSE: get the current git status
def get_git_status():
    """Get the status of a ``git`` repository as a boolean value
    """
    # get path to .git directory from current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    basepath = pathlib.Path(filename).absolute().parent.parent
    gitpath = basepath.joinpath('.git')
    # build command
    cmd = ['git', f'--git-dir={gitpath}', 'status', '--porcelain']
    with warnings.catch_warnings():
        return bool(subprocess.check_output(cmd))

# PURPOSE: recursively split a url path
def url_split(s: str):
    """
    Recursively split a url path into a list

    Parameters
    ----------
    s: str
        url string
    """
    head, tail = posixpath.split(s)
    if head in ('http:','https:','ftp:','s3:'):
        return s,
    elif head in ('', posixpath.sep):
        return tail,
    return url_split(head) + (tail,)

# PURPOSE: convert file lines to arguments
def convert_arg_line_to_args(arg_line):
    """
    Convert file lines to arguments

    Parameters
    ----------
    arg_line: str
        line string containing a single argument and/or comments
    """
    # remove commented lines and after argument comments
    for arg in re.sub(r'\#(.*?)$',r'',arg_line).split():
        if not arg.strip():
            continue
        yield arg

# PURPOSE: build a logging instance with a specified name
def build_logger(name: str, **kwargs):
    """
    Builds a logging instance with the specified name

    Parameters
    ----------
    name: str
        name of the logger
    format: str
        event description message format
    level: int, default logging.CRITICAL
        lowest-severity log message logger will handle
    propagate: bool, default False
        events logged will be passed to higher level handlers
    stream: obj or NoneType, default None
        specified stream to initialize StreamHandler
    """
    # set default arguments
    kwargs.setdefault('format', '%(levelname)s:%(name)s:%(message)s')
    kwargs.setdefault('level', logging.CRITICAL)
    kwargs.setdefault('propagate',False)
    kwargs.setdefault('stream',None)
    # build logger
    logger = logging.getLogger(name)
    logger.setLevel(kwargs['level'])
    logger.propagate = kwargs['propagate']
    # create and add handlers to logger
    if not logger.handlers:
        # create handler for logger
        handler = logging.StreamHandler(stream=kwargs['stream'])
        formatter = logging.Formatter(kwargs['format'])
        handler.setFormatter(formatter)
        # add handler to logger
        logger.addHandler(handler)
    return logger

# PURPOSE: returns the Unix timestamp value for a formatted date string
def get_unix_time(
        time_string: str,
        format: str = '%Y-%m-%d %H:%M:%S'
    ):
    """
    Get the Unix timestamp value for a formatted date string

    Parameters
    ----------
    time_string: str
        formatted time string to parse
    format: str, default '%Y-%m-%d %H:%M:%S'
        format for input time string
    """
    try:
        parsed_time = time.strptime(time_string.rstrip(), format)
    except (TypeError, ValueError):
        pass
    else:
        return calendar.timegm(parsed_time)

# PURPOSE: rounds a number to an even number less than or equal to original
def even(value: float):
    """
    Rounds a number to an even number less than or equal to original

    Parameters
    ----------
    value: float
        number to be rounded
    """
    return 2*int(value//2)

# PURPOSE: make a copy of a file with all system information
def copy(
        source: str | pathlib.Path,
        destination: str | pathlib.Path,
        move: bool = False,
        **kwargs
    ):
    """
    Copy or move a file with all system information

    Parameters
    ----------
    source: str
        source file
    destination: str
        copied destination file
    move: bool, default False
        remove the source file
    """
    source = pathlib.Path(source).expanduser().absolute()
    destination = pathlib.Path(destination).expanduser().absolute()
    # log source and destination
    logging.info(f'{str(source)} -->\n\t{str(destination)}')
    shutil.copyfile(source, destination)
    shutil.copystat(source, destination)
    # remove the original file if moving
    if move:
        source.unlink()

# PURPOSE: check ftp connection
def check_ftp_connection(
        HOST: str,
        username: str | None = None,
        password: str | None = None
    ):
    """
    Check internet connection with ftp host

    Parameters
    ----------
    HOST: str
        remote ftp host
    username: str or NoneType
        ftp username
    password: str or NoneType
        ftp password
    """
    # attempt to connect to ftp host
    try:
        f = ftplib.FTP(HOST)
        f.login(username, password)
        f.voidcmd("NOOP")
    except IOError:
        raise RuntimeError('Check internet connection')
    except ftplib.error_perm:
        raise RuntimeError('Check login credentials')
    else:
        return True

# PURPOSE: list a directory on a ftp host
def ftp_list(
        HOST: str | list,
        username: str | None = None,
        password: str | None = None,
        timeout: int | None = None,
        basename: bool = False,
        pattern: str | None = None,
        sort: bool = False
    ):
    """
    List a directory on a ftp host

    Parameters
    ----------
    HOST: str or list
        remote ftp host path split as list
    username: str or NoneType
        ftp username
    password: str or NoneType
        ftp password
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    basename: bool, default False
        return the file or directory basename instead of the full path
    pattern: str or NoneType, default None
        regular expression pattern for reducing list
    sort: bool, default False
        sort output list

    Returns
    -------
    output: list
        items in a directory
    mtimes: list
        last modification times for items in the directory
    """
    # verify inputs for remote ftp host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try to connect to ftp host
    try:
        ftp = ftplib.FTP(HOST[0],timeout=timeout)
    except (socket.gaierror,IOError):
        raise RuntimeError(f'Unable to connect to {HOST[0]}')
    else:
        ftp.login(username,password)
        # list remote path
        output = ftp.nlst(posixpath.join(*HOST[1:]))
        # get last modified date of ftp files and convert into unix time
        mtimes = [None]*len(output)
        # iterate over each file in the list and get the modification time
        for i,f in enumerate(output):
            try:
                # try sending modification time command
                mdtm = ftp.sendcmd(f'MDTM {f}')
            except ftplib.error_perm:
                # directories will return with an error
                pass
            else:
                # convert the modification time into unix time
                mtimes[i] = get_unix_time(mdtm[4:], format="%Y%m%d%H%M%S")
        # reduce to basenames
        if basename:
            output = [posixpath.basename(i) for i in output]
        # reduce using regular expression pattern
        if pattern:
            i = [i for i,f in enumerate(output) if re.search(pattern,f)]
            # reduce list of listed items and last modified times
            output = [output[indice] for indice in i]
            mtimes = [mtimes[indice] for indice in i]
        # sort the list
        if sort:
            i = [i for i,j in sorted(enumerate(output), key=lambda i: i[1])]
            # sort list of listed items and last modified times
            output = [output[indice] for indice in i]
            mtimes = [mtimes[indice] for indice in i]
        # close the ftp connection
        ftp.close()
        # return the list of items and last modified times
        return (output, mtimes)

# PURPOSE: download a file from a ftp host
def from_ftp(
        HOST: str | list,
        username: str | None = None,
        password: str | None = None,
        timeout: int | None = None,
        local: str | pathlib.Path | None = None,
        hash: str = '',
        chunk: int = 8192,
        verbose: bool = False,
        fid=sys.stdout,
        mode: oct = 0o775
    ):
    """
    Download a file from a ftp host

    Parameters
    ----------
    HOST: str or list
        remote ftp host path
    username: str or NoneType
        ftp username
    password: str or NoneType
        ftp password
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    local: str, pathlib.Path or NoneType, default None
        path to local file
    hash: str, default ''
        MD5 hash of local file
    chunk: int, default 8192
        chunk size for transfer encoding
    verbose: bool, default False
        print file transfer information
    fid: obj, default sys.stdout
        open file object to print if verbose
    mode: oct, default 0o775
        permissions mode of output local file

    Returns
    -------
    remote_buffer: obj
        BytesIO representation of file
    """
    # create logger
    loglevel = logging.INFO if verbose else logging.CRITICAL
    logging.basicConfig(stream=fid, level=loglevel)
    # verify inputs for remote ftp host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try downloading from ftp
    try:
        # try to connect to ftp host
        ftp = ftplib.FTP(HOST[0], timeout=timeout)
    except (socket.gaierror,IOError):
        raise RuntimeError(f'Unable to connect to {HOST[0]}')
    else:
        ftp.login(username,password)
        # remote path
        ftp_remote_path = posixpath.join(*HOST[1:])
        # copy remote file contents to bytesIO object
        remote_buffer = io.BytesIO()
        ftp.retrbinary(f'RETR {ftp_remote_path}',
            remote_buffer.write, blocksize=chunk)
        remote_buffer.seek(0)
        # save file basename with bytesIO object
        remote_buffer.filename = HOST[-1]
        # generate checksum hash for remote file
        remote_hash = hashlib.md5(remote_buffer.getvalue()).hexdigest()
        # get last modified date of remote file and convert into unix time
        mdtm = ftp.sendcmd(f'MDTM {ftp_remote_path}')
        remote_mtime = get_unix_time(mdtm[4:], format="%Y%m%d%H%M%S")
        # compare checksums
        if local and (hash != remote_hash):
            # convert to absolute path
            local = pathlib.Path(local).expanduser().absolute()
            # create directory if non-existent
            local.parent.mkdir(mode=mode, parents=True, exist_ok=True)
            # print file information
            args = (posixpath.join(*HOST), str(local))
            logging.info('{0} -->\n\t{1}'.format(*args))
            # store bytes to file using chunked transfer encoding
            remote_buffer.seek(0)
            with local.open(mode='wb') as f:
                shutil.copyfileobj(remote_buffer, f, chunk)
            # change the permissions mode
            local.chmod(mode)
            # keep remote modification time of file and local access time
            os.utime(local, (local.stat().st_atime, remote_mtime))
        # close the ftp connection
        ftp.close()
        # return the bytesIO object
        remote_buffer.seek(0)
        return remote_buffer

def _create_default_ssl_context() -> ssl.SSLContext:
    """Creates the default SSL context
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    _set_ssl_context_options(context)
    context.options |= ssl.OP_NO_COMPRESSION
    return context

def _create_ssl_context_no_verify() -> ssl.SSLContext:
    """Creates an SSL context for unverified connections
    """
    context = _create_default_ssl_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

def _set_ssl_context_options(context: ssl.SSLContext) -> None:
    """Sets the default options for the SSL context
    """
    if sys.version_info >= (3, 10) or ssl.OPENSSL_VERSION_INFO >= (1, 1, 0, 7):
        context.minimum_version = ssl.TLSVersion.TLSv1_2
    else:
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1

# default ssl context
_default_ssl_context = _create_ssl_context_no_verify()

# PURPOSE: check internet connection
def check_connection(
        HOST: str,
        context: ssl.SSLContext = _default_ssl_context,
    ):
    """
    Check internet connection with http host

    Parameters
    ----------
    HOST: str
        remote http host
    context: obj, default pyTMD.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    """
    # attempt to connect to http host
    try:
        urllib2.urlopen(HOST, timeout=20, context=context)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise RuntimeError(exc.reason) from exc
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        raise RuntimeError('Check internet connection') from exc
    else:
        return True

# PURPOSE: list a directory on an Apache http Server
def http_list(
        HOST: str | list,
        timeout: int | None = None,
        context: ssl.SSLContext = _default_ssl_context,
        parser = lxml.etree.HTMLParser(),
        format: str = '%Y-%m-%d %H:%M',
        pattern: str = '',
        sort: bool = False,
        **kwargs
    ):
    """
    List a directory on an Apache http Server

    Parameters
    ----------
    HOST: str or list
        remote http host path
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default pyTMD.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    parser: obj, default lxml.etree.HTMLParser()
        HTML parser for ``lxml``
    format: str, default '%Y-%m-%d %H:%M'
        format for input time string
    pattern: str, default ''
        regular expression pattern for reducing list
    sort: bool, default False
        sort output list

    Returns
    -------
    colnames: list
        column names in a directory
    collastmod: list
        last modification times for items in the directory
    """
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try listing from http
    try:
        # Create and submit request.
        request = urllib2.Request(posixpath.join(*HOST), **kwargs)
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise RuntimeError(exc.reason) from exc
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        msg = 'List error from {0}'.format(posixpath.join(*HOST))
        raise Exception(msg) from exc
    else:
        # read and parse request for files (column names and modified times)
        tree = lxml.etree.parse(response, parser)
        colnames = tree.xpath('//tr/td[not(@*)]//a/@href')
        # get the Unix timestamp value for a modification time
        collastmod = [get_unix_time(i,format=format)
            for i in tree.xpath('//tr/td[@align="right"][1]/text()')]
        # reduce using regular expression pattern
        if pattern:
            i = [i for i,f in enumerate(colnames) if re.search(pattern, f)]
            # reduce list of column names and last modified times
            colnames = [colnames[indice] for indice in i]
            collastmod = [collastmod[indice] for indice in i]
        # sort the list
        if sort:
            i = [i for i,j in sorted(enumerate(colnames), key=lambda i: i[1])]
            # sort list of column names and last modified times
            colnames = [colnames[indice] for indice in i]
            collastmod = [collastmod[indice] for indice in i]
        # return the list of column names and last modified times
        return (colnames, collastmod)

# PURPOSE: download a file from a http host
def from_http(
        HOST: str | list,
        timeout: int | None = None,
        context: ssl.SSLContext = _default_ssl_context,
        local: str | pathlib.Path | None = None,
        hash: str = '',
        chunk: int = 16384,
        verbose: bool = False,
        fid = sys.stdout,
        mode: oct = 0o775,
        **kwargs
    ):
    """
    Download a file from a http host

    Parameters
    ----------
    HOST: str or list
        remote http host path split as list
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default pyTMD.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    local: str, pathlib.Path or NoneType, default None
        path to local file
    hash: str, default ''
        MD5 hash of local file
    chunk: int, default 16384
        chunk size for transfer encoding
    verbose: bool, default False
        print file transfer information
    fid: obj, default sys.stdout
        open file object to print if verbose
    mode: oct, default 0o775
        permissions mode of output local file

    Returns
    -------
    remote_buffer: obj
        BytesIO representation of file
    """
    # create logger
    loglevel = logging.INFO if verbose else logging.CRITICAL
    logging.basicConfig(stream=fid, level=loglevel)
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try downloading from http
    try:
        # Create and submit request.
        request = urllib2.Request(posixpath.join(*HOST), **kwargs)
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except:
        raise Exception('Download error from {0}'.format(posixpath.join(*HOST)))
    else:
        # copy remote file contents to bytesIO object
        remote_buffer = io.BytesIO()
        shutil.copyfileobj(response, remote_buffer, chunk)
        remote_buffer.seek(0)
        # save file basename with bytesIO object
        remote_buffer.filename = HOST[-1]
        # generate checksum hash for remote file
        remote_hash = hashlib.md5(remote_buffer.getvalue()).hexdigest()
        # compare checksums
        if local and (hash != remote_hash):
            # convert to absolute path
            local = pathlib.Path(local).expanduser().absolute()
            # create directory if non-existent
            local.parent.mkdir(mode=mode, parents=True, exist_ok=True)
            # print file information
            args = (posixpath.join(*HOST), str(local))
            logging.info('{0} -->\n\t{1}'.format(*args))
            # store bytes to file using chunked transfer encoding
            remote_buffer.seek(0)
            with local.open(mode='wb') as f:
                shutil.copyfileobj(remote_buffer, f, chunk)
            # change the permissions mode
            local.chmod(mode)
        # return the bytesIO object
        remote_buffer.seek(0)
        return remote_buffer

# PURPOSE: load a JSON response from a http host
def from_json(
        HOST: str | list,
        timeout: int | None = None,
        context: ssl.SSLContext = _default_ssl_context
    ) -> dict:
    """
    Load a JSON response from a http host

    Parameters
    ----------
    HOST: str or list
        remote http host path split as list
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default pyTMD.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    """
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try loading JSON from http
    try:
        # Create and submit request for JSON response
        request = urllib2.Request(posixpath.join(*HOST))
        request.add_header('Accept', 'application/json')
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise RuntimeError(exc.reason) from exc
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        msg = 'Load error from {0}'.format(posixpath.join(*HOST))
        raise Exception(msg) from exc
    else:
        # load JSON response
        return json.loads(response.read())

# PURPOSE: list a directory on IERS https Server
def iers_list(
        HOST: str | list,
        timeout: int | None = None,
        context: ssl.SSLContext = _default_ssl_context,
        parser = lxml.etree.HTMLParser()
    ):
    """
    List a directory on IERS Bulletin-A https server

    Parameters
    ----------
    HOST: str or list
        remote http host path
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default pyTMD.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    parser: obj, default lxml.etree.HTMLParser()
        HTML parser for ``lxml``

    Returns
    -------
    colnames: list
        column names in a directory
    collastmod: list
        last modification times for items in the directory
    """
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try listing from http
    try:
        # Create and submit request.
        request = urllib2.Request(posixpath.join(*HOST))
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise RuntimeError(exc.reason) from exc
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        msg = 'List error from {0}'.format(posixpath.join(*HOST))
        raise Exception(msg) from exc
    else:
        # read and parse request for files (column names and modified times)
        tree = lxml.etree.parse(response, parser)
        colnames = tree.xpath('//tr/td[@class="$tdclass"][4]//a/@href')
        # get the Unix timestamp value for a modification time
        collastmod = [get_unix_time(i,format='%Y-%m-%d')
            for i in tree.xpath('//tr/td[@class="$tdclass"][2]/span/text()')]
        # sort list of column names and last modified times in reverse order
        # return the list of column names and last modified times
        return (colnames[::-1], collastmod[::-1])

def from_jpl_ssd(
        kernel='de440s.bsp',
        timeout: int | None = None,
        context: ssl.SSLContext = _default_ssl_context,
        local: str | pathlib.Path | None = None,
        hash: str = '',
        chunk: int = 16384,
        verbose: bool = False,
        mode: oct = 0o775
    ):
    """
    Download `planetary ephemeride kernels`__ from the JPL Solar
    System Dynamics server

    .. __: https://ssd.jpl.nasa.gov/planets/eph_export.html

    Parameters
    ----------
    kernel: str
        JPL kernel file to download
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default pyTMD.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    hash: str, default ''
        MD5 hash of local file
    chunk: int, default 16384
        chunk size for transfer encoding
    verbose: bool, default False
        print file transfer information
    mode: oct, default 0o775
        permissions mode of output local file
    """
    # determine which kernel file to download
    if (local is None):
        # local path to kernel file
        local = get_cache_path(kernel)
    elif (kernel is None) and (local is not None):
        # verify inputs for remote http host
        local = pathlib.Path(local).expanduser().absolute()
        kernel = local.name
    # remote host path to kernel file
    HOST = ['https://ssd.jpl.nasa.gov','ftp','eph','planets','bsp',kernel]
    # get kernel file from remote host
    logging.info('Downloading JPL Planetary Ephemeride Kernel File')
    from_http(HOST, timeout=timeout, context=context, local=local,
        hash=hash, chunk=chunk, verbose=verbose, mode=mode)

# PURPOSE: list a directory on the University of Hawaii SLC Server
def uhslc_list(
        HOST: str | list,
        timeout: int | None = None,
        context: ssl.SSLContext = _default_ssl_context,
        parser = lxml.etree.HTMLParser(),
        pattern: str = '',
        sort: bool = False
    ):
    """
    List a directory from the University of Hawaii Sea Level Center

    Parameters
    ----------
    HOST: str or list
        remote http host path
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default pyTMD.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    parser: obj, default lxml.etree.HTMLParser()
        HTML parser for ``lxml``
    pattern: str, default ''
        regular expression pattern for reducing list
    sort: bool, default False
        sort output list

    Returns
    -------
    colnames: list
        column names in a directory
    """
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try listing from http
    try:
        # Create and submit request.
        request = urllib2.Request(posixpath.join(*HOST))
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise RuntimeError(exc.reason) from exc
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        msg = 'List error from {0}'.format(posixpath.join(*HOST))
        raise Exception(msg) from exc
    else:
        # read and parse request for files
        tree = lxml.etree.parse(response, parser)
        colnames = tree.xpath('//a/text()')
        # reduce using regular expression pattern
        if pattern:
            i = [i for i,f in enumerate(colnames) if re.search(pattern, f)]
            # reduce list of column names
            colnames = [colnames[indice] for indice in i]
        # sort the list
        if sort:
            i = [i for i,j in sorted(enumerate(colnames), key=lambda i: i[1])]
            # sort list of column names
            colnames = [colnames[indice] for indice in i]
        # return the list of column names
        return colnames
