#!/usr/bin/env python
u"""
test_utilities.py (04/2026)
Verify file utility functions

UPDATE HISTORY:
    Updated 04/2026: add coverage for URL class, format/compression
        detection, hashing, git functions, and other utilities
    Written 12/2020
"""
import io
import gzip
import logging
import pathlib
import tempfile
import warnings
import pytest
import pyTMD.datasets
import pyTMD.utilities


def test_hash():
    # get hash of compressed file
    ocean_pole_tide_file = pyTMD.utilities.get_cache_path(
        'opoleloadcoefcmcor.txt.gz')
    # fetch file if it doesn't exist
    if not ocean_pole_tide_file.exists():
        pyTMD.datasets.fetch_iers_opole(
            directory=ocean_pole_tide_file.parent
        )
    TEST = pyTMD.utilities.get_hash(ocean_pole_tide_file)
    assert (TEST == '9c66edc2d0fbf627e7ae1cb923a9f0e5')
    # get hash of uncompressed file
    with gzip.open(ocean_pole_tide_file) as fid:
        TEST = pyTMD.utilities.get_hash(io.BytesIO(fid.read()))
        assert (TEST == 'cea08f83d613ed8e1a81f3b3a9453721')

_default_directory = pyTMD.utilities.get_cache_path()
def test_valid_url():
    # test over some valid urls
    URLS = [
        'https://arcticdata.io/',
        'http://www.esr.org/research/polar-tide-models',
        's3://pytmd-scratch/CATS2008.zarr'
    ]
    for URL in URLS:
        url = pyTMD.utilities.Path(URL).resolve()
        assert pyTMD.utilities.is_valid_url(url)
    # test over some file paths
    PATHS = [
        pathlib.PurePosixPath('/home/user/data/CATS2008/grid_CATS2008'),
        pathlib.PureWindowsPath('C://Users/user/data/CATS2008/grid_CATS2008'),
        _default_directory.joinpath('CATS2008','grid_CATS2008')
    ]
    for PATH in PATHS:
        path = pyTMD.utilities.Path(PATH).resolve()
        assert not pyTMD.utilities.is_valid_url(path)


# ---------------------------------------------------------------------------
# get_data_path and get_cache_path
# ---------------------------------------------------------------------------
def test_get_data_path_list():
    """get_data_path should accept a list of path components"""
    p = pyTMD.utilities.get_data_path(["data", "doodson.json"])
    assert isinstance(p, pathlib.Path)
    assert p.name == "doodson.json"


def test_get_data_path_str():
    """get_data_path should accept a plain string"""
    p = pyTMD.utilities.get_data_path("data")
    assert isinstance(p, pathlib.Path)
    assert p.name == "data"


def test_get_data_path_pathlib():
    """get_data_path should accept a pathlib.Path"""
    p = pyTMD.utilities.get_data_path(pathlib.Path("data"))
    assert isinstance(p, pathlib.Path)


def test_get_cache_path_none():
    """get_cache_path with no argument returns the app cache directory"""
    p = pyTMD.utilities.get_cache_path()
    assert isinstance(p, pathlib.Path)


def test_get_cache_path_list():
    """get_cache_path should accept a list"""
    p = pyTMD.utilities.get_cache_path(["subdir", "file.txt"])
    assert isinstance(p, pathlib.Path)
    assert p.name == "file.txt"


def test_get_cache_path_str():
    """get_cache_path should accept a plain string"""
    p = pyTMD.utilities.get_cache_path("test_subdir")
    assert isinstance(p, pathlib.Path)
    assert p.name == "test_subdir"


# ---------------------------------------------------------------------------
# import_dependency and dependency_available
# ---------------------------------------------------------------------------
def test_import_dependency_present():
    """import_dependency should return module when it exists"""
    mod = pyTMD.utilities.import_dependency("numpy")
    import numpy as np_check
    assert mod is np_check


def test_import_dependency_missing_no_raise():
    """import_dependency should not raise when raise_exception=False"""
    mod = pyTMD.utilities.import_dependency(
        "nonexistent_module_xyz", raise_exception=False
    )
    # returns a dummy class when missing and not raising
    assert mod is not None


def test_import_dependency_missing_raise():
    """import_dependency should raise when raise_exception=True"""
    with pytest.raises(ImportError):
        pyTMD.utilities.import_dependency(
            "nonexistent_module_xyz", raise_exception=True
        )


def test_dependency_available_present():
    """dependency_available should return True for installed packages"""
    assert pyTMD.utilities.dependency_available("numpy")


def test_dependency_available_missing():
    """dependency_available should return False for missing packages"""
    assert not pyTMD.utilities.dependency_available("nonexistent_module_xyz")


def test_dependency_available_with_minversion():
    """dependency_available should check minimum version"""
    # numpy is definitely >= 1.0.0
    assert pyTMD.utilities.dependency_available("numpy", minversion="1.0.0")


# ---------------------------------------------------------------------------
# URL class
# ---------------------------------------------------------------------------
def test_url_from_string():
    """URL should be created from a URL string"""
    url = pyTMD.utilities.URL("https://example.com/data/file.txt")
    assert str(url) == "https://example.com/data/file.txt"
    assert repr(url) == "https://example.com/data/file.txt"


def test_url_from_parts_list():
    """URL.from_parts should accept a list"""
    url = pyTMD.utilities.URL.from_parts(
        ["https://example.com", "data", "file.txt"]
    )
    assert isinstance(url, pyTMD.utilities.URL)


def test_url_from_parts_str():
    """URL.from_parts should accept a string"""
    url = pyTMD.utilities.URL.from_parts("https://example.com/data")
    assert isinstance(url, pyTMD.utilities.URL)


def test_url_joinpath():
    """URL.joinpath should append path segments"""
    url = pyTMD.utilities.URL("https://example.com/data")
    url2 = url.joinpath("subdir", "file.txt")
    assert "subdir" in str(url2)
    assert "file.txt" in str(url2)


def test_url_truediv():
    """URL / operator should join paths"""
    url = pyTMD.utilities.URL("https://example.com/data")
    url2 = url / "file.txt"
    assert "file.txt" in str(url2)


def test_url_div():
    """URL __div__ should be equivalent to __truediv__"""
    url = pyTMD.utilities.URL("https://example.com/data")
    url2 = url.__div__("file.txt")
    assert "file.txt" in str(url2)


def test_url_resolve():
    """URL.resolve should return a URL"""
    url = pyTMD.utilities.URL("https://example.com/data/file.txt")
    resolved = url.resolve()
    assert isinstance(resolved, pyTMD.utilities.URL)


def test_url_is_file():
    """URL.is_file should return False"""
    url = pyTMD.utilities.URL("https://example.com/file.txt")
    assert url.is_file() is False


def test_url_is_dir():
    """URL.is_dir should return False"""
    url = pyTMD.utilities.URL("https://example.com/dir/")
    assert url.is_dir() is False


def test_url_name():
    """URL.name property should return the file name"""
    url = pyTMD.utilities.URL("https://example.com/data/file.txt")
    assert url.name == "file.txt"


def test_url_stem():
    """URL.stem property should return the stem"""
    url = pyTMD.utilities.URL("https://example.com/data/file.txt")
    assert url.stem == "file"


def test_url_scheme():
    """URL.scheme property should return the scheme"""
    url = pyTMD.utilities.URL("https://example.com/data/file.txt")
    assert url.scheme == "https://"


def test_url_parent():
    """URL.parent property should return the parent URL"""
    url = pyTMD.utilities.URL("https://example.com/data/file.txt")
    parent = url.parent
    assert isinstance(parent, pyTMD.utilities.URL)
    assert "data" in str(parent)


def test_url_parents():
    """URL.parents should return a list of parent URLs"""
    url = pyTMD.utilities.URL("https://example.com/data/subdir/file.txt")
    parents = url.parents
    assert isinstance(parents, list)
    assert len(parents) > 0


def test_url_parts():
    """URL.parts should include the scheme, netloc, and path segments"""
    url = pyTMD.utilities.URL("https://example.com/data/file.txt")
    parts = url.parts
    assert "https://" in parts
    assert "example.com" in parts


# ---------------------------------------------------------------------------
# detect_format and detect_compression
# ---------------------------------------------------------------------------
def test_detect_format_ascii_asc():
    """detect_format should return 'ascii' for .asc files"""
    assert pyTMD.utilities.detect_format("m2.asc") == "ascii"


def test_detect_format_ascii_d():
    """detect_format should return 'ascii' for .d files"""
    assert pyTMD.utilities.detect_format("m2.d") == "ascii"


def test_detect_format_ascii_gz():
    """detect_format should return 'ascii' for .asc.gz files"""
    assert pyTMD.utilities.detect_format("m2.asc.gz") == "ascii"


def test_detect_format_netcdf():
    """detect_format should return 'netcdf' for .nc files"""
    assert pyTMD.utilities.detect_format("m2.nc") == "netcdf"


def test_detect_format_netcdf_gz():
    """detect_format should return 'netcdf' for .nc.gz files"""
    assert pyTMD.utilities.detect_format("m2.nc.gz") == "netcdf"


def test_detect_format_unknown():
    """detect_format should raise ValueError for unknown extensions"""
    with pytest.raises(ValueError, match="Unrecognized file format"):
        pyTMD.utilities.detect_format("m2.xyz")


def test_detect_compression_gz():
    """detect_compression should return True for .gz files"""
    assert pyTMD.utilities.detect_compression("file.nc.gz") is True


def test_detect_compression_plain():
    """detect_compression should return False for non-compressed files"""
    assert pyTMD.utilities.detect_compression("file.nc") is False


# ---------------------------------------------------------------------------
# compressuser
# ---------------------------------------------------------------------------
def test_compressuser_home_path():
    """compressuser should compress home-relative paths with ~"""
    home = pathlib.Path.home()
    test_path = home / "test_file.txt"
    result = pyTMD.utilities.compressuser(test_path)
    # should start with ~
    assert str(result).startswith("~")


def test_compressuser_absolute_path():
    """compressuser should return the absolute path for non-home paths"""
    result = pyTMD.utilities.compressuser("/tmp/test_file.txt")
    # /tmp is not under home, so returned as-is (absolute)
    assert isinstance(result, pathlib.Path)


# ---------------------------------------------------------------------------
# get_hash (without external files)
# ---------------------------------------------------------------------------
def test_get_hash_bytesio():
    """get_hash should compute the hash of a BytesIO object"""
    data = b"hello world"
    buf = io.BytesIO(data)
    import hashlib
    expected = hashlib.md5(data).hexdigest()
    result = pyTMD.utilities.get_hash(buf)
    assert result == expected


def test_get_hash_bytesio_sha256():
    """get_hash should support sha256 algorithm"""
    data = b"test data"
    buf = io.BytesIO(data)
    import hashlib
    expected = hashlib.sha256(data).hexdigest()
    result = pyTMD.utilities.get_hash(buf, algorithm="sha256")
    assert result == expected


def test_get_hash_missing_file():
    """get_hash should return empty string for a missing file"""
    result = pyTMD.utilities.get_hash("/nonexistent/path/to/file.txt")
    assert result == ""


def test_get_hash_local_file():
    """get_hash should compute hash for an existing file"""
    import hashlib
    data = b"file content for hashing"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        expected = hashlib.md5(data).hexdigest()
        result = pyTMD.utilities.get_hash(tmp_path)
        assert result == expected
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)


def test_get_hash_unsupported_algorithm():
    """get_hash should raise ValueError for invalid algorithm"""
    buf = io.BytesIO(b"data")
    with pytest.raises(ValueError, match="Invalid hashing algorithm"):
        pyTMD.utilities.get_hash(buf, algorithm="nonexistent_algo")


def test_get_hash_invalid_type():
    """get_hash should return empty string for invalid type"""
    result = pyTMD.utilities.get_hash(12345)
    assert result == ""


# ---------------------------------------------------------------------------
# get_git_revision_hash and get_git_status
# ---------------------------------------------------------------------------
def test_git_revision_hash():
    """get_git_revision_hash should return a non-empty string"""
    result = pyTMD.utilities.get_git_revision_hash()
    assert isinstance(result, str)
    assert len(result) > 0


def test_git_revision_hash_short():
    """get_git_revision_hash with short=True should return a shorter hash"""
    full = pyTMD.utilities.get_git_revision_hash(short=False)
    short = pyTMD.utilities.get_git_revision_hash(short=True)
    assert len(short) <= len(full)


def test_git_status():
    """get_git_status should return a boolean"""
    result = pyTMD.utilities.get_git_status()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# url_split
# ---------------------------------------------------------------------------
def test_url_split_https():
    """url_split should split an HTTPS URL at /"""
    parts = pyTMD.utilities.url_split("https://example.com/data/file.txt")
    assert "https://example.com" in parts
    assert "data" in parts
    assert "file.txt" in parts


def test_url_split_simple():
    """url_split should handle a simple path"""
    parts = pyTMD.utilities.url_split("data/file.txt")
    assert "file.txt" in parts


# ---------------------------------------------------------------------------
# convert_arg_line_to_args
# ---------------------------------------------------------------------------
def test_convert_arg_line_to_args_basic():
    """convert_arg_line_to_args should yield args from a line"""
    args = list(pyTMD.utilities.convert_arg_line_to_args("--flag value"))
    assert "--flag" in args
    assert "value" in args


def test_convert_arg_line_to_args_comment():
    """convert_arg_line_to_args should strip comments"""
    args = list(pyTMD.utilities.convert_arg_line_to_args("--flag  # comment"))
    assert "--flag" in args
    assert "comment" not in str(args)


def test_convert_arg_line_to_args_empty():
    """convert_arg_line_to_args should handle empty/comment-only lines"""
    args = list(pyTMD.utilities.convert_arg_line_to_args("# comment only"))
    assert args == []


# ---------------------------------------------------------------------------
# build_logger
# ---------------------------------------------------------------------------
def test_build_logger():
    """build_logger should return a logging.Logger"""
    logger = pyTMD.utilities.build_logger("test_pytmd_logger")
    assert isinstance(logger, logging.Logger)


def test_build_logger_level():
    """build_logger should respect the level parameter"""
    logger = pyTMD.utilities.build_logger(
        "test_pytmd_level_logger", level=logging.DEBUG
    )
    assert logger.level == logging.DEBUG


# ---------------------------------------------------------------------------
# get_unix_time
# ---------------------------------------------------------------------------
def test_get_unix_time_valid():
    """get_unix_time should parse a valid time string"""
    result = pyTMD.utilities.get_unix_time("2000-01-01 00:00:00")
    assert result is not None
    assert isinstance(result, int)


def test_get_unix_time_invalid():
    """get_unix_time should return None for an invalid string"""
    result = pyTMD.utilities.get_unix_time("not-a-date")
    assert result is None


def test_get_unix_time_custom_format():
    """get_unix_time should accept a custom format"""
    result = pyTMD.utilities.get_unix_time(
        "01/01/2000", format="%m/%d/%Y"
    )
    assert result is not None


# ---------------------------------------------------------------------------
# even
# ---------------------------------------------------------------------------
def test_even_integer():
    """even should round down to nearest even integer"""
    assert pyTMD.utilities.even(4) == 4
    assert pyTMD.utilities.even(5) == 4
    assert pyTMD.utilities.even(6) == 6
    assert pyTMD.utilities.even(7) == 6


def test_even_float():
    """even should handle float input"""
    assert pyTMD.utilities.even(3.9) == 2
    assert pyTMD.utilities.even(4.0) == 4


# ---------------------------------------------------------------------------
# copy utility
# ---------------------------------------------------------------------------
def test_copy_file():
    """copy should create a duplicate of the source file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as src:
        src.write(b"test data")
        src_path = src.name
    dst_path = src_path + "_copy.txt"
    try:
        pyTMD.utilities.copy(src_path, dst_path)
        assert pathlib.Path(dst_path).exists()
        assert pathlib.Path(dst_path).read_bytes() == b"test data"
    finally:
        pathlib.Path(src_path).unlink(missing_ok=True)
        pathlib.Path(dst_path).unlink(missing_ok=True)


def test_copy_move_file():
    """copy with move=True should remove the source file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as src:
        src.write(b"move data")
        src_path = src.name
    dst_path = src_path + "_moved.txt"
    try:
        pyTMD.utilities.copy(src_path, dst_path, move=True)
        assert pathlib.Path(dst_path).exists()
        assert not pathlib.Path(src_path).exists()
    finally:
        pathlib.Path(dst_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# reify class decorator
# ---------------------------------------------------------------------------
def test_reify_decorator():
    """reify should cache computed property values"""
    call_count = []

    class MyClass:
        @pyTMD.utilities.reify
        def computed(self):
            call_count.append(1)
            return 42

    obj = MyClass()
    # first access computes and caches
    val1 = obj.computed
    # second access uses cached value
    val2 = obj.computed
    assert val1 == 42
    assert val2 == 42
    assert len(call_count) == 1


def test_reify_class_access():
    """Accessing reify on the class itself should return the descriptor"""
    class MyClass:
        @pyTMD.utilities.reify
        def prop(self):
            return 99

    descriptor = MyClass.prop
    assert isinstance(descriptor, pyTMD.utilities.reify)

