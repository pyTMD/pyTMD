import pytest
import pathlib
from pyTMD.utilities import get_cache_path

# default working data directory for tide models
_default_directory = get_cache_path()

def pytest_addoption(parser):
    parser.addoption("--directory", action="store", help="Directory for test data", default=_default_directory, type=pathlib.Path)

@pytest.fixture(scope="session")
def directory(request):
    """ Returns Data Directory """
    return request.config.getoption("--directory")
