import pytest
import inspect
import pathlib
from pyTMD.utilities import get_data_path

# default working data directory for tide models
_default_directory = get_data_path('data')

def pytest_addoption(parser):
    parser.addoption("--directory", action="store", help="Directory for test data", default=_default_directory, type=pathlib.Path)
    parser.addoption("--aws-access", action="store", help="AWS Access Key ID")
    parser.addoption("--aws-secret", action="store", help="AWS Secret Key")
    parser.addoption("--aws-region", action="store", help="AWS Region Name")

@pytest.fixture(scope="session")
def directory(request):
    """ Returns Data Directory """
    return request.config.getoption("--directory")

@pytest.fixture(scope="session")
def aws_access_key_id(request):
    """ Returns AWS Access Key ID """
    return request.config.getoption("--aws-access")

@pytest.fixture(scope="session")
def aws_secret_access_key(request):
    """ Returns AWS Secret Key """
    return request.config.getoption("--aws-secret")

@pytest.fixture(scope="session")
def aws_region_name(request):
    """ Returns AWS Region Name """
    return request.config.getoption("--aws-region")
