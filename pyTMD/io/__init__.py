"""
Input/output functions for reading and writing tidal data
"""
from .ATLAS import *
from .FES import *
from .GOT import *
from .OTIS import *
from .IERS import *
from .NOAA import *
from .constituents import constituents
from .model import (
    model,
    load_database,
    extract_constants,
    read_constants,
    interpolate_constants
)

# Attempt to import the dataset module for xarray support
try:
    from .dataset import dataset
except (ImportError, AttributeError):
    dataset = None
