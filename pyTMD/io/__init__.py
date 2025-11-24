"""
Input/output functions for reading and writing tidal data
"""
import os
from .ATLAS import *
from .FES import *
from .GOT import *
from .OTIS import *
from .IERS import *
from .NOAA import *
from .dataset import *
from .model import model, load_database

# set environmental variable for anonymous s3 access
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'