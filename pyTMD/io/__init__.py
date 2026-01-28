"""
Input/output functions for reading and writing tidal data
"""

import os
from . import ATLAS
from . import FES
from . import GOT
from . import OTIS
from . import IERS
from . import NOAA
from . import dataset
from .model import model, load_database

# set environmental variable for anonymous s3 access
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
