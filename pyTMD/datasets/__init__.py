"""
Input/output functions for reading and writing tidal data
"""

from .fetch_arcticdata import fetch_arcticdata
from .fetch_aviso_fes import fetch_aviso_fes
from .fetch_box_tpxo import fetch_box_tpxo
from .fetch_gsfc_got import fetch_gsfc_got
from .fetch_iers_opole import fetch_iers_opole
from .fetch_jpl_ssd import fetch_jpl_ssd
from .fetch_test_data import fetch_test_data
from .reduce_otis import reduce_otis

# create fetch class to group fetching functions
fetch = type("fetch", (), {})
fetch.arcticdata = fetch_arcticdata
fetch.aviso_fes = fetch_aviso_fes
fetch.box_tpxo = fetch_box_tpxo
fetch.gsfc_got = fetch_gsfc_got
fetch.iers_opole = fetch_iers_opole
fetch.jpl_ssd = fetch_jpl_ssd
fetch.test_data = fetch_test_data
