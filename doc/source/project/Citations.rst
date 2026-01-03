====================
Citation Information
====================

References
##########

This work was initially supported by an appointment to the NASA Postdoctoral
Program (NPP) at NASA Goddard Space Flight Center (GSFC), administered by
Universities Space Research Association (USRA) under contract with NASA.
It is currently supported under the NASA Cryospheric Sciences Program
(Grant Numbers `80NSSC22K0379`_ and `80NSSC21K0911`_).

.. admonition:: Please consider citing our article in the Journal of Open Source Software (JOSS):

    T. C. Sutterley, S. L. Howard, L. Padman, and M. Siegfried,
    "pyTMD: Python-based tidal prediction software," 
    *Journal of Open Source Software*, 10(116), 8566 (2025).
    `doi: 10.21105/joss.08566 <https://doi.org/10.21105/joss.08566>`_

.. _80NSSC22K0379: https://app.dimensions.ai/details/grant/grant.13212262
.. _80NSSC21K0911: https://app.dimensions.ai/details/grant/grant.13210572

Dependencies
############

This software is also dependent on other commonly used Python packages:

- `h5netcdf: Pythonic interface to netCDF4 via h5py <https://h5netcdf.org/>`_
- `lxml: processing XML and HTML in Python <https://pypi.python.org/pypi/lxml>`_
- `numpy: Scientific Computing Tools For Python <https://www.numpy.org>`_
- `pint: Python package to define, operate and manipulate physical quantities <https://pypi.org/project/Pint/>`_
- `platformdirs: Python module for determining platform-specific directories <https://pypi.org/project/platformdirs/>`_
- `pyproj: Python interface to PROJ library <https://pypi.org/project/pyproj/>`_
- `scipy: Scientific Tools for Python <https://www.scipy.org/>`_
- `timescale: Python tools for time and astronomical calculations <https://pypi.org/project/timescale/>`_
- `xarray: N-D labeled arrays and datasets in Python <https://docs.xarray.dev/en/stable/>`_

Optional Dependencies
---------------------

- `cartopy: Python package designed for geospatial data processing <https://scitools.org.uk/cartopy/docs/latest/>`_
- `dask: Parallel computing with task scheduling <https://www.dask.org/>`_
- `ipyleaflet: Jupyter / Leaflet bridge enabling interactive maps <https://github.com/jupyter-widgets/ipyleaflet>`_
- `ipywidgets: interactive HTML widgets for Jupyter notebooks and IPython <https://ipywidgets.readthedocs.io/en/latest/>`_
- `jplephem: Astronomical Ephemeris for Python <https://pypi.org/project/jplephem/>`_
- `matplotlib: Python 2D plotting library <https://matplotlib.org/>`_
- `obstore: Simple, high-throughput Python interface for object storage <https://developmentseed.org/obstore>`_
- `pandas: Python Data Analysis Library <https://pandas.pydata.org/>`_
- `pyarrow: Apache Arrow Python bindings <https://arrow.apache.org/docs/python/>`_
- `s3fs: Pythonic file interface to S3 built on top of botocore <https://s3fs.readthedocs.io/en/latest/>`_
- `zarr: Chunked, compressed, N-dimensional arrays in Python <https://zarr.readthedocs.io/en/stable/>`_

Credits
#######

The Tidal Model Driver (TMD) Matlab Toolbox was developed by Laurie Padman, Lana Erofeeva and Susan Howard.
An updated version of the TMD Matlab Toolbox (TMD3) was developed by Chad Greene.
The OSU Tidal Inversion Software (OTIS) and OSU Tidal Prediction Software (OTPS) were developed by
Lana Erofeeva and Gary Egbert (`copyright OSU <http://volkov.oce.orst.edu/tides/COPYRIGHT.pdf>`_,
licensed for non-commercial use).
The NASA Goddard Space Flight Center (GSFC) PREdict Tidal Heights (PERTH3) software was developed by
Richard Ray and Remko Scharroo.
An updated and more versatile version of the NASA GSFC tidal prediction software (PERTH5) was developed by Richard Ray.

Data Citations
##############

Internally, ``pyTMD`` includes datasets from the following:

.. bibliography::
    :list: bullet
    :filter: False

    Cartwright:1971iz
    Cartwright:1973em
    Desai:2015jr
    Hartmann:1995jp
    Petit:2010tp
    Ray:2014fu
    Tamura:1987tp


Disclaimer
##########

This package includes software developed at NASA Goddard Space Flight Center (GSFC) and the University
of Washington Applied Physics Laboratory (UW-APL).
It is not sponsored or maintained by the Universities Space Research Association (USRA), AVISO or NASA.

.. warning::
    Outputs from this software should be used for scientific or technical purposes only.
    This software should not be used for coastal navigation or *any application that may risk life or property*.

.. |auml|    unicode:: U+00E4 .. LATIN SMALL LETTER A WITH DIAERESIS
.. |uuml|    unicode:: U+00FC .. LATIN SMALL LETTER U WITH DIAERESIS

