=====
ATLAS
=====

- Reads netCDF format tidal solutions provided by Oregon State University and ESR
- Spatially interpolates tidal constituents to input coordinates

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.io
    amp,ph,D,c = pyTMD.io.ATLAS.extract_constants(ilon, ilat, grid_file, model_files,
       type='z',
       method='spline')

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/io/ATLAS.py

.. autofunction:: pyTMD.io.ATLAS.extract_constants

.. autofunction:: pyTMD.io.ATLAS.read_constants

.. autofunction:: pyTMD.io.ATLAS.interpolate_constants

.. autofunction:: pyTMD.io.ATLAS.read_netcdf_grid

.. autofunction:: pyTMD.io.ATLAS.read_netcdf_file

.. autofunction:: pyTMD.io.ATLAS.read_netcdf_elevation

.. autofunction:: pyTMD.io.ATLAS.read_netcdf_transport

.. autofunction:: pyTMD.io.ATLAS.output_netcdf_grid

.. autofunction:: pyTMD.io.ATLAS.output_netcdf_elevation

.. autofunction:: pyTMD.io.ATLAS.output_netcdf_transport

.. autofunction:: pyTMD.io.ATLAS._extend_array

.. autofunction:: pyTMD.io.ATLAS._extend_matrix

.. autofunction:: pyTMD.io.ATLAS._crop

.. autofunction:: pyTMD.io.ATLAS._shift
