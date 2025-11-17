====
OTIS
====

- Reads OTIS format tidal solutions provided by Oregon State University and ESR

   * multi-constituent ``OTIS`` binary 
   * ``ATLAS-compact`` binary
   * single-constituent ``OTIS`` binary
   * ``TMD3`` consolidated netCDF4

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.io
    ds = pyTMD.io.OTIS.open_dataset(model_file, grid_file, format='OTIS', type='z')

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/io/OTIS.py

.. autofunction:: pyTMD.io.OTIS.open_dataset

.. autofunction:: pyTMD.io.OTIS.open_mfdataset

.. autofunction:: pyTMD.io.OTIS.open_otis_dataset

.. autofunction:: pyTMD.io.OTIS.open_atlas_dataset

.. autofunction:: pyTMD.io.OTIS.open_tmd3_dataset

.. autofunction:: pyTMD.io.OTIS.open_otis_grid

.. autofunction:: pyTMD.io.OTIS.open_otis_elevation

.. autofunction:: pyTMD.io.OTIS.open_otis_transport

.. autofunction:: pyTMD.io.OTIS.open_atlas_grid

.. autofunction:: pyTMD.io.OTIS.open_atlas_elevation

.. autofunction:: pyTMD.io.OTIS.open_atlas_transport

.. autofunction:: pyTMD.io.OTIS.read_raw_binary

.. autofunction:: pyTMD.io.OTIS.write_raw_binary

.. autoclass:: pyTMD.io.OTIS.OTISDataset
   :members:

.. autoclass:: pyTMD.io.OTIS.OTISDataTree
   :members:

.. autoclass:: pyTMD.io.OTIS.ATLASDataset
   :members:
