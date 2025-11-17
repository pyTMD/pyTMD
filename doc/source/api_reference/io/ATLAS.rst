=====
ATLAS
=====

- Reads ``ATLAS-netcdf`` tidal solutions provided by Oregon State University

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.io
    ds = pyTMD.io.ATLAS.open_dataset(model_files, grid_file, type='z')

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/io/ATLAS.py

.. autofunction:: pyTMD.io.ATLAS.open_dataset

.. autofunction:: pyTMD.io.ATLAS.open_mfdataset

.. autofunction:: pyTMD.io.ATLAS.open_atlas_grid

.. autofunction:: pyTMD.io.ATLAS.open_atlas_dataset

.. autoclass:: pyTMD.io.ATLAS.ATLASDataset
   :members:

.. autoclass:: pyTMD.io.ATLAS.ATLASDataTree
   :members:
