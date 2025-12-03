===
GOT
===

- Reads files for Richard Ray's Goddard Ocean Tide (GOT) models

   * ``GOT-ascii``
   * ``GOT-netcdf``

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.io
    ds = pyTMD.io.GOT.open_mfdataset(model_files, group='z', format=format)

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/io/GOT.py

.. autofunction:: pyTMD.io.GOT.open_mfdataset

.. autofunction:: pyTMD.io.GOT.open_got_dataset

.. autofunction:: pyTMD.io.GOT.open_got_ascii

.. autofunction:: pyTMD.io.GOT.open_got_netcdf

.. autoclass:: pyTMD.io.GOT.GOTDataset
   :members:
