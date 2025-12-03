===
FES
===

- Reads Finite Element Solution (FES), Empirical Ocean Tide (EOT), and Hamburg direct data Assimilation Methods for Tides (HAMTIDE) models

   * ``FES-ascii``
   * ``FES-netcdf``

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.io
    ds = pyTMD.io.FES.open_mfdataset(model_files, group='z', format=format)

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/io/FES.py


.. autofunction:: pyTMD.io.FES.open_mfdataset

.. autofunction:: pyTMD.io.FES.open_fes_dataset

.. autofunction:: pyTMD.io.FES.open_fes_ascii

.. autofunction:: pyTMD.io.FES.open_fes_netcdf

.. autoclass:: pyTMD.io.FES.FESDataset
   :members:
