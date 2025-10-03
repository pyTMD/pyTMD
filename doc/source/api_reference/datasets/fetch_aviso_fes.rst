===============
fetch_aviso_fes
===============

- Downloads the FES (Finite Element Solution) global tide model from `AVISO <https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html>`_
- Decompresses the model tar files into the constituent files and auxiliary files
- Must have `data access to tide models from AVISO <https://www.aviso.altimetry.fr/en/data/data-access.html>`_

.. note::
    *FES outputs are licensed for scientific purposes only*

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/datasets/fetch_aviso_fes.py

.. autofunction:: pyTMD.datasets.fetch_aviso_fes

CLI
===

.. argparse::
    :module: pyTMD.datasets.fetch_aviso_fes
    :func: arguments
    :prog: fetch_aviso_fes.py
    :nodescription:
    :nodefault:
