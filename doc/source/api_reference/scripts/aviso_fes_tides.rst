==================
aviso_fes_tides.py
==================

- Downloads the FES (Finite Element Solution) global tide model from `AVISO <https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html>`_
- Decompresses the model tar files into the constituent files and auxiliary files
- Must have `data access to tide models from AVISO <https://www.aviso.altimetry.fr/en/data/data-access.html>`_

.. note::
    *FES outputs are licensed for scientific purposes only*

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/scripts/aviso_fes_tides.py

Calling Sequence
################

.. argparse::
    :module: pyTMD.scripts.aviso_fes_tides
    :func: arguments
    :prog: aviso_fes_tides.py
    :nodescription:
    :nodefault:
