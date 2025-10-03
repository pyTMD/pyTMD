===============
verify_box_tpxo
===============

- Verifies TPXO9-atlas global tide models downloaded from the `box file sharing service <https://developer.box.com/guides/>`_
- Compares ``sha1`` hashes to verify the binary or netCDF4 files

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/datasets/verify_box_tpxo.py

.. autofunction:: pyTMD.datasets.verify_box_tpxo

CLI
===

.. argparse::
    :module: pyTMD.datasets.verify_box_tpxo
    :func: arguments
    :prog: verify_box_tpxo.py
    :nodescription:
    :nodefault:
