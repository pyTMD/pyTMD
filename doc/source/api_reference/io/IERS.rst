====
IERS
====

- Reads ocean pole load tide coefficients provided by IERS as computed by :cite:t:`Desai:2002ev` and :cite:t:`Desai:2015jr`
- See `materials from Chapter 7 of the IERS Conventions <https://webtai.bipm.org/iers/convupdt/convupdt_c7.html>`_
- `Ocean Pole Tide File <ftp://maia.usno.navy.mil/conventions/2010/2010_update/chapter7/additional_info/opoleloadcoefcmcor.txt.gz>`_

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.io
    import pyTMD.utilities
    ocean_pole_tide_file = pyTMD.utilities.get_data_path(['data','opoleloadcoefcmcor.txt.gz'])
    ur,un,ue,glon,glat = pyTMD.io.IERS.read_binary_file(model_file=ocean_pole_tide_file)

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/io/IERS.py

.. autofunction:: pyTMD.io.IERS.extract_coefficients

.. autofunction:: pyTMD.io.IERS.read_binary_file
