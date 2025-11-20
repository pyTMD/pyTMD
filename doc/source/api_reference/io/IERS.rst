====
IERS
====

- Reads ocean pole load tide coefficients provided by IERS as computed by :cite:t:`Desai:2002ev` and :cite:t:`Desai:2015jr`
- See `materials from Chapter 7 of the IERS Conventions <https://webtai.bipm.org/iers/convupdt/convupdt_c7.html>`_

.. tip::
    `Ocean Pole Tide file <ftp://maia.usno.navy.mil/conventions/2010/2010_update/chapter7/additional_info/opoleloadcoefcmcor.txt.gz>`_
    is accessible from the US Naval Observatory (USNO)

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.io
    ds = pyTMD.io.IERS.open_dataset()

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/io/IERS.py

.. autofunction:: pyTMD.io.IERS.open_dataset
