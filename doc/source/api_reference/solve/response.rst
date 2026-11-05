========================
``pyTMD.solve.response``
========================

- Routines for estimating tidal constituents using the response method

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.solve
    ds = pyTMD.solve.response(time, lon, lat, h, constituents)

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/solve/response.py

.. autofunction:: pyTMD.solve.response

.. autofunction:: pyTMD.solve.response._build_design_matrix

.. autofunction:: pyTMD.solve.response._legendre_addition

.. autofunction:: pyTMD.solve.response._gravitational

.. autofunction:: pyTMD.solve.response._radiational

.. autofunction:: pyTMD.solve.response._kappa
