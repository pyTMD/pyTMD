====
NOAA
====

- Query and parsing functions for the `NOAA Tides and Currents webservices API <https://tidesandcurrents.noaa.gov/>`_

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.io
    stations = pyTMD.io.NOAA.prediction_stations()

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/io/NOAA.py

.. autofunction:: pyTMD.io.NOAA.build_query

.. autofunction:: pyTMD.io.NOAA.from_xml

.. autofunction:: pyTMD.io.NOAA.prediction_stations

.. autofunction:: pyTMD.io.NOAA.harmonic_constituents

.. autofunction:: pyTMD.io.NOAA.water_level
