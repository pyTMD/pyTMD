====
NOAA
====

- Query and parsing functions for the `NOAA Tides and Currents webservices API <https://tidesandcurrents.noaa.gov/>`_

Calling Sequence
----------------

.. code-block:: python

    import pyTMD.io
    api = 'tidepredictionstations'
    xpath = pyTMD.io.NOAA._xpaths[api]
    url, namespaces = pyTMD.io.NOAA.build_query(api)
    stations = pyTMD.io.NOAA.from_xml(url, xpath=xpath, namespaces=namespaces)

`Source code`__

.. __: https://github.com/pyTMD/pyTMD/blob/main/pyTMD/io/NOAA.py

.. autofunction:: pyTMD.io.NOAA.build_query

.. autofunction:: pyTMD.io.NOAA.from_xml
