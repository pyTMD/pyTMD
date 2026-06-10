.. _time-zones:

Time Zones
==========

:ref:`Tide tables <high-low-water>` are often provided in local time, but ``pyTMD`` uses times in UTC for its internal calculations.
While the ``timescale`` library handles the conversion into UTC, it does not have the built-in functionality to export into different time zones.
Luckily, the ``datetime`` and ``zoneinfo`` libraries provide the tools necessary for this conversion.

.. plot:: ./user_guide/time-zones.py
    :include-source: True
    :align: center
